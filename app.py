import os
import queue
import random
import subprocess
import time
import math
import struct
from pathlib import Path
import wave
import requests
import re
import json
import shutil
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yt_dlp
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

# 引入情境、Pydantic 契約與後端決策核心
from senarios import SCENARIO_PRESETS
from schemas import AgentState, ActionType, UIAction, UIPayload, AgentResponsePayload
from engine import LiveInteractionEngine
from youtube_stream import YouTubeChatListener, extract_video_id, fetch_youtube_concurrent_viewers

# 自動刷新組件 (若環境未安裝則降級)
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ==============================================================================
# 載入金鑰 & 環境變數讀取
# ------------------------------------------------------------------------------
# 💡 API 運作機制：
#   1. Gemini API Key：提供主對話生成、繁中即時翻譯、情感批次校正與音訊多模態 STT 轉錄。
#   2. YouTube Data API v3 Key：提供頻道頭像、直播即時標題以及查詢當前在線人數。
# ⚠️ 易卡殼細節：
#   1. 使用 Path(__file__).parent.resolve() 鎖定絕對路徑，防止終端機 CWD 偏移找不到 .env。
#   2. 使用 try-except 包裹 st.secrets.get，避免本地無 secrets.toml 時噴 FileNotFoundError。
# ==============================================================================
current_dir = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=current_dir / "GEMINI_API_KEY.env")
load_dotenv(dotenv_path=current_dir / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

youtube_api_key = os.getenv("YOUTUBE_API_KEY")
if not youtube_api_key:
    try:
        youtube_api_key = st.secrets.get("YOUTUBE_API_KEY")
    except Exception:
        pass

rate_limit_tier = os.getenv("GEMINI_RATE_LIMIT_TIER", "Free")
if not rate_limit_tier:
    try:
        rate_limit_tier = st.secrets.get("GEMINI_RATE_LIMIT_TIER", "Free")
    except Exception:
        rate_limit_tier = "Free"

global_model_name = "models/gemini-3.5-flash-lite"

st.set_page_config(page_title="AI Agent FSM & Generative UI Demo", layout="wide")

# ==============================================================================
# CSS 注入：Hype-bot 專屬綠底與防反灰穿透
# ==============================================================================
st.markdown("""
<style>
/* 徹底關閉 Streamlit 運算時的元件反灰淡出效果，維持 SPA 流暢視覺 */
[data-testid="stAppViewBlockContainer"] [data-stale="true"],
.element-container:has([data-stale="true"]),
div[data-testid="stVerticalBlock"] > div[style*="opacity"] {
    opacity: 1 !important;
    filter: none !important;
    transition: none !important;
}

/* 🟢 Hype-bot 專屬深綠對話框底板 */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(16, 50, 40, 0.85) 0%, rgba(20, 65, 50, 0.95) 100%) !important;
    border: 1px solid #1E7E58 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    margin-bottom: 8px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) p {
    color: #F0FFF4 !important;
    margin-bottom: 0px !important;
}

/* ⚪ User (觀眾) 無底板純淨樣式 */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]),
div[data-testid="stChatMessage"]:has(img) {
    background: transparent !important;
    border: none !important;
    padding: 6px 4px !important;
    box-shadow: none !important;
}

.lang-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    background-color: #0E4833;
    color: #A3E635;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 6px;
    border: 1px solid #22C55E;
}

.translation-sub {
    font-size: 0.85rem;
    color: #86EFAC !important;
    border-top: 1px dashed rgba(255, 255, 255, 0.2);
    padding-top: 6px;
    margin-top: 6px;
}

/* User 訊息卡片懸停 (Hover) 才顯示翻譯按鈕 */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stButton"],
div[data-testid="stChatMessage"]:has(img) div[data-testid="stButton"] {
    opacity: 0 !important;
    transition: opacity 0.2s ease-in-out !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]):hover div[data-testid="stButton"],
div[data-testid="stChatMessage"]:has(img):hover div[data-testid="stButton"] {
    opacity: 1 !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 狀態機與全域 Session State 初始化
# ------------------------------------------------------------------------------
# ⚠️ 易卡殼細節：
#   1. LiveInteractionEngine 必須只初始化一次存入 session_state，避免頻繁重建洗掉節流時間戳。
#   2. 預設名單必須完整賦予 status="Active" 與 review_note，否則 HITL Console 會拋 KeyError。
# ==============================================================================
if api_key and "engine" not in st.session_state:
    st.session_state.engine = LiveInteractionEngine(
        api_key=api_key,
        model_name=global_model_name
    )
if "listener" not in st.session_state:
    st.session_state.listener = None
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "last_message_time" not in st.session_state:
    st.session_state.last_message_time = time.time()
if "message_timestamps" not in st.session_state:
    st.session_state.message_timestamps = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "author": "Hype-bot",
            "content": "Multilingual live chat is ready. Waiting for stream...",
            "translation": "Multilingual live chat is ready. Waiting for stream...",
            "lang": "en",
            "ui_actions": []
        }
    ]
if "current_fsm_state" not in st.session_state:
    st.session_state.current_fsm_state = AgentState.ACTIVE_INTERACTION
if "decision_trace" not in st.session_state:
    st.session_state.decision_trace = "System Standby"

# Dashboard 初始化
if "token_stats" not in st.session_state:
    st.session_state.token_stats = {
        "prompt_tokens": 0,
        "candidate_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0
    }

if "sentiment_logs" not in st.session_state:
    # 紀錄結構: [{"time": "20:01:10", "author": "UserA", "content": "好讚", "score": 2, "cumulative": 2}]
    st.session_state.sentiment_logs = []

#  確保若有預設範例或後續寫入時，都有 status 與 review_note
if "moderation_actions" not in st.session_state:
    st.session_state.moderation_actions = [
        {
            "time": "20:02:15",
            "author": "@SpamBot_99",
            "content": "點此連結領取免費禮券 https://bit.ly/claim-free",
            "action": "Permanent Ban",
            "reason": "Phishing / Spam Advertisement",
            "severity": "Critical",
            "status": "Active",
            "review_note": "Auto Enforced"
        },
        {
            "time": "20:04:20",
            "author": "@TrollMaster",
            "content": "全家死光爛主播 退費詐欺！",
            "action": "Permanent Ban",
            "reason": "Defamation / Fraud Allegation",
            "severity": "Critical",
            "status": "Active",
            "review_note": "Auto Enforced"
        }
    ]

# 環比指標基線快取
# Session State & Safty Monitoring 初始化
current_init_time = time.time()

# 1. Heat Score 5 分鐘環比基準
if "heat_history_base" not in st.session_state:
    st.session_state.heat_history_base = {
        "timestamp": current_init_time,
        "score": 0
    }
if "heat_delta_str" not in st.session_state:
    st.session_state.heat_delta_str = "0.0%"
# 2. Viewers 5 分鐘環比基準
if "viewers_history_base" not in st.session_state:
    st.session_state.viewers_history_base = {
        "timestamp": current_init_time,
        "count": 0
    }
if "viewers_delta_str" not in st.session_state:
    st.session_state.viewers_delta_str = "0"
# 3. Total Messages 1 分鐘環比基準
if "msg_history_base" not in st.session_state:
    st.session_state.msg_history_base = {
        "timestamp": current_init_time,
        "total": 0
    }
if "msg_delta_str" not in st.session_state:
    st.session_state.msg_delta_str = "+0 /min"

# Live chatroom 標題變動
if "stream_title" not in st.session_state:
    st.session_state.stream_title = ""
if "channel_title" not in st.session_state:
    st.session_state.channel_title = ""
if "channel_icon_url" not in st.session_state:
    st.session_state.channel_icon_url = ""
# 針對手動翻譯功能，記錄最後一次翻譯的時間戳
if "last_manual_translate_time" not in st.session_state:
    st.session_state.last_manual_translate_time = 0.0

# 追蹤音訊切片背景工作實體
if "audio_process" not in st.session_state:
    st.session_state.audio_process = None
if "audio_thread_stop_event" not in st.session_state:
    st.session_state.audio_thread_stop_event = threading.Event()

# 初始化 STT 切片資料庫與 3 分鐘自動評級計時器
if "stt_chunks" not in st.session_state:
    st.session_state.stt_chunks = []
if "audit_last_eval_time" not in st.session_state:
    st.session_state.audit_last_eval_time = time.time()
if "last_match_audit_status" not in st.session_state:
    st.session_state.last_match_audit_status = {
        "last_time_str": "Pending",
        "last_timestamp": 0.0,
        "state": "STANDBY",
        "detail": "Waiting for live stream connection and accumulating slices...",
        "evaluated_chunks_count": 0
    }
if "processed_chunks_set" not in st.session_state:
    st.session_state.processed_chunks_set = set()
if "last_stt_call_time" not in st.session_state:
    st.session_state.last_stt_call_time = 0.0
if "last_calibration_time" not in st.session_state:
    st.session_state.last_calibration_time = time.time()

# ==============================================================================
# 🛠️ 輔助函式庫 (表情清洗、語意評分、風控與 Token 統計)
# ------------------------------------------------------------------------------
# 💡 運作機制：
#   1. 表情清洗 (clean_message_text)：避免 Unicode 表情貼圖干擾正負向關鍵字匹配。
#   2. 第一階段關鍵字評分 (evaluate_message_sentiment)：快速判定五級分（+2, +1, -1, -2），0分略過。
#   3. 第二階段 AI 校正 (calibrate_sentiments_with_llm)：每 5 分鐘調用 Gemini 校正語境反諷。
#   4. 風控審查 (check_and_apply_moderation)：攔截釣魚外連與仇恨誹謗，自動熔斷並存檔。
#   5. LLMOps Token 統計 (update_token_metrics_from_response)：累積成本並反映於儀表板。
# ==============================================================================

# 補上清洗函式
def clean_message_text(text: str) -> str:
    """去除 YouTube 短碼與 Unicode 表情符號"""
    text_clean = re.sub(r":_?[a-zA-Z0-9_\-]+:", "", text)
    emoji_pattern = re.compile("[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50]+", flags=re.UNICODE)
    return emoji_pattern.sub("", text_clean).strip()

# 在此處插入五級分判定與動態寫入 sentiment_logs 的函式
def evaluate_message_sentiment(user: str, message: str) -> None:
    """第一階段：即時關鍵字匹配 (標記為 Matched)"""
    cleaned_msg = clean_message_text(message)
    if not cleaned_msg:
        return

    msg_lower = message.lower()
    score = 0
    
    # 1. 優先攔截否定句 (避免 "not good"、"よくない" 被後續的 "good"、"よい" 判定為正向)
    negation_patterns = [
        # English
        "not good", "not ok", "not great", "def not", "no good", "yeah not", "not worth", "isn't good",
        # 繁中 / 簡中
        "不好", "不行", "不太行", "不推", "不怎麼樣", "沒很好", "不是很", "不值得",
        # 日本語 (含平假名、口語縮寫)
        "よくない", "良くない", "よくねえ", "微妙", "ダメ", "だめ", "いまいち", "イマイチ", "微妙かも"
    ]

    if any(neg in msg_lower for neg in negation_patterns):
        score = -1
    else:
        # ----------------------------------------------------
        # 積極正向 (+2): 強烈推薦、狂熱、驚豔
        # ----------------------------------------------------
        positive_strong = [
            # English
            "great", "amazing", "clutch", "pog", "poggers", "hype", "love it", "must buy", "goat", "insane",
            # 中文
            "太讚", "神作", "必買", "超好用", "太強", "愛了", "太神", "推爆", "買爆", "太頂",
            # 日本語 (含直播彈幕流行語)
            "最高", "神回", "神ゲー", "神作", "天才", "すげえ", "凄すぎる", "買い確定", "うますぎ", "尊い"
        ]

        # ----------------------------------------------------
        # 正向 (+1): 一般認同、良好反饋、直播助興
        # ----------------------------------------------------
        positive_weak = [
            # English
            "good", "nice", "cool", "ok", "fine", "neat", "liked", "gg", "w", "naisu",
            # 中文
            "推", "不錯", "讚", "好看", "可以", "好用", "舒服", "滿意", "有料", "挺好",
            # 日本語
            "よい", "良い", "いいね", "すき", "好き", "よき", "ナイス", "助かる", "草", "かわいい", "うまい", "おめ", "草", "ｗ", "えぐ", "エグイ"
        ]

        # ----------------------------------------------------
        # 極度負向 (-2): 欺詐、惡意攻擊、重大產品質量問題
        # ----------------------------------------------------
        negative_strong = [
            # English
            "scam", "trash", "fake", "terrible", "worst", "fraud", "horrible", "refund", "garbage",
            # 中文
            "爛", "退費", "詐欺", "騙錢", "難用", "垃圾", "騙人", "快逃", "太爛", "黑心",
            # 日本語
            "詐欺", "最悪", "ゴミ", "返金", "クソ", "くそ", "役立たず", "二度と買わん", "通報"
        ]

        # ----------------------------------------------------
        # 負向 (-1): 規格或體驗不符預期、速度慢、抱怨
        # ----------------------------------------------------
        negative_weak = [
            # English
            "meh", "expensive", "cooked", "slow", "bad", "mid", "lag", "bug", "l",
            # 中文
            "普通", "還好", "有點貴", "太慢", "偏貴", "卡頓", "延遲", "失望", "一般般",
            # 日本語
            "高い", "遅い", "重い", "ラグい", "微妙", "バグ", "がっかり", "いらね", "微妙だな"
        ]

        if any(k in msg_lower for k in positive_strong):
            score = 2
        elif any(k in msg_lower for k in positive_weak):
            score = 1
        elif any(k in msg_lower for k in negative_strong):
            score = -2
        elif any(k in msg_lower for k in negative_weak):
            score = -1
        else:
            score = 0

    if score == 0:
        return

    prev_total = st.session_state.sentiment_logs[-1]["cumulative"] if st.session_state.sentiment_logs else 0
    new_total = prev_total + score
    now_ts = time.time()
    time_str = time.strftime("%H:%M:%S", time.localtime(now_ts))

    st.session_state.sentiment_logs.append({
        "timestamp_raw": now_ts,     # 用於 5 分鐘窗口篩選
        "time": time_str,
        "author": user,
        "content": message,
        "score": score,
        "cumulative": new_total,
        "state": "Matched"           # 初始狀態：Matched
    })

# 校正後重新計算整張表的 Total Score，確保圖表與走勢完全對齊
def recompute_cumulative_scores() -> None:
    """校正後重新計算整張表的 Total Score，確保圖表與走勢完全對齊"""
    running_total = 0
    for row in st.session_state.sentiment_logs:
        running_total += row["score"]
        row["cumulative"] = running_total

# 提取 Gemini API 回傳的 Token 數據並累加至 session_state
def update_token_metrics_from_response(res, viewer_message: str = "") -> None:
    """提取 Gemini API 回傳的 Token 數據並累加至 session_state"""
    if not res:
        return

    # 1. 嘗試從各種路徑讀取 usage_metadata
    usage = getattr(res, "usage_metadata", None)
    if not usage and hasattr(res, "__dict__"):
        usage = res.__dict__.get("usage_metadata")
    if not usage and hasattr(res, "response"):
        usage = getattr(res.response, "usage_metadata", None)

    p_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
    c_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

    # 2. 保底估算 (若 Fallback 或尚未取到 usage)
    if p_tok == 0 and c_tok == 0:
        msg_len = len(str(viewer_message))
        p_tok = 1200 + max(int(msg_len * 1.5), 10)
        reply_len = len(str(getattr(res, "display_message", "")))
        c_tok = max(int(reply_len * 1.8), 80)

    # 3. 寫入 Session State
    st.session_state.token_stats["prompt_tokens"] += p_tok
    st.session_state.token_stats["candidate_tokens"] += c_tok
    st.session_state.token_stats["total_tokens"] += (p_tok + c_tok)

    # 4. 依 Flash 官方費率累加成本 (輸入 $0.075/1M, 輸出 $0.30/1M)
    call_cost = (p_tok * 0.000000075) + (c_tok * 0.00000030)
    st.session_state.token_stats["estimated_cost_usd"] += call_cost

# 實時安全審查與風控裁決引擎 (Trust & Safety Engine)
def check_and_apply_moderation(user: str, message: str) -> bool:
    """
    依據 4 大危險語境判定並即時處置：
    回傳 True 代表觸發處置 (違規)；False 代表正常發言。
    """
    msg_lower = message.lower()
    now_ts = time.time()
    time_str = time.strftime("%H:%M:%S", time.localtime(now_ts))
    action, reason, severity = None, None, "Normal"

    # 1. 惡意廣告與釣魚外連 (Regex 偵測網址或特定通訊行銷詞彙)
    url_pattern = r"(https?://\S+|bit\.ly/\S+|www\.\S+|\.com|\.xyz|加賴|私訊領)"
    if re.search(url_pattern, msg_lower):
        action = "Permanent Ban"
        reason = "Phishing / Spam Advertisement"
        severity = "Critical"

    # 2. 商業欺詐與惡意毀謗
    elif any(k in msg_lower for k in ["退費詐欺", "scam", "fraud", "fake", "騙局", "黑心", "假貨", "騙錢"]):
        action = "Permanent Ban"
        reason = "Defamation / Fraud Allegation"
        severity = "Critical"

    # 3. 仇恨謾罵與人身攻擊
    elif any(k in msg_lower for k in ["全家", "去死", "垃圾主播", "kys", "trash streamer", "fuck", "bitch"]):
        action = "Mute 30 Mins"
        reason = "Toxicity & Harassment"
        severity = "High"

    # 4. 若觸發違規，寫入風控處置名單 (Audit Log)
    if action:
        # 避免重複懲處同一位已在名單上的用戶
        existing_bans = [m["author"] for m in st.session_state.moderation_actions if m["status"] == "Active"]
        if user not in existing_bans:
            st.session_state.moderation_actions.append({
                "time": time_str,
                "author": user,
                "content": message,
                "action": action,
                "reason": reason,
                "severity": severity,
                "status": "Active",            # Active (生效中) / Cleared (已人工撤回)
                "review_note": "Auto Enforced" # 審計備註
            })
        return True
    return False

# 每 5 分鐘批次校正過去 5 分鐘內標記為 'Matched' 的紀錄
def calibrate_sentiments_with_llm(client, model_name: str) -> bool:
    """第二階段：每 5 分鐘批次校正過去 5 分鐘內標記為 'Matched' 的紀錄"""
    now = time.time()
    # 篩選過去 300 秒且狀態為 Matched 的項目索引
    target_indices = [
        idx for idx, row in enumerate(st.session_state.sentiment_logs)
        if row.get("state") == "Matched" and (now - row.get("timestamp_raw", now) <= 300)
    ]

    if not target_indices:
        return False

    # 組裝批次校正 Prompt
    eval_payload = [
        {"id": idx, "message": st.session_state.sentiment_logs[idx]["content"]}
        for idx in target_indices
    ]

    prompt = f"""
You are a precise live chat sentiment analysis evaluator.
Evaluate the exact sentiment for each viewer message based on true context and nuances (e.g. sarcasm, negations like 'not good', slang like 'cooked'):
Scores MUST be an integer:
+2: Highly positive
+1: Positive
 0: Neutral / Irrelevant
-1: Negative
-2: Highly negative

Return strict JSON array of objects:
[
  {{"id": 0, "score": -1}},
  ...
]

Messages to evaluate:
{json.dumps(eval_payload, ensure_ascii=False)}
"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.1}
        )
        
        # 讀取並累加 Token
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            p_tok = response.usage_metadata.prompt_token_count or 0
            c_tok = response.usage_metadata.candidates_token_count or 0
            st.session_state.token_stats["prompt_tokens"] += p_tok
            st.session_state.token_stats["candidate_tokens"] += c_tok
            st.session_state.token_stats["total_tokens"] += (p_tok + c_tok)
            st.session_state.token_stats["estimated_cost_usd"] += (p_tok * 0.000000075) + (c_tok * 0.00000030)

        calibrated_results = json.loads(response.text)
        updated = False

        for item in calibrated_results:
            msg_id = item.get("id")
            ai_score = int(item.get("score", 0))
            if msg_id in target_indices:
                row = st.session_state.sentiment_logs[msg_id]
                # 標記已校正
                row["state"] = "Calibrated"
                # 若判定不同，更新分數
                if row["score"] != ai_score:
                    row["score"] = ai_score
                    updated = True

        # 若有分數異動，全表重算累積 Total Score
        recompute_cumulative_scores()
        return updated
    except Exception as e:
        st.session_state.decision_trace = f"[AI CALIBRATION ERROR] {str(e)}"
        return False

# 獲取直播的 Video Title、Channel Title 與頻道頭像 URL
def fetch_youtube_stream_details(api_key: str, video_id: str) -> dict:
    """獲取直播的 Video Title、Channel Title 與頻道頭像 URL"""
    if not api_key or not video_id:
        return {}
    
    try:
        # 1. 抓取影片詳情 (包含 video title 與 channelId)
        video_url = "https://www.googleapis.com/youtube/v3/videos"
        v_params = {
            "part": "snippet",
            "id": video_id,
            "key": api_key
        }
        v_resp = requests.get(video_url, params=v_params, timeout=5).json()
        
        items = v_resp.get("items", [])
        if not items:
            return {}
            
        snippet = items[0].get("snippet", {})
        stream_title = snippet.get("title", "Live Stream")
        channel_title = snippet.get("channelTitle", "YouTuber")
        channel_id = snippet.get("channelId", "")

        # 2. 抓取頻道詳情以獲取高解析度 Profile Icon
        channel_icon_url = ""
        if channel_id:
            c_url = "https://www.googleapis.com/youtube/v3/channels"
            c_params = {
                "part": "snippet",
                "id": channel_id,
                "key": api_key
            }
            c_resp = requests.get(c_url, params=c_params, timeout=5).json()
            c_items = c_resp.get("items", [])
            if c_items:
                channel_icon_url = c_items[0].get("snippet", {}).get("thumbnails", {}).get("default", {}).get("url", "")

        return {
            "stream_title": stream_title,
            "channel_title": channel_title,
            "channel_icon_url": channel_icon_url
        }
    except Exception as e:
        print(f"Error fetching stream details: {e}")
        return {}

# 手動請求翻譯單條訊息 (使用輕量 Flash 模型)
def translate_single_message(text: str) -> str:
    """手動請求專用：使用輕量 Flash 模型將外文即時翻為繁體中文"""
    if not text or not api_key:
        return ""
    
    # 檢查全域 2.0 秒防連點機制
    now = time.time()
    last_call = st.session_state.get("last_manual_translate_time", 0.0)
    if now - last_call < 2.0:
        st.toast(f"⏳ 請求過於頻繁，請等待 {2.0 - (now - last_call):.1f} 秒後再試", icon="⚠️")
        return ""

    # 記錄本次請求時間戳記
    st.session_state.last_manual_translate_time = now

    prompt = f"Translate this live chat comment into fluent Traditional Chinese (colloquial style). Return ONLY the translated string:\nComment: {text}"
    try:
        if "engine" in st.session_state and hasattr(st.session_state.engine, "client"):
            client = st.session_state.engine.client
            model = st.session_state.engine.model
        else:
            client = genai.Client(api_key=api_key)
            model = global_model_name

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.2}
        )

        # 同步計入 LLMOps Token 統計
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            p_tok = response.usage_metadata.prompt_token_count or 0
            c_tok = response.usage_metadata.candidates_token_count or 0
            st.session_state.token_stats["prompt_tokens"] += p_tok
            st.session_state.token_stats["candidate_tokens"] += c_tok
            st.session_state.token_stats["total_tokens"] += (p_tok + c_tok)
            st.session_state.token_stats["estimated_cost_usd"] += (p_tok * 0.000000075) + (c_tok * 0.00000030)

        return response.text.strip()
    except Exception as e:
        st.toast(f"翻譯失敗: {str(e)}", icon="❌")
        return ""

# 預算治理策略動態決策器 (Budget Policy Decision Engine)
def get_budget_governance_policy(current_cost: float, budget_cap: float) -> dict:
    ratio = current_cost / budget_cap if budget_cap > 0 else 1.0
    
    if ratio >= 1.0:
        return {
            "level": "CRITICAL",
            "fsm_target": AgentState.VIOLATION_CIRCUIT_BREAKER,
            "allow_llm": False,
            "min_interval": 999.0,
            "badge": "🚨 100% Circuit Breaker",
            "description": "Already cut off external large model API calls, activating purely local rule backup."
        }
    elif ratio >= 0.85:
        return {
            "level": "HIGH",
            "fsm_target": AgentState.ACTIVE_INTERACTION,
            "allow_llm": True,
            "min_interval": 10.0,  # 嚴格冷卻
            "badge": "⚠️ 85% Cost Throttling",
            "description": "Extended cooldown interval lengthened to 10 seconds, processing only VIP or critical risk control events."
        }
    elif ratio >= 0.60:
        return {
            "level": "MEDIUM",
            "fsm_target": AgentState.ACTIVE_INTERACTION,
            "allow_llm": True,
            "min_interval": 5.0,   # 一般冷卻
            "badge": "🟡 60% Cost Warning",
            "description": "Reduced Prompt history length, compressing Token consumption."
        }
    else:
        return {
            "level": "NOMINAL",
            "fsm_target": AgentState.ACTIVE_INTERACTION,
            "allow_llm": True,
            "min_interval": 2.0,   # 極速響應
            "badge": "✅ Normal Operation",
            "description": "Budget water level is sufficient, full functionality real-time interaction."
        }

# ==============================================================================
# 雙軌音訊切片管線 (Real Stream vs Virtual Simulation)
# ------------------------------------------------------------------------------
# 💡 運作機制：
#   1. Real Mode (本機真實抓流)：透過 yt-dlp 解析串流並叫起 FFmpeg 進行 16kHz Mono PCM WAV 切片。
#   2. Virtual Simulation (雲端 Demo 推薦)：純 Python 生成合法 PCM WAV 音訊檔，不受 YouTube 機房 429 封鎖。
# ⚠️ 易卡殼細節：
#   1. 跨平台進程終止：Windows 必須使用 taskkill 砍除樹狀進程；Linux 則調用 proc.kill()，避免 File Handle 鎖定。
#   2. 單一定義原則：杜絕多重宣告 get_live_audio_url 導致 Cookies 被覆蓋的邏輯漏洞。
# ==============================================================================

# 調用本機 yt-dlp 抓取 YouTube 直播音訊 m3u8 串流
def get_live_audio_url_native(youtube_url: str) -> str:
    """調用本機 yt-dlp 抓取 YouTube 直播音訊 m3u8 串流"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'skip_download': True,
    }
    cookies_str = None
    try:
        cookies_str = st.secrets.get("YOUTUBE_COOKIES")
    except Exception:
        pass

    if cookies_str and cookies_str.strip():
        cookie_file_path = "yt_cookies.txt"
        with open(cookie_file_path, "w", encoding="utf-8") as f:
            f.write(cookies_str.strip())
        ydl_opts['cookiefile'] = cookie_file_path

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

# 喚起背景全域 FFmpeg 子程序進行音訊切片
def start_audio_slicing_pipeline_real(audio_stream_url: str, segment_seconds: int = 20, output_dir: str = "audio_chunks"):
    """叫起背景全域 FFmpeg 子程序進行音訊切片"""
    os.makedirs(output_dir, exist_ok=True)
    out_pattern = os.path.join(output_dir, "chunk_%04d.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", audio_stream_url,
        "-vn",
        "-af", "pan=mono|c0=0.5*c0+0.5*c1",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-segment_format", "wav",
        "-reset_timestamps", "1",
        out_pattern
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return process

# 純 Python 產生合法 16kHz PCM MAV 音訊檔 (零外部二進為工具依賴)
def generate_synthetic_speech_wav(filepath: Path, duration_secs: int = 15):
    """純 Python 產生合法 16kHz PCM WAV 音訊檔 (零外部二進位工具依賴)"""
    sample_rate = 16000
    n_samples = sample_rate * duration_secs
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        raw_data = bytearray()
        for i in range(n_samples):
            val = int(2500 * math.sin(2 * math.pi * 320 * (i / sample_rate)) * (0.6 + 0.4 * math.sin(2 * math.pi * 3 * (i / sample_rate))))
            raw_data.extend(struct.pack('<h', val))
        wf.writeframes(raw_data)

# 純 Python 虛擬串流排程器 (Streamlit Cloud 100% 穩定避開 YouTube 風控)
def start_virtual_audio_simulation(output_dir: str = "audio_chunks", interval_secs: int = 20, stop_event: threading.Event = None):
    """純 Python 虛擬串流排程器 (Streamlit Cloud 100% 穩定避開 YouTube 風控)"""
    os.makedirs(output_dir, exist_ok=True)

    def _worker():
        idx = 0
        while not (stop_event and stop_event.is_set()):
            chunk_file = Path(output_dir) / f"chunk_{idx:04d}.wav"
            generate_synthetic_speech_wav(chunk_file, duration_secs=15)
            idx += 1
            for _ in range(interval_secs):
                if stop_event and stop_event.is_set():
                    break
                time.sleep(1)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread

# 徹底終止 FFmpeg 進程並釋放檔案鎖定 (免手動在 PowerShell 執行 taskkill)
def force_kill_ffmpeg_safely(proc=None):
    """
    徹底終止 FFmpeg 進程並釋放檔案鎖定 (免手動在 PowerShell 執行 taskkill)
    """
    # 1. 精準針對當前 Popen 物件終止進程樹
    if proc and hasattr(proc, "pid"):
        try:
            if os.name == "nt":  # Windows 環境
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False
                )
            else:  # Linux / Streamlit Cloud 環境
                proc.kill()
        except Exception:
            pass

    # 2. Windows 本機保底：徹底清除所有因重複點擊而殘留的孤兒 ffmpeg.exe
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/IM", "ffmpeg.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

    # 關鍵緩衝：讓 Windows 核心釋放 File Handle，避免隨後的安全刪除噴 PermissionError
    time.sleep(0.3)

# 跨平台安全終止子程序 (Windows taskkill + Linux kill)
def force_kill_process_cross_platform(proc):
    """跨平台安全終止子程序 (Windows taskkill + Linux kill)"""
    if not proc:
        return
    try:
        if os.name == 'nt':
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    time.sleep(0.2)

# 安全清空目錄內容，略過被作業系統占用的殘留檔案
def safe_purge_directory(dir_path: Path):
    """安全清空目錄內容，略過被作業系統佔用的殘留檔案"""
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        return
    for item in dir_path.glob("*"):
        try:
            if item.is_file():
                item.unlink(missing_ok=True)
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

# ==============================================================================
# STT Chunk 與 Chatroom Msg 關聯評分與跨區向前回溯演算法 (3 mins 週期)
# ------------------------------------------------------------------------------
# 💡 運作機制：
#   1. Gemini Multimodal Ingestion：直接將音訊二進位 bytes 送給 Gemini 轉錄原始逐字稿。
#   2. 跨時段分桶對齊：扣除 7 秒直播網路延遲，將切片時段與觀眾彈幕分桶對照，結算主播話術引起的正負向反應。
# ==============================================================================

# 本機檔案儲存模組 STT Chunk Log & Chatroom Msg Log (JSON)
BASE_DIR = Path(__file__).parent.resolve()
DIR_STT_LOG = BASE_DIR / "STT Chunk Log"
DIR_CHAT_LOG = BASE_DIR / "Chatroom Msg Log"
# 自動建立目錄
DIR_STT_LOG.mkdir(parents=True, exist_ok=True)
DIR_CHAT_LOG.mkdir(parents=True, exist_ok=True)

# 存放單個 STT 切片至 'STT Chunk Log' 目錄下
def save_stt_chunk_log(chunk: dict):
    """將單個 STT 切片及其評級資訊存至 'STT Chunk Log' 目錄下"""
    try:
        chunk_id = chunk.get("chunk_id", f"chunk_{int(time.time())}")
        file_path = DIR_STT_LOG / f"{chunk_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save STT Chunk log: {e}")

# 將觀眾留言批次追加存入 'Chatroom Msg Log' 的 JSON 檔中
def append_chatroom_msg_log(messages: list):
    """將觀眾留言批次追加存入 'Chatroom Msg Log' 檔案中 (JSONL 格式)"""
    if not messages:
        return
    try:
        date_str = time.strftime("%Y%m%d")
        file_path = DIR_CHAT_LOG / f"live_chat_{date_str}.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            for msg in messages:
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "author": msg.get("author", ""),
                    "content": msg.get("message", msg.get("content", "")),
                    "avatar": msg.get("avatar", "")
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to save Chatroom Msg log: {e}")

# 使用 Gemini 多模態 API 將 20s 切片音檔轉錄為逐字稿
def transcribe_audio_chunk_with_gemini(audio_path: Path) -> str:
    """使用 Gemini 多模態 API 將 20s 切片音檔轉錄為原始語言逐字稿"""
    if not api_key:
        return "(未配置 GEMINI_API_KEY)"
    try:
        # 1. 統一客戶端實體
        if "engine" in st.session_state and hasattr(st.session_state.engine, "client"):
            client = st.session_state.engine.client
        else:
            client = genai.Client(api_key=api_key)

        # 2. 統一模型名稱：使用穩定支援多模態的 global_model_name
        model_name = global_model_name

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        if len(audio_bytes) < 10_000:
            return "(主播靜默 / 背景音樂)"

        prompt = """
You are a livestream speech-to-text transcriber.
Transcribe the speech verbatim in its original spoken language (e.g. Japanese, Chinese, English).
If no intelligible human voice can be heard, return "".
Output ONLY the transcription text without quotation marks or notes.
"""
        response = None
        # 針對 429 (ResourceExhausted) 與 503 加入重試防護
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                        prompt
                    ]
                )
                if response and response.text:
                    break
            except Exception as req_err:
                err_msg = str(req_err)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    time.sleep(3.0 * (attempt + 1))
                elif "503" in err_msg or "UNAVAILABLE" in err_msg:
                    time.sleep(1.5)
                else:
                    raise req_err

        if response and response.text:
            return response.text.strip()
        else:
            return "(無語音或背景音樂)"

    except Exception as e:
        err_str = str(e)
        if "429" in err_str:
            return "(STT 失敗: 429 配額頻率超限)"
        return f"(STT 失敗: {type(e).__name__} - {err_str[:25]})"

# 掃描 audio_chunks 目錄，找出尚未被 STT 辨識的切片進行轉錄
def process_pending_audio_chunks():
    """掃描 audio_chunks 目錄，找出尚未被 STT 辨識的切片進行轉錄"""
    now = time.time()
    # 關鍵保護：距離上次 STT 呼叫未滿 45 秒，直接退出，不浪費 RPM！
    if now - st.session_state.get("last_stt_call_time", 0.0) < 45.0:
        return

    audio_dir = Path("audio_chunks")
    if not audio_dir.exists():
        return

    if "processed_chunks_set" not in st.session_state:
        st.session_state.processed_chunks_set = set()

    wav_files = sorted(list(audio_dir.glob("chunk_*.wav")))
    # 排除當前正在被 FFmpeg 寫入的最後一片
    completed_wavs = wav_files[:-1] if len(wav_files) > 1 else []
    # 先快速檢查是否有未處理的切片，若全部都已處理完，直接 return！
    new_chunks = [w for w in completed_wavs if w.stem not in st.session_state.processed_chunks_set]
    if not new_chunks:
        return

    target_wav = new_chunks[-1]
        # 檔案大小未達標 (尚未寫入完畢) 則跳過
    try:
        if target_wav.stat().st_size < 10_000:
            return
    except OSError:
        return
    chunk_id = target_wav.stem
    st.session_state.last_stt_call_time = now  # 鎖定冷卻時間戳

    # 執行轉錄
    text = transcribe_audio_chunk_with_gemini(target_wav)
    file_mtime = target_wav.stat().st_mtime
    chunk_data = {
        "chunk_id": chunk_id,
        "start_ts": file_mtime - 180.0,
        "end_ts": file_mtime,
        "time_window": f"{time.strftime('%H:%M:%S', time.localtime(file_mtime - 180.0))} - {time.strftime('%H:%M:%S', time.localtime(file_mtime))}",
        "text": text if text else "(主播靜默 / 背景音樂)",
        "pos_count": 0,
        "neg_count": 0,
        "net_score": 0,
        "status": "Pending Alignment"
    }

    # 寫入狀態與磁碟
    st.session_state.stt_chunks.append(chunk_data)
    st.session_state.processed_chunks_set.add(chunk_id)
    save_stt_chunk_log(chunk_data)

# 與 app.py 有嚴重出入，待處理
def evaluate_chunk_chat_correlation():
    """利用時間戳倒排索引將彈幕分桶，並調用 Gemini 批次結算 3 分鐘切片評級"""
    now_ts = time.time()
    time_str = time.strftime("%H:%M:%S", time.localtime(now_ts))

    # 1. 條件檢查：若無切片或無留言，記錄為略過
    if not st.session_state.get("stt_chunks") or not st.session_state.get("chat_history"):
        st.session_state.last_match_audit_status.update({
            "last_time_str": time_str,
            "last_timestamp": now_ts,
            "state": "SKIPPED_NO_DATA",
            "detail": "切片或留言緩衝區為空，略過本次評級",
            "evaluated_chunks_count": 0
        })
        st.session_state.audit_last_eval_time = now_ts
        return

    user_msgs = [c for c in st.session_state.chat_history if c.get("role") == "user"]
    if not user_msgs:
        st.session_state.last_match_audit_status.update({
            "last_time_str": time_str,
            "last_timestamp": now_ts,
            "state": "SKIPPED_NO_USER_MSG",
            "detail": "3 分鐘內無觀眾發言，略過評估",
            "evaluated_chunks_count": 0
        })
        st.session_state.audit_last_eval_time = now_ts
        return

    chunks = st.session_state.stt_chunks
    LOOKBACK_DELAY = 7.0

    # 1. 建立各 Chunk 的彈幕分桶 (Bucket)
    chunk_buckets = {c["chunk_id"]: [] for c in chunks}

    for msg in user_msgs:
        # 讀取留言接收時間戳 (若無則以當前時間保底)
        msg_ts = msg.get("timestamp_raw", time.time())
        est_spoken_ts = msg_ts - LOOKBACK_DELAY

        # 改為安全讀取防護：
        matched = False
        for idx, chunk in enumerate(chunks):
            start_ts = chunk.get("start_ts", 0)
            end_ts = chunk.get("end_ts", 0)
            if start_ts <= est_spoken_ts <= end_ts:
                chunk_buckets[chunk["chunk_id"]].append(msg["content"])
                matched = True
                break
        
        # 若未命中且距離上一個 chunk 結束時間在 10 秒內，向前回溯歸入前一個 Chunk
        if not matched and len(chunks) >= 2:
            for idx in range(1, len(chunks)):
                prev_end = chunks[idx-1].get("end_ts", 0)
                if 0 <= (est_spoken_ts - prev_end) <= 10.0:
                    chunk_buckets[chunks[idx-1]["chunk_id"]].append(msg["content"])
                    break

    # 2. 挑選需要評估的切片 (最近 9 片，相當於過去 3 分鐘)
    recent_eval_payload = []
    for chunk in chunks[-9:]:
        cid = chunk["chunk_id"]
        chunk_text = chunk.get("text", "")

        # 若該切片為靜音/純音樂，直接在本地結算，不送給 Gemini
        if not chunk_text or any(empty_mark in chunk_text for empty_mark in ["(無語音", "(主播靜默", "背景音樂"]):
            chunk["pos_count"] = 0
            chunk["neg_count"] = 0
            chunk["net_score"] = 0
            chunk["status"] = "No Speech (Skipped)"
            # 同步將此狀態回寫至 STT Chunk Log
            save_stt_chunk_log(chunk)
            continue

        # 真正有主播發言的切片，才抓取對應彈幕打包
        msgs_in_bucket = chunk_buckets.get(cid, [])[-6:]
        recent_eval_payload.append({
            "chunk_id": cid,
            "host_speech": chunk_text[:80],
            "viewer_reactions": msgs_in_bucket
        })

    # 若這 3 分鐘內主播完全沒說話，則略過後續的 LLM 呼叫
    if not recent_eval_payload:
        st.session_state.last_match_audit_status.update({
            "last_time_str": time_str,
            "last_timestamp": now_ts,
            "state": "NO_HOST_SPEECH",
            "detail": "區間內切片皆為靜音或音樂，已在本地自動歸零，略過 LLM 呼叫",
            "evaluated_chunks_count": 0
        })
        st.session_state.audit_last_eval_time = now_ts
        return

    # 3. 呼叫 Gemini 進行批次語意結算 (安全防禦版)
    res_map = {}  # 預先在外面初始化，保證下方讀取時絕不噴 UnboundLocalError
    is_fallback_used = False

    try:
        client = st.session_state.engine.client if "engine" in st.session_state else genai.Client(api_key=api_key)
        model = st.session_state.engine.model if "engine" in st.session_state else global_model_name

        prompt = f"""
You are an expert livestream audience sentiment alignment judge.
Below is a sequence of host speech chunks and corresponding viewer reactions bucketed by latency windows.

Scoring Rules per Viewer Message:
1. Base Engagement: Every valid message gets +1 simply for speaking/participating.
2. Positive Sentiment: Additional +1 bonus (Total = +2 for this message).
3. Negative Sentiment: Counts as -1 penalty (Total = -1 for this message).
4. Neutral Sentiment: Only gets base +1 (Total = +1 for this message).

Calculate for each chunk:
- pos_count: Number of positive messages.
- neg_count: Number of negative messages.
- net_score: Sum of all message scores for this chunk (sum of base engagement + sentiment modifications).

Return ONLY a strict JSON array of objects:
[
  {{"chunk_id": "chunk_0001", "pos_count": 3, "neg_count": 1, "net_score": 5}}
]

Data to evaluate:
{json.dumps(recent_eval_payload, ensure_ascii=False)}
"""
        response = None
        # 指數退避重試最多 3 次 (對抗 503 與連線抖動)
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={"response_mime_type": "application/json", "temperature": 0.1}
                )
                if response and response.text:
                    break
            except Exception as net_err:
                time.sleep(1.5 * (attempt + 1))
                if attempt == 2:
                    raise net_err

        if response and response.text:
            update_token_metrics_from_response(response, viewer_message="[BATCH_AUDIT_EVAL]")
            results = json.loads(response.text)
            res_map = {item["chunk_id"]: item for item in results}
        else:
            raise RuntimeError("API returned empty response.")

    except Exception as eval_err:
        # 當連線或解析失敗時，啟動本機規則保底填入 res_map
        is_fallback_used = True
        print(f"[Audit Fallback] 批次對齊進入本地保底: {eval_err}")
        for item in recent_eval_payload:
            cid = item["chunk_id"]
            pos = 0
            neg = 0
            chunk_total_score = 0
            
            for text in item["viewer_reactions"]:
                t_low = text.lower()
                # 規則 3: 只要有發話，基礎分 +1
                msg_score = 1
                # 規則 1 & 2: 正負向判定
                if any(w in t_low for w in [
                    # English
                    "good", "nice", "cool", "ok", "fine", "neat", "liked", "gg", "w", "naisu",
                    # 中文
                    "推", "不錯", "讚", "好看", "可以", "好用", "舒服", "滿意", "有料", "挺好",
                    # 日本語
                    "よい", "良い", "いいね", "すき", "好き", "よき", "ナイス", "助かる", "草", "かわいい", "うまい", "おめ", "草", "ｗ", "えぐ", "エグイ"
                ]):
                    pos += 1
                    msg_score += 1  # 累加 +1，總計 +2
                elif any(w in t_low for w in [
                    # English
                    "scam", "trash", "fake", "terrible", "worst", "fraud", "horrible", "refund", "garbage",
                    # 中文
                    "爛", "退費", "詐欺", "騙錢", "難用", "垃圾", "騙人", "快逃", "太爛", "黑心",
                    # 日本語
                    "詐欺", "最悪", "ゴミ", "返金", "クソ", "くそ", "役立たず", "二度と買わん", "通報"
                ]):
                    neg += 1
                    msg_score = -1  # 負面覆蓋為 -1

                chunk_total_score += msg_score

            res_map[cid] = {
                "chunk_id": cid,
                "pos_count": pos,
                "neg_count": neg,
                "net_score": chunk_total_score
            }

    # 4. 更新評分結果並持久化存檔 (此時 res_map 必不為 None)
    eval_count = 0
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in res_map:
            chunk["pos_count"] = res_map[cid].get("pos_count", 0)
            chunk["neg_count"] = res_map[cid].get("neg_count", 0)
            chunk["net_score"] = res_map[cid].get("net_score", 0)
            chunk["status"] = "Evaluated (Fallback)" if is_fallback_used else "Evaluated"
            save_stt_chunk_log(chunk)
            eval_count += 1

    # 5. 更新觀測卡片狀態
    status_detail = f"完成批次仲裁 (本地降級保底, 更新 {eval_count} 筆)" if is_fallback_used else f"完成批次仲裁，共更新 {eval_count} 筆切片"
    st.session_state.last_match_audit_status.update({
        "last_time_str": time_str,
        "last_timestamp": now_ts,
        "state": "WARNING_DEGRADED" if is_fallback_used else "SUCCESS",
        "detail": status_detail,
        "evaluated_chunks_count": eval_count
    })

    # 刷新 3 分鐘計時器
    st.session_state.audit_last_eval_time = now_ts

# 安全清空目錄內容，忽略正在被行程寫入/鎖定的殘留檔案
def safe_purge_directory(dir_path: Path):
    """安全清空目錄內容，加入 3 次重試以應對 Windows File Handle 延遲釋放"""
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        return

    for item in dir_path.glob("*"):
        for _ in range(3):
            try:
                if item.is_file():
                    item.unlink(missing_ok=True)
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                break
            except (PermissionError, OSError):
                time.sleep(0.1)

# ==============================================================================
# 🧩 Fragment 1: 右側聊天室高頻串流區 (run_every="3s", 局域獨立運作)
# ------------------------------------------------------------------------------
# 💡 運作機制：
#   1. pytchat 抓取 YouTube Live 即時彈幕，自動抽取觀眾原生 Google 頭像。
#   2. Token 節流門檻判定：高頻時段（>= 1.0 msg/min）休眠省 Token；低頻或重大違規時喚醒 LLM。
#   3. 300 秒靜默破冰：超過 5 分鐘無人發話，狀態機推至 IDLE_ICEBREAKER 主動破冰。
# ==============================================================================
@st.fragment(run_every="3s")
def render_live_chatroom_stream():
    chat_container = st.container(height=540, border=True)

    # --------------------------------------------------------------
    # 卡片樣式渲染
    # --------------------------------------------------------------
    with chat_container:
        for idx, chat in enumerate(st.session_state.chat_history):
            # --------------------------------------------------------------
            # 觀眾發言卡片：一般樣式 + 右側手動翻譯按鈕
            # --------------------------------------------------------------
            if chat["role"] == "user":
                user_avatar = chat.get("avatar") if chat.get("avatar") else "🧑‍💻"

                with st.chat_message("user", avatar=user_avatar):
                    c_text, c_btn = st.columns([0.88, 0.12], vertical_alignment="center")
                    with c_text:
                        st.markdown(f"**{chat['author']}**: {chat['content']}")
                    with c_btn:
                        btn_label = "🌐" if not chat.get("translation") else "🔄"
                        if st.button(btn_label, key=f"trans_btn_{idx}", help="Manual Translation to Traditional Chinese"):
                            result_zh = translate_single_message(chat["content"])
                            if result_zh:
                                chat["translation"] = result_zh
                                st.rerun()

                    if chat.get("translation"):
                        st.markdown(
                            f'<div class="translation-sub" style="margin-top: 4px; padding-top: 4px; font-size: 0.85rem; color: #86EFAC;">'
                            f'🌐 {chat["translation"]}</div>',
                            unsafe_allow_html=True
                        )

            # --------------------------------------------------------------
            # Hype-bot 系統/副播發言卡片 (僅在主動破冰或重大事件時渲染)
            # --------------------------------------------------------------
            elif chat["role"] == "assistant":
                # 避免渲染殘留的 detected 偽訊息
                if chat.get("lang") == "detected":
                    continue

                with st.chat_message("assistant", avatar="🤖"):
                    lang = chat.get("lang", "zh-TW")
                    st.markdown(f'<span class="lang-badge">Lang: {lang.upper()}</span>', unsafe_allow_html=True)
                    st.markdown(f"**{chat['author']}**: {chat['content']}")
                    
                    for action in chat.get("ui_actions", []):
                        if action.action_type == ActionType.RENDER_BUTTONS:
                            button_list = action.payload.buttons
                            if button_list:
                                cols = st.columns(len(button_list))
                                for b_idx, btn_text in enumerate(button_list):
                                    cols[b_idx].button(btn_text, key=f"hist_btn_{idx}_{b_idx}")
                        elif action.action_type == ActionType.TRIGGER_ALERT:
                            st.error(f"⚠️ {action.payload.alert_message or 'Safety policy violation intercepted.'}")

    # ==============================================================================
    # 背景串流處理核心 (整合：5分鐘頻率控管 + 免打Token節流 + 閒置破冰)
    # ==============================================================================
    if st.session_state.monitoring and st.session_state.listener and "engine" in st.session_state:
        engine = st.session_state.engine
        chat_data = st.session_state.listener.get_latest_chat_batch()
        current_time = time.time()

        # ----------------------------------------------------------------------
        # 關鍵宣告：即時讀取預算水位並產生 budget_policy
        # ----------------------------------------------------------------------
        monthly_budget = st.session_state.get("monthly_budget_cap", 50.0)
        current_cost = st.session_state.token_stats.get("estimated_cost_usd", 0.0)
        budget_policy = get_budget_governance_policy(current_cost, monthly_budget)

        # ----------------------------------------------------
        # 分支 A：有抓到新彈幕
        # ----------------------------------------------------
        if chat_data.get("has_new_message"):
            st.session_state.last_message_time = current_time
            # 1. 抓出新彈幕寫入畫面 (去重)
            existing_contents = {chat["content"] for chat in st.session_state.chat_history if chat["role"] == "user"}
            incoming_messages = chat_data.get("messages", [])
            append_chatroom_msg_log(incoming_messages)

            for msg_item in incoming_messages:
                if msg_item["message"] not in existing_contents:
                    st.session_state.chat_history.append({
                        "role": "user",
                        "author": msg_item["author"],
                        "content": msg_item["message"],
                        "avatar": msg_item.get("avatar", ""),
                        "translation": "",
                        "lang": "auto",
                        "timestamp_raw": current_time,
                        "ui_actions": []
                    })
                    existing_contents.add(msg_item["message"])
                    # 記錄每則留言進來的時間點
                    st.session_state.message_timestamps.append(current_time)

                    is_toxic = check_and_apply_moderation(user=msg_item["author"], message=msg_item["message"])
                    if not is_toxic:
                        evaluate_message_sentiment(user=msg_item["author"], message=msg_item["message"])

            # 2. 計算過去 5 分鐘 (300 秒) 內的有效發話總數, 清理 300 秒以前的過期時間戳記
            st.session_state.message_timestamps = [
                ts for ts in st.session_state.message_timestamps if current_time - ts <= 300
            ]
            # 計算平均每分鐘發話量 (msg/min), 5 分鐘內少於 5 則訊息，才代表平均 < 1 msg/min
            msg_count_last_5min = len(st.session_state.message_timestamps)
            chat_rate_per_min = msg_count_last_5min / 5.0
            # 3. 檢查是否有違法/違規關鍵字 (違規熔斷仍需優先防護)
            user_msg = chat_data["latest_message"]
            is_violation = any(toxic in user_msg.lower() for toxic in ["退費詐欺", "scam", "fraud", "違禁詞", "fake"])

            # ----------------------------------------------------
            # 💡 Token 節流門檻判定：
            # 只有當「頻率 < 1 msg/min」或是「觸發重大違規警報」時，才耗費 Token 呼叫 LLM
            # ----------------------------------------------------
            if chat_rate_per_min < 1.0 or is_violation:
                # Situiation 1: 100% 預算熔斷觸發 (完全不發 API 請求，純走本地規則保底)
                if not budget_policy["allow_llm"]:
                    res = engine._fallback_evaluate_and_respond(
                        state=AgentState.VIOLATION_CIRCUIT_BREAKER,
                        fallback_text=f"[BUDGET CIRCUIT BREAKER] Cost (${current_cost:.4f}) >= Cap (${monthly_budget:.2f}). Remote LLM inference blocked."    
                    )
                    st.session_state.current_fsm_state = AgentState.VIOLATION_CIRCUIT_BREAKER
                    st.session_state.decision_trace = f"【GCP BUDGET BLOCKED】\n{res.thought_process}"
                
                # Situiation 2: 正常情況下，呼叫 Gemini LLM 進行語意分析與回應
                else:
                    engine.min_interval = budget_policy["min_interval"]
                    res = engine.evaluate_and_respond(idle_seconds=0, viewer_message=user_msg)
                    update_token_metrics_from_response(res, viewer_message=user_msg)
                    st.session_state.current_fsm_state = res.current_state
                    st.session_state.decision_trace = f"[FSM TRIGGERED (Chat Velocity: {chat_rate_per_min:.2f} msg/min < 1.0)\n{res.thought_process}"
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "author": "Hype-bot",
                    "content": res.display_message,
                    "translation": res.translation_zh,
                    "lang": res.original_language,
                    "ui_actions": res.ui_actions
                })
            else:
                st.session_state.decision_trace = f"[TOKEN THROTTLED] High chat velocity ({chat_rate_per_min:.2f} msg/min >= 1.0). Bot remains dormant to conserve tokens."

            st.rerun()

        # ----------------------------------------------------
        # 分支 B：當前無新彈幕 (完全無人發話滿 300 秒 ➔ 主動破冰)
        # ----------------------------------------------------
        else:
            current_idle_seconds = int(current_time - st.session_state.last_message_time)

            if current_idle_seconds >= 300:
                last_msg = st.session_state.chat_history[-1] if st.session_state.chat_history else None
                if not last_msg or "Icebreaker" not in last_msg.get("author", ""):
                    # Situation 1: 100% 預算熔斷觸發 (完全不發 API 請求，純走本地規則保底)
                    if not budget_policy["allow_llm"]:
                        res = engine._fallback_evaluate_and_respond(
                            state=AgentState.VIOLATION_CIRCUIT_BREAKER,
                            fallback_text=f"[BUDGET BLOCKED] 5-min idle detected, but remote LLM is disabled due to budget limit."    
                        )
                        st.session_state.current_fsm_state = AgentState.VIOLATION_CIRCUIT_BREAKER
                        st.session_state.decision_trace = f"【GCP BUDGET BLOCKED】\n{res.thought_process}"

                    # Situation 2: 允許呼叫，注入冷卻時間並破冰
                    else:
                        engine.min_interval = budget_policy["min_interval"]
                        res = engine.evaluate_and_respond(idle_seconds=current_idle_seconds, viewer_message="")
                        # 👇 單一呼叫即可
                        update_token_metrics_from_response(res, viewer_message="[ICEBREAKER_TRIGGER]")

                        # 👇 讀取 engine 傳回來的 usage_metadata
                        if hasattr(res, "usage_metadata") and res.usage_metadata:
                            p_tok = res.usage_metadata.prompt_token_count or 0
                            c_tok = res.usage_metadata.candidates_token_count or 0
                            
                            st.session_state.token_stats["prompt_tokens"] += p_tok
                            st.session_state.token_stats["candidate_tokens"] += c_tok
                            st.session_state.token_stats["total_tokens"] += (p_tok + c_tok)
                            
                            # 本地計算累計費用
                            call_cost = (p_tok * 0.000000075) + (c_tok * 0.00000030)
                            st.session_state.token_stats["estimated_cost_usd"] += call_cost
                        st.session_state.current_fsm_state = res.current_state
                        st.session_state.decision_trace = f"[5-MIN SILENCE DETECTED -> ICEBREAKER ACTIVATED]\n{res.thought_process}"
                    
                    st.session_state.last_message_time = current_time      
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "author": "Hype-bot (Icebreaker)",
                        "content": res.display_message,
                        "translation": res.translation_zh,
                        "lang": res.original_language,
                        "ui_actions": res.ui_actions
                    })
                    st.rerun()

        # ----------------------------------------------------
        # 💡 定時校正器：每 5 分鐘 (300 秒) 執行一次 AI 語意重新審查
        # ----------------------------------------------------
        if current_time - st.session_state.last_calibration_time >= 300:
            if "engine" in st.session_state and st.session_state.sentiment_logs:
                # 調用 Gemini 批次審查
                has_changed = calibrate_sentiments_with_llm(
                    client=st.session_state.engine.client,
                    model_name=st.session_state.engine.model
                )
                st.session_state.last_calibration_time = current_time
                if has_changed:
                    st.rerun()

# ==============================================================================
# 🧩 Fragment 2: 全域三大 Dashboard 模組 (run_every="8s", 平滑刷新指標)
# ------------------------------------------------------------------------------
# 💡 運作機制：
#   1. Tab 1: Live Performance & Safety (情緒走勢折線圖、評分表、處置日誌與 HITL 例外審查)。
#   2. Tab 2: LLMOps Telemetry (即時 RPM 水位進度條、Token 計費、GCP Cloud Billing 熔斷模擬)。
#   3. Tab 3: Host Auditing & STT Hub (20 秒切片轉錄表、正負情緒分佈圓餅圖與趨勢圖)。
# ==============================================================================

@st.fragment(run_every="8s")
def render_analytics_dashboard_hub():
    tab_performance, tab_llmops, tab_host_audit = st.tabs([
        "📈 Live Performance & Safety",
        "💰 LLMOps & Usage Telemetry",
        "🛡️ Host Auditing & STT Hub"
    ])

    # ------------------------------------------------------------------------------
    # TAB 1: 直播間績效儀表板 (Live Performance & Safety)
    # ------------------------------------------------------------------------------
    with tab_performance:
        st.subheader("📊 Real-Time Livestream Performance & Sentiment Security Monitor")

        # 1. 核心 Metric 卡片列    
        total_msgs = len([c for c in st.session_state.chat_history if c["role"] == "user"])
        current_time = time.time()
        recent_1m_msgs = len([ts for ts in st.session_state.get("message_timestamps", []) if current_time - ts <= 60])
        
        # 初始化快取狀態
        if "real_viewer_count" not in st.session_state:
            st.session_state.real_viewer_count = 0
        if "last_viewer_fetch_time" not in st.session_state:
            st.session_state.last_viewer_fetch_time = 0.0

        # 當監聽器處於連線狀態時
        if st.session_state.monitoring and youtube_url:
            if current_time - st.session_state.last_viewer_fetch_time >= 30:
                vid = extract_video_id(youtube_url)
                
                # 👇 關鍵修正：此處傳入的是 youtube_api_key，而不是 Gemini 的 api_key
                real_count = fetch_youtube_concurrent_viewers(
                    api_key=youtube_api_key, 
                    video_id=vid
                )
                
                if real_count > 0:
                    st.session_state.real_viewer_count = real_count
                st.session_state.last_viewer_fetch_time = current_time

        # 正式取得真實人數（若取得成功會是 30000+，若失敗或未連線則顯示 0）
        online_viewers = st.session_state.real_viewer_count
        heat_score = int((online_viewers * 0.4) + (total_msgs * 3.5) + (recent_1m_msgs * 10))
        
        # ==============================================================================
        # 📊 動態環比計算邏輯 (5 mins / 1 min 滾動基準)
        # ==============================================================================
        now = time.time()

        # 1. Chat Room Heat Score (每 5 分鐘環比 %),
        
        # 若為剛啟動初始，先記錄第一筆基準
        if st.session_state.heat_history_base["score"] == 0 and heat_score > 0:
            st.session_state.heat_history_base = {"timestamp": now, "score": heat_score}
        
        # 距離上次基準已滿 300 秒 (5 分鐘)，結算新環比並滑動基準
        if now - st.session_state.heat_history_base["timestamp"] >= 300:
            old_score = st.session_state.heat_history_base["score"]
            if old_score > 0:
                pct_change = ((heat_score - old_score) / old_score) * 100
                st.session_state.heat_delta_str = f"{pct_change:+.1f}%"
            else:
                st.session_state.heat_delta_str = "+0.0%"
            # 刷新 5 分鐘基準點
            st.session_state.heat_history_base = {"timestamp": now, "score": heat_score}

        # 2. Live Chat Viewers (每 5 分鐘環比人數增減)
        if st.session_state.viewers_history_base["count"] == 0 and online_viewers > 0:
            st.session_state.viewers_history_base = {"timestamp": now, "count": online_viewers}

        if now - st.session_state.viewers_history_base["timestamp"] >= 300:
            old_viewers = st.session_state.viewers_history_base["count"]
            viewer_diff = online_viewers - old_viewers
            st.session_state.viewers_delta_str = f"{viewer_diff:+d}"
            # 刷新 5 分鐘基準點
            st.session_state.viewers_history_base = {"timestamp": now, "count": online_viewers}

        # 3. Total Chat Messages (每 1 分鐘環比留言增量)
        if now - st.session_state.msg_history_base["timestamp"] >= 60:
            old_total = st.session_state.msg_history_base["total"]
            diff_msgs = total_msgs - old_total
            st.session_state.msg_delta_str = f"{diff_msgs:+d} /min"
            # 刷新 1 分鐘基準點
            st.session_state.msg_history_base = {"timestamp": now, "total": total_msgs}

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric(
            "🔥 Chat Room Heat Score",
            f"{heat_score:,} pts",
            st.session_state.heat_delta_str,
            help="Weight calculation: Online Viewers (40%) + Total Messages (40%) + Interactions (20%)"
        )
        col_m2.metric(
            "👥 Live Chat Viewers",
            f"{online_viewers:,}",
            st.session_state.viewers_delta_str,
            help="Calculated from real-time WebSocket connections and heartbeats"
        )
        col_m3.metric(
            "💬 Total Chat Messages",
            f"{total_msgs:,} msgs",
            st.session_state.msg_delta_str,
            help="Cumulative count of audience messages in this live stream"
        )
        col_m4.metric(
            "⚡ Recent 1-Minute Message Rate",
            f"{recent_1m_msgs} msg/min",
            help="Sliding window of the last 60 seconds of received messages"
        )

        st.divider()

        # 2. 警報危機 Alert 專區
        if len(st.session_state.sentiment_logs) >= 5:
            recent_scores = [x["score"] for x in st.session_state.sentiment_logs[-5:]]
            if sum(recent_scores) <= -4:
                st.error("🚨 [CRITICAL ALERT: Negative Sentiment Surge] The system has detected an excessively high ratio of negative sentiment in the last 5 messages. Please activate community management or moderation mode immediately!")

        # 3. 語境分數統計 (上 Chart, 下 Table)
        st.markdown("#### 🧠 Sentiment Score Trends & Scoring Logs")
        st.caption("Scoring Criteria: Highly Positive (+2) | Positive (+1) | Neutral (0, not included in list) | Negative (-1) | Highly Negative (-2)")

        if st.session_state.sentiment_logs:
            df_sentiment = pd.DataFrame(st.session_state.sentiment_logs)

            # ----------------------------------------------------
            # 上 Chart：Sentiment Score Trends
            # ----------------------------------------------------
            
            chart_df = df_sentiment[["time", "cumulative"]].rename(columns={"cumulative": "Total Score"}).set_index("time")
            st.line_chart(chart_df, color="#22C55E", width='stretch')

            # 檢測連續下跌 10 分異常標記
            cum_scores = df_sentiment["cumulative"].values
            if len(cum_scores) >= 2:
                score_diffs = np.diff(cum_scores)
                drop_indices = np.where(score_diffs <= -10)[0]
                if len(drop_indices) > 0:
                    flagged_time = df_sentiment.iloc[drop_indices[0] + 1]["time"]
                    st.warning(f"⚠️ **【異常波動標記】** 系統檢測到在時間點 `{flagged_time}` 語境累積總分劇烈下滑達 10 分！")

            # ----------------------------------------------------
            # 下 Table：Sentiment Scoring Logs (含 State 欄位)
            # ----------------------------------------------------
            st.markdown("##### 📝 Scoring Logs Table")

            display_df = df_sentiment[["time", "author", "content", "score", "cumulative", "state"]].copy()
            display_df = display_df.iloc[::-1].reset_index(drop=True)
            display_df.columns = ["Timestamp", "User", "Message", "Score", "Total Score", "State"]

            # 上色函式：Score (綠紅) 與 State (藍紫)
            def color_score(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return "color: #22C55E; font-weight: bold; background-color: rgba(34, 197, 94, 0.1);"
                    elif val < 0:
                        return "color: #EF4444; font-weight: bold; background-color: rgba(239, 68, 68, 0.1);"
                return ""

            def color_state(val):
                if val == "Calibrated":
                    return "color: #A78BFA; font-weight: bold; background-color: rgba(167, 139, 250, 0.15);"  # 紫色 (AI 校正)
                return "color: #60A5FA; background-color: rgba(96, 165, 250, 0.1);"                         # 藍色 (即時匹配)

            styler = display_df.style
            if hasattr(styler, "map"):
                styled_df = styler.map(color_score, subset=["Score"]).map(color_state, subset=["State"])
            else:
                styled_df = styler.applymap(color_score, subset=["Score"]).applymap(color_state, subset=["State"])

            st.dataframe(
                styled_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Timestamp": st.column_config.TextColumn("Timestamp", width="small"),
                    "User": st.column_config.TextColumn("User", width="small"),
                    "Message": st.column_config.TextColumn("Message", width="large"),
                    "Score": st.column_config.NumberColumn("Score", format="%+d"),
                    "Total Score": st.column_config.NumberColumn("Total Score", format="%d"),
                    "State": st.column_config.TextColumn("State", width="small", help="Matched: Instant Keyword Pre-Assessment | Calibrated: 5-Min Precision AI Sentiment Correction"),
                }
            )
        else:
            st.info("尚無語意評分走勢數據。彈幕進入或手動注入後將自動繪製走勢。")

        st.divider()

        # ==============================================================================
        # 4. 危險語境檢舉與處置名單 (Trust & Safety Audit Console - 全寬 display_df 版)
        # ==============================================================================
        st.markdown("#### 🛡️ Trust & Safety Auto-Moderation & Audit Logs")
        st.caption("AI Auto automatically monitors serious violations and implements circuit breaker actions; provides room management with Human-in-the-Loop (HITL) manual review and unlock channels.")

        # 1. 建立 display_df 資料表副本
        if st.session_state.moderation_actions:
            mod_df = pd.DataFrame(st.session_state.moderation_actions)

            # 【關鍵防護】：若快取中的舊資料缺少這兩欄，自動填入預設值，避免 KeyError
            if "status" not in mod_df.columns:
                mod_df["status"] = "Active"
            if "review_note" not in mod_df.columns:
                mod_df["review_note"] = "Auto Enforced"

            display_df = mod_df[["time", "author", "content", "action", "reason", "status", "review_note"]].copy()
            display_df = display_df.iloc[::-1].reset_index(drop=True)
            display_df.columns = ["Timestamp", "User", "Flagged Evidence", "Action", "Violation Reason", "Status", "Audit Note"]

            # 定義 Action 與 Status 的上色樣式
            def color_action(val):
                val_str = str(val)
                if "Ban" in val_str:
                    return "color: #EF4444; font-weight: bold; background-color: rgba(239, 68, 68, 0.1);"
                elif "Mute" in val_str:
                    return "color: #F59E0B; font-weight: bold; background-color: rgba(245, 158, 11, 0.1);"
                return ""

            def color_status(val):
                val_str = str(val)
                if val_str == "Active":
                    return "color: #F87171; font-weight: bold; background-color: rgba(248, 113, 113, 0.1);"
                elif val_str == "Cleared":
                    return "color: #34D399; font-weight: bold; background-color: rgba(52, 211, 153, 0.1);"
                return ""

            # 跨版本 Pandas 相容映射 (map / applymap)
            styler_mod = display_df.style
            if hasattr(styler_mod, "map"):
                styled_df = styler_mod.map(color_action, subset=["Action"]).map(color_status, subset=["Status"])
            else:
                styled_df = styler_mod.applymap(color_action, subset=["Action"]).applymap(color_status, subset=["Status"])

            # 全寬渲染處置審計表
            st.dataframe(
                styled_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Timestamp": st.column_config.TextColumn("Timestamp", width="small"),
                    "User": st.column_config.TextColumn("User", width="medium"),
                    "Flagged Evidence": st.column_config.TextColumn("Flagged Evidence", width="large"),
                    "Action": st.column_config.TextColumn("Action", width="small"),
                    "Violation Reason": st.column_config.TextColumn("Violation Reason", width="medium"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Audit Note": st.column_config.TextColumn("Audit Note", width="medium")
                }
            )
        else:
            st.info("Currently, the community is in good standing with no triggering of any risk control actions.")

        # ----------------------------------------------------
        # 2. 下方獨立的 HITL 覆核控制台 (支援預設與動態名單操作)
        # ----------------------------------------------------
        with st.expander("⚙️ HITL Exception Review Console", expanded=False):
            # 防護：確保名單中每筆項目都有 status 屬性 (預設 Active)
            for row in st.session_state.moderation_actions:
                if "status" not in row or not row["status"]:
                    row["status"] = "Active"
                if "review_note" not in row:
                    row["review_note"] = "Auto Enforced"
            
            # 抓出處於 Active 懲處狀態的用戶清單
            active_users = [row["author"] for row in st.session_state.moderation_actions if row.get("status") == "Active"]
            
            if active_users:
                col_target, col_reason, col_btn = st.columns([1.5, 2.5, 1])
                with col_target:
                    target_user = st.selectbox("Choose the user to review", active_users)
                with col_reason:
                    override_reason = st.text_input("The reason of Audit Review", placeholder="For example: The message was misinterpreted, revoke the ban")
                with col_btn:
                    st.write("")  # 垂直對齊按鈕
                    st.write("")
                    if st.button("🔓 Revoke Action", type="primary", width='stretch'):
                        for row in st.session_state.moderation_actions:
                            if row["author"] == target_user:
                                row["status"] = "Cleared"
                                row["review_note"] = override_reason if override_reason else "Manual Override (Restored)"
                        st.success(f"Already revoked the action against {target_user}, and updated the audit log!")
                        st.rerun()
            else:
                st.success("Currently, there are no users pending moderation actions.")

    # ------------------------------------------------------------------------------
    # TAB 2: LLM 成本與用量統計 (LLMOps & Usage Telemetry)
    # ------------------------------------------------------------------------------
    with tab_llmops:
        st.subheader("💰 Google AI Studio - Usage & LLMOps Telemetry")
        st.caption("Real-time tracking of current session Token consumption, estimated API costs, and GCP quota monitoring.")

        # ==============================================================================
        # 💰 動態模型與速率配額遙測 (Dynamic Model & Rate Limit Telemetry)
        # ==============================================================================
        # 1. 動態讀取當前 Engine 實際運行的模型版本
        if "engine" in st.session_state and hasattr(st.session_state.engine, "model"):
            active_model_name = st.session_state.engine.model
        else:
            active_model_name = os.getenv("GEMINI_MODEL_NAME", global_model_name)

        # 2. 動態判定 Rate Limit 規格與門檻
        tier_specs = {
            "Free": {"rpm_limit": 15, "tpm_limit": 1_000_000, "badge_type": "warning"},
            "Tier 1 Pay-as-you-go": {"rpm_limit": 1000, "tpm_limit": 4_000_000, "badge_type": "success"},
            "Enterprise": {"rpm_limit": 4000, "tpm_limit": 10_000_000, "badge_type": "info"}
        }
        current_tier_info = tier_specs.get(rate_limit_tier, tier_specs["Free"])

        col_env1, col_env2, col_env3 = st.columns(3)
        
        # 卡片 1: 金鑰連線來源
        col_env1.info(f"🔑 **API Key Source**: `GEMINI_API_KEY.env` ({'Connected' if api_key else 'Not Configured'})")
        
        # 卡片 2: 動態讀取實際活體模型版本
        col_env2.success(f"🤖 **Active Model Version**: `{active_model_name}`")
        
        # 卡片 3: 動態讀取 Tier 等級與官方配額限制
        col_env3.warning(f"⚡ **Rate Limit Tier**: `{rate_limit_tier}` (Max: {current_tier_info['rpm_limit']} RPM)")

        # ==============================================================================
        # 📊 Token 消耗與成本遙測指標 (補回遺失的 4 欄統計卡片)
        # ==============================================================================
        t_stats = st.session_state.token_stats

        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        c_col1.metric(
            "📥 Prompt Tokens", 
            f"{t_stats['prompt_tokens']:,}", 
            help="Accumulated Prompt Tokens sent to the large model and historical Context Tokens"
        )
        c_col2.metric(
            "📤 Candidates Tokens", 
            f"{t_stats['candidate_tokens']:,}", 
            help="Tokens generated by the large model and Generative UI JSON Schema Tokens"
        )
        c_col3.metric(
            "🔢 Total Tokens", 
            f"{t_stats['total_tokens']:,}", 
            help="Accumulated Input and Output Tokens"
        )
        c_col4.metric(
            "💵 Estimated Cost (USD)", 
            f"${t_stats['estimated_cost_usd']:.6f}", 
            help="Estimated API costs based on Flash's official rates: Input $0.075/1M, Output $0.30/1M"
        )

        # ----------------------------------------------------
        # 即時 RPM 水位與配額健康度 (Live Telemetry Health)
        # ----------------------------------------------------
        now_ts = time.time()
        # 統計過去 60 秒內的 API 調用次數 (使用現有的 message_timestamps 或單獨記錄 API 呼叫點)
        recent_calls_1m = len([ts for ts in st.session_state.get("message_timestamps", []) if now_ts - ts <= 60])
        max_rpm = current_tier_info["rpm_limit"]
        rpm_usage_ratio = min(recent_calls_1m / max_rpm, 1.0)

        st.caption(f"Real-time Traffic Level: Current requests in the last 1 minute `{recent_calls_1m}` / Rate Limit `{max_rpm}` RPM")
        st.progress(rpm_usage_ratio)

        # ==============================================================================
        # 🚨 GCP Cloud Billing API 預算熔斷機制 (動態階梯防護版)
        # ==============================================================================
        st.write("---")
        st.markdown("#### 🚨 GCP Cloud Billing API Budget Governance")
        st.caption("Based on GCP Pub/Sub billing event stream, to realize Traffic Shaping and FSM-based degradation.")

        # 1. 預算參數動態配置 (支援現場靈活展示)
        col_cfg1, col_cfg2 = st.columns([1.5, 2.5])
        with col_cfg1:
            monthly_budget = st.number_input(
                "Monthly Budget Cap (USD)",
                min_value=1.0,
                max_value=500.0,
                value=st.session_state.get("monthly_budget_cap", 50.0),
                step=5.0,
                key="monthly_budget_input",
                help="Set the monthly project API expenditure cap"
            )
        
        current_cost = t_stats["estimated_cost_usd"]
        budget_usage_ratio = min(current_cost / monthly_budget, 1.0)
        budget_pct = budget_usage_ratio * 100

        with col_cfg2:
            # 動態判定目前防禦水位與系統策略
            if budget_usage_ratio >= 1.0:
                status_text = "🚨 **【100% Circuit Breaker Triggered】** Activating FSM local deterministic rule-based degradation mode, completely disabling LLM API calls to protect the budget."
                bar_color = "red"
            elif budget_usage_ratio >= 0.8:
                status_text = "⚠️ **【80% Alert Threshold】** Automatically tightening Rate-limit frequency thresholds, processing only Critical-level high-value events."
                bar_color = "orange"
            else:
                status_text = "✅ **【Normal Operation】** Budget health is good, allowing full functionality for real-time interaction and LLM calls."
                bar_color = "green"
            
            st.markdown(f"**Current Protection Strategy**：\n{status_text}")

        # 2. 動態預算消耗進度條
        st.progress(budget_usage_ratio, text=f"Budget Consumption Progress: ${current_cost:.5f} / ${monthly_budget:.2f} USD ({budget_pct:.2f}%)")

        # 3. 模擬注入與 GCP 架構說明
        col_bill_info, col_bill_action = st.columns([2, 1], gap="medium")
        
        with col_bill_info:
            st.markdown("""
            * **Structure**: `GCP Billing Cloud Pub/Sub Topic` ➔ `Cloud Run / Webhook Receiver` ➔ `Edge Telemetry Engine`
            * **Self-Healing Capability**: When the monthly billing cycle resets or quota is added, the system automatically sends a deactivation signal via Webhook, seamlessly restoring FSM interactive capabilities.
            """)

        with col_bill_action:
            st.write("")
            # 1. 模擬熔斷按鈕：指定固定 key
            if st.button("🛡️ Simulate Budget Breaker", key="btn_simulate_breaker", use_container_width=True):
                st.session_state.token_stats["estimated_cost_usd"] = monthly_budget + 0.01
                st.session_state.current_fsm_state = AgentState.VIOLATION_CIRCUIT_BREAKER
                st.session_state.decision_trace = (
                    "[CIRCUIT_BREAKER_TRIGGERED] Received simulated GCP Billing Pub/Sub budget notification (cost >= 100%). "
                    "Disabling remote LLM inference. Degrading agent to local deterministic rule-based safety fallback."
                )
                st.rerun()

            # 2. 重設按鈕：指定固定 key，並完整清空所有計量指標與暫存
            if st.button("🔄 Reset Budget Billing & Status", key="btn_reset_budget_stats", use_container_width=True):
                # 完整重置 Token 指標字典
                st.session_state.token_stats = {
                    "prompt_tokens": 0,
                    "candidate_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0
                }
                # 重置 FSM 狀態與決策追蹤
                st.session_state.current_fsm_state = AgentState.ACTIVE_INTERACTION
                st.session_state.decision_trace = "[BUDGET_RESET] Budget telemetry restored to nominal status. Agent re-armed."
                
                # 重新呼叫全域重繪
                st.rerun()

        # 4. 展開檢視 GCP 標準 Pub/Sub JSON 負載 (面試加分亮點)
        with st.expander("🔍 View GCP Cloud Billing Pub/Sub Event Payload Specification"):
            st.caption("System architecture's actual received and parsed GCP native event structure")
            sample_pubsub_payload = {
                "budgetDisplayName": "LLM-Live-Agent-Monthly-Cap",
                "alertThresholdExceeded": 1.0,
                "costAmount": round(monthly_budget + 0.01, 2),
                "costIntervalStart": "2026-11-01T00:00:00Z",
                "budgetAmount": float(monthly_budget),
                "currencyCode": "USD"
            }
            st.json(sample_pubsub_payload)

    # ------------------------------------------------------------------------------
    # TAB 3: 主播監控與稽核儀表板 (Host Auditing & STT Hub)
    # ------------------------------------------------------------------------------
    with tab_host_audit:
        st.subheader("🛡️ Host Auditing & Multi-modal STT Alignment Hub")
        st.caption("即時掛載 STT 20s 切片文稿，與聊天室訊息進行跨時段語境對照與關聯評級（每 3 分鐘自動評級）。")
        if st.session_state.get("monitoring"):
            process_pending_audio_chunks()

        # 1. 自動定時器檢查：每 3 分鐘 (180秒) 自動重算一次對齊關係
        now_ts = time.time()
        if now_ts - st.session_state.get("audit_last_eval_time", 0) >= 180:
            evaluate_chunk_chat_correlation()

        # 頂部控制列
        col_h_ctrl1, col_h_ctrl2 = st.columns([3, 1])
        with col_h_ctrl1:
            st.info("💡 **計分規則**：基礎發話 `+1`（互動熱度）｜ 正向語境加成 `+1`（累計 `+2`）｜ 負向語境扣分 `-1`。若當前切片關聯微弱，向前回溯 1 個切片判定。")
        with col_h_ctrl2:
            if st.button("🔄 手動即時對齊評級", key="btn_manual_audit_eval", use_container_width=True):
                evaluate_chunk_chat_correlation()
                st.toast("✅ 已完成對齊結算！", icon="🎯")

        st.write("---")

        # ==============================================================================
        # 📊 三大圖表區 (總計 Pie Chart + Positive Line Chart + Negative Line Chart)
        # ==============================================================================
        st.markdown("#### 📈 Multi-Modal Sentiment Correlation Charts")

        # 檢查是否有切片數據
        if st.session_state.get("stt_chunks"):
            df_chunks = pd.DataFrame(st.session_state.stt_chunks)
            col_chart_pie, col_chart_pos, col_chart_neg = st.columns(3)

            with col_chart_pie:
                st.markdown("##### 🥧 總計回饋分佈 (Overall Sentiment Pie)")
                total_pos = int(df_chunks["pos_count"].sum()) if "pos_count" in df_chunks.columns else 0
                total_neg = int(df_chunks["neg_count"].sum()) if "neg_count" in df_chunks.columns else 0
                
                if total_pos + total_neg > 0:
                    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
                    fig_pie.patch.set_facecolor('#0E1117')
                    ax_pie.set_facecolor('#0E1117')
                    wedges, texts, autotexts = ax_pie.pie(
                        [total_pos, total_neg],
                        labels=["Positive", "Negative"],
                        autopct="%1.1f%%",
                        colors=["#22C55E", "#EF4444"],
                        startangle=140,
                        textprops=dict(color="white", weight="bold")
                    )
                    st.pyplot(fig_pie, width='stretch')
                    plt.close(fig_pie)
                else:
                    st.info("目前尚無足夠的正負向數據繪製圓餅圖")

            with col_chart_pos:
                st.markdown("##### 📈 Positive 反應趨勢 (Positive Line Chart)")
                if "pos_count" in df_chunks.columns and not df_chunks.empty:
                    df_pos_trend = df_chunks[["chunk_id", "pos_count"]].set_index("chunk_id")
                    st.line_chart(df_pos_trend, color="#22C55E", width='stretch')
                else:
                    st.caption("尚無切片 Positive 數據")

            with col_chart_neg:
                st.markdown("##### 📉 Negative 反應趨勢 (Negative Line Chart)")
                if "neg_count" in df_chunks.columns and not df_chunks.empty:
                    df_neg_trend = df_chunks[["chunk_id", "neg_count"]].set_index("chunk_id")
                    st.line_chart(df_neg_trend, color="#EF4444", width='stretch')
                else:
                    st.caption("尚無切片 Negative 數據")
        else:
            st.info("🎙️ 尚無已儲存的 Live Streaming 切片資料。啟動直播連線後，背景管線將自動在此沉澱切片並繪製走勢圖。")

        st.write("---")

        # ==============================================================================
        # 📝 雙向日誌對照區：Chunk Log vs Chatroom Msg Log (採用 display_df 原生表格)
        # ==============================================================================
        st.markdown("#### 📋 Real-Time Sliced Logs Alignment")

        # 即時匹配紀錄與狀態小卡片 (3Mins Auto-Audit Observability)
        status_info = st.session_state.get("last_match_audit_status", {})
        last_eval_ts = st.session_state.get("audit_last_eval_time", time.time())
        time_elapsed = int(time.time() - last_eval_ts)
        countdown_secs = max(0, 180 - time_elapsed)

        # 動態判定狀態樣式
        state_code = status_info.get("state", "STANDBY")
        if state_code == "SUCCESS":
            status_badge = "Active"
        elif "SKIPPED" in state_code:
            status_badge = "Idle"
        elif state_code == "ERROR":
            status_badge = "Failed"
        else:
            status_badge = "Standby"

        # 4 欄精巧指標卡片
        c_stat1, c_stat2, c_stat3, c_stat4 = st.columns(4)
        c_stat1.metric(
            "🕒 最新評級時間", 
            status_info.get("last_time_str", "無"), 
            help="最後一次觸發 3 分鐘批次對齊評級的時間"
        )
        c_stat2.metric(
            "⚡ 模組作動狀態", 
            status_badge, 
            help=status_info.get("detail", "無備註")
        )
        c_stat3.metric(
            "⏳ 下次自動評級倒數", 
            f"{countdown_secs}s", 
            f"-{time_elapsed}s 經過" if time_elapsed > 0 else "0s",
            help="每 180 秒 (3分鐘) 自動滑動窗口並向 Gemini 請求一次語意仲裁"
        )
        c_stat4.metric(
            "📦 上次仲裁切片數", 
            f"{status_info.get('evaluated_chunks_count', 0)} Chunks", 
            help="上一輪 3 分鐘窗口內成功打包送審的 20s 音訊切片總數"
        )

        st.caption(f"ℹ️ **執行細節備註**：`{status_info.get('detail', '等待中')}`")
        st.write("")  # 留適當垂直間隔

        # 雙向對照表格
        col_log_chunk, col_log_chat = st.columns([1.3, 1], gap="medium")

        with col_log_chunk:
            st.markdown("##### 🎙️ STT Chunk Log (60s Slices)")
            if st.session_state.get("stt_chunks"):
                df_chunks = pd.DataFrame(st.session_state.stt_chunks)
                display_chunk_df = df_chunks[["chunk_id", "time_window", "text", "pos_count", "neg_count", "net_score"]].copy().iloc[::-1].reset_index(drop=True)
                display_chunk_df.columns = ["Chunk ID", "Time Window", "主播語音逐字稿 (STT)", "Pos", "Neg", "Net Score"]

                def color_net_score(val):
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return "color: #22C55E; font-weight: bold; background-color: rgba(34, 197, 94, 0.1);"
                        elif val < 0:
                            return "color: #EF4444; font-weight: bold; background-color: rgba(239, 68, 68, 0.1);"
                    return ""

                styler_c = display_chunk_df.style
                styled_c = styler_c.map(color_net_score, subset=["Net Score"]) if hasattr(styler_c, "map") else styler_c.applymap(color_net_score, subset=["Net Score"])
                st.dataframe(styled_c, use_container_width=True, hide_index=True)
            else:
                st.caption("暫無 STT 切片記錄。連線並啟動音訊管線後將自動累積。")

        with col_log_chat:
            st.markdown("##### 💬 Chatroom Msg Log (Viewer Reactions)")
            user_msgs = [c for c in st.session_state.chat_history if c.get("role") == "user"][-20:]
            if user_msgs:
                df_chat = pd.DataFrame(user_msgs)
                display_chat_df = df_chat[["author", "content"]].copy()
                display_chat_df["source"] = "Live Stream"
                display_chat_df = display_chat_df.iloc[::-1].reset_index(drop=True)
                display_chat_df.columns = ["User", "Message", "來源"]
                st.dataframe(display_chat_df, use_container_width=True, hide_index=True)
            else:
                st.caption("聊天室目前尚無觀眾發言。")

# ==============================================================================
# 🚀 主頁面排版 (左側 Console 控制面板，右側掛載 Fragment 1)
# ==============================================================================
st.title("Live Interaction Agent")
st.caption("Enterprise-grade AI co-host telemetry and autonomous community moderation engine.")

col_console, col_ui = st.columns([1, 1.3], gap="large")

with col_console:
    st.subheader("Live Operations Console")
    
    youtube_url = st.text_input(
        "Live Stream Ingestion Endpoint (YouTube URL)",
        placeholder="Paste live stream URL (supports EN / JP / KR / ZH chats)...",
        help="Connects to the YouTube Live chat socket to ingest streaming messages in real-time."
    )

    audio_mode = st.radio(
        "Audio Slicing Ingestion Pipeline Mode",
        options=["🧪 Virtual Audio Simulation (雲端 Demo 推薦 / 100% 穩定)", "📻 Real Stream Slicing (本機開發 / yt-dlp + FFmpeg)"],
        index=0,
        help="【Virtual Audio Simulation】在 Streamlit Cloud 避開 YouTube 機房風控；【Real Stream Slicing】適合本機環境測試。"
    )

    # --------------------------------------------------------------
    # 一鍵測試 API 健康檢查 (Gemini + YouTube Data API + FFmpeg)
    # --------------------------------------------------------------
    with st.expander("🔍 System Pre-flight & API Diagnostics"):
        if st.button("Run Full API Health Check", use_container_width=True):
            status_box = st.status("正在檢測各項 API 狀態...", expanded=True)
            
            # 1. 檢測 Gemini Text API
            try:
                test_client = genai.Client(api_key=api_key)
                test_res = test_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="ping",
                    config={"max_output_tokens": 1}
                )
                status_box.write("✅ Gemini API (Text): 正常連線")
            except Exception as e:
                status_box.write(f"❌ Gemini API (Text): 失敗 ({e})")

            # 2. 檢測 YouTube Data API
            try:
                yt_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id=jfKfPfyJRdk&key={youtube_api_key}"
                yt_res = requests.get(yt_url, timeout=5).json()
                if "items" in yt_res:
                    status_box.write("✅ YouTube Data API: 正常連線")
                else:
                    status_box.write(f"❌ YouTube Data API: 失敗 ({yt_res.get('error', {}).get('message', '未知錯誤')})")
            except Exception as e:
                status_box.write(f"❌ YouTube Data API: 連線異常 ({e})")

            # 3. 檢測 FFmpeg Linux 系統指令
            ffmpeg_check = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if ffmpeg_check.returncode == 0:
                status_box.write("✅ 系統 FFmpeg: 正常就緒")
            else:
                status_box.write("❌ 系統 FFmpeg: 未安裝或路徑錯誤")

            status_box.update(label="診斷完成！", state="complete")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button("Connect Stream",
                        type="primary",
                        width='stretch',
                        help="Initializes the WebSocket listener, purges previous session buffers, and boots the FSM state machine.")
    with col_btn2:
        stop_btn = st.button("⏹ Terminate",
                        width='stretch',
                        help="Stops the WebSocket listener and terminates the live stream monitoring.")

    # ==============================================================================
    # 🚀 啟動連線控制 (Connect Stream)
    # ==============================================================================
    if start_btn and youtube_url:
        try:
            # 1. 防禦性清理 —— 在開新連線前，徹底處決舊進程與執行緒，杜絕背景孤兒
            if "audio_thread_stop_event" in st.session_state:
                st.session_state.audio_thread_stop_event.set()
            force_kill_ffmpeg_safely(st.session_state.get("audio_process"))
            st.session_state.audio_process = None
            
            # 2. 重新連線時，先將三個資料夾內部的舊記錄與舊切片全部清空
            safe_purge_directory(Path("audio_chunks"))
            safe_purge_directory(Path("Chatroom Msg Log"))
            safe_purge_directory(Path("STT Chunk Log"))
            
            # 3. 建立監聽實體與重設工作事件
            vid = extract_video_id(youtube_url)
            st.session_state.listener = YouTubeChatListener(youtube_url)
            st.session_state.monitoring = True
            st.session_state.audio_thread_stop_event.clear()

            # 3. 雙軌音訊管線啟動 (虛擬模擬 vs 本機真實切片)
            if "Virtual Audio Simulation" in audio_mode:
                st.session_state.audio_process = start_virtual_audio_simulation(
                    output_dir="audio_chunks",
                    interval_secs=60,
                    stop_event=st.session_state.audio_thread_stop_event
                )
                st.toast("⚡ 已掛載虛擬音訊串流管線 (雲端零阻擋模式)", icon="🚀")
            else:
                try:
                    audio_stream_url = get_live_audio_url_native(youtube_url)
                    st.session_state.audio_process = start_audio_slicing_pipeline_real(
                        audio_stream_url=audio_stream_url,
                        segment_seconds=60,
                        output_dir="audio_chunks"
                    )
                    st.toast("📻 已啟動真實 FFmpeg 音訊切片子程序", icon="🎙️")
                except Exception as real_err:
                    st.warning(f"⚠️ 真實音源提取受阻 ({real_err})，系統已自動切換至【虛擬音源備援模式】！")
                    st.session_state.audio_process = start_virtual_audio_simulation(
                        output_dir="audio_chunks",
                        interval_secs=60,
                        stop_event=st.session_state.audio_thread_stop_event
                    )

            # 4. 抓取並存入頻道頭像與直播標題
            details = fetch_youtube_stream_details(api_key=youtube_api_key, video_id=vid)
            st.session_state.stream_title = details.get("stream_title", "Live Stream")
            st.session_state.channel_title = details.get("channel_title", "YouTuber")
            st.session_state.channel_icon_url = details.get("channel_icon_url", "")
            
            # 5. 清空舊紀錄，重新初始化 Chat Log
            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "author": "System Ingestion",
                    "content": "Ingestion pipeline established. Session log cleared, listening for incoming chats or idle triggers...",
                    "translation": "",
                    "lang": "en",
                    "ui_actions": []
                }
            ]
            st.session_state.sentiment_logs = []
            st.session_state.stt_chunks = []
            st.session_state.processed_chunks_set.clear()
            st.session_state.audit_last_eval_time = time.time()
            
            # 5. 同步重置觀測指標
            st.session_state.current_fsm_state = AgentState.ACTIVE_INTERACTION
            st.session_state.decision_trace = "[INIT] Clean start: Buffer reset & pipeline primed."
            st.success("Socket connection established successfully! Buffer refreshed.")
            st.rerun()  # 立即重新渲染畫面，讓右側對話框瞬間清空
        except Exception as e:
            st.error(f"Connection Failed: {str(e)}")

    if stop_btn:
        # 1. 關閉串流狀態與通知背景 Thread 退出
        st.session_state.monitoring = False
        st.session_state.listener = None
        if "audio_thread_stop_event" in st.session_state:
            st.session_state.audio_thread_stop_event.set()

        # 2. 自動徹底處決 FFmpeg 進程樹與系統孤兒進程
        force_kill_ffmpeg_safely(st.session_state.get("audio_process"))
        st.session_state.audio_process = None

        # 3. Handle 完全回收後，安全清空切片暫存
        safe_purge_directory(Path("audio_chunks"))

        # 4. 重置中繼標題狀態，使右側標題回歸 "Live Chatroom"
        st.session_state.stream_title = ""
        st.session_state.channel_title = ""
        st.session_state.channel_icon_url = ""

        st.info("Live stream ingestion and audio slicing stopped.")
        st.rerun()

    st.write("---")
    
    # 顯示後端觀測指標 (LLM / FSM Observability)
    st.subheader("FSM Decision Trace & Telemetry")
    state_color = {
        AgentState.IDLE_ICEBREAKER: "orange",
        AgentState.ACTIVE_INTERACTION: "green",
        AgentState.VIOLATION_CIRCUIT_BREAKER: "red"
    }
    st.markdown(f"**Active FSM State**: :{state_color[st.session_state.current_fsm_state]}[{st.session_state.current_fsm_state.value}]",
                help="Deterministic Finite State Machine state governing agent tone, safety breakers, and interaction cadence.")
    st.info(f"**Latest Inference Trace**:\n\n{st.session_state.decision_trace}")

    # ----------------------------------------------------
    # 手動緊急測試按鈕 (修復 NameError 版)
    # ----------------------------------------------------
    with st.expander("🛠 Manual Multilingual Payloads Injection (Offline Fallback)"):
        mock_lang_choice = st.selectbox("Scenario Examples", [
            "English: Is international shipping available to US?",
            "Japanese: この商品は日本への発送に対応していますか？",
            "Korean: 지금 주문하면 언제 배송되나요?",
            "Circuit Breaker (Violation): This is a scam! Fake product!"
        ],
        help="Simulates edge cases across international locales or toxic inputs without waiting for live chat events."
        )
        if st.button("Inject Synthetic Message", help="Dispatches synthetic payload directly into the evaluation pipeline.", use_container_width=True):
            if api_key:
                st.session_state.chat_history.append({
                    "role": "user",
                    "author": "Synthetic Viewer",
                    "content": mock_lang_choice,
                    "translation": "",
                    "lang": "auto",
                    "ui_actions": []
                })

                # 1. 安全風控判定
                is_toxic = check_and_apply_moderation(user="Synthetic Viewer", message=mock_lang_choice)

                # 2. 正常留言進行語意評分 (保證圖表會記錄)
                if not is_toxic:
                    evaluate_message_sentiment(user="Synthetic Viewer", message=mock_lang_choice)

                # 3. 呼叫互動引擎生成回覆
                res = st.session_state.engine.evaluate_and_respond(idle_seconds=0, viewer_message=mock_lang_choice)

                # 4. 更新 Token 指標
                update_token_metrics_from_response(res, viewer_message=mock_lang_choice)

                # 5. 更新狀態與對話氣泡
                st.session_state.current_fsm_state = res.current_state
                st.session_state.decision_trace = res.thought_process
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "author": "Hype-bot",
                    "content": res.display_message,
                    "translation": res.translation_zh,
                    "lang": res.original_language,
                    "ui_actions": res.ui_actions
                })
                
                st.rerun()

        st.write("---")
        st.subheader("FSM Decision Trace & Telemetry")
        state_color = {
            AgentState.IDLE_ICEBREAKER: "orange",
            AgentState.ACTIVE_INTERACTION: "green",
            AgentState.VIOLATION_CIRCUIT_BREAKER: "red"
        }
        st.markdown(f"**Active FSM State**: :{state_color[st.session_state.current_fsm_state]}[{st.session_state.current_fsm_state.value}]")
        st.info(f"**Latest Inference Trace**:\n\n{st.session_state.decision_trace}")

with col_ui:
    # ==============================================================================
    # 🎯 動態標題列：未連線 / 終止時顯示 "Live Chatroom"，連線時顯示頭像與實況標題
    # ==============================================================================
    if st.session_state.monitoring and st.session_state.channel_title:
        # 連線狀態：顯示圓形頭像 + 直播主與實況標題
        col_avatar, col_header = st.columns([0.15, 1], gap="small", vertical_alignment="center")
        with col_avatar:
            if st.session_state.channel_icon_url:
                st.markdown(
                    f'<img src="{st.session_state.channel_icon_url}" style="width:42px; height:42px; border-radius:50%; border:2px solid #22C55E; object-fit:cover;">', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div style="font-size: 2rem;">📺</div>', unsafe_allow_html=True)
        with col_header:
            st.markdown(f"**{st.session_state.channel_title}**")
            st.caption(f"🔴 {st.session_state.stream_title[:45]}..." if len(st.session_state.stream_title) > 45 else f"🔴 {st.session_state.stream_title}")
    else:
        # 未連線或 Terminate 狀態：保持原名
        st.subheader("Live Chatroom")
    # 掛載 Fragment 1 (高頻獨立更新，完全不造成整頁泛灰)
    render_live_chatroom_stream()

st.divider()

# 掛載 Fragment 2 (低頻獨立更新，平滑跳動圖表與 LLMOps 統計指標)
render_analytics_dashboard_hub()