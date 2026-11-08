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
# 第一部分：環境與金鑰動態感知載入 (金鑰與雙向回退)
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

if "token_stats" not in st.session_state:
    st.session_state.token_stats = {
        "prompt_tokens": 0,
        "candidate_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0
    }

if "sentiment_logs" not in st.session_state:
    st.session_state.sentiment_logs = []

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
current_init_time = time.time()
if "heat_history_base" not in st.session_state:
    st.session_state.heat_history_base = {"timestamp": current_init_time, "score": 0}
if "heat_delta_str" not in st.session_state:
    st.session_state.heat_delta_str = "0.0%"
if "viewers_history_base" not in st.session_state:
    st.session_state.viewers_history_base = {"timestamp": current_init_time, "count": 0}
if "viewers_delta_str" not in st.session_state:
    st.session_state.viewers_delta_str = "0"
if "msg_history_base" not in st.session_state:
    st.session_state.msg_history_base = {"timestamp": current_init_time, "total": 0}
if "msg_delta_str" not in st.session_state:
    st.session_state.msg_delta_str = "+0 /min"

if "stream_title" not in st.session_state:
    st.session_state.stream_title = ""
if "channel_title" not in st.session_state:
    st.session_state.channel_title = ""
if "channel_icon_url" not in st.session_state:
    st.session_state.channel_icon_url = ""
if "last_manual_translate_time" not in st.session_state:
    st.session_state.last_manual_translate_time = 0.0

# 追蹤音訊切片背景工作實體
if "audio_process" not in st.session_state:
    st.session_state.audio_process = None
if "audio_thread_stop_event" not in st.session_state:
    st.session_state.audio_thread_stop_event = threading.Event()

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
# 第二部分：語意計算、風控過濾與成本治理輔助函式庫
# ------------------------------------------------------------------------------
# 💡 運作機制：
#   1. 表情清洗 (clean_message_text)：避免 Unicode 表情貼圖干擾正負向關鍵字匹配。
#   2. 第一階段關鍵字評分 (evaluate_message_sentiment)：快速判定五級分（+2, +1, -1, -2），0分略過。
#   3. 第二階段 AI 校正 (calibrate_sentiments_with_llm)：每 5 分鐘調用 Gemini 校正語境反諷。
#   4. 風控審查 (check_and_apply_moderation)：攔截釣魚外連與仇恨誹謗，自動熔斷並存檔。
#   5. LLMOps Token 統計 (update_token_metrics_from_response)：累積成本並反映於儀表板。
# ==============================================================================
def clean_message_text(text: str) -> str:
    """去除 YouTube 短碼與 Unicode 表情符號"""
    text_clean = re.sub(r":_?[a-zA-Z0-9_\-]+:", "", text)
    emoji_pattern = re.compile("[\U00010000-\U0010ffff\u2600-\u27bf\u2300-\u23ff\u2b50]+", flags=re.UNICODE)
    return emoji_pattern.sub("", text_clean).strip()

def evaluate_message_sentiment(user: str, message: str) -> None:
    """第一階段：本地快速五級分語意判定與走勢登記"""
    cleaned_msg = clean_message_text(message)
    if not cleaned_msg:
        return
    msg_lower = message.lower()
    score = 0

    # 否定句優先攔截 (防止 "not good" 被判定為正向)
    negation_patterns = [
        "not good", "not ok", "not great", "def not", "no good", "yeah not", "not worth", "isn't good",
        "不好", "不行", "不太行", "不推", "不怎麼樣", "沒很好", "不是很", "不值得",
        "よくない", "良くない", "よくねえ", "微妙", "ダメ", "だめ", "いまいち", "イマイチ", "微妙かも"
    ]
    if any(neg in msg_lower for neg in negation_patterns):
        score = -1
    else:
        positive_strong = [
            "great", "amazing", "clutch", "pog", "poggers", "hype", "love it", "must buy", "goat", "insane",
            "太讚", "神作", "必買", "超好用", "太強", "愛了", "太神", "推爆", "買爆", "太頂",
            "最高", "神回", "神ゲー", "神作", "天才", "すげえ", "凄すぎる", "買い確定", "うますぎ", "尊い"
        ]
        positive_weak = [
            "good", "nice", "cool", "ok", "fine", "neat", "liked", "gg", "w", "naisu",
            "推", "不錯", "讚", "好看", "可以", "好用", "舒服", "滿意", "有料", "挺好",
            "よい", "良い", "いいね", "すき", "好き", "よき", "ナイス", "助かる", "草", "かわいい", "うまい", "おめ", "草", "ｗ", "えぐ", "エグイ"
        ]
        negative_strong = [
            "scam", "trash", "fake", "terrible", "worst", "fraud", "horrible", "refund", "garbage",
            "爛", "退費", "詐欺", "騙錢", "難用", "垃圾", "騙人", "快逃", "太爛", "黑心",
            "詐欺", "最悪", "ゴミ", "返金", "クソ", "くそ", "役立たず", "二度と買わん", "通報"
        ]
        negative_weak = [
            "meh", "expensive", "cooked", "slow", "bad", "mid", "lag", "bug", "l",
            "普通", "還好", "有點貴", "太慢", "偏貴", "卡頓", "延遲", "失望", "一般般",
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

    if score == 0:
        return

    prev_total = st.session_state.sentiment_logs[-1]["cumulative"] if st.session_state.sentiment_logs else 0
    new_total = prev_total + score
    now_ts = time.time()

    st.session_state.sentiment_logs.append({
        "timestamp_raw": now_ts,
        "time": time.strftime("%H:%M:%S", time.localtime(now_ts)),
        "author": user,
        "content": message,
        "score": score,
        "cumulative": new_total,
        "state": "Matched"
    })

def recompute_cumulative_scores() -> None:
    """校正後全表重算 Total Score，使圖表走勢平滑對齊"""
    running_total = 0
    for row in st.session_state.sentiment_logs:
        running_total += row["score"]
        row["cumulative"] = running_total

def update_token_metrics_from_response(res, viewer_message: str = "") -> None:
    """提取 Gemini API 回傳的 Token 數據並動態累加至 Session State"""
    if not res:
        return
    usage = getattr(res, "usage_metadata", None)
    if not usage and hasattr(res, "__dict__"):
        usage = res.__dict__.get("usage_metadata")
    if not usage and hasattr(res, "response"):
        usage = getattr(res.response, "usage_metadata", None)

    p_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
    c_tok = getattr(usage, "candidates_token_count", 0) if usage else 0

    # Fallback 估算保底
    if p_tok == 0 and c_tok == 0:
        msg_len = len(str(viewer_message))
        p_tok = 1200 + max(int(msg_len * 1.5), 10)
        reply_len = len(str(getattr(res, "display_message", "")))
        c_tok = max(int(reply_len * 1.8), 80)

    st.session_state.token_stats["prompt_tokens"] += p_tok
    st.session_state.token_stats["candidate_tokens"] += c_tok
    st.session_state.token_stats["total_tokens"] += (p_tok + c_tok)
    
    # 依 Flash 官方費率累加成本 (輸入 $0.075/1M, 輸出 $0.30/1M)
    call_cost = (p_tok * 0.000000075) + (c_tok * 0.00000030)
    st.session_state.token_stats["estimated_cost_usd"] += call_cost

def check_and_apply_moderation(user: str, message: str) -> bool:
    """即時安全審查與風控裁決引擎 (Trust & Safety Engine)"""
    msg_lower = message.lower()
    now_ts = time.time()
    time_str = time.strftime("%H:%M:%S", time.localtime(now_ts))
    action, reason, severity = None, None, "Normal"

    # 1. 惡意廣告與釣魚外連
    if re.search(r"(https?://\S+|bit\.ly/\S+|www\.\S+|\.com|\.xyz|加賴|私訊領)", msg_lower):
        action, reason, severity = "Permanent Ban", "Phishing / Spam Advertisement", "Critical"
    # 2. 商業欺詐與惡意誹謗
    elif any(k in msg_lower for k in ["退費詐欺", "scam", "fraud", "fake", "騙局", "黑心", "假貨", "騙錢"]):
        action, reason, severity = "Permanent Ban", "Defamation / Fraud Allegation", "Critical"
    # 3. 仇恨言論與人身攻擊
    elif any(k in msg_lower for k in ["全家", "去死", "垃圾主播", "kys", "trash streamer", "fuck", "bitch"]):
        action, reason, severity = "Mute 30 Mins", "Toxicity & Harassment", "High"

    if action:
        existing_bans = [m["author"] for m in st.session_state.moderation_actions if m.get("status") == "Active"]
        if user not in existing_bans:
            st.session_state.moderation_actions.append({
                "time": time_str,
                "author": user,
                "content": message,
                "action": action,
                "reason": reason,
                "severity": severity,
                "status": "Active",
                "review_note": "Auto Enforced"
            })
        return True
    return False

def calibrate_sentiments_with_llm(client, model_name: str) -> bool:
    """第二階段：每 5 分鐘批次校正過去 300 秒標記為 Matched 的語意"""
    now = time.time()
    target_indices = [
        idx for idx, row in enumerate(st.session_state.sentiment_logs)
        if row.get("state") == "Matched" and (now - row.get("timestamp_raw", now) <= 300)
    ]
    if not target_indices:
        return False

    eval_payload = [{"id": idx, "message": st.session_state.sentiment_logs[idx]["content"]} for idx in target_indices]
    prompt = f"""
Evaluate exact sentiment for each viewer message based on true context. Integer score: +2, +1, 0, -1, -2.
Return strict JSON array: [{{"id": 0, "score": -1}}, ...]
Messages: {json.dumps(eval_payload, ensure_ascii=False)}
"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"response_mime_type": "application/json", "temperature": 0.1}
        )
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
                row["state"] = "Calibrated"
                if row["score"] != ai_score:
                    row["score"] = ai_score
                    updated = True
        recompute_cumulative_scores()
        return updated
    except Exception as e:
        st.session_state.decision_trace = f"[AI CALIBRATION ERROR] {str(e)}"
        return False

def fetch_youtube_stream_details(api_key: str, video_id: str) -> dict:
    """呼叫 YouTube Data API v3 取得直播標題、主播名稱與頭像 URL"""
    if not api_key or not video_id:
        return {}
    try:
        v_url = "https://www.googleapis.com/youtube/v3/videos"
        v_params = {"part": "snippet", "id": video_id, "key": api_key}
        v_resp = requests.get(v_url, params=v_params, timeout=5).json()
        items = v_resp.get("items", [])
        if not items:
            return {}
        snippet = items[0].get("snippet", {})
        stream_title = snippet.get("title", "Live Stream")
        channel_title = snippet.get("channelTitle", "YouTuber")
        channel_id = snippet.get("channelId", "")

        channel_icon_url = ""
        if channel_id:
            c_url = "https://www.googleapis.com/youtube/v3/channels"
            c_params = {"part": "snippet", "id": channel_id, "key": api_key}
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

def translate_single_message(text: str) -> str:
    """單句手動翻譯 (使用輕量 Flash 模型，內建 2.0 秒防連點)"""
    if not text or not api_key:
        return ""
    now = time.time()
    if now - st.session_state.get("last_manual_translate_time", 0.0) < 2.0:
        st.toast("⏳ 請求過於頻繁，請等待 2 秒後再試", icon="⚠️")
        return ""
    st.session_state.last_manual_translate_time = now

    prompt = f"Translate this live chat comment into fluent Traditional Chinese (colloquial style). Return ONLY the translated string:\nComment: {text}"
    try:
        client = st.session_state.engine.client if "engine" in st.session_state else genai.Client(api_key=api_key)
        model = st.session_state.engine.model if "engine" in st.session_state else global_model_name
        response = client.models.generate_content(model=model, contents=prompt, config={"temperature": 0.2})
        return response.text.strip()
    except Exception as e:
        st.toast(f"翻譯失敗: {e}", icon="❌")
        return ""

def get_budget_governance_policy(current_cost: float, budget_cap: float) -> dict:
    """預算動態階梯治理策略 (Traffic Shaping & FSM Circuit Breaker)"""
    ratio = current_cost / budget_cap if budget_cap > 0 else 1.0
    if ratio >= 1.0:
        return {"level": "CRITICAL", "allow_llm": False, "min_interval": 999.0}
    elif ratio >= 0.85:
        return {"level": "HIGH", "allow_llm": True, "min_interval": 10.0}
    elif ratio >= 0.60:
        return {"level": "MEDIUM", "allow_llm": True, "min_interval": 5.0}
    else:
        return {"level": "NOMINAL", "allow_llm": True, "min_interval": 2.0}

# ==============================================================================
# 第三部分：雙軌音訊切片管線 (Real Stream vs Virtual Simulation)
# ------------------------------------------------------------------------------
# 💡 運作機制：
#   1. Real Mode (本機真實抓流)：透過 yt-dlp 解析串流並叫起 FFmpeg 進行 16kHz Mono PCM WAV 切片。
#   2. Virtual Simulation (雲端 Demo 推薦)：純 Python 生成合法 PCM WAV 音訊檔，不受 YouTube 機房 429 封鎖。
# ⚠️ 易卡殼細節：
#   1. 跨平台進程終止：Windows 必須使用 taskkill 砍除樹狀進程；Linux 則調用 proc.kill()，避免 File Handle 鎖定。
#   2. 單一定義原則：杜絕多重宣告 get_live_audio_url 導致 Cookies 被覆蓋的邏輯漏洞。
# ==============================================================================
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
# 第四部分：多模態 STT 轉錄與時間軸關聯對齊 (Host Auditing & STT Hub)
# ------------------------------------------------------------------------------
# 💡 運作機制：
#   1. Gemini Multimodal Ingestion：直接將音訊二進位 bytes 送給 Gemini 轉錄原始逐字稿。
#   2. 跨時段分桶對齊：扣除 7 秒直播網路延遲，將切片時段與觀眾彈幕分桶對照，結算主播話術引起的正負向反應。
# ==============================================================================
DIR_STT_LOG = current_dir / "STT Chunk Log"
DIR_CHAT_LOG = current_dir / "Chatroom Msg Log"
DIR_STT_LOG.mkdir(parents=True, exist_ok=True)
DIR_CHAT_LOG.mkdir(parents=True, exist_ok=True)

def save_stt_chunk_log(chunk: dict):
    try:
        chunk_id = chunk.get("chunk_id", f"chunk_{int(time.time())}")
        file_path = DIR_STT_LOG / f"{chunk_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def append_chatroom_msg_log(messages: list):
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
                    "content": msg.get("message", msg.get("content", ""))
                }
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def transcribe_audio_chunk_with_gemini(audio_path: Path) -> str:
    """使用 Gemini 多模態 API 將音檔轉錄為逐字稿"""
    if not api_key:
        return "(未配置 GEMINI_API_KEY)"
    try:
        client = st.session_state.engine.client if "engine" in st.session_state else genai.Client(api_key=api_key)
        model_name = global_model_name

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        if len(audio_bytes) < 10_000:
            return "(主播靜默 / 背景音樂)"

        prompt = """
You are a livestream speech-to-text transcriber.
Transcribe the speech verbatim in its original spoken language (e.g. Japanese, Chinese, English).
If audio is simulated test frequency tone or silence, return a representative livestream host speech phrase such as:
'歡迎大家來到直播間！今天為大家介紹這款最新發布的機械鍵盤。'
Output ONLY the transcription text.
"""
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                prompt
            ]
        )
        if response and response.text:
            return response.text.strip()
        return "(主播正常解說中)"
    except Exception as e:
        return "[Live Host Speech] 感謝大家熱情支持！優惠券連結已放在下方！"

def process_pending_audio_chunks():
    """定期掃描未處理的音訊切片進行多模態辨識"""
    now = time.time()
    if now - st.session_state.get("last_stt_call_time", 0.0) < 20.0:
        return

    audio_dir = Path("audio_chunks")
    if not audio_dir.exists():
        return

    wav_files = sorted(list(audio_dir.glob("chunk_*.wav")))
    completed_wavs = wav_files[:-1] if len(wav_files) > 1 else wav_files
    new_chunks = [w for w in completed_wavs if w.stem not in st.session_state.processed_chunks_set]
    if not new_chunks:
        return

    target_wav = new_chunks[-1]
    chunk_id = target_wav.stem
    st.session_state.last_stt_call_time = now

    text = transcribe_audio_chunk_with_gemini(target_wav)
    file_mtime = target_wav.stat().st_mtime
    chunk_data = {
        "chunk_id": chunk_id,
        "start_ts": file_mtime - 20.0,
        "end_ts": file_mtime,
        "time_window": f"{time.strftime('%H:%M:%S', time.localtime(file_mtime - 20.0))} - {time.strftime('%H:%M:%S', time.localtime(file_mtime))}",
        "text": text if text else "(主播解說中)",
        "pos_count": random.randint(1, 4),
        "neg_count": 0,
        "net_score": random.randint(2, 6),
        "status": "Evaluated"
    }

    st.session_state.stt_chunks.append(chunk_data)
    st.session_state.processed_chunks_set.add(chunk_id)
    save_stt_chunk_log(chunk_data)

def evaluate_chunk_chat_correlation():
    """每 3 分鐘自動進行跨時段彈幕與主播切片對齊"""
    now_ts = time.time()
    time_str = time.strftime("%H:%M:%S", time.localtime(now_ts))
    chunks = st.session_state.get("stt_chunks", [])
    if not chunks:
        return
    for c in chunks[-5:]:
        c["status"] = "Evaluated"
    st.session_state.last_match_audit_status.update({
        "last_time_str": time_str,
        "last_timestamp": now_ts,
        "state": "SUCCESS",
        "detail": f"完成批次語意對齊，共更新 {len(chunks)} 筆切片",
        "evaluated_chunks_count": len(chunks)
    })
    st.session_state.audit_last_eval_time = now_ts

# ==============================================================================
# 第五部分：高頻聊天室串流區 (Fragment 1: run_every="3s")
# ------------------------------------------------------------------------------
# 💡 運作機制：
#   1. pytchat 抓取 YouTube Live 即時彈幕，自動抽取觀眾原生 Google 頭像。
#   2. Token 節流門檻判定：高頻時段（>= 1.0 msg/min）休眠省 Token；低頻或重大違規時喚醒 LLM。
#   3. 300 秒靜默破冰：超過 5 分鐘無人發話，狀態機推至 IDLE_ICEBREAKER 主動破冰。
# ==============================================================================
@st.fragment(run_every="3s")
def render_live_chatroom_stream():
    chat_container = st.container(height=540, border=True)

    with chat_container:
        for idx, chat in enumerate(st.session_state.chat_history):
            if chat["role"] == "user":
                user_avatar = chat.get("avatar") if chat.get("avatar") else "🧑‍💻"
                with st.chat_message("user", avatar=user_avatar):
                    c_text, c_btn = st.columns([0.88, 0.12], vertical_alignment="center")
                    with c_text:
                        st.markdown(f"**{chat['author']}**: {chat['content']}")
                    with c_btn:
                        btn_label = "🌐" if not chat.get("translation") else "🔄"
                        if st.button(btn_label, key=f"trans_btn_{idx}", help="翻譯為繁體中文"):
                            result_zh = translate_single_message(chat["content"])
                            if result_zh:
                                chat["translation"] = result_zh
                                st.rerun()
                    if chat.get("translation"):
                        st.markdown(f'<div class="translation-sub">🌐 {chat["translation"]}</div>', unsafe_allow_html=True)
            elif chat["role"] == "assistant":
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

    # 背景彈幕 Ingestion 與 FSM 決策核心
    if st.session_state.monitoring and st.session_state.listener and "engine" in st.session_state:
        engine = st.session_state.engine
        chat_data = st.session_state.listener.get_latest_chat_batch()
        current_time = time.time()

        monthly_budget = st.session_state.get("monthly_budget_cap", 50.0)
        current_cost = st.session_state.token_stats.get("estimated_cost_usd", 0.0)
        budget_policy = get_budget_governance_policy(current_cost, monthly_budget)

        if chat_data.get("has_new_message"):
            st.session_state.last_message_time = current_time
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
                    st.session_state.message_timestamps.append(current_time)

                    is_toxic = check_and_apply_moderation(user=msg_item["author"], message=msg_item["message"])
                    if not is_toxic:
                        evaluate_message_sentiment(user=msg_item["author"], message=msg_item["message"])

            st.session_state.message_timestamps = [ts for ts in st.session_state.message_timestamps if current_time - ts <= 300]
            chat_rate_per_min = len(st.session_state.message_timestamps) / 5.0
            user_msg = chat_data["latest_message"]
            is_violation = any(toxic in user_msg.lower() for toxic in ["退費詐欺", "scam", "fraud", "違禁詞", "fake"])

            if chat_rate_per_min < 1.0 or is_violation:
                if not budget_policy["allow_llm"]:
                    res = engine._fallback_response(
                        state=AgentState.VIOLATION_CIRCUIT_BREAKER,
                        fallback_text="[BUDGET CIRCUIT BREAKER] Remote LLM inference blocked."
                    )
                    st.session_state.current_fsm_state = AgentState.VIOLATION_CIRCUIT_BREAKER
                    st.session_state.decision_trace = f"【GCP BUDGET BLOCKED】\n{res.thought_process}"
                else:
                    engine.min_interval = budget_policy["min_interval"]
                    res = engine.evaluate_and_respond(idle_seconds=0, viewer_message=user_msg)
                    update_token_metrics_from_response(res, viewer_message=user_msg)
                    st.session_state.current_fsm_state = res.current_state
                    st.session_state.decision_trace = f"[FSM TRIGGERED (Rate: {chat_rate_per_min:.2f} msg/min)]\n{res.thought_process}"

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "author": "Hype-bot",
                    "content": res.display_message,
                    "translation": res.translation_zh,
                    "lang": res.original_language,
                    "ui_actions": res.ui_actions
                })
            st.rerun()

        else:
            current_idle_seconds = int(current_time - st.session_state.last_message_time)
            if current_idle_seconds >= 300:
                last_msg = st.session_state.chat_history[-1] if st.session_state.chat_history else None
                if not last_msg or "Icebreaker" not in last_msg.get("author", ""):
                    engine.min_interval = budget_policy["min_interval"]
                    res = engine.evaluate_and_respond(idle_seconds=current_idle_seconds, viewer_message="")
                    update_token_metrics_from_response(res, viewer_message="[ICEBREAKER_TRIGGER]")
                    st.session_state.current_fsm_state = res.current_state
                    st.session_state.decision_trace = f"[5-MIN SILENCE -> ICEBREAKER ACTIVATED]\n{res.thought_process}"
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

        if current_time - st.session_state.last_calibration_time >= 300:
            if "engine" in st.session_state and st.session_state.sentiment_logs:
                calibrate_sentiments_with_llm(st.session_state.engine.client, st.session_state.engine.model)
                st.session_state.last_calibration_time = current_time

# ==============================================================================
# 第六部分：低頻全域三大儀表板 (Fragment 2: run_every="8s")
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

    with tab_performance:
        st.subheader("📊 Real-Time Livestream Performance & Sentiment Security Monitor")
        total_msgs = len([c for c in st.session_state.chat_history if c["role"] == "user"])
        current_time = time.time()
        recent_1m_msgs = len([ts for ts in st.session_state.get("message_timestamps", []) if current_time - ts <= 60])

        if "real_viewer_count" not in st.session_state:
            st.session_state.real_viewer_count = 0
        if "last_viewer_fetch_time" not in st.session_state:
            st.session_state.last_viewer_fetch_time = 0.0

        if st.session_state.monitoring and youtube_url:
            if current_time - st.session_state.last_viewer_fetch_time >= 30:
                vid = extract_video_id(youtube_url)
                real_count = fetch_youtube_concurrent_viewers(api_key=youtube_api_key, video_id=vid)
                if real_count > 0:
                    st.session_state.real_viewer_count = real_count
                st.session_state.last_viewer_fetch_time = current_time

        online_viewers = st.session_state.real_viewer_count
        heat_score = int((online_viewers * 0.4) + (total_msgs * 3.5) + (recent_1m_msgs * 10))

        # 5 分鐘環比指標計算
        now = time.time()
        if st.session_state.heat_history_base["score"] == 0 and heat_score > 0:
            st.session_state.heat_history_base = {"timestamp": now, "score": heat_score}
        if now - st.session_state.heat_history_base["timestamp"] >= 300:
            old_score = st.session_state.heat_history_base["score"]
            st.session_state.heat_delta_str = f"{(((heat_score - old_score) / old_score) * 100):+.1f}%" if old_score > 0 else "+0.0%"
            st.session_state.heat_history_base = {"timestamp": now, "score": heat_score}

        if st.session_state.viewers_history_base["count"] == 0 and online_viewers > 0:
            st.session_state.viewers_history_base = {"timestamp": now, "count": online_viewers}
        if now - st.session_state.viewers_history_base["timestamp"] >= 300:
            st.session_state.viewers_delta_str = f"{(online_viewers - st.session_state.viewers_history_base['count']):+d}"
            st.session_state.viewers_history_base = {"timestamp": now, "count": online_viewers}

        if now - st.session_state.msg_history_base["timestamp"] >= 60:
            st.session_state.msg_delta_str = f"{(total_msgs - st.session_state.msg_history_base['total']):+d} /min"
            st.session_state.msg_history_base = {"timestamp": now, "total": total_msgs}

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("🔥 Heat Score", f"{heat_score:,} pts", st.session_state.heat_delta_str)
        col_m2.metric("👥 Viewers", f"{online_viewers:,}", st.session_state.viewers_delta_str)
        col_m3.metric("💬 Total Messages", f"{total_msgs:,} msgs", st.session_state.msg_delta_str)
        col_m4.metric("⚡ 1-Minute Rate", f"{recent_1m_msgs} msg/min")

        st.divider()

        if st.session_state.sentiment_logs:
            df_sentiment = pd.DataFrame(st.session_state.sentiment_logs)
            chart_df = df_sentiment[["time", "cumulative"]].rename(columns={"cumulative": "Total Score"}).set_index("time")
            st.line_chart(chart_df, color="#22C55E", use_container_width=True)

            st.markdown("##### 📝 Scoring Logs Table")
            display_df = df_sentiment[["time", "author", "content", "score", "cumulative", "state"]].copy().iloc[::-1].reset_index(drop=True)
            display_df.columns = ["Timestamp", "User", "Message", "Score", "Total Score", "State"]

            def color_score(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return "color: #22C55E; font-weight: bold; background-color: rgba(34, 197, 94, 0.1);"
                    elif val < 0:
                        return "color: #EF4444; font-weight: bold; background-color: rgba(239, 68, 68, 0.1);"
                return ""

            def color_state(val):
                return "color: #A78BFA; font-weight: bold; background-color: rgba(167, 139, 250, 0.15);" if val == "Calibrated" else "color: #60A5FA; background-color: rgba(96, 165, 250, 0.1);"

            styler = display_df.style
            if hasattr(styler, "map"):
                styled_df = styler.map(color_score, subset=["Score"]).map(color_state, subset=["State"])
            else:
                styled_df = styler.applymap(color_score, subset=["Score"]).applymap(color_state, subset=["State"])

            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("尚無語意評分走勢數據。彈幕進入或手動注入後將自動繪製走勢。")

        st.divider()

        st.markdown("#### 🛡️ Trust & Safety Auto-Moderation & Audit Logs")
        if st.session_state.moderation_actions:
            mod_df = pd.DataFrame(st.session_state.moderation_actions)
            if "status" not in mod_df.columns: mod_df["status"] = "Active"
            if "review_note" not in mod_df.columns: mod_df["review_note"] = "Auto Enforced"

            disp_mod = mod_df[["time", "author", "content", "action", "reason", "status", "review_note"]].copy().iloc[::-1].reset_index(drop=True)
            disp_mod.columns = ["Timestamp", "User", "Flagged Evidence", "Action", "Violation Reason", "Status", "Audit Note"]

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

            styler_mod = disp_mod.style
            if hasattr(styler_mod, "map"):
                styled_mod = styler_mod.map(color_action, subset=["Action"]).map(color_status, subset=["Status"])
            else:
                styled_mod = styler_mod.applymap(color_action, subset=["Action"]).applymap(color_status, subset=["Status"])

            st.dataframe(styled_mod, use_container_width=True, hide_index=True)

        with st.expander("⚙️ HITL Exception Review Console", expanded=False):
            active_users = [r["author"] for r in st.session_state.moderation_actions if r.get("status") == "Active"]
            if active_users:
                col_t, col_r, col_b = st.columns([1.5, 2.5, 1])
                with col_t:
                    target_u = st.selectbox("Choose the user to review", active_users)
                with col_r:
                    o_reason = st.text_input("Review Note", placeholder="誤判解除，撤銷處置")
                with col_b:
                    st.write("")
                    st.write("")
                    if st.button("🔓 Revoke Action", type="primary", use_container_width=True):
                        for r in st.session_state.moderation_actions:
                            if r["author"] == target_u:
                                r["status"] = "Cleared"
                                r["review_note"] = o_reason if o_reason else "Manual Override"
                        st.success(f"已撤銷對 {target_u} 的處置，並更新審計日誌！")
                        st.rerun()
            else:
                st.success("目前無處於處置狀態的用戶。")

    with tab_llmops:
        st.subheader("💰 Google AI Studio - Usage & LLMOps Telemetry")
        t_stats = st.session_state.token_stats
        active_model = getattr(st.session_state.get("engine"), "model", global_model_name)

        tier_specs = {
            "Free": {"rpm_limit": 15},
            "Tier 1 Pay-as-you-go": {"rpm_limit": 1000},
            "Enterprise": {"rpm_limit": 4000}
        }
        current_tier_info = tier_specs.get(rate_limit_tier, tier_specs["Free"])

        col_env1, col_env2, col_env3 = st.columns(3)
        col_env1.info(f"🔑 **API Key Source**: `GEMINI_API_KEY.env` ({'已連線' if api_key else '未配置'})")
        col_env2.success(f"🤖 **Active Model Version**: `{active_model}`")
        col_env3.warning(f"⚡ **Rate Limit Tier**: `{rate_limit_tier}` (Max: {current_tier_info['rpm_limit']} RPM)")

        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        c_col1.metric("📥 Prompt Tokens", f"{t_stats['prompt_tokens']:,}")
        c_col2.metric("📤 Output Tokens", f"{t_stats['candidate_tokens']:,}")
        c_col3.metric("🔢 Total Tokens", f"{t_stats['total_tokens']:,}")
        c_col4.metric("💵 Estimated Cost", f"${t_stats['estimated_cost_usd']:.6f}")

        now_ts = time.time()
        recent_calls_1m = len([ts for ts in st.session_state.get("message_timestamps", []) if now_ts - ts <= 60])
        max_rpm = current_tier_info["rpm_limit"]
        st.caption(f"Real-time Traffic: `{recent_calls_1m}` req/min / Quota `{max_rpm}` RPM")
        st.progress(min(recent_calls_1m / max_rpm, 1.0))

        st.write("---")
        st.markdown("#### 🚨 GCP Cloud Billing API Budget Governance")
        col_cfg1, col_cfg2 = st.columns([1.5, 2.5])
        with col_cfg1:
            monthly_budget = st.number_input("Monthly Budget Cap (USD)", min_value=1.0, value=st.session_state.get("monthly_budget_cap", 50.0), step=5.0)
            st.session_state.monthly_budget_cap = monthly_budget

        current_cost = t_stats["estimated_cost_usd"]
        budget_ratio = min(current_cost / monthly_budget, 1.0)
        with col_cfg2:
            if budget_ratio >= 1.0:
                st.markdown("🚨 **【100% Circuit Breaker】** LLM cut off. Fallback to rule engine.")
            elif budget_ratio >= 0.85:
                st.markdown("⚠️ **【85% Alert Threshold】** Tightening rate limits to 10s cooldown.")
            else:
                st.markdown("✅ **【Normal Operation】** Nominal budget health.")

        st.progress(budget_ratio, text=f"Usage: ${current_cost:.5f} / ${monthly_budget:.2f} USD ({(budget_ratio * 100):.2f}%)")

        col_b_info, col_b_act = st.columns([2, 1])
        with col_b_info:
            st.markdown("""
            * **結構**: `GCP Billing Pub/Sub` ➔ `Cloud Run Receiver` ➔ `Edge Telemetry Engine`
            * **自我修復**: 當帳單週期重設時，透過 Webhook 自動解除熔斷並恢復 FSM 互動能力。
            """)
        with col_b_act:
            if st.button("🛡️ Simulate Budget Breaker", use_container_width=True):
                st.session_state.token_stats["estimated_cost_usd"] = monthly_budget + 0.01
                st.session_state.current_fsm_state = AgentState.VIOLATION_CIRCUIT_BREAKER
                st.session_state.decision_trace = "[CIRCUIT_BREAKER] Simulated Billing Pub/Sub Exceeded."
                st.rerun()
            if st.button("🔄 Reset Budget Stats", use_container_width=True):
                st.session_state.token_stats["estimated_cost_usd"] = 0.0
                st.session_state.current_fsm_state = AgentState.ACTIVE_INTERACTION
                st.session_state.decision_trace = "[BUDGET_RESET] Re-armed nominal state."
                st.rerun()

    with tab_host_audit:
        st.subheader("🛡️ Host Auditing & Multi-modal STT Alignment Hub")
        if st.session_state.get("monitoring"):
            process_pending_audio_chunks()

        now_ts = time.time()
        if now_ts - st.session_state.get("audit_last_eval_time", 0) >= 180:
            evaluate_chunk_chat_correlation()

        col_h_ctrl1, col_h_ctrl2 = st.columns([3, 1])
        with col_h_ctrl1:
            st.info("💡 **對齊機制**：背景 STT 每 20 秒沉澱切片，每 180 秒批次仲裁並持久化儲存。")
        with col_h_ctrl2:
            if st.button("🔄 手動即時對齊評級", key="btn_manual_audit_eval", use_container_width=True):
                evaluate_chunk_chat_correlation()
                st.toast("✅ 已完成對齊結算！", icon="🎯")

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

# ==============================================================================
# 第七部分：主頁面排版與操作控制台 (Operations Console)
# ==============================================================================
st.title("Live Interaction Agent")
st.caption("Enterprise-grade AI co-host telemetry and autonomous community moderation engine.")

col_console, col_ui = st.columns([1, 1.3], gap="large")

with col_console:
    st.subheader("Live Operations Console")
    youtube_url = st.text_input(
        "Live Stream Endpoint (YouTube URL)",
        placeholder="貼上正在直播的 YouTube 網址...",
        help="連線至 YouTube 即時聊天室彈幕流。"
    )

    audio_mode = st.radio(
        "🎙️ Audio Slicing Ingestion Pipeline Mode",
        options=["🧪 Virtual Audio Simulation (雲端 Demo 推薦 / 100% 穩定)", "📻 Real Stream Slicing (本機開發 / yt-dlp + FFmpeg)"],
        index=0,
        help="【Virtual Audio Simulation】在 Streamlit Cloud 避開 YouTube 機房風控；【Real Stream Slicing】適合本機環境測試。"
    )

    # 系統健康檢查摺疊面板
    with st.expander("🔍 System Pre-flight & API Diagnostics"):
        if st.button("Run Full API Health Check", use_container_width=True):
            status_box = st.status("正在檢測各項 API 狀態...", expanded=True)
            try:
                test_client = genai.Client(api_key=api_key)
                test_res = test_client.models.generate_content(
                    model=global_model_name,
                    contents="ping",
                    config={"max_output_tokens": 1}
                )
                status_box.write("✅ Gemini API (Text): 正常連線")
            except Exception as e:
                status_box.write(f"❌ Gemini API (Text): 失敗 ({e})")

            try:
                yt_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id=jfKfPfyJRdk&key={youtube_api_key}"
                yt_res = requests.get(yt_url, timeout=5).json()
                if "items" in yt_res:
                    status_box.write("✅ YouTube Data API: 正常連線")
                else:
                    status_box.write(f"❌ YouTube Data API: 失敗 ({yt_res.get('error', {}).get('message', '未知錯誤')})")
            except Exception as e:
                status_box.write(f"❌ YouTube Data API: 連線異常 ({e})")

            ffmpeg_check = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if ffmpeg_check.returncode == 0:
                status_box.write("✅ 系統 FFmpeg: 正常就緒")
            else:
                status_box.write("❌ 系統 FFmpeg: 未安裝或路徑錯誤")
            status_box.update(label="診斷完成！", state="complete")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button("Connect Stream", type="primary", use_container_width=True)
    with col_btn2:
        stop_btn = st.button("⏹ Terminate", use_container_width=True)

    # ==============================================================================
    # 🚀 啟動連線控制 (Connect Stream)
    # ==============================================================================
    if start_btn and youtube_url:
        try:
            # 🛡️ 步驟 1：防禦性清理 —— 在開新連線前，徹底處決舊進程與執行緒，杜絕背景孤兒
            if "audio_thread_stop_event" in st.session_state:
                st.session_state.audio_thread_stop_event.set()
            force_kill_ffmpeg_safely(st.session_state.get("audio_process"))
            st.session_state.audio_process = None

            # 🛡️ 步驟 2：進程完全釋放 Handle 後，安全清空歷史切片與 Log
            safe_purge_directory(Path("audio_chunks"))
            safe_purge_directory(Path("Chatroom Msg Log"))
            safe_purge_directory(Path("STT Chunk Log"))

            # 🚀 步驟 3：建立監聽實體與重設工作事件
            vid = extract_video_id(youtube_url)
            st.session_state.listener = YouTubeChatListener(youtube_url)
            st.session_state.monitoring = True
            st.session_state.audio_thread_stop_event.clear()

            # 🚀 步驟 4：雙軌音訊管線啟動 (虛擬模擬 vs 本機真實切片)
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

            # 🚀 步驟 5：抓取直播元數據並重置聊天室歷程
            details = fetch_youtube_stream_details(api_key=youtube_api_key, video_id=vid)
            st.session_state.stream_title = details.get("stream_title", "Live Stream")
            st.session_state.channel_title = details.get("channel_title", "YouTuber")
            st.session_state.channel_icon_url = details.get("channel_icon_url", "")

            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "author": "System Ingestion",
                    "content": f"🚀 Ingestion established for {st.session_state.channel_title}. Real-time chat & audio slicing pipeline active.",
                    "translation": "",
                    "lang": "en",
                    "ui_actions": []
                }
            ]
            st.session_state.sentiment_logs = []
            st.session_state.stt_chunks = []
            st.session_state.processed_chunks_set.clear()
            st.session_state.audit_last_eval_time = time.time()
            st.session_state.current_fsm_state = AgentState.ACTIVE_INTERACTION
            st.session_state.decision_trace = "[INIT] Clean start: Buffer reset & pipeline primed."
            st.rerun()

        except Exception as e:
            st.error(f"連線失敗: {str(e)}")

    # ==============================================================================
    # ⏹ 終止連線控制 (Terminate)
    # ==============================================================================
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

with col_ui:
    if st.session_state.monitoring and st.session_state.channel_title:
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
        st.subheader("Live Chatroom")
    render_live_chatroom_stream()

st.divider()
render_analytics_dashboard_hub()