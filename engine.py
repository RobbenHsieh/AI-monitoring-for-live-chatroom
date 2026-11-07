import time
import logging
from turtle import st
from urllib import response
from google import genai
from google.genai import types
from google.genai.errors import APIError

from schemas import AgentState, ActionType, UIAction, UIPayload, AgentResponsePayload

logger = logging.getLogger(__name__)
global_model_name = "models/gemini-2.5-flash-lite"  # 預設模型名稱，可透過環境變數 GEMINI_MODEL_NAME 覆蓋

class LiveInteractionEngine:
    def __init__(self, api_key: str, model_name: str = global_model_name):
        self.client = genai.Client(api_key=api_key)
        self.model = model_name
        self.last_call_time = 0.0
        self.min_interval = 2.0  # 關鍵防護：每次請求強制間隔至少 2 秒 (相當於最高 30 RPM，安全避開 15 RPM 上限)

    def evaluate_and_respond(self, idle_seconds: int, viewer_message: str) -> AgentResponsePayload:
        now = time.time()
        elapsed = now - self.last_call_time

        # --- 節流機制：若距離上次呼叫未滿 5 秒，直接走本地降級，不浪費 API 額度 ---
        if elapsed < self.min_interval:
            wait_needed = round(self.min_interval - elapsed, 1)
            logger.warning(f"觸發頻率保護，冷卻中（剩餘 {wait_needed} 秒）")
            # 依然給予狀態判定，但不打擾 LLM
            return self._fallback_response(
                AgentState.ACTIVE_INTERACTION if viewer_message else AgentState.IDLE_ICEBREAKER,
                f"【頻率節流】呼叫過於頻繁，系統保護中 (冷卻剩餘 {wait_needed}s)"
            )

        # ==============================================================================
        # 👇 1. 將閒置判定門檻改為 300 秒 (5 分鐘)
        # ==============================================================================
        if idle_seconds >= 300 and not viewer_message.strip():
            target_state = AgentState.IDLE_ICEBREAKER
        elif any(toxic in viewer_message.lower() for toxic in ["退費詐欺", "scam", "fraud", "違禁詞", "fake"]):
            target_state = AgentState.VIOLATION_CIRCUIT_BREAKER
        else:
            target_state = AgentState.ACTIVE_INTERACTION

        # ==============================================================================
        # 👇 2. 在 Prompt 中明確指示破冰長度為 1-3 句話
        # ==============================================================================
        system_instruction = f"""
You are a professional, charismatic live stream co-host and community moderator (Hype-bot).
Current system state: {target_state.value}.

【Language Guidelines】:
1. Detect the original language of the viewer (original_language).
2. display_message must match the detected language (or English/Traditional Chinese if initiating).

【State Behavior Guidelines】:
- IDLE_ICEBREAKER:
  * Triggered after 5 minutes of chat silence.
  * You MUST speak exactly 1 to 3 engaging sentences (1-3 sentences) to reignite chat activity.
  * Ask an open-ended question or bring up an interesting topic related to the current stream.
  * Provide 2~3 interactive button options in ui_actions.
- ACTIVE_INTERACTION:
  * Warm, concise, and helpful responses to viewer questions (1-2 sentences).
- VIOLATION_CIRCUIT_BREAKER:
  * Strictly neutral/official tone reminding community guidelines. No buttons.
"""
        prompt = f"Idle Duration: {idle_seconds}s\nViewer Message: {viewer_message}"

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                response_mime_type="application/json",
                response_schema=AgentResponsePayload,
            )

            # 更新呼叫時間戳記
            self.last_call_time = time.time()
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            # 1. 先透過 Pydantic 反序列化 JSON 文本
            result_payload = AgentResponsePayload.model_validate_json(response.text)

            # 2. 將 usage_metadata 動態附加到 Pydantic 實體上 (如果 Pydantic 模型有開放或允許動態屬性)
            usage = getattr(response, "usage_metadata", None)
            result_payload.__dict__["usage_metadata"] = usage

            # 3. 回傳包含 Token 數據的物件給前端
            return result_payload

        except APIError as e:
            logger.error(f"Google GenAI API 錯誤 (代碼 {e.code}): {e.message}")
            return self._fallback_response(target_state, f"【API 錯誤】{e.message}")

        except Exception as e:
            logger.error(f"非預期例外錯誤: {str(e)}")
            return self._fallback_response(target_state, f"【系統例外】{type(e).__name__}: {str(e)}")
            

    def _fallback_response(self, state: AgentState, fallback_text: str) -> AgentResponsePayload:
        actions = []
        if state == AgentState.IDLE_ICEBREAKER:
            actions = [
                UIAction(
                    action_type=ActionType.RENDER_BUTTONS,
                    payload=UIPayload(buttons=["🔥 Trending Items", "🎁 Claim Stream Voucher"])
                )
            ]
        elif state == AgentState.VIOLATION_CIRCUIT_BREAKER:
            actions = [
                UIAction(
                    action_type=ActionType.TRIGGER_ALERT,
                    payload=UIPayload(alert_level="warning", alert_message="Community guidelines triggered.")
                )
            ]

        return AgentResponsePayload(
            current_state=state,
            thought_process=f"[FALLBACK TRIGGERED] 原因: {fallback_text}",
            original_language="zh-TW",
            display_message="系統目前載入預設互動內容 / System fallback active.",
            translation_zh="系統目前載入預設互動內容。",
            ui_actions=actions
        )

# ----------------- 獨立驗證連線測試 -----------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # 1. 載入 .env 檔案
    load_dotenv(dotenv_path=r"C:\Users\sky00\Desktop\PythonDir\GEMINI_API_KEY.env")
    test_api_key = os.getenv("GEMINI_API_KEY")

    print("\n🔍 === 開始檢查 Google GenAI 連線狀態 ===")
    
    # 2. 檢查 API Key 是否成功讀取
    if not test_api_key:
        print("❌ 錯誤：找不到 GEMINI_API_KEY！請確認專案根目錄下有 .env 檔案，且內容為 GEMINI_API_KEY=AIzaSy...")
        exit(1)
    else:
        # 印出前後幾碼確認有讀到值，中間遮蔽
        masked_key = test_api_key[:6] + "..." + test_api_key[-4:]
        print(f"  API Key 讀取成功：{masked_key}")

    # 3. 初始化引擎並發送測試請求
    try:
        print("🚀 正在向 Google GenAI 發送請求 (模型: " + global_model_name + ")...")
        # 若使用的是其他模型名稱，例如 gemini-1.5-flash，可在此處調整
        engine = LiveInteractionEngine(api_key=test_api_key, model_name=global_model_name)
        
        # 模擬一次正常的用戶問句
        result = engine.evaluate_and_respond(
            idle_seconds=10, 
            viewer_message="這件外套有防水功能嗎？"
        )

        print("\n🎉 === 連線測試成功！回傳結果如下 ===")
        print(f"狀態機狀態: {result.current_state}")
        print(f"決策邏輯 (Thought): {result.thought_process}")
        print(f"主播回覆文字: {result.display_message}")
        print(f"UI 動態元件: {result.ui_actions}")

        # 檢查是否不幸落入 fallback
        if "[FALLBACK TRIGGERED]" in result.thought_process:
            print("\n⚠️ 注意：雖然沒有崩潰，但觸發了本地降級防護 (Fallback)！請查看上方決策邏輯中的具體原因。")
        else:
            print("\n 完美！Google GenAI 結構化輸出運作正常！")

    except Exception as e:
        print(f"\n❌ 執行過程中發生未捕捉的異常: {type(e).__name__} - {str(e)}")