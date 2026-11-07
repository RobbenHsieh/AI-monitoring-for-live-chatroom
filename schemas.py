from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field

class AgentState(str, Enum):
    IDLE_ICEBREAKER = "IDLE_ICEBREAKER"           # 沉默破冰
    ACTIVE_INTERACTION = "ACTIVE_INTERACTION"     # 正常互動
    VIOLATION_CIRCUIT_BREAKER = "CIRCUIT_BREAKER" # 違規熔斷

class ActionType(str, Enum):
    RENDER_TEXT = "RENDER_TEXT"
    RENDER_BUTTONS = "RENDER_BUTTONS"
    TRIGGER_ALERT = "TRIGGER_ALERT"

# 關鍵修正：將原本的 dict 替換為明確欄位的 Model，避開 additionalProperties 限制
class UIPayload(BaseModel):
    buttons: List[str] = Field(default_factory=list, description="要渲染的按鈕文字清單")
    alert_level: str = Field(default="info", description="警告等級: info, warning, error")
    alert_message: str = Field(default="", description="警示或彈窗內容")

class UIAction(BaseModel):
    action_type: ActionType
    payload: UIPayload = Field(default_factory=UIPayload, description="UI 所需參數")

class AgentResponsePayload(BaseModel):
    current_state: AgentState
    thought_process: str = Field(description="狀態決策依據或思維鏈")
    original_language: str = Field(default="zh-TW", description="觀眾發言語言，例如 en, ja, ko, zh-TW")
    display_message: str = Field(description="向直播間呈現的回覆文字（務必與觀眾發言使用相同語言）")
    translation_zh: str = Field(default="", description="display_message 的繁體中文對照翻譯（方便後台管理人員審閱）")
    ui_actions: List[UIAction] = Field(default_factory=list, description="動態 UI 元件清單")

class AgentResponsePayload(BaseModel):
    current_state: AgentState
    thought_process: str
    original_language: str
    display_message: str
    translation_zh: str
    ui_actions: List[UIAction] = []
    usage_metadata: Optional[Any] = None  # 👈 加上這行供 engine 塞入 Token 數據