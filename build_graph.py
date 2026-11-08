import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# =============================================================
# command: python build_graph.py
# =============================================================

# 載入金鑰
current_dir = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=current_dir / "GEMINI_API_KEY.env")
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ 請確認 .env 中有 GEMINI_API_KEY")
    exit(1)

client = genai.Client(api_key=api_key)

# 1. 蒐集本機專案關鍵程式碼
project_files = ["schemas.py", "engine.py", "app.py", "youtube_stream.py"]
code_context = ""

for file_name in project_files:
    file_path = Path(file_name)
    if file_path.exists():
        code_context += f"\n\n--- 檔案: {file_name} ---\n" + file_path.read_text(encoding="utf-8")

print("🔍 正在請 Gemini 3.5-flash-lite 分析專案架構並生成 Graphviz DOT 語法...")

# 2. 讓 Gemini 分析模組相依性並產出 DOT 格式
prompt = f"""
分析以下 Python 直播 Agent 專案的程式碼。
請產出一個標準、美觀且具備層次感的 Graphviz DOT 語言語法（以 digraph ArchitectureGraph 開頭）。

要求：
1. 必須在 graph、node、edge 屬性中全局指定中文字型：
   node [fontname="Microsoft JhengHei"];
   edge [fontname="Microsoft JhengHei"];
   graph [fontname="Microsoft JhengHei"];
2. 節點要包含主要檔案、關鍵 Class 與函式（如 LiveInteractionEngine, AgentState, YouTubeChatListener, FSM 狀態流轉）。
3. 用箭頭標示資料傳遞與引用關係（Imports, Calls, State Transition）。
4. 只回傳純 DOT 語法代碼區塊，不要包含額外文字。

原始碼如下：
{code_context}
"""

response = client.models.generate_content(
    model="models/gemini-3.5-flash-lite",
    contents=prompt
)

dot_code = response.text.replace("```dot", "").replace("```graphviz", "").replace("```", "").strip()

# 3. 存檔並調用本機的 Graphviz 轉成 SVG/PNG
dot_file = Path("architecture.dot")
dot_file.write_text(dot_code, encoding="utf-8")
print("✅ DOT 檔案已生成：architecture.dot")

try:
    subprocess.run(["dot", "-Tpng", "architecture.dot", "-o", "architecture_graph.png"], check=True)
    print("🎉 成功！已使用本地 Graphviz 渲染出架構圖：architecture_graph.png")
except Exception as e:
    print(f"⚠️ 調用本地 dot 指令失敗: {e}，但你可以將 architecture.dot 內容複製至線上 Graphviz 預覽。")