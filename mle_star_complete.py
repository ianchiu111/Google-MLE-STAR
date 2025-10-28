# mle_star_tool_node.py
from __future__ import annotations
import os, re, shutil, subprocess, datetime, json
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, validator

# LangChain / LangGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage

# ---------- 參數 Schema ----------
class MLEStarArgs(BaseModel):
    dataset: str = Field(..., description="CSV 資料集路徑")
    target: str = Field(..., description="目標欄位名稱")
    output_dir: Optional[str] = Field(None, description="模型與輸出目錄，例如 ./models/churn")
    name: Optional[str] = Field(None, description='實驗名稱，例如 "revenue-prediction"')

    # STAR 流程控制
    search_iterations: int = Field(3, ge=1, description="搜尋迭代次數（預設 3）")
    refinement_iterations: int = Field(5, ge=1, description="精煉迭代次數（預設 5）")
    max_agents: int = Field(6, ge=1, description="可併發 agent 上限（預設 6）")
    interactive: bool = Field(False, description="是否使用互動模式（預設 False）")
    chaining: bool = Field(True, description="是否啟用 stream chaining（預設 True）")
    timeout_ms: int = Field(4 * 60 * 60 * 1000, description="CLI 逾時（毫秒），預設 4 小時")
    verbose: bool = Field(False, description="顯示詳細日誌（--verbose）")
    output_format: Literal["text", "stream-json"] = Field(
        "text", description="CLI 輸出格式（預設 text；stream-json 用於串連）"
    )

    # ★ 這裡是你要的「不用 Claude 模型，而接 OAI-compatible 端點（Ollama）」設定
    llm_base_url: Optional[str] = Field(
        "http://127.0.0.1:11434/v1",
        description="OpenAI-compatible base URL（Ollama 預設 http://127.0.0.1:11434/v1）",
    )
    llm_api_key: Optional[str] = Field(
        "ollama",
        description="OpenAI-compatible 端點的 API key/token（Ollama 可填佔位字串）",
    )
    llm_model: Optional[str] = Field(
        "qwen2.5:7b-instruct",
        description="要走的模型 slug（會寫入 ANTHROPIC_MODEL 讓 CLI 認得）",
    )
    use_claude_cli: bool = Field(
        True,
        description="是否啟用 Claude Code CLI 整合旗標（--claude）。即使不是 Anthropic，也建議保持 True 讓 MLE-STAR 正常運轉。",
    )

# ---------- Tool 實作 ----------
@tool("mle_star", args_schema=MLEStarArgs)
def mle_star_tool(
    dataset: str,
    target: str,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    search_iterations: int = 3,
    refinement_iterations: int = 5,
    max_agents: int = 6,
    interactive: bool = False,
    chaining: bool = True,
    timeout_ms: int = 4 * 60 * 60 * 1000,
    verbose: bool = False,
    output_format: str = "text",

    # ollama QWEN Model INSTEAD OF CLAUDE Code API
    llm_base_url: Optional[str] = "http://127.0.0.1:11434/v1",
    llm_api_key: Optional[str] = "ollama",
    llm_model: Optional[str] = "qwen2.5:7b-instruct",
    use_claude_cli: bool = True,
) -> Dict[str, Any]:
    """
    以 CLI 方式執行 MLE-STAR Workflow，並回傳結果摘要與日誌路徑。

    重點：
      - 我們仍使用 `claude-flow automation mle-star` 的自動化指令。
      - 但把 Claude Code 的 BASE_URL / TOKEN / MODEL 指到你的 OpenAI-compatible 端點（例如本機 Ollama）。
        依官方文件，設定 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL 即可。 
    """

    # 1) 找到 CLI（優先使用全域 claude-flow；否則退回 npx）
    cmd: list[str]
    cf = shutil.which("claude-flow")
    if cf:
        cmd = [cf, "automation", "mle-star"]
    else:
        npx = shutil.which("npx")
        if not npx:
            raise RuntimeError("找不到 'claude-flow' 或 'npx'，請先依官方文件完成安裝。")
        cmd = [npx, "claude-flow@alpha", "automation", "mle-star"]

    # 2) 組合旗標（依官方 wiki）
    cmd += ["--dataset", dataset, "--target", target]
    if use_claude_cli:
        # 啟用 CLI 的 LLM 整合；實際模型將由環境變數決定（可指到 Ollama）
        cmd += ["--claude"]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cmd += ["--output", output_dir]
    if name:
        cmd += ["--name", name]
    if search_iterations:
        cmd += ["--search-iterations", str(search_iterations)]
    if refinement_iterations:
        cmd += ["--refinement-iterations", str(refinement_iterations)]
    if max_agents:
        cmd += ["--max-agents", str(max_agents)]
    if interactive:
        cmd += ["--interactive"]
    else:
        cmd += ["--non-interactive"]
    cmd += ["--output-format", output_format]
    cmd += ["--timeout", str(timeout_ms)]
    if chaining:
        cmd += ["--chaining"]
    else:
        cmd += ["--no-chaining"]
    if verbose:
        cmd += ["--verbose"]

    # 3) 建立子行程環境：把 OAI-compatible 端點寫進 Claude Code 的環境變數
    #    參考官方“Using Claude Code with Open Models”：只要提供 ANTHROPIC_BASE_URL / AUTH_TOKEN / MODEL 即可。
    env = os.environ.copy()
    if llm_base_url:
        env["ANTHROPIC_BASE_URL"] = llm_base_url
    if llm_api_key:
        env["ANTHROPIC_AUTH_TOKEN"] = llm_api_key
    if llm_model:
        env["ANTHROPIC_MODEL"] = llm_model

    # 基本健檢：若使用 CLI 整合但沒任何可用的身分/端點資訊，就提醒
    if use_claude_cli and not (
        env.get("ANTHROPIC_API_KEY") or (env.get("ANTHROPIC_BASE_URL") and env.get("ANTHROPIC_AUTH_TOKEN"))
    ):
        raise RuntimeError(
            "請提供 ANTHROPIC_API_KEY（雲端 Anthropic）或 llm_base_url + llm_api_key（本機/代理端點）。"
        )

    # 4) 執行並擷取輸出（stdout + stderr）
    completed = subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        env=env,  # ★ 關鍵：把上面注入的 OAI-compatible 設定傳給 CLI
    )
    merged_out = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")

    # 5) 解析摘要
    exec_id = None
    m = re.search(r"Execution ID:\s*(\S+)", merged_out)
    if m:
        exec_id = m.group(1)

    tasks_done = None
    m2 = re.search(r"Results:\s*(\d+/\d+)\s*tasks\s*completed", merged_out, re.I)
    if m2:
        tasks_done = m2.group(1)

    # 6) 寫入日誌檔
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = output_dir or "."
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"mle_star_{ts}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(merged_out)

    # 只回傳末尾 200 行，避免訊息過長
    tail_lines = merged_out.strip().splitlines()[-200:]
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": cmd,
        "execution_env": {
            "ANTHROPIC_BASE_URL": env.get("ANTHROPIC_BASE_URL"),
            "ANTHROPIC_MODEL": env.get("ANTHROPIC_MODEL"),
            # 不回傳 token
        },
        "execution_id": exec_id,
        "tasks_completed": tasks_done,
        "log_path": log_path,
        "output_dir": output_dir,
        "tail": "\n".join(tail_lines),
    }

# ---------- 把 Tool 包成 ToolNode ----------
mle_star_tool_node = ToolNode([mle_star_tool])

from langgraph.graph import StateGraph, START, END, MessagesState
# ---------- （選用）最小 Graph：直接從 ToolNode 開始執行 ----------
def build_minimal_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("mle_star", mle_star_tool_node)
    graph.add_edge(START, "mle_star")
    graph.add_edge("mle_star", END)
    return graph.compile()

def run_star_via_toolnode(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    手動構造一個 AIMessage，指定要呼叫 'mle_star'，把參數丟給 ToolNode。
    回傳執行後的 state（其中最後一則 message 會是 ToolMessage，content 即為 tool 的回傳 dict）。
    """
    app = build_minimal_graph()
    ai = AIMessage(
        content="Run MLE-STAR",
        tool_calls=[{"name": "mle_star", "args": args, "id": "call_mle_star_1"}],
    )
    out_state = app.invoke({"messages": [ai]})
    last_msg = out_state["messages"][-1]
    try:
        payload = json.loads(last_msg.content) if isinstance(last_msg.content, str) else last_msg.content
    except Exception:
        payload = {"raw": last_msg.content}
    return payload

# ---------- （範例）實際呼叫 ----------
if __name__ == "__main__":
    example_args = {
        "dataset": "data/train.csv",
        "target": "Sales",
        "output_dir": "data/",
        "name": "sales-prediction",
        "search_iterations": 5,
        "refinement_iterations": 8,
        "max_agents": 8,
        "interactive": False,
        "chaining": True,
        "timeout_ms": 4 * 60 * 60 * 1000,
        "verbose": True,
        "output_format": "text",

        # ★ 你的本機 Ollama（OpenAI-compatible）模型設定
        "llm_base_url": "http://127.0.0.1:11434/v1",
        "llm_api_key": "ollama",
        "llm_model": "qwen2.5:7b-instruct",
        "use_claude_cli": True,  # 仍啟用 CLI 整合，但實際模型是上面這個
    }
    result = run_star_via_toolnode(example_args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
