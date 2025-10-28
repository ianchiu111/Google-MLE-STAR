# mle_star_tool_node.py
from __future__ import annotations
import os, re, shutil, subprocess, datetime, json
from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv(".env")

# LangChain / LangGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import AIMessage

# ---------- 參數 Schema（精簡） ----------
class MLEStarArgs(BaseModel):
    dataset: str = Field(..., description="CSV 資料集路徑")
    target: str = Field(..., description="目標欄位名稱")
    output_dir: Optional[str] = Field(None, description="模型與輸出目錄，例如 ./models/churn")
    name: Optional[str] = Field(None, description='實驗名稱，例如 "revenue-prediction"')

    # STAR 流程控制
    search_iterations: int = Field(3, ge=1)
    refinement_iterations: int = Field(5, ge=1)
    max_agents: int = Field(6, ge=1)
    interactive: bool = Field(False)
    chaining: bool = Field(True)
    timeout_ms: int = Field(4 * 60 * 60 * 1000)
    verbose: bool = Field(False)
    output_format: Literal["text", "stream-json"] = Field("text")

    # 固定用你的 OAI-compatible（Ollama）端點
    llm_base_url: str = Field("http://127.0.0.1:11434/v1")
    llm_api_key: str = Field("ollama")
    llm_model: str = Field("qwen2.5:7b-instruct")

# ---------- Tool：永遠使用 claude-flow + --claude ----------
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
    llm_base_url: str = os.getenv("llm_base_url"),
    llm_api_key: str = os.getenv("llm_api_key"),
    llm_model: str = os.getenv("llm_model"),
) -> Dict[str, Any]:

    """
    use Claude-flow to run MLE-STAR process for the given dataset and target.
    """

    # 1) 指令列：優先用全域 claude-flow；否則用 npx（加 --yes 並鎖版本）
    cf = shutil.which("claude-flow")
    if cf:
        cmd = [cf, "automation", "mle-star"]
    else:
        npx = shutil.which("npx")
        if not npx:
            raise RuntimeError("找不到 'claude-flow' 或 'npx'。請先安裝：npm i -g claude-flow@2.7.26")
        cmd = [npx, "--yes", "claude-flow@2.7.26", "automation", "mle-star"]

    # 2) 固定帶 --claude，其餘旗標依參數組裝
    cmd += ["--dataset", dataset, "--target", target, "--claude"]
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cmd += ["--output", output_dir]
    if name:
        cmd += ["--name", name]
    cmd += [
        "--search-iterations", str(search_iterations),
        "--refinement-iterations", str(refinement_iterations),
        "--max-agents", str(max_agents),
        "--output-format", output_format,
        "--timeout", str(timeout_ms),
    ]
    cmd += ["--interactive" if interactive else "--non-interactive"]
    cmd += ["--chaining" if chaining else "--no-chaining"]
    if verbose:
        cmd += ["--verbose"]

    # 3) 子行程環境：直寫到 Claude Code 讀取的三個變數
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = llm_base_url
    env["ANTHROPIC_AUTH_TOKEN"] = llm_api_key
    env["ANTHROPIC_MODEL"] = llm_model

    # 4) 執行與日誌
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
    merged_out = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")

    # 5) 摘要資訊（可選解析）
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

    tail_lines = merged_out.strip().splitlines()[-200:]
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": cmd,
        "execution_id": exec_id,
        "tasks_completed": tasks_done,
        "log_path": log_path,
        "tail": "\n".join(tail_lines),
    }

# ---------- ToolNode 與最小 Graph ----------
mle_star_tool_node = ToolNode([mle_star_tool])

def build_minimal_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("mle_star", mle_star_tool_node)
    graph.add_edge(START, "mle_star")
    graph.add_edge("mle_star", END)
    return graph.compile()

def run_star_via_toolnode(args: Dict[str, Any]) -> Dict[str, Any]:
    app = build_minimal_graph()
    ai = AIMessage(content="Run MLE-STAR",
                   tool_calls=[{"name": "mle_star", "args": args, "id": "call_mle_star_1"}])
    out_state = app.invoke({"messages": [ai]})
    last_msg = out_state["messages"][-1]
    try:
        payload = json.loads(last_msg.content) if isinstance(last_msg.content, str) else last_msg.content
    except Exception:
        payload = {"raw": last_msg.content}
    return payload

# ---------- 範例直接跑 ----------
if __name__ == "__main__":
    example_args = {
        "dataset": "data/train.csv",
        "target": "Sales",
        "output_dir": "data/",
        "name": "sales-prediction",
        "search_iterations": 3,
        "refinement_iterations": 5,
        "max_agents": 6,
        "interactive": False,
        "chaining": True,
        "timeout_ms": 4 * 60 * 60 * 1000,
        "verbose": True,
        "output_format": "text",

        # 你的本機 Ollama（OpenAI-compatible）模型設定
        # "llm_base_url": "http://127.0.0.1:11434/v1",
        # "llm_api_key": "ollama",
        # "llm_model": "qwen2.5:7b-instruct",
        "llm_base_url": os.getenv("llm_base_url"),
        "llm_api_key": os.getenv("llm_api_key"),
        "llm_model": os.getenv("llm_model")
    }
    result = run_star_via_toolnode(example_args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
