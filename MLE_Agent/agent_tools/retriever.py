from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

import requests
from typing import Annotated
from bs4 import BeautifulSoup
from readability import Document
import logging

import re
from urllib.parse import urlparse

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s :: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("deep_research")


def log_state_delta(before: dict, after: dict):
    """簡單比較 state 關鍵欄位的變化，避免輸出過大內容"""
    def lens(d, k): return len(d.get(k, {})) if isinstance(d.get(k, {}), dict) else len(d.get(k, []))
    msg = (
        f"messages={len(after.get('messages', []))} "
        f"hits={len(after.get('hits', []))} "
        f"pages={lens(after, 'pages')} "
        f"notes={len(after.get('notes', []))} "
        f"citations={len(after.get('citations', []))} "
        f"rounds={after.get('rounds', 0)} "
        f"stop={after.get('stop', False)} "
        f"done={after.get('done', False)}"
    )
    logger.info("[STATE] " + msg)


# ---------- 狀態 ----------
class RState(TypedDict):
    messages: List[BaseMessage]
    queries: List[str]
    hits: List[Dict[str, Any]]      # [{'title','url','snippet'}]
    pages: Dict[str, str]           # url -> cleaned_text
    notes: List[str]                # 摘要/重點
    citations: List[Dict[str, Any]] # [{'url','quote'}]
    done: bool
    stop: bool                      # 新增：由 LLM 發出 STOP/REPORT 訊號時設為 True
    rounds: int                     # 新增：疊代輪數


# ---------- 可調參數 ----------
MAX_ROUNDS = 8        # 迴圈再多也會硬停（避免 GraphRecursionError）
HITS_TARGET = 8       # 命中數達標 + 有一定筆記就可報告
MIN_NOTE_ROUNDS = 2   # 至少做兩輪 extract 再收斂
QUERY_HISTORY_MAX = 30
QUERY_TOPK_PER_ROUND = 2       # 每輪最多新增 2 條查詢
AUTO_FETCH_TOPK = 3            # 每輪自動抓前 3 個多樣化來源的頁面


# ---------- API Key ----------
my_TavilyClient_key   = "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"


# ---------- 工具：搜尋 ----------
tavily = TavilyClient(api_key=my_TavilyClient_key)
@tool("web_search", return_direct=False)
def web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """用搜尋引擎找相關頁面，回傳 [{'title','url','snippet'}]"""
    try:
        logger.info("[TOOL:web_search] q=%s k=%s", query, k)
        res = tavily.search(query=query, max_results=k)
        out = []
        for r in res.get("results", []):
            out.append({"title": r.get("title",""),
                        "url": r.get("url",""),
                        "snippet": r.get("content","")})
        logger.info("[TOOL:web_search] %s results", len(out))
        return out
    except Exception as e:
        logger.exception("[TOOL:web_search] failed q=%s err=%s", query, e)
        return []


# ---------- 工具：抓頁面 ----------
@tool("fetch_page", return_direct=False)
def fetch_page(url: str) -> str:
    """抓取網頁並清洗成可讀文字"""
    headers = {"User-Agent": "Mozilla/5.0 (deep-research/0.1)"}
    resp = requests.get(url, headers=headers, timeout=25)
    resp.raise_for_status()
    html = resp.text
    doc = Document(html)
    cleaned_html = doc.summary()
    text = BeautifulSoup(cleaned_html, "html.parser").get_text("\n")
    return text[:120000]


# ---------- LLM ----------
llm = ChatOpenAI(
    base_url="http://127.0.0.1:11434/v1",  # ← 用 127.0.0.1 與 11434
    api_key="ollama",                      # ← 佔位即可
    model="qwen2.5:7b-instruct",         # ← 跟你 pull 的模型一致
    temperature=0,
).bind_tools([web_search, fetch_page])


# System prompt for the orchestrator plan_node
SYSTEM_ORCHESTRATOR = """
你是 Deep Research Orchestrator。可用工具：
1) web_search(query: str, k: int=5)
2) fetch_page(url: str)

策略（務必遵守）：
- 每輪至多產生 1~2 條**全新且不重複**的搜尋式；避免與歷史查詢相同或僅是同義重寫。
- 多樣化查詢維度：關鍵詞（學術 vs. 產業）、運算子（site:, filetype:pdf）、時序（2024..2025）、同義詞與縮寫（e.g., demand forecasting, M5/M4, XGBoost, LightGBM, CatBoost, TFT, N-BEATS, DeepAR, PatchTST, TimesNet, Chronos, Moirai）。
- 若已取得多個來源能相互印證，請回覆「REPORT」。

操作規則：
- 需要搜尋 → 呼叫 web_search(query=..., k=...)
- 需要抓正文 → 呼叫 fetch_page(url=...)
- 準備彙整 → 只回覆「REPORT」
- 不可再有價值 → 只回覆「STOP」

禁止：
- 重複歷史查詢內容（包含僅微小變形）。
- 除工具呼叫與 REPORT/STOP 外的冗長敘述。
"""


# ---------- 節點：規劃/決策 ----------

def normalize_query(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q

def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""

def diversify_hits(hits, k=3):
    """選出 domain 多樣化的前 k 個連結"""
    chosen, seen = [], set()
    for h in hits:
        d = domain_of(h.get("url",""))
        if not h.get("url"): 
            continue
        if d in seen: 
            continue
        chosen.append(h)
        seen.add(d)
        if len(chosen) >= k:
            break
    return chosen


def plan_node(state: RState):
    before = dict(state)
    past_q = [normalize_query(q) for q in state.get("queries", [])][-QUERY_HISTORY_MAX:]
    obs = (
        "目前查詢（歷史，請勿重複）: {q}\n"
        "已蒐集連結(前10): {urls}\n"
        "已有重點段落數: {n}\n"
        "提示：若資料足夠請回覆 REPORT；若確定無法再拓展請回覆 STOP。"
    ).format(
        q=past_q,
        urls=[h["url"] for h in state.get("hits", [])][:10],
        n=len(state.get("notes", [])),
    )

    logger.info("[PLAN] rounds=%s hits=%s pages=%s notes=%s",
                state.get("rounds", 0),
                len(state.get("hits", [])),
                len(state.get("pages", {})),
                len(state.get("notes", [])))

    msg = llm.invoke([SystemMessage(content=SYSTEM_ORCHESTRATOR),
                      HumanMessage(content=obs)])

    new_state = {"messages": state["messages"] + [msg]}
    content = (getattr(msg, "content", "") or "").strip().upper()
    if content in {"REPORT", "STOP"}:
        new_state["stop"] = True
        logger.info("[PLAN] LLM signaled %s", content)

    new_state["rounds"] = state.get("rounds", 0) + 1
    after = dict(state, **new_state)
    log_state_delta(before, after)
    return new_state


# ---------- 節點：執行工具（由 LLM 決定呼叫 web_search/fetch_page） ----------
def act_node(state: RState):
    before = dict(state)
    last = state["messages"][-1]
    out = {}
    if hasattr(last, "tool_calls") and last.tool_calls:
        out_hits = state["hits"][:]
        out_pages = dict(state["pages"])
        msgs = state["messages"][:]

        for call in last.tool_calls:
            if call["name"] == "web_search":
                raw_q = call["args"].get("query", "")
                q = normalize_query(raw_q)
                # 若重複，直接跳過並警告
                if q in {normalize_query(x) for x in state.get("queries", [])}:
                    logger.warning("[ACT] DUP_QUERY skipped: %s", q)
                    continue

                logger.info("[ACT] -> web_search q=%s k=%s", q, call["args"].get("k", 5))
                res = web_search.invoke({"query": q, "k": call["args"].get("k", 5)})
                out_hits.extend(res)
                state["queries"].append(q)  # 確認記錄下「真的執行過」的查詢
                msgs.append(AIMessage(content=f"[web_search] {len(res)} results"))

                # 列出本輪新命中
                new_count = 0
                for h in res:
                    title = (h.get("title") or "").strip()
                    url = (h.get("url") or "").strip()
                    dom = domain_of(url)
                    if not url: 
                        continue
                    logger.info("[HITS] %s | %s | %s", title[:100], url, dom)
                    new_count += 1
                logger.info("[HITS] new=%s (total=%s)", new_count, len(out_hits))

                # 自動抓多樣化前 N 個頁面
                to_fetch = diversify_hits(res, k=AUTO_FETCH_TOPK)
                for h in to_fetch:
                    u = h.get("url")
                    if not u or u in out_pages:
                        continue
                    logger.info("[AUTO_FETCH] %s", u)
                    try:
                        text = fetch_page.invoke({"url": u})
                        out_pages[u] = text
                        msgs.append(AIMessage(content=f"[fetch_page] {u} len={len(text)}"))
                    except Exception as e:
                        logger.exception("[AUTO_FETCH] failed url=%s err=%s", u, e)


            elif call["name"] == "fetch_page":
                url = call["args"]["url"]
                logger.info("[ACT] -> fetch_page url=%s", url)
                try:
                    text = fetch_page.invoke(call["args"])
                    out_pages[url] = text
                    msgs.append(AIMessage(content=f"[fetch_page] {url} len={len(text)}"))
                except Exception as e:
                    logger.exception("[ACT] fetch_page failed url=%s err=%s", url, e)
                    msgs.append(AIMessage(content=f"[fetch_page] ERROR {url}: {e}"))

        out = {"messages": msgs, "hits": out_hits, "pages": out_pages}
    else:
        logger.info("[ACT] no tool_calls in last message")
    after = dict(state, **out)
    log_state_delta(before, after)
    return out



# ---------- 節點：抽取與寫筆記（含引註） ----------
def extract_node(state: RState):
    before = dict(state)
    urls_for_llm = list(state["pages"].keys())[:6]
    bundle = "\n\n".join(
        f"URL: {u}\n---\n{state['pages'][u][:5000]}" for u in urls_for_llm
    ) or "（目前沒有正文）"

    sys = "請從網頁文字中整理出適合 AI Engineer (AI 開發者) 適合實作的機器學習演算法與程式碼範例。"
    msg = llm.invoke(
        [HumanMessage(content=f"{sys}\n\n{bundle}\n\n請條列3-6點洞見，每點末尾附(來源:URL)。")]
    )

    new_notes = (state["notes"] or []) + [msg.content]
    new_cites = (state["citations"] or []) + [{"url": u, "quote": "see notes"} for u in urls_for_llm]

    # 達標條件：hits 足夠 & 至少兩輪 extract；或 plan_node 發出 stop；或超過 MAX_ROUNDS
    done = (
        (len(state.get("hits", [])) >= HITS_TARGET and len(new_notes) >= MIN_NOTE_ROUNDS)
        or state.get("stop", False)
        or state.get("rounds", 0) >= MAX_ROUNDS
    )

    out = {"messages": state["messages"] + [msg],
           "notes": new_notes,
           "citations": new_cites,
           "done": done}
    after = dict(state, **out)
    logger.info("[EXTRACT] notes+=1 urls_included=%s done=%s", len(urls_for_llm), done)
    log_state_delta(before, after)
    return out



# ---------- 節點：收斂產出最終報告 ----------
def report_node(state: RState):
    before = dict(state)

    # 1) 把 notes 與 citations 串成輸入
    refs = [c['url'] for c in state.get("citations", []) if c.get('url')]
    unique_refs = []
    seen = set()
    for u in refs:
        if u not in seen:
            unique_refs.append(u); seen.add(u)

    body = "\n\n".join(state.get("notes", [])) or "（目前沒有筆記內容）"
    ref_block = "\n".join(f"- {u}" for u in unique_refs) or "- （尚無引用）"

    # 2) 明確規範輸出格式：分類 → 每類至少 1 個可執行 code block → 最後列 References
    prompt = f"""
你是一名資深 AI 工程師。請根據「研究筆記」彙整**可直接實作的**模型與範例碼，並附上參考連結。

【輸出格式要求】
1) 分類：將演算法依性質分組。
2) 每一類至少給 1~3 個「可執行」Python 範例（以 ```python 區塊輸出），必要時包含 `pip install`、`import`、資料前處理、fit/predict、簡短評估（如 MAPE/RMSE）。
3) 如果是 GitHub 專案，給出最小可運行範例（可用假資料或簡化版），並附該 repo 連結。
4) 在每個小節最後以「Sources: <URL1>, <URL2>」列出用到的來源。
5) 文末單獨列出 **References**（去重後的 URL 清單）。

【研究筆記】
{body}

【候選引用（去重後）】
{ref_block}

請嚴格遵守上面的輸出格式，直接產出結果。
"""

    logger.info("[REPORT] composing final report with %s notes and %s citations",
                len(state.get("notes", [])), len(state.get("citations", [])))

    msg = llm.invoke([HumanMessage(content=prompt)])
    out = {"messages": state["messages"] + [msg]}
    after = dict(state, **out)
    log_state_delta(before, after)
    return out



# ---------- (web research + deep research) workflow  ----------

def should_continue(state: RState):
    if state.get("rounds", 0) >= MAX_ROUNDS:
        logger.warning("[FLOW] Reached MAX_ROUNDS=%s -> report", MAX_ROUNDS)
        return "report"
    return "report" if state.get("done") else "plan"

def create_retriever_workflow():
    graph = StateGraph(RState)
    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("extract", extract_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "extract")
    graph.add_conditional_edges("extract", should_continue, {"plan": "plan", "report": "report"})
    graph.add_edge("report", END)

    retriever_workflow = graph.compile()

    return retriever_workflow


@tool
def retriever_tool(
    user_query: Annotated[str, "string query"]
):
    """
    use this tool to call retriever_tool function.
    Args:
        user_query (str) :
        string query.
    Returns:
        A ML note including latest Machine Learning models and code examples for prediction from Github.
    """
    logger.info("✴️ Starting retriever_tool with query: %s", user_query)
    init: RState = {
        "messages": [HumanMessage(content= user_query )],
        "queries": ["Machine Learning Models on Github", 
                    "Code Example for Machine Learning Models",
                    "Machine Learning Models Algorithms"
        ],
        "hits": [], "pages": {}, "notes": [], "citations": [],
        "done": False, "stop": False, "rounds": 0,
    }
    retriever_workflow = create_retriever_workflow()

    response = retriever_workflow.invoke(init, config={"recursion_limit": 20})
    final_response = response["messages"][-1].content

    return final_response


# ---------- 執行 ----------

# user_query = """
# 幫我搜尋 Github 上最新的 Machine Learning 模型演算法與程式碼範例以進行進行預測
# """

# if __name__ == "__main__":
#     init: RState = {
#         "messages": [HumanMessage(content= user_query )],
#         "queries": ["Machine Learning Models on Github", 
#                     "Code Example for Machine Learning Models for prediction",],
#         "hits": [], "pages": {}, "notes": [], "citations": [],
#         "done": False, "stop": False, "rounds": 0,
#     }
#     # 小建議：也可以給 invoke 一個較小的 recursion_limit，以便提早發現規劃問題
#     retriever_workflow = create_retriever_workflow()
#     final = retriever_workflow.invoke(init, config={"recursion_limit": 20})
#     print(final["messages"][-1].content)

