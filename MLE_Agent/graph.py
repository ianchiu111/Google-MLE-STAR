import os
import json
import numpy as np
import pandas as pd
from typing import Annotated, Literal, List, Union

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_experimental.utilities import PythonREPL


from agent_prompts import (
    get_retriever_agent_prompt,
    get_code_generator_agent_prompt
)
from agent_tools.retriever import retriever_tool

from dotenv import load_dotenv
load_dotenv(".env")


# =========================  Model Config  ===================================
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url="http://127.0.0.1:11434/v1",  # ← 用 127.0.0.1 與 11434
    api_key="ollama",                      # ← 佔位即可
    model="qwen2.5:7b-instruct",         # ← 跟你 pull 的模型一致
    temperature=0,
)


# ========================== Tools ===================================

repl = PythonREPL()
@tool
def run_python_code(
    code: Annotated[str, "Python code"],
):
    """
    use this tool to execute python code 
    """
    print("✴️llm is executing python code:\n", code)
    try:
        outcome = repl.run(code)
        print("✴️llm python code execution outcome:\n", outcome)
        return json.dumps({
            "code you have generated": f"{code}",
            "success": outcome
            }, ensure_ascii=False)

    except Exception as e:
        print("✴️llm python code execution error:\n", e)
        outcome = {"error": str(e)}
        return json.dumps({
            "code you have generated": f"{code}",
            "error": outcome
            }, ensure_ascii=False)


# ================================================================


def create_graph():
    checkpoint = InMemorySaver()

    workflow = StateGraph(MessagesState)


    retriever_agent_node = create_agent("retriever_agent")
    code_generator_agent_node = create_agent("code_generator_agent")
    # ablation_agent_node = create_agent("ablation_agent")
    # refinement_agent_node = create_agent("refinement_agent")
    # ensemble_agent_node = create_agent("ensemble_agent")
    # validation_agent_node = create_agent("validation_agent")
    # report_agent_node = create_agent("report_agent")

    workflow.add_node("retriever_agent", retriever_agent_node)
    workflow.add_node("code_generator_agent", code_generator_agent_node)
    # workflow.add_node("ablation_agent", ablation_agent_node)
    # workflow.add_node("refinement_agent", refinement_agent_node)
    # workflow.add_node("ensemble_agent", ensemble_agent_node)
    # workflow.add_node("validation_agent", validation_agent_node)
    # workflow.add_node("report_agent", report_agent_node)


    workflow.add_edge(START, "retriever_agent")
    workflow.add_edge("retriever_agent", "code_generator_agent")

    workflow.add_edge("code_generator_agent", END)
    # workflow.add_edge("foundation_agent", "ablation_agent")
    # workflow.add_edge("ablation_agent", "refinement_agent")

    # workflow.add_conditional_edges(
    #     "refinement_agent",
    #     decide_next_agent, # function to decide next agent
    #     {
    #         "needs_refinement": "ablation_agent",
    #         "ready_for_ensemble": "ensemble_agent",
    #     },
    # )
    # workflow.add_edge("ensemble_agent", "validation_agent")
    # workflow.add_edge("validation_agent", "report_agent")
    # workflow.add_edge("report_agent", END)

    graph = workflow.compile(checkpointer=checkpoint)
    return graph


def keyword_agent_process_tool(input:str):
    """
    Prefer Query_Dict when provided. 
    Fallback to user_query only if bundle is missing.
    """

    graph = create_graph()

    thread_id = f"test-session-{np.random.randint(1, 1000)}"

    response = graph.invoke(   
        {"messages": [HumanMessage(content=input)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    return response["messages"][-1].content




def create_agent(agent_name: str):
    match agent_name:

        case "retriever_agent":

            retriever_agent_prompt = get_retriever_agent_prompt()
            retriever_agent = create_react_agent(
                model,
                tools=[retriever_tool],
                prompt=retriever_agent_prompt
            )

            def retriever_agent_node(state: MessagesState) -> Command[Literal["code_generator_agent"]]:
                
                print("✴️  Logger for retriever_agent: Current state messages:", state["messages"][-1].content)
                result = retriever_agent.invoke(state)
                print("✴️  Logger for retriever_agent: Agent result:", result["messages"][-1].content)

                return Command(
                    update={
                        "messages": [
                            AIMessage(content=result["messages"][-1].content, name="retriever_agent")
                        ]
                    },
                    goto="code_generator_agent",
                )
            return retriever_agent_node


        case "code_generator_agent":

            code_generator_agent_prompt = get_code_generator_agent_prompt()
            code_generator_agent = create_react_agent(
                model,
                tools=[run_python_code],
                prompt=code_generator_agent_prompt
            )

            def code_generator_agent_node(state: MessagesState) -> Command[Literal["__end__"]]:

                print("✴️  Logger for code_generator_agent: Current state messages:", state["messages"][-1].content)
                result = code_generator_agent.invoke(state)
                print("✴️  Logger for code_generator_agent: result:", result)

                return Command(
                    update={
                        "messages": [
                            AIMessage(content=result["messages"][-1].content, name="code_generator_agent")
                        ]
                    },
                    goto="__end__",
                )
            return code_generator_agent_node


if __name__ == "__main__":
    # first_test_input = "請幫我產生分析 dataframe 的 python code 範例，資料路徑為 'data/store.csv'。"

    second_test_input = """
請幫我利用 python code 進行 Machine Learning，利用 data/store.csv 進行特徵工程並以 data/train.csv 中的 Sales 欄位為預測目標。
以下是資料集的描述，請幫我完成所有步驟。
並講所有相關的結果儲存在資料夾 data/information_from_agent/ 的資料夾中。

File 1:
data/train.csv - historical sales data

Data Schema:
1. Store - a unique Id for each store
2. Date - the date of the sales record
3. DayOfWeek - the day of the week
4. Customers - the number of customers on a given day
5. Sales - the turnover for any given day (this is what you are predicting)
6. Open - an indicator for whether the store was open: 0 = closed, 1 = open
7. StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
8. SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
9. Promo - indicates whether a store is running a promo on that day

---

File 2:
data/store.csv - supplemental information about the stores

Data Schema:
1. Store - a unique Id for each store
2. StoreType - differentiates between 4 different store models: a, b, c, d
3. Assortment - describes an assortment level: a = basic, b = extra, c = extended
4. CompetitionDistance - distance in meters to the nearest competitor store
5. CompetitionOpenSinceMonth - gives the approximate month of the time the nearest competitor was opened
6. CompetitionOpenSinceYear - gives the approximate year of the time the nearest competitor was opened
7. Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
8. Promo2SinceWeek - describes the calendar week when the store started participating in Promo2
9. Promo2SinceYear - describes the year when the store started participating in Promo2
10. PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
"""
    response = keyword_agent_process_tool(second_test_input)
    print("final response:", response)
