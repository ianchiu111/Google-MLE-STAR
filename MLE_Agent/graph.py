import os
import json
import numpy as np
import pandas as pd
from typing import Annotated, Literal, List, Union

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_experimental.utilities import PythonREPL


from MLE_Agent.agent_prompts import (
    get_retriever_agent_prompt,
    get_code_generator_agent_prompt,
    get_run_python_code_agent_prompt
)
from MLE_Agent.agent_tools.retriever import retriever_tool

import logging
LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s :: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("deep_research")

from dotenv import load_dotenv
load_dotenv(".env")


# =========================  Model Config  ===================================
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url="http://127.0.0.1:11434/v1",  # ← 用 127.0.0.1 與 11434
    api_key="ollama",                      # ← 佔位即可
    model="qwen2.5:7b-instruct",         # ← 跟你 pull 的模型一致
    temperature=0,
    streaming = True
)


# ========================== Tools ===================================

repl = PythonREPL()
@tool
def run_python_code(
    code: Annotated[str, "Only python code can be executed here."],
):
    """
    use this tool to execute python code 
    """
    outcome = repl.run(code)
    logger.info("✴️llm python code execution outcome:\n%s", outcome)
    
    file_path = "data/information_from_agent/python_code_execution_results.json"
    new_data = {"code": code, "outcome": outcome}

    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump([new_data], f, indent=4)
    else: # if file exists, append the new record
        with open(file_path, "r+") as f:
            try:
                # Load existing data, or initialize if file is empty
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [] # Or handle error appropriately
            except json.JSONDecodeError:
                existing_data = []
            
            existing_data.append(new_data)
            f.seek(0) # Rewind to the beginning of the file
            f.truncate() # Clear the file content
            json.dump(existing_data, f, indent=4)

    return "This set of python code is executed."


# ================================================================


def create_graph():
    checkpoint = InMemorySaver()

    workflow = StateGraph(MessagesState)


    retriever_agent_node = create_agent("retriever_agent")
    code_generator_agent_node = create_agent("code_generator_agent")
    run_python_code_agent_node = create_agent("run_python_code_agent")

    # ablation_agent_node = create_agent("ablation_agent")
    # refinement_agent_node = create_agent("refinement_agent")
    # ensemble_agent_node = create_agent("ensemble_agent")
    # validation_agent_node = create_agent("validation_agent")
    # report_agent_node = create_agent("report_agent")

    workflow.add_node("retriever_agent", retriever_agent_node)
    workflow.add_node("code_generator_agent", code_generator_agent_node)
    workflow.add_node("run_python_code_agent", run_python_code_agent_node)
    # workflow.add_node("ablation_agent", ablation_agent_node)
    # workflow.add_node("refinement_agent", refinement_agent_node)
    # workflow.add_node("ensemble_agent", ensemble_agent_node)
    # workflow.add_node("validation_agent", validation_agent_node)
    # workflow.add_node("report_agent", report_agent_node)


    workflow.add_edge(START, "retriever_agent")
    workflow.add_edge("retriever_agent", "run_python_code_agent")
    # workflow.add_edge("code_generator_agent", "run_python_code_agent")
    workflow.add_edge("run_python_code_agent", END)

    # workflow.add_conditional_edges("extract", should_continue, {"plan": "plan", "report": "report"})


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


def mle_star_process_tool(input:str):
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

            def retriever_agent_node(state: MessagesState) -> Command[Literal["run_python_code_agent"]]:

                logger.info("✴️  Input for retriever_agent: %s", state["messages"][-1].content)
                result = retriever_agent.invoke(state)
                logger.info("✴️  Output from retriever_agent: %s", result["messages"][-1].content)

                return Command(
                    update={
                        "messages": [
                            AIMessage(content=result["messages"][-1].content, name="retriever_agent")
                        ]
                    },
                    goto="run_python_code_agent",
                )
            return retriever_agent_node


        case "code_generator_agent":

            code_generator_agent_prompt = get_code_generator_agent_prompt()
            code_generator_agent = create_react_agent(
                model,
                tools=[],
                prompt=code_generator_agent_prompt
            )

            def code_generator_agent_node(state: MessagesState) -> Command[Literal["run_python_code_agent"]]:

                logger.info("✴️  Input for code_generator_agent: %s", state["messages"][-1].content)
                result = code_generator_agent.invoke(state)
                logger.info("✴️  Output from code_generator_agent: %s", result["messages"][-1].content)

                return Command(
                    update={
                        "messages": [
                            AIMessage(content=result["messages"][-1].content, name="code_generator_agent")
                        ]
                    },
                    goto="run_python_code_agent",
                )
            return code_generator_agent_node

        case "run_python_code_agent":

            run_python_code_agent_prompt = get_run_python_code_agent_prompt()
            run_python_code_agent = create_react_agent(
                model,
                tools=[run_python_code],
                prompt=run_python_code_agent_prompt
            )

            def run_python_code_agent_node(state: MessagesState) -> Command[Literal["__end__"]]:

                logger.info("✴️  Input for run_python_code_agent: %s", state["messages"][-1].content)

                python_code = state["messages"][-1].content
                result = run_python_code_agent.invoke({"messages": [{
                                                    "role": "user",
                                                    "content": f" please use `run_python_code` with python code {python_code}"
                                                }]},)

                logger.info("✴️  Output from run_python_code_agent: %s", result["messages"][-1].content)

                return Command(
                    update={
                        "messages": [
                            AIMessage(content=result["messages"][-1].content, name="run_python_code_agent")
                        ]
                    },
                    goto="__end__",
                )
            return run_python_code_agent_node


