import traceback
from typing import Literal, TypedDict, Optional, List, Dict, Any
import json
import os
import uuid
from dotenv import load_dotenv
import logging
from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from Langgraph.Prompts.web_search_agent_prompt import WEB_SEARCH_AGENT_PROMPT
from Langgraph.Prompts.scraper_agent_prompt import SCRAPER_AGENT_PROMPT
from Langgraph.Prompts.generator_agent_prompt import GENERATOR_AGENT_PROMPT

from Langgraph.utils.AI_SaaS_utils.azure_openai_helper import get_azure_openai_instance
from Langgraph.utils.DataForseo.web_search import web_search
from Langgraph.utils.DataForseo.content_parser import parse_content
from Langgraph.utils.file_utils import read_file_content, create_temp_file_tool
from Langgraph.PIS.get_model_info import get_model_info_by_name

load_dotenv(".env")

checkpoint = InMemorySaver()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    user_input: str
    content_type: Optional[Literal["title", "content"]]
    extracted_info: Optional[Dict[str, Any]]
    web_search_results: Optional[List[Dict[str, Any]]]
    scraped_content: Optional[List[Dict[str, Any]]]
    model_info: Optional[List[Dict[str, Any]]]  # Model information from PIS
    output: Optional[str]
    error_count: Optional[int]
    errors: Optional[List[str]]
    temp_files: Optional[List[str]]  # Track temporary files for cleanup
    done: Optional[bool]


def add_error_to_state(state: GraphState, error_msg: str) -> GraphState:
    """Add error to state and increment error count"""
    errors = state.get("errors", []) or []
    errors.append(error_msg)
    error_count = state.get("error_count", 0) or 0
    return {**state, "errors": errors, "error_count": error_count + 1}


def cleanup_temp_files(temp_files: List[str]):
    """Clean up temporary files"""
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up file {file_path}: {e}")


def create_graph():
    """Create content generation workflow graph"""

    try:
        web_search_agent_node = create_agent("web_search_agent")
        scraper_agent_node = create_agent("scraper_agent")
        generator_agent_node = create_agent("generator_agent")

        def route_after_web_search(
            state: GraphState,
        ) -> Literal["scraper_agent", "generator_agent"]:
            """Route based on content type after web search"""
            content_type = state.get("content_type")
            if content_type == "title":
                return "generator_agent"
            else:
                return "scraper_agent"

        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("web_search_agent", web_search_agent_node)
        workflow.add_node("scraper_agent", scraper_agent_node)
        workflow.add_node("generator_agent", generator_agent_node)

        # Add edges
        workflow.add_edge(START, "web_search_agent")
        workflow.add_conditional_edges(
            "web_search_agent",
            route_after_web_search,
            {"scraper_agent": "scraper_agent", "generator_agent": "generator_agent"},
        )
        workflow.add_edge("scraper_agent", "generator_agent")
        workflow.add_edge("generator_agent", END)

        graph = workflow.compile(checkpointer=checkpoint)
        return graph

    except Exception as e:
        logger.error(f"Critical error creating graph: {e}")
        traceback.print_exc()
        raise


@tool
def content_generation_tool(user_query: str):
    """
    Use this tool to invoke the graph and get content generation results
    """
    print("⚠️ User query: ", user_query)

    try:
        graph = create_graph()

        response = graph.invoke(
            {"user_input": user_query, "temp_files": [], "done": False},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )

        # Clean up temporary files after processing
        temp_files = response.get("temp_files", [])
        if temp_files:
            cleanup_temp_files(temp_files)

        print("\n\n===========✴️ Content Generation Results ✴️===========")
        print(f"Final output: {response.get('output', 'No output generated')}")
        print("\n\n============✴️ End Results ✴️============")

        generation_result = {
            "done": response.get("done", True),
            "output": response.get("output", ""),
            "goto": "Transfer back to supervisor"   ## 絕對不能刪，刪了會完蛋
        }

        return json.dumps(generation_result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Content generation tool error: {e}")
        traceback.print_exc()

        error_result = {
            "done": True,
            "output": f"Content generation failed: {str(e)}",
        }

        return json.dumps(error_result, ensure_ascii=False)


def create_agent(agent_name: str):
    """Create specified agent"""

    unparallel_model = get_azure_openai_instance(
        deployment_name="gpt-4o", temperature=0, parallel_tool_calling=False
    )

    model = get_azure_openai_instance(
        deployment_name="gpt-4o",
        temperature=0,
    )

    match agent_name:
        case "web_search_agent":
            web_search_agent = create_react_agent(
                model=unparallel_model,
                tools=[web_search],
                prompt=WEB_SEARCH_AGENT_PROMPT,
            )

            def web_search_agent_node(state: GraphState) -> GraphState:
                """Web search agent node - analyzes intent and performs web search"""
                print("🔄️ Web Search Agent processing...")

                try:
                    user_input = state["user_input"]

                    # Invoke web_search_agent
                    result = web_search_agent.invoke(
                        {"messages": [HumanMessage(content=user_input)]}
                    )

                    agent_response = result["messages"][-1].content
                    print(f"✴️ Web Search Agent response: {agent_response}")

                    # Parse agent response JSON
                    response_data = json.loads(agent_response)

                    return {
                        **state,
                        "content_type": response_data.get("user_intent"),
                        "extracted_info": {
                            "keywords": response_data.get("keywords", []),
                            "topic": response_data.get("topic", ""),
                            "model": response_data.get("model", ""),
                        },
                        "web_search_results": response_data.get(
                            "web_search_results", []
                        ),
                    }
                except Exception as e:
                    error_msg = f"Unexpected error in web search agent: {e}"
                    logger.error(f"{error_msg}")
                    print(f"❌ {error_msg}")
                    traceback.print_exc()
                    return {
                        **state,
                        "content_type": "content",
                        "web_search_results": [],
                        **add_error_to_state(state, error_msg),
                    }

            return web_search_agent_node

        case "scraper_agent":
            scraper_agent = create_react_agent(
                model=unparallel_model,
                tools=[parse_content, create_temp_file_tool, get_model_info_by_name],
                prompt=SCRAPER_AGENT_PROMPT,
            )

            def scraper_agent_node(state: GraphState) -> GraphState:
                """Scraper agent node - scrapes all URLs from web search results"""
                print("🔄️ Scraper Agent processing...")

                try:
                    web_search_results = state.get("web_search_results", [])
                    urls_to_scrape = [
                        result.get("url")
                        for result in web_search_results or []
                        if result and result.get("url")
                    ]

                    if not urls_to_scrape:
                        print("❌ No URLs found to scrape")
                        return {**state, "scraped_content": []}

                    # Build input for scraper_agent
                    extracted_info = state.get("extracted_info", {}) or {}
                    scraper_input = {
                        "urls_to_scrape": urls_to_scrape,
                        "web_search_results": web_search_results,
                        "model": extracted_info.get("model", []),
                    }

                    result = scraper_agent.invoke(
                        {
                            "messages": [
                                HumanMessage(
                                    content=json.dumps(
                                        scraper_input, ensure_ascii=False
                                    )
                                )
                            ]
                        }
                    )

                    agent_response = result["messages"][-1].content
                    print(f"✴️ Scraper Agent response: {agent_response}")

                    # Try to parse JSON response
                    scraped_data = json.loads(agent_response)

                    # Handle new format with both scraped_content and model_info
                    if isinstance(scraped_data, dict):
                        scraped_content = scraped_data.get("scraped_content", [])
                        model_info = scraped_data.get("model_info", [])
                    elif isinstance(scraped_data, list):
                        # Backward compatibility with old format
                        scraped_content = scraped_data
                        model_info = []
                    else:
                        scraped_content = [scraped_data]
                        model_info = []

                    # Track temporary files for cleanup
                    temp_files = state.get("temp_files", []) or []
                    for item in scraped_content:
                        if isinstance(item, dict) and "file_path" in item:
                            temp_files.append(item["file_path"])

                except json.JSONDecodeError as e:
                    print("Error: ", e)
                    print("❌ Failed to parse Scraper Agent response, using raw text")
                    logger.error(f"Error: {e}")
                    scraped_content = [{"content": agent_response}]
                    model_info = []
                    temp_files = state.get("temp_files", []) or []

                except Exception as e:
                    error_msg = f"Unexpected error in scraper agent: {e}"
                    logger.error(f"{error_msg}")
                    print(f"❌ {error_msg}")
                    traceback.print_exc()
                    return {
                        **state,
                        "scraped_content": [],
                        "model_info": [],
                        **add_error_to_state(state, error_msg),
                    }

                return {
                    **state,
                    "scraped_content": scraped_content,
                    "model_info": model_info,
                    "temp_files": temp_files,
                }

            return scraper_agent_node

        case "generator_agent":
            generator_agent = create_react_agent(
                model=model, tools=[read_file_content], prompt=GENERATOR_AGENT_PROMPT
            )

            def generator_agent_node(state: GraphState) -> GraphState:
                """Generator agent node - generates final content based on available data"""
                print("🔄️ Generator Agent processing...")

                # Build comprehensive input for generator
                generator_input = {
                    "user_input": state.get("user_input", ""),
                    "content_type": state.get("content_type", "content"),
                    "extracted_info": state.get("extracted_info", {}),
                    "web_search_results": state.get("web_search_results", []),
                    "scraped_content": state.get("scraped_content", []),
                    "model_info": state.get("model_info", []),
                }

                try:
                    result = generator_agent.invoke(
                        {
                            "messages": [
                                HumanMessage(
                                    content=json.dumps(
                                        generator_input, ensure_ascii=False
                                    )
                                )
                            ]
                        }
                    )

                    output = result["messages"][-1].content
                    print(f"✴️ Generator Agent final output: {output}")
                except Exception as e:
                    logger.error(f"Unexpected error in generator agent: {e}")
                    return {
                        **state,
                        "output": f"Unexpected error in generator agent: {e}",
                    }

                return {**state, "output": output, "done": True}

            return generator_agent_node

        case _:
            raise ValueError(f"Unknown agent name: {agent_name}")


# Usage examples
if __name__ == "__main__":
    # Test title generation
    title_query = "Generate an attractive advertising title for my AI product, targeting business executives, keywords: AI, automation, efficiency"
    result = content_generation_tool(title_query)
    print(f"Title generation result: {result}")

    # Test content generation
    content_query = "Write an advertising description for my AI product, emphasizing how it helps businesses improve efficiency, tone should be professional but approachable"
    result = content_generation_tool(content_query)
    print(f"Content generation result: {result}")
