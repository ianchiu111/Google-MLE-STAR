#!/usr/bin/env python3
"""
MLE-STAR Agent Framework - Complete Single File Implementation
Machine Learning Engineering Agent via Search and Targeted Refinement

This framework implements an intelligent multi-agent system that:
- Uses Web Search to find task-specific ML models and techniques
- Generates and executes Python code for ML tasks
- Leverages LangGraph for agent orchestration
- Based on Google Cloud's MLE-STAR research

Author: Claude Code
Version: 1.0.0
"""

import os
import json
from typing import Annotated, Literal
from typing_extensions import TypedDict

# LangChain & LangGraph imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Experimental tools
from langchain_experimental.utilities import PythonREPL

# Web search
from duckduckgo_search import DDGS

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the MLE-STAR framework"""

    # LLM Configuration (using Ollama by default)
    LLM_BASE_URL = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:11434/v1")
    LLM_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")  # Ollama doesn't need real key
    LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    LLM_TEMPERATURE = 0

    # Alternative: Use OpenAI ChatGPT
    # LLM_BASE_URL = "https://api.openai.com/v1"
    # LLM_API_KEY = os.getenv("OPENAI_API_KEY")
    # LLM_MODEL = "gpt-4o-mini"

    # Web Search Configuration
    SEARCH_REGION = "tw-zh"
    SEARCH_SAFESEARCH = "off"
    SEARCH_TIMELIMIT = "m"  # m = month, d = day, w = week, y = year
    SEARCH_MAX_RESULTS = 5

    # Output directories
    OUTPUT_DIR = "data/information_from_agent/"


# ============================================================================
# AGENT PROMPTS
# ============================================================================

WEB_SEARCH_AGENT_PROMPT = """You are an elite Machine Learning/AI Researcher and Information Specialist.

Your Mission:
Search the web for the latest and most effective machine learning techniques, approaches, and best practices relevant to the user's specific query or task.

Focus Areas:
1. Model Selection: Find state-of-the-art models and algorithms for the task
2. Feature Engineering: Discover effective feature engineering techniques
3. Data Preprocessing: Identify best practices for data cleaning and preparation
4. Hyperparameter Tuning: Locate optimal hyperparameter strategies
5. Evaluation Metrics: Find appropriate metrics for the task

Output Format:
Organize your findings clearly by technique, including:
- Description of the approach
- Why it's effective for this task
- Links to references (papers, documentation, tutorials)
- Code examples when available

Always search for the most recent and relevant information. Use the web_search tool to find information.
"""

CODE_GENERATOR_AGENT_PROMPT = """You are an elite Machine Learning Engineer and Data Scientist.

Your Mission (4 Steps):

STEP 1: Load Dataset & Understand Data
- Load the dataset from the provided path
- Explore data characteristics: shape, columns, data types, missing values
- Perform basic statistical analysis
- Visualize key patterns and distributions

STEP 2: Data Cleaning & Feature Engineering
- Handle missing values appropriately
- Encode categorical variables
- Create new features based on domain knowledge
- Scale/normalize features if needed
- Handle outliers if necessary

STEP 3: Train Models & Evaluate
- Split data into train/validation/test sets
- Train multiple ML models (try at least 3 different algorithms)
- Tune hyperparameters using cross-validation
- Compare model performance
- Select the best model
- Generate predictions

Libraries Available:
- scikit-learn (sklearn)
- XGBoost
- LightGBM
- CatBoost
- TensorFlow
- PyTorch
- pandas, numpy

STEP 4: Save Results
- Save all generated code to files
- Save trained models
- Save predictions and evaluation metrics
- Save visualizations
- Output location: data/information_from_agent/

Evaluation Metrics:
- Classification: Accuracy, Precision, Recall, F1-score, ROC-AUC
- Regression: RMSE, MAE, R², MAPE

IMPORTANT:
1. Generate complete, executable Python code
2. Handle errors gracefully
3. Document your code with comments
4. Save all outputs to the specified directory
5. Use the run_python_code tool to execute your code
6. If code fails, analyze the error and fix it
7. Iterate until you get working code

Generate code step by step and execute it using the run_python_code tool.
"""


# ============================================================================
# TOOLS
# ============================================================================

# Initialize Python REPL for code execution
repl = PythonREPL()

@tool
def web_search(query: Annotated[str, "The specific machine learning technique, algorithm, or approach to search for"]) -> str:
    """
    Search the web for machine learning techniques, models, and best practices.

    Args:
        query: The ML technique or approach to search for

    Returns:
        Formatted search results with titles, snippets, and URLs
    """
    # Enhance query for better ML results
    enhanced_query = f"What models are effective for {query}"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                enhanced_query,
                region=Config.SEARCH_REGION,
                safesearch=Config.SEARCH_SAFESEARCH,
                timelimit=Config.SEARCH_TIMELIMIT,
                max_results=Config.SEARCH_MAX_RESULTS,
            ))

        if not results:
            return f"No search results found for: {query}"

        # Format results
        formatted_results = f"Search Results for: {query}\n\n"
        for idx, result in enumerate(results, 1):
            formatted_results += f"{idx}. {result['title']}\n"
            formatted_results += f"   URL: {result['href']}\n"
            formatted_results += f"   {result['body']}\n\n"

        return formatted_results

    except Exception as e:
        return f"Error during web search: {str(e)}"


@tool
def run_python_code(code: Annotated[str, "Complete Python code to execute"]) -> str:
    """
    Execute Python code in a persistent REPL environment.

    Args:
        code: Python code string to execute

    Returns:
        JSON string with execution results or error message
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        # Execute the code
        outcome = repl.run(code)

        return json.dumps({
            "status": "success",
            "code_executed": code,
            "output": outcome if outcome else "Code executed successfully (no output)"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "code_executed": code,
            "error": str(e),
            "error_type": type(e).__name__
        }, indent=2)


# ============================================================================
# LLM & AGENT SETUP
# ============================================================================

def create_llm():
    """Create and configure the LLM instance"""
    return ChatOpenAI(
        base_url=Config.LLM_BASE_URL,
        api_key=Config.LLM_API_KEY,
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
    )


def create_agent(agent_name: str, llm, tools: list, system_prompt: str):
    """
    Create a ReAct agent with specified tools and prompt.

    Args:
        agent_name: Name of the agent
        llm: Language model instance
        tools: List of tools available to the agent
        system_prompt: System prompt for the agent

    Returns:
        Configured agent
    """
    return create_react_agent(
        llm,
        tools=tools,
        state_modifier=system_prompt
    )


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

def create_mle_star_graph():
    """
    Create the MLE-STAR agent workflow using LangGraph.

    Workflow:
    START → web_search_agent → code_generator_agent → END

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize LLM
    llm = create_llm()

    # Create agents with their specific tools and prompts
    web_search_agent_executor = create_agent(
        agent_name="web_search_agent",
        llm=llm,
        tools=[web_search],
        system_prompt=WEB_SEARCH_AGENT_PROMPT
    )

    code_generator_agent_executor = create_agent(
        agent_name="code_generator_agent",
        llm=llm,
        tools=[run_python_code],
        system_prompt=CODE_GENERATOR_AGENT_PROMPT
    )

    # Create the workflow graph
    workflow = StateGraph(MessagesState)

    # Add agent nodes
    workflow.add_node("web_search_agent", web_search_agent_executor)
    workflow.add_node("code_generator_agent", code_generator_agent_executor)

    # Define the workflow edges
    workflow.add_edge(START, "web_search_agent")
    workflow.add_edge("web_search_agent", "code_generator_agent")
    workflow.add_edge("code_generator_agent", END)

    # Compile with checkpointer for state management
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    return graph


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_mle_star(query: str, thread_id: str = "default-thread"):
    """
    Execute the MLE-STAR framework with a given query.

    Args:
        query: User's machine learning task description
        thread_id: Unique identifier for this conversation thread

    Returns:
        Final response from the agent workflow
    """
    print("=" * 80)
    print("MLE-STAR Agent Framework")
    print("=" * 80)
    print(f"\nQuery: {query}\n")
    print("-" * 80)

    # Create the graph
    graph = create_mle_star_graph()

    # Configure thread for state management
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 50
    }

    # Prepare input
    input_message = {
        "messages": [HumanMessage(content=query)]
    }

    # Execute the workflow
    print("\n🔍 Starting Web Search Agent...\n")

    final_state = None
    for event in graph.stream(input_message, config, stream_mode="values"):
        # Get the last message
        if "messages" in event and len(event["messages"]) > 0:
            last_message = event["messages"][-1]

            # Print agent outputs
            if hasattr(last_message, "content") and last_message.content:
                print(f"\n{last_message.content}\n")
                print("-" * 80)

        final_state = event

    print("\n✅ Workflow Completed!\n")
    print("=" * 80)

    return final_state


def print_usage_examples():
    """Print usage examples for the framework"""
    print("""
Usage Examples:
===============

Example 1: Sales Prediction
----------------------------
query = '''
Please use Python to perform Machine Learning on data/train.csv (Rossmann Store Sales).
Use 'Sales' as the prediction target.
Perform feature engineering and train multiple models.
Save all results to data/information_from_agent/
'''

Example 2: Classification Task
-------------------------------
query = '''
I have a dataset at data/classification.csv.
Please build a classification model to predict the 'target' column.
Try RandomForest, XGBoost, and LightGBM.
Save the best model and evaluation metrics.
'''

Example 3: Custom Analysis
---------------------------
query = '''
Analyze data/customer_data.csv and build a churn prediction model.
Perform EDA, feature engineering, and model selection.
Use cross-validation and save results.
'''

Running the Framework:
----------------------
if __name__ == "__main__":
    query = "Your ML task description here..."
    results = run_mle_star(query, thread_id="session-001")
    """)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check if query is provided as command line argument
    if len(sys.argv) > 1:
        # Join all arguments as the query
        user_query = " ".join(sys.argv[1:])
    else:
        # Default example query
        print("No query provided. Using default example...\n")
        user_query = """
Please use Python to perform Machine Learning on data/train.csv.
This is the Rossmann Store Sales dataset.
Use 'Sales' as the prediction target.

Steps:
1. Load and explore the data
2. Perform data cleaning and feature engineering
3. Train multiple ML models (RandomForest, XGBoost, LightGBM)
4. Evaluate and compare models
5. Save all results, models, and visualizations to data/information_from_agent/

Also load data/store.csv for additional store features if helpful.
"""

    # Run the framework
    try:
        results = run_mle_star(user_query, thread_id="mle-star-session-001")

        print("\n" + "=" * 80)
        print("Framework Execution Summary")
        print("=" * 80)
        print(f"\nTotal messages exchanged: {len(results.get('messages', []))}")
        print(f"\nCheck output directory: {Config.OUTPUT_DIR}")
        print("\nTo run with custom query:")
        print("  python mle_star_complete.py 'Your custom ML task here'")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
