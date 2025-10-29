# MLE-STAR Agent Tool Integration Guide

## Overview

The `mle_star_agent_tool.py` is a comprehensive Python implementation of the MLE-STAR (Machine Learning Engineering via Search and Targeted Refinement) workflow. It provides a unified tool for orchestrating 7 specialized ML agents across 4 workflow phases.

## Architecture

### 4 Workflow Phases

```
Phase 1: DISCOVERY & FOUNDATION (Parallel)
├── Web Search Agent
│   └── Researches state-of-the-art ML approaches
└── Foundation Agent
    └── Analyzes dataset and creates baselines

Phase 2: ANALYSIS & REFINEMENT (Sequential)
├── Ablation Agent
│   └── Identifies critical components
└── Refinement Agent
    └── Deep optimization of critical components

Phase 3: ENSEMBLE & VALIDATION (Parallel)
├── Ensemble Agent
│   └── Creates sophisticated model combinations
└── Validation Agent
    └── Ensures production-readiness

Phase 4: PRODUCTION DEPLOYMENT
└── Deployment Agent
    └── Prepares deployment packages
```

### Agent Roles

1. **Web Search Agent**: Elite ML researcher - searches for winning solutions and benchmarks
2. **Foundation Agent**: Expert data scientist - analyzes data and creates baselines
3. **Ablation Agent**: ML optimization specialist - isolates high-impact components
4. **Refinement Agent**: Advanced optimization specialist - applies deep refinement techniques
5. **Ensemble Agent**: Ensemble learning expert - creates sophisticated model combinations
6. **Validation Agent**: Quality assurance specialist - ensures production-readiness
7. **Deployment Agent**: ML deployment specialist - prepares for production

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

The tool requires:
- LangChain & LangGraph
- Pandas & NumPy
- OpenAI API (or local Ollama)
- DuckDuckGo Search API
- Python 3.9+

### Environment Setup

Create a `.env` file:

```env
llm_base_url = "http://127.0.0.1:11434/v1"
llm_api_key = "ollama"
llm_model = "qwen2.5:7b-instruct"
```

Or use environment variables:
```bash
export LLM_BASE_URL="http://127.0.0.1:11434/v1"
export LLM_API_KEY="ollama"
export LLM_MODEL="qwen2.5:7b-instruct"
```

## Usage

### Basic Usage

#### Method 1: Direct Function Call

```python
from mle_star_agent_tool import mle_star_process_tool

# Run the workflow
results = mle_star_process_tool(
    task_description="Predict store sales for Rossmann stores",
    dataset_path="data/train.csv"
)

# Access results
print(f"Task: {results['task_description']}")
print(f"Completed: {results['completion_timestamp']}")
```

#### Method 2: Using MLESTARWorkflow Class

```python
from mle_star_agent_tool import MLESTARWorkflow, MLESTARConfig

# Create configuration
config = MLESTARConfig(
    output_dir="data/information_from_agent",
    temperature=0.0,
    llm_model="qwen2.5:7b-instruct"
)

# Initialize workflow
workflow = MLESTARWorkflow(config)

# Run the workflow
results = workflow.run(
    task_description="Your ML task description",
    dataset_path="path/to/data.csv"
)
```

#### Method 3: Async Execution

```python
import asyncio
from mle_star_agent_tool import mle_star_process_tool_async

async def main():
    results = await mle_star_process_tool_async(
        task_description="Predict store sales",
        dataset_path="data/train.csv"
    )
    return results

results = asyncio.run(main())
```

### Command Line Interface

```bash
# Basic usage
python mle_star_agent_tool.py --task "Predict store sales" --dataset data/train.csv

# With custom output directory
python mle_star_agent_tool.py \
    --task "Your ML task" \
    --dataset data/your_data.csv \
    --output-dir ./results

# With custom model
python mle_star_agent_tool.py \
    --task "Your ML task" \
    --dataset data/your_data.csv \
    --model gpt-4
```

## Integration Examples

### 1. Integration with LangChain Agents

```python
from langchain.agents import Tool
from mle_star_agent_tool import mle_star_process_tool

# Create a LangChain Tool
mle_star_tool = Tool(
    name="MLE-STAR Workflow",
    func=mle_star_process_tool,
    description="Complete ML engineering workflow with 7 agents across 4 phases. "
                "Use for end-to-end ML pipeline development.",
    args_schema={
        "type": "object",
        "properties": {
            "task_description": {
                "type": "string",
                "description": "Description of the ML task"
            },
            "dataset_path": {
                "type": "string",
                "description": "Path to the dataset CSV file"
            }
        },
        "required": ["task_description"]
    }
)

# Use in an agent
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
agent = initialize_agent(
    [mle_star_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Run agent with MLE-STAR tool
response = agent.run("Develop an ML solution for store sales prediction")
```

### 2. Integration with LangGraph

```python
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from mle_star_agent_tool import mle_star_process_tool

# Add MLE-STAR as a node in a larger workflow
@tool
def trigger_mle_star(task: str, dataset: str):
    """Trigger the MLE-STAR workflow for complex ML tasks"""
    return mle_star_process_tool(task, dataset)

# Integrate into your graph
workflow = StateGraph(YourStateType)
workflow.add_node("mle_star_step", mle_star_node)
workflow.add_edge("previous_step", "mle_star_step")
workflow.add_edge("mle_star_step", "next_step")
```

### 3. Integration with Custom Agents

```python
from mle_star_agent_tool import MLESTARWorkflow, MLESTARConfig
from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI

# Create a custom agent that uses MLE-STAR
def create_ml_engineer_agent():
    llm = ChatOpenAI(model="gpt-4")

    @tool
    def run_ml_pipeline(task: str, dataset: str):
        """Run the complete MLE-STAR ML engineering pipeline"""
        workflow = MLESTARWorkflow()
        return workflow.run(task, dataset)

    tools = [run_ml_pipeline, ...other_tools...]
    agent = create_react_agent(llm, tools, prompt)
    return agent
```

### 4. Integration with Batch Processing

```python
from mle_star_agent_tool import mle_star_process_tool
import pandas as pd
import json

# Process multiple datasets
datasets = [
    ("Task 1: Sales Prediction", "data/sales.csv"),
    ("Task 2: Customer Churn", "data/churn.csv"),
    ("Task 3: Fraud Detection", "data/fraud.csv"),
]

results_list = []
for task_desc, dataset_path in datasets:
    print(f"Processing: {task_desc}")
    results = mle_star_process_tool(task_desc, dataset_path)
    results_list.append(results)

# Save all results
with open("batch_results.json", "w") as f:
    json.dump(results_list, f, indent=2, default=str)
```

## Configuration

### MLESTARConfig Options

```python
from mle_star_agent_tool import MLESTARConfig

config = MLESTARConfig(
    # LLM Configuration
    llm_base_url="http://127.0.0.1:11434/v1",  # Local Ollama or OpenAI endpoint
    llm_api_key="ollama",                        # API key
    llm_model="qwen2.5:7b-instruct",             # Model name

    # Agent Configuration
    temperature=0.0,                             # 0=deterministic, 1=creative
    max_iterations=10,                           # Max iterations per agent
    timeout_seconds=3600,                        # Timeout for workflow

    # Output Configuration
    output_dir="data/information_from_agent"     # Output directory
)

workflow = MLESTARWorkflow(config)
```

## Tool Functions

The tool provides 8 specialized functions:

### 1. web_search(query: str) -> str
Searches the web for ML techniques and best practices.

```python
results = web_search("best practices for XGBoost hyperparameter tuning")
```

### 2. analyze_dataset(dataset_path: str) -> str
Analyzes dataset characteristics.

```python
analysis = analyze_dataset("data/train.csv")
```

### 3. execute_python_code(code: str) -> str
Executes Python code for data processing and model training.

```python
code = """
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
"""
output = execute_python_code(code)
```

### 4. generate_feature_importance(model_results: str) -> str
Analyzes feature importance from model results.

```python
importance = generate_feature_importance(json_model_results)
```

### 5. perform_ablation_study(model_config: str) -> str
Performs ablation study on model components.

```python
study = perform_ablation_study(json_config)
```

### 6. optimize_components(components: str) -> str
Optimizes critical components using advanced techniques.

```python
optimized = optimize_components(json_components)
```

### 7. create_ensemble(models: str) -> str
Creates ensemble models with stacking and averaging.

```python
ensemble = create_ensemble(json_models)
```

### 8. validate_robustness(model_config: str) -> str
Validates model robustness through comprehensive testing.

```python
validation = validate_robustness(json_config)
```

## Output Structure

The workflow produces results in the following structure:

```
data/information_from_agent/
├── mle_star_results_YYYYMMDD_HHMMSS.json  # Main results file
├── mle_star_YYYYMMDD-HHMMSS.log           # Execution log
└── [Optional outputs from agents]
    ├── eda_results.csv
    ├── cleaned_data.csv
    ├── model_comparison.csv
    ├── best_model.pkl
    ├── predictions.csv
    └── feature_importance.png
```

### Result JSON Structure

```json
{
  "task_description": "Predict store sales for Rossmann stores",
  "dataset_path": "data/train.csv",
  "workflow_messages": [
    {
      "role": "human",
      "content": "Initial task description"
    },
    {
      "role": "ai",
      "content": "Web Search Agent findings..."
    },
    ...
  ],
  "completion_timestamp": "2025-10-29T12:34:56.789123",
  "workflow_state": {
    "phase": "discovery",
    "current_step": "initialization",
    "research_findings": {...},
    "baseline_models": {...},
    "error_log": []
  }
}
```

## Error Handling

The tool includes comprehensive error handling:

```python
from mle_star_agent_tool import mle_star_process_tool

try:
    results = mle_star_process_tool(task, dataset)
    if "error" in results:
        print(f"Workflow error: {results['error']}")
        print(f"Error log: {results['error_log']}")
except Exception as e:
    print(f"Exception occurred: {e}")
```

## Logging

The tool logs all operations to both file and console:

```python
import logging
from mle_star_agent_tool import logger

# Access the logger
logger.info("Custom log message")
logger.error("Error occurred")

# Log files are created in: mle_star_YYYYMMDD-HHMMSS.log
```

## Performance Expectations

Based on the MLE-STAR research:

- **Runtime**: 2-4 hours depending on dataset complexity
- **Performance Improvement**: 14.1% higher model performance
- **Development Speed**: 21x faster than traditional development
- **Research Coverage**: 5x more comprehensive
- **Reproducibility**: 95% success rate

## Supported Problem Types

- Classification (Binary and Multi-class)
- Regression (Continuous prediction)
- Time Series Forecasting
- Anomaly Detection
- Clustering

## Dataset Requirements

- **Size**: 10K to 10M rows
- **Features**: 10 to 10K columns
- **Format**: CSV with headers
- **Quality**: Recommended preprocessing for production data

## Troubleshooting

### Issue: Model Not Found
**Solution**: Ensure the LLM model is downloaded in Ollama:
```bash
ollama pull qwen2.5:7b-instruct
```

### Issue: Connection Error
**Solution**: Verify Ollama is running:
```bash
ollama serve
```

### Issue: Memory Issues
**Solution**: Use a smaller model or increase system memory:
```python
config = MLESTARConfig(llm_model="mistral:7b")  # Smaller model
```

### Issue: Timeout
**Solution**: Increase timeout or simplify the task:
```python
config = MLESTARConfig(timeout_seconds=7200)  # 2 hours
```

## Advanced Usage

### Custom Configuration

```python
from mle_star_agent_tool import MLESTARWorkflow, MLESTARConfig

# Custom configuration for specific use case
config = MLESTARConfig(
    llm_base_url="https://api.openai.com/v1",  # Use OpenAI
    llm_api_key="sk-...",
    llm_model="gpt-4",
    temperature=0.1,  # More deterministic
    max_iterations=5,
    timeout_seconds=1800
)

workflow = MLESTARWorkflow(config)
results = workflow.run("Your task", "data/your_data.csv")
```

### Chaining Multiple Workflows

```python
from mle_star_agent_tool import mle_star_process_tool

# Run multiple workflows sequentially
tasks = [
    ("Sales Prediction", "data/sales.csv"),
    ("Customer Segmentation", "data/customers.csv"),
    ("Churn Prediction", "data/churn.csv"),
]

workflows_results = {}
for task_name, dataset in tasks:
    print(f"Running: {task_name}")
    results = mle_star_process_tool(task_name, dataset)
    workflows_results[task_name] = results
```

## Best Practices

1. **Start with Clear Task Descriptions**: Be specific about your ML objective
2. **Prepare Clean Data**: Pre-process data if possible to save workflow time
3. **Monitor Resource Usage**: Watch CPU, memory during execution
4. **Save Results**: Results are auto-saved but keep backups
5. **Review Agent Outputs**: Check the log files for insights
6. **Iterate**: Use results to refine your approach
7. **Validate Externally**: Don't rely solely on the tool's validation

## API Reference

### Main Function

```python
def mle_star_process_tool(
    task_description: str,
    dataset_path: Optional[str] = None,
    config: Optional[MLESTARConfig] = None
) -> Dict[str, Any]
```

### Async Function

```python
async def mle_star_process_tool_async(
    task_description: str,
    dataset_path: Optional[str] = None,
    config: Optional[MLESTARConfig] = None
) -> Dict[str, Any]
```

### Workflow Class

```python
class MLESTARWorkflow:
    def __init__(self, config: MLESTARConfig = None)
    def run(self, task_description: str, dataset_path: Optional[str] = None) -> Dict[str, Any]
```

## Contributing

To extend the tool:

1. Add new tools by creating functions with `@tool` decorator
2. Modify agent prompts in the agent creation functions
3. Add new phases by extending the graph
4. Submit pull requests with improvements

## License

This implementation is part of the Google-MLE-STAR project.

## References

- **Original Paper**: Machine Learning Engineering via Search and Targeted Refinement
- **Claude Flow**: https://github.com/ruvnet/claude-flow
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **MLE-STAR Workflow**: https://github.com/ruvnet/claude-flow/wiki/MLE-STAR-Workflow

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the log files in `mle_star_*.log`
3. Open an issue on the project repository
