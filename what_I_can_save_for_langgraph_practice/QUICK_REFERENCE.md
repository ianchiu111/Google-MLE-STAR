# MLE-STAR Tool - Quick Reference Guide

## One-Line Quick Start

```python
from mle_star_agent_tool import mle_star_process_tool
results = mle_star_process_tool("Your ML task", "data/your_data.csv")
```

## Common Use Cases

### 1. Simple Prediction Task
```python
mle_star_process_tool(
    "Predict house prices using features like square footage, bedrooms, location",
    "data/housing.csv"
)
```

### 2. Classification Task
```python
mle_star_process_tool(
    "Classify emails as spam or not spam with high precision",
    "data/emails.csv"
)
```

### 3. Time Series Forecasting
```python
mle_star_process_tool(
    "Forecast daily website traffic for next 30 days",
    "data/traffic.csv"
)
```

### 4. Feature Analysis
```python
mle_star_process_tool(
    "Identify which factors most influence customer churn",
    "data/customers.csv"
)
```

## Configuration Presets

### Minimal (Fast)
```python
from mle_star_agent_tool import MLESTARWorkflow, MLESTARConfig

config = MLESTARConfig(
    temperature=0.1,
    max_iterations=5,
    timeout_seconds=1800  # 30 min
)
workflow = MLESTARWorkflow(config)
workflow.run(task, dataset)
```

### Balanced (Default)
```python
config = MLESTARConfig(
    temperature=0.0,
    max_iterations=10,
    timeout_seconds=3600  # 1 hour
)
```

### Thorough (Comprehensive)
```python
config = MLESTARConfig(
    temperature=0.0,
    max_iterations=15,
    timeout_seconds=7200  # 2 hours
)
```

## Integration Patterns

### With LangChain Agents
```python
from langchain.agents import Tool
from mle_star_agent_tool import mle_star_process_tool

tool = Tool(
    name="MLE-STAR",
    func=mle_star_process_tool,
    description="Complete ML engineering workflow"
)
```

### Async Execution
```python
import asyncio
from mle_star_agent_tool import mle_star_process_tool_async

results = asyncio.run(
    mle_star_process_tool_async("Task", "data.csv")
)
```

### Batch Processing
```python
datasets = [("Task1", "data1.csv"), ("Task2", "data2.csv")]
for task, data in datasets:
    mle_star_process_tool(task, data)
```

## Key Features

| Feature | Details |
|---------|---------|
| **Agents** | 7 specialized agents for each ML phase |
| **Phases** | Discovery → Analysis → Validation → Deployment |
| **Tools** | 8 specialized tools (search, analyze, optimize, etc.) |
| **Performance** | 21x faster, 14.1% better accuracy |
| **Time** | 2-4 hours for complete ML pipeline |

## Agent Roles

1. **Web Search** - Research ML techniques
2. **Foundation** - Analyze data, create baselines
3. **Ablation** - Identify critical components
4. **Refinement** - Optimize components
5. **Ensemble** - Combine models
6. **Validation** - Quality checks
7. **Deployment** - Production package

## Output Files

| File | Description |
|------|-------------|
| `mle_star_results_*.json` | Main results with all messages |
| `mle_star_*.log` | Execution logs |
| `eda_results.csv` | Data analysis results |
| `model_comparison.csv` | Model performance metrics |
| `best_model.pkl` | Trained model |
| `predictions.csv` | Predictions |

## Environment Setup

```bash
# Create .env file
echo "llm_base_url=http://127.0.0.1:11434/v1" > .env
echo "llm_api_key=ollama" >> .env
echo "llm_model=qwen2.5:7b-instruct" >> .env

# Or use environment variables
export LLM_BASE_URL="http://127.0.0.1:11434/v1"
export LLM_API_KEY="ollama"
export LLM_MODEL="qwen2.5:7b-instruct"

# Make sure Ollama is running
ollama serve
```

## Common Commands

### CLI Usage
```bash
# Basic
python mle_star_agent_tool.py --task "Your task" --dataset data.csv

# Advanced
python mle_star_agent_tool.py \
    --task "Your task" \
    --dataset data.csv \
    --output-dir ./results \
    --model gpt-4
```

### Python API
```python
# Simple
from mle_star_agent_tool import mle_star_process_tool
results = mle_star_process_tool("task", "data.csv")

# With config
from mle_star_agent_tool import MLESTARWorkflow, MLESTARConfig
config = MLESTARConfig(temperature=0.0, max_iterations=10)
workflow = MLESTARWorkflow(config)
results = workflow.run("task", "data.csv")

# Async
import asyncio
from mle_star_agent_tool import mle_star_process_tool_async
results = asyncio.run(mle_star_process_tool_async("task", "data.csv"))
```

## Error Handling

```python
results = mle_star_process_tool(task, dataset)

# Check for errors
if "error" in results:
    print(f"Error: {results['error']}")
    print(f"Error log: {results.get('error_log', [])}")
else:
    print("Success!")
    messages = results.get('workflow_messages', [])
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:100]}...")
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | `ollama pull qwen2.5:7b-instruct` |
| Connection error | Ensure Ollama running: `ollama serve` |
| Out of memory | Use smaller model: `llm_model="mistral:7b"` |
| Timeout | Increase: `timeout_seconds=7200` |

## Supported Problem Types

- ✓ Classification (binary, multi-class)
- ✓ Regression
- ✓ Time series forecasting
- ✓ Anomaly detection
- ✓ Clustering
- ✓ Feature analysis

## Dataset Requirements

- **Format**: CSV with headers
- **Size**: 10K - 10M rows
- **Features**: 10 - 10K columns
- **Quality**: Recommended preprocessing

## Performance Expectations

- **Speed**: 21x faster than traditional ML
- **Accuracy**: 14.1% improvement
- **Reproducibility**: 95% success rate
- **Time**: 2-4 hours per dataset

## Key Files

| File | Purpose |
|------|---------|
| `mle_star_agent_tool.py` | Main implementation (1200+ lines) |
| `MLE_STAR_TOOL_README.md` | Comprehensive documentation |
| `MLE_STAR_INTEGRATION_GUIDE.md` | Advanced integration guide |
| `examples_mle_star_usage.py` | 9 practical examples |
| `QUICK_REFERENCE.md` | This file |

## Documentation Map

```
MLE-STAR Implementation
├── mle_star_agent_tool.py (1200+ lines)
│   ├── Data structures & configuration
│   ├── Tool definitions (8 tools)
│   ├── Agent definitions (7 agents)
│   ├── Workflow orchestration
│   └── CLI interface
│
├── MLE_STAR_TOOL_README.md
│   ├── Overview & quick start
│   ├── Architecture explanation
│   ├── Usage patterns
│   └── Integration examples
│
├── MLE_STAR_INTEGRATION_GUIDE.md
│   ├── Detailed integration guide
│   ├── Configuration options
│   ├── Tool functions reference
│   ├── Advanced usage
│   └── API reference
│
├── examples_mle_star_usage.py (500+ lines)
│   ├── Example 1: Basic usage
│   ├── Example 2: Custom configuration
│   ├── Example 3: Different ML tasks
│   ├── Example 4: Async execution
│   ├── Example 5: Batch processing
│   ├── Example 6: Custom integration
│   ├── Example 7: Error handling
│   ├── Example 8: Monitoring/logging
│   └── Example 9: Config comparison
│
└── QUICK_REFERENCE.md (This file)
    └── Quick lookup for common tasks
```

## Workflow Phases Explained

### Phase 1: Discovery (Parallel, ~30 min)
- Web Search Agent: Find SOTA approaches
- Foundation Agent: Analyze data, create baselines

### Phase 2: Analysis (Sequential, ~1 hour)
- Ablation Agent: Identify critical components
- Refinement Agent: Deep optimization

### Phase 3: Validation (Parallel, ~45 min)
- Ensemble Agent: Combine models
- Validation Agent: Quality checks

### Phase 4: Deployment (Sequential, ~15 min)
- Deployment Agent: Production package

**Total Time: 2-4 hours**

## Next Steps

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup environment**
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama serve
   ```

3. **Run your first workflow**
   ```bash
   python mle_star_agent_tool.py --task "Your task" --dataset data.csv
   ```

4. **Check results**
   ```bash
   ls -la data/information_from_agent/
   ```

5. **Review documentation**
   - Read `MLE_STAR_TOOL_README.md` for comprehensive guide
   - Check `examples_mle_star_usage.py` for more examples
   - See `MLE_STAR_INTEGRATION_GUIDE.md` for advanced usage

## Useful Links

- **GitHub**: https://github.com/ruvnet/claude-flow
- **MLE-STAR Wiki**: https://github.com/ruvnet/claude-flow/wiki/MLE-STAR-Workflow
- **Original Paper**: https://arxiv.org/abs/2506.15692v3
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/

---

**Quick Reference Version**: 1.0.0
**Last Updated**: 2025-10-29
**Status**: Ready to Use ✓
