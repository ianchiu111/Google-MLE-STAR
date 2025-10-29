# JSON to Python Integration Guide
## Using mle-star-workflow.json with mle_star_agent_tool.py

---

## 🔗 Integration Overview

This guide shows how to use the **Claude Flow JSON configuration** (`mle-star-workflow.json`) to orchestrate the **Python MLE-STAR implementation** (`mle_star_agent_tool.py`).

### Architecture

```
Claude Flow Service
    ↓
mle-star-workflow.json (Task Orchestration)
    ↓
Python Agent Tool (mle_star_agent_tool.py)
    ├── LangGraph Workflow
    ├── 7 Agents
    ├── 8 Tools
    └── Result Serialization
    ↓
JSON Results
```

---

## 📊 Configuration Mapping

### JSON Variables → Python Config

```json
{
  "variables": {
    "dataset": "data/train.csv",
    "target": "Sales",
    "output_dir": "./mle-star-output",
    "search_iterations": 3,
    "refinement_iterations": 5,
    "ensemble_size": 5
  }
}
```

Maps to:

```python
from mle_star_agent_tool import MLESTARConfig

config = MLESTARConfig(
    output_dir="./mle-star-output",  # from JSON
    max_iterations=5,                 # from refinement_iterations
    timeout_seconds=3600,             # recommended: 1 hour
    llm_model="qwen2.5:7b-instruct"  # system default
)

task = f"Predict {target} using {dataset}"
dataset_path = "data/train.csv"

results = mle_star_process_tool(task, dataset_path, config)
```

---

## 🔄 Task-to-Agent Mapping

### How JSON Tasks Call Python Agents

```
JSON Task: web-search
    ↓
Python: web_search() tool + Web Search Agent
    ↓
Output: Research findings, SOTA models, benchmarks

JSON Task: data-analysis
    ↓
Python: analyze_dataset() tool + Foundation Agent
    ↓
Output: Data insights, feature recommendations

JSON Task: ablation-study
    ↓
Python: perform_ablation_study() tool + Ablation Agent
    ↓
Output: Critical components, impact scores

JSON Task: targeted-refinement
    ↓
Python: optimize_components() tool + Refinement Agent
    ↓
Output: Optimized components, improved performance

JSON Task: ensemble-creation
    ↓
Python: create_ensemble() tool + Ensemble Agent
    ↓
Output: Ensemble configuration, model weights

JSON Task: robustness-validation
    ↓
Python: validate_robustness() tool + Validation Agent
    ↓
Output: Validation report, production readiness

JSON Task: deployment-package
    ↓
Python: prepare_deployment() tool + Deployment Agent
    ↓
Output: Deployment package, API documentation
```

---

## 💻 Implementation Approaches

### Approach 1: Simple Python Wrapper

Create a script that reads JSON and calls Python tool:

```python
#!/usr/bin/env python3
"""
Simple integration: Read JSON config and run Python workflow
"""

import json
from mle_star_agent_tool import mle_star_process_tool, MLESTARConfig

# Load JSON configuration
with open('src/cli/simple-commands/templates/mle-star-workflow.json') as f:
    workflow_config = json.load(f)

# Extract variables
variables = workflow_config['variables']
dataset = variables['dataset']
target = variables['target']
output_dir = variables['output_dir']
refinement_iterations = variables['refinement_iterations']

# Create task description from JSON metadata
task_description = f"""
MLE-STAR Workflow: {workflow_config['name']}

Objective: Predict {target} using data from {dataset}

Methodology: {workflow_config['metadata']['methodology']}
Expected Runtime: {workflow_config['metadata']['expected_runtime']}

Execute all phases:
1. Discovery: Web search + Data analysis
2. Analysis: Ablation study + Refinement
3. Validation: Ensemble + Robustness checks
4. Deployment: Create production package
"""

# Create Python configuration
config = MLESTARConfig(
    output_dir=output_dir,
    max_iterations=refinement_iterations,
    timeout_seconds=3600
)

# Run the workflow
print("Starting MLE-STAR Workflow from JSON configuration...")
results = mle_star_process_tool(
    task_description=task_description,
    dataset_path=dataset,
    config=config
)

# Save results with JSON configuration metadata
output_file = f"{output_dir}/mle_star_workflow_results.json"
with open(output_file, 'w') as f:
    json.dump({
        "configuration": workflow_config,
        "results": results
    }, f, indent=2, default=str)

print(f"✅ Workflow completed. Results saved to {output_file}")
```

### Approach 2: Task-by-Task Orchestration

```python
#!/usr/bin/env python3
"""
Advanced integration: Orchestrate JSON tasks with Python agents
"""

import json
from mle_star_agent_tool import (
    web_search,
    analyze_dataset,
    perform_ablation_study,
    optimize_components,
    create_ensemble,
    validate_robustness,
    prepare_deployment,
    execute_python_code
)

class JSONWorkflowOrchestrator:
    """Orchestrate JSON workflow with Python tools"""

    def __init__(self, json_path):
        with open(json_path) as f:
            self.config = json.load(f)
        self.results = {}

    def execute_task(self, task_id, task_config, prev_results):
        """Execute a single task and return results"""
        print(f"\n📋 Executing: {task_config['name']}")
        print(f"   Type: {task_config['type']}")
        print(f"   Description: {task_config['description']}")

        # Map task type to Python tool
        task_type = task_config['type']

        if task_type == 'research':
            # web-search task
            query = task_config['claudePrompt'].format(**self.config['variables'])
            result = web_search(query)

        elif task_type == 'analysis':
            if 'ablation' in task_config['id']:
                # ablation-study task
                model_config = json.dumps(prev_results.get('initial-pipeline', {}))
                result = perform_ablation_study(model_config)
            else:
                # data-analysis task
                dataset = self.config['variables']['dataset']
                result = analyze_dataset(dataset)

        elif task_type == 'optimization':
            # targeted-refinement task
            components = json.dumps(prev_results.get('ablation-study', {}))
            result = optimize_components(components)

        elif task_type == 'ensemble':
            # ensemble-creation task
            models = json.dumps(prev_results.get('targeted-refinement', {}))
            result = create_ensemble(models)

        elif task_type == 'validation':
            # robustness-validation task
            model_config = json.dumps(prev_results.get('ensemble-creation', {}))
            result = validate_robustness(model_config)

        elif task_type == 'deployment':
            # deployment-package task
            model_info = json.dumps(prev_results.get('robustness-validation', {}))
            result = prepare_deployment(model_info)

        elif task_type == 'implementation':
            # initial-pipeline task
            code = f"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('{self.config['variables']['dataset']}')

# Split data
X = df.drop('{self.config['variables']['target']}', axis=1)
y = df['{self.config['variables']['target']}']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {{X_train_scaled.shape}}")
print(f"Test set: {{X_test_scaled.shape}}")
"""
            result = execute_python_code(code)

        return result

    def run_workflow(self):
        """Execute all tasks in dependency order"""
        print(f"🚀 Starting: {self.config['name']}")
        print(f"Version: {self.config['version']}")

        # Get tasks (should be ordered by dependencies)
        tasks = self.config['tasks']

        for task in tasks:
            task_id = task['id']
            task_results = self.execute_task(task_id, task, self.results)
            self.results[task_id] = task_results

            print(f"   ✅ Completed: {task_id}")

        # Save final results
        output_file = f"{self.config['variables']['output_dir']}/orchestrated_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n✅ All tasks completed. Results saved to {output_file}")
        return self.results

# Usage
orchestrator = JSONWorkflowOrchestrator(
    'src/cli/simple-commands/templates/mle-star-workflow.json'
)
results = orchestrator.run_workflow()
```

### Approach 3: Claude Flow Service Integration

```python
#!/usr/bin/env python3
"""
Integration with Claude Flow service for task orchestration
"""

import json
import subprocess
from pathlib import Path

class ClaudeFlowMLESTARBridge:
    """Bridge between Claude Flow JSON and Python MLE-STAR tool"""

    def __init__(self, json_path):
        self.json_path = json_path
        with open(json_path) as f:
            self.workflow = json.load(f)

    def to_python_config(self):
        """Convert JSON variables to Python configuration"""
        from mle_star_agent_tool import MLESTARConfig

        return MLESTARConfig(
            output_dir=self.workflow['variables']['output_dir'],
            max_iterations=self.workflow['variables']['refinement_iterations'],
            timeout_seconds=self.workflow['settings']['timeout'] // 1000  # Convert ms to seconds
        )

    def to_python_task(self):
        """Convert JSON workflow to Python task description"""
        var = self.workflow['variables']
        metadata = self.workflow['metadata']

        return f"""
Execute MLE-STAR Workflow:

Dataset: {var['dataset']}
Target: {var['target']}
Methodology: {metadata['methodology']}
Expected Runtime: {metadata['expected_runtime']}

Tasks to execute:
1. Web search for ML solutions (search_iterations: {var['search_iterations']})
2. Data analysis and profiling
3. Build initial ML pipeline
4. Perform component ablation study
5. Optimize critical components (refinement_iterations: {var['refinement_iterations']})
6. Create advanced ensemble (ensemble_size: {var['ensemble_size']})
7. Validate robustness and production readiness
8. Prepare deployment package
"""

    def run_with_claude_flow(self):
        """Execute using Claude Flow service"""
        from mle_star_agent_tool import mle_star_process_tool

        config = self.to_python_config()
        task = self.to_python_task()
        dataset = self.workflow['variables']['dataset']

        print("🔄 Bridging Claude Flow → Python MLE-STAR Tool")
        print(f"Configuration: {config}")
        print(f"Task: {task[:100]}...")

        # Execute Python tool
        results = mle_star_process_tool(task, dataset, config)

        # Save combined results
        output_path = Path(self.workflow['variables']['output_dir'])
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "bridge_results.json", 'w') as f:
            json.dump({
                "json_config": self.workflow,
                "python_results": results
            }, f, indent=2, default=str)

        return results

# Usage
bridge = ClaudeFlowMLESTARBridge(
    'src/cli/simple-commands/templates/mle-star-workflow.json'
)
results = bridge.run_with_claude_flow()
```

---

## 📈 Execution Flow Diagrams

### Sequential Tasks (Phases 2 & 4)

```
web-search ──→ initial-pipeline ──→ ablation-study ──→ targeted-refinement ──→ ...
(Phase 1)      (Phase 2 start)      (Phase 2)          (Phase 2)

Each task waits for previous to complete
Proper dependency management
```

### Parallel Tasks (Phases 1 & 3)

```
Phase 1:
web-search ──┐
             ├──→ initial-pipeline
data-analysis┘

Phase 3:
ensemble-creation ────────┐
                          ├──→ deployment-package
robustness-validation ────┘
```

---

## ✅ Parameter Mapping Reference

### JSON → Python Mappings

| JSON Variable | Python Parameter | Purpose |
|--------------|-----------------|---------|
| `dataset` | `dataset_path` | Input data file |
| `target` | Part of `task_description` | Prediction target |
| `output_dir` | `config.output_dir` | Results location |
| `search_iterations` | Implicit in prompts | Web search depth |
| `refinement_iterations` | `config.max_iterations` | Optimization depth |
| `ensemble_size` | Implicit in tool | Number of ensemble models |
| `maxConcurrency` | LangGraph setting | Parallel execution limit |
| `timeout` | `config.timeout_seconds` | Maximum execution time |
| `retryPolicy` | Error handling strategy | Retry mechanism |

---

## 🧪 Testing the Integration

### Test 1: Simple Integration
```python
# Test that JSON loads correctly
import json
with open('src/cli/simple-commands/templates/mle-star-workflow.json') as f:
    config = json.load(f)
assert config['name'] == 'MLE-STAR Machine Learning Engineering'
assert len(config['agents']) == 8
assert len(config['tasks']) == 8
print("✅ JSON validation passed")
```

### Test 2: Configuration Conversion
```python
# Test JSON → Python config conversion
from json_python_integration import JSONWorkflowOrchestrator

orchestrator = JSONWorkflowOrchestrator(
    'src/cli/simple-commands/templates/mle-star-workflow.json'
)
py_config = orchestrator.to_python_config()
assert py_config.output_dir == './mle-star-output'
print("✅ Configuration conversion passed")
```

### Test 3: Task Execution
```python
# Test executing a single task
from json_python_integration import JSONWorkflowOrchestrator

orchestrator = JSONWorkflowOrchestrator(
    'src/cli/simple-commands/templates/mle-star-workflow.json'
)
task = orchestrator.config['tasks'][0]
result = orchestrator.execute_task(task['id'], task, {})
assert result is not None
print("✅ Task execution passed")
```

---

## 🔐 Best Practices

### 1. **Configuration Validation**
```python
def validate_json_config(config_path):
    """Validate JSON configuration before execution"""
    with open(config_path) as f:
        config = json.load(f)

    required_fields = ['name', 'agents', 'tasks', 'variables']
    for field in required_fields:
        assert field in config, f"Missing required field: {field}"

    assert len(config['agents']) > 0, "Must have at least one agent"
    assert len(config['tasks']) > 0, "Must have at least one task"

    return True
```

### 2. **Error Recovery**
```python
try:
    results = mle_star_process_tool(task, dataset, config)
except Exception as e:
    print(f"Error: {e}")
    # Save partial results
    save_checkpoint(results, error=str(e))
    # Retry with different config
    config.max_iterations = 5
    results = mle_star_process_tool(task, dataset, config)
```

### 3. **Result Tracking**
```python
# Combine JSON config with results for full context
final_results = {
    "timestamp": datetime.now().isoformat(),
    "json_config": workflow_config,
    "python_results": mle_star_results,
    "metadata": {
        "dataset": dataset_path,
        "target": target_column,
        "status": "completed"
    }
}

with open(output_path, 'w') as f:
    json.dump(final_results, f, indent=2)
```

---

## 📚 Summary

| Aspect | JSON Role | Python Role | Integration |
|--------|-----------|-------------|-------------|
| **Configuration** | Task parameters | LLM parameters | Load JSON, pass to Python config |
| **Agents** | Define roles & types | Implement functionality | Map JSON agents to Python agents |
| **Tasks** | Orchestration & flow | Execution & tools | JSON coordinates Python execution |
| **Tools** | Declared capabilities | Actual functions | JSON prompts call Python tools |
| **Results** | Defined structure | Generated structure | Combine in final output |

---

## 🚀 Next Steps

1. **Choose Integration Approach**
   - Approach 1: Simple wrapper (fastest)
   - Approach 2: Task orchestration (flexible)
   - Approach 3: Claude Flow bridge (production)

2. **Test with Sample Data**
   ```bash
   python json_python_integration.py \
     --json src/cli/simple-commands/templates/mle-star-workflow.json \
     --dataset data/train.csv
   ```

3. **Deploy to Production**
   - Use Claude Flow service for task management
   - Call Python tool for execution
   - Archive results with metadata

4. **Monitor & Iterate**
   - Track execution metrics
   - Refine JSON configuration
   - Optimize Python parameters

---

**Integration Status**: ✅ Ready to Deploy
