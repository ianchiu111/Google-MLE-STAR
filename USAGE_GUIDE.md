# MLE-STAR Complete - Single File Implementation

## Quick Start Guide

This guide shows you how to use the **complete single-file implementation** of the MLE-STAR agent framework (`mle_star_complete.py`).

---

## What is MLE-STAR?

**MLE-STAR** (Machine Learning Engineering Agent via Search and Targeted Refinement) is an intelligent multi-agent system that:

- 🔍 **Searches** the web for relevant ML techniques and models
- 💻 **Generates** Python code for your ML tasks
- ⚙️ **Executes** code and iterates until successful
- 📊 **Saves** all results, models, and visualizations
- 🤖 **Orchestrates** multiple AI agents using LangGraph

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `langchain_core`, `langchain_openai`, `langgraph` - Agent framework
- `duckduckgo-search` - Web search capability
- `langchain-experimental` - Python code execution
- `pandas`, `numpy`, `scikit-learn` - ML libraries

### 2. Set Up LLM

**Option A: Ollama (Local, Free)**
```bash
# Install Ollama: https://ollama.ai
ollama pull qwen2.5:7b-instruct

# Start Ollama server (runs on http://localhost:11434)
ollama serve
```

**Option B: OpenAI ChatGPT**
```bash
# Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

Then edit `mle_star_complete.py` lines 58-62 to use OpenAI configuration.

---

## Basic Usage

### Method 1: Run with Default Example

```bash
python mle_star_complete.py
```

This runs the default Rossmann Store Sales prediction task.

### Method 2: Provide Custom Query

```bash
python mle_star_complete.py "Your ML task description here"
```

Example:
```bash
python mle_star_complete.py "Build a classification model for data/iris.csv, predict 'species' column, save results"
```

### Method 3: Use as Python Module

```python
from mle_star_complete import run_mle_star

query = """
Please analyze data/customer_churn.csv and build a churn prediction model.
Steps:
1. Load and explore the data
2. Clean data and engineer features
3. Train RandomForest, XGBoost, and LightGBM models
4. Compare performance and select the best model
5. Save everything to data/information_from_agent/
"""

results = run_mle_star(query, thread_id="my-session-001")
```

---

## Real-World Examples

### Example 1: Sales Forecasting

```python
from mle_star_complete import run_mle_star

query = """
Load data/train.csv (Rossmann Store Sales dataset).
Target: Predict 'Sales' column.

Tasks:
1. Merge with data/store.csv for additional store features
2. Feature engineering: extract date features, create ratios
3. Handle missing values and outliers
4. Train multiple regression models: Linear Regression, RandomForest, XGBoost, LightGBM
5. Use cross-validation to tune hyperparameters
6. Evaluate with RMSE, MAE, R²
7. Save best model, predictions, and visualizations

Output to: data/information_from_agent/
"""

results = run_mle_star(query)
```

### Example 2: Binary Classification

```python
query = """
Build a binary classification model for data/credit_default.csv
Predict 'default' column (0 or 1).

Steps:
1. EDA: Check class balance, correlations, distributions
2. Handle imbalanced classes (SMOTE or class weights)
3. Feature selection using feature importance
4. Train: LogisticRegression, RandomForest, XGBoost
5. Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC
6. Generate ROC curves and confusion matrices
7. Save all outputs

Output location: data/information_from_agent/
"""

results = run_mle_star(query)
```

### Example 3: Multi-Class Classification

```python
query = """
Classify iris flowers using data/iris.csv
Target: 'species' column (setosa, versicolor, virginica)

Tasks:
1. Load and visualize data (pairplot, correlation heatmap)
2. Split data: 70% train, 15% validation, 15% test
3. Try multiple classifiers: SVM, RandomForest, XGBoost, Neural Network
4. Use GridSearchCV for hyperparameter tuning
5. Generate classification reports and confusion matrices
6. Save the best model as a pickle file
7. Create visualization of decision boundaries

Save everything to: data/information_from_agent/
"""

results = run_mle_star(query)
```

### Example 4: Time Series Analysis

```python
query = """
Analyze time series data in data/stock_prices.csv
Predict next 30 days of stock prices.

Steps:
1. Load data and check for stationarity (ADF test)
2. Create lag features and rolling statistics
3. Split data chronologically (no shuffle)
4. Try: ARIMA, Prophet, LSTM, XGBoost with time features
5. Evaluate with RMSE, MAE, MAPE
6. Plot actual vs predicted values
7. Generate forecast for next 30 days

Output: data/information_from_agent/
"""

results = run_mle_star(query)
```

---

## Configuration Options

### Modify LLM Settings

Edit the `Config` class in `mle_star_complete.py`:

```python
class Config:
    # Change model
    LLM_MODEL = "gpt-4o-mini"  # or "qwen2.5:7b-instruct"

    # Adjust temperature (0 = deterministic, 1 = creative)
    LLM_TEMPERATURE = 0.2

    # Change output directory
    OUTPUT_DIR = "outputs/"
```

### Modify Web Search Settings

```python
class Config:
    SEARCH_REGION = "us-en"  # Change region
    SEARCH_TIMELIMIT = "w"   # w=week, m=month, y=year
    SEARCH_MAX_RESULTS = 10  # Get more results
```

---

## Understanding the Workflow

The framework executes in two stages:

```
┌─────────────────────────────────────────────┐
│  1. WEB SEARCH AGENT                        │
│  - Analyzes your ML task                    │
│  - Searches for relevant techniques         │
│  - Finds state-of-the-art models            │
│  - Recommends best practices                │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  2. CODE GENERATOR AGENT                    │
│  - Loads and explores your data             │
│  - Performs feature engineering             │
│  - Trains multiple ML models                │
│  - Evaluates and compares models            │
│  - Saves all results and code               │
└─────────────────────────────────────────────┘
```

---

## Output Files

All results are saved to `data/information_from_agent/`:

```
data/information_from_agent/
├── eda_results.csv              # Exploratory data analysis
├── cleaned_data.csv             # Processed dataset
├── feature_importance.png       # Feature importance plot
├── model_comparison.csv         # Model performance metrics
├── best_model.pkl               # Trained model (pickle)
├── predictions.csv              # Model predictions
├── confusion_matrix.png         # Classification results
├── roc_curve.png                # ROC curve
└── generated_code.py            # All generated Python code
```

---

## Troubleshooting

### Issue: "Connection refused" error

**Solution:** Make sure Ollama is running:
```bash
ollama serve
```

### Issue: "Module not found" error

**Solution:** Install missing dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Code execution fails

**Solution:** Check the error message in the output. The agent will attempt to fix and retry. You can also:
1. Check if the data file path is correct
2. Ensure required columns exist in the dataset
3. Verify sufficient memory for large datasets

### Issue: Web search returns no results

**Solution:**
1. Check internet connection
2. Try a different search region in Config
3. Simplify your query (agent will extract key terms)

---

## Advanced Usage

### Custom Agent Prompts

Modify the prompts to specialize the agents:

```python
# Make the code generator focus on deep learning
CODE_GENERATOR_AGENT_PROMPT = """
You are an expert Deep Learning Engineer.
Use TensorFlow and PyTorch to build neural networks.
Focus on: CNNs for images, RNNs for sequences, Transformers for NLP.
...
"""
```

### Add Custom Tools

```python
@tool
def load_pretrained_model(model_name: Annotated[str, "Model name"]) -> str:
    """Load a pretrained model from HuggingFace"""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name)
    return f"Loaded {model_name} successfully"

# Add to code generator agent tools
code_generator_agent_executor = create_agent(
    agent_name="code_generator_agent",
    llm=llm,
    tools=[run_python_code, load_pretrained_model],  # Added custom tool
    system_prompt=CODE_GENERATOR_AGENT_PROMPT
)
```

### Session Management

Use thread IDs to maintain conversation history:

```python
# Session 1
run_mle_star("Build a model for data1.csv", thread_id="session-001")

# Session 2 (separate context)
run_mle_star("Build a model for data2.csv", thread_id="session-002")

# Continue session 1
run_mle_star("Now tune hyperparameters", thread_id="session-001")
```

---

## Performance Tips

1. **Use smaller models for faster iteration**
   - `qwen2.5:7b-instruct` is faster than `qwen2.5:32b`
   - `gpt-4o-mini` is faster than `gpt-4o`

2. **Limit search results for faster research**
   ```python
   SEARCH_MAX_RESULTS = 3  # Fewer results = faster search
   ```

3. **Use specific queries**
   - Good: "Build XGBoost classifier for imbalanced data"
   - Bad: "Do machine learning on my data"

4. **Start with small datasets**
   - Test with 1000 rows first
   - Scale up once working

---

## Comparison with Modular Version

| Feature | Single File (`mle_star_complete.py`) | Modular Version (`MLE_Agent/`) |
|---------|--------------------------------------|--------------------------------|
| **Setup** | ✅ One file, easy to run | ❌ Multiple imports needed |
| **Portability** | ✅ Copy one file anywhere | ❌ Need entire directory |
| **Customization** | ⚠️ Edit one large file | ✅ Modify individual modules |
| **Readability** | ⚠️ ~400 lines in one file | ✅ Organized by component |
| **Best For** | Quick start, demos, sharing | Production, team development |

---

## Next Steps

1. **Try the default example**
   ```bash
   python mle_star_complete.py
   ```

2. **Experiment with your own data**
   ```bash
   python mle_star_complete.py "Analyze my_data.csv and predict target_column"
   ```

3. **Customize for your domain**
   - Modify agent prompts for specific industries
   - Add domain-specific tools
   - Adjust search parameters

4. **Extend the framework**
   - Add ablation agent for iterative refinement
   - Implement ensemble methods
   - Create validation agent for model testing

---

## Resources

- **Original Paper:** [Google MLE-STAR on arXiv](https://arxiv.org/abs/2506.15692v3)
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **Ollama:** https://ollama.ai
- **DuckDuckGo Search:** https://github.com/deedy5/duckduckgo_search

---

## Support

For issues or questions:
1. Check the error message in the console output
2. Review the generated code in `data/information_from_agent/`
3. Verify your LLM is running (`ollama serve` or OpenAI key)
4. Ensure dataset paths are correct

---

**Happy Machine Learning!** 🚀
