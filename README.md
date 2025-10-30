# Welcome 
Hi, I'm YuChen Chiu. 

This project will going to use the paper written by google cloud team to practice **Machine Learning Engineering Agent via Search and Targeted Refinement Workflow** by two approaches below:

1. Langgraph + Ollama in Multi-Agent System
2. Claude-Flow Service in Sales Prediction Workflow


## 1️⃣ Langgraph + Ollama in Multi-Agent System
### 📚 Reference
* Paer Reading
    * [Google Cloud - MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement](https://arxiv.org/abs/2506.15692v3) 
* Claude-Flow
    * [Github Repo](https://github.com/ruvnet/claude-flow)
    * [Wiki to understand the detail concepts)](https://github.com/ruvnet/claude-flow/wiki/Agent-System-Overview)
* Code Examples
    * [Web Search example - Langgraph with Ollama](https://github.com/john-adeojo/graph_websearch_agent)

### 📁 Practice Dataset
* [Rossmann Store Sales Prediction](https://www.kaggle.com/competitions/rossmann-store-sales/)

### 🤖 Machine‑Learning AI Agent Framework and Concepts

#### (一) System Workflow

Overview of MLE-STAR

(a) **AI Research stage**: MLE-STAR use **Web Search** to retrieves task-specific models to generate an initial solution.

(b) **Outer Loop: Target Issue stage**: Find out which target code block of ML components can be better via [**ablation study**](https://blog.csdn.net/flyfish1986/article/details/104812229).

(c) **Inner Loop: Code Block Refinement**: Iteratively refine the target code block until where the improved solution of inner loop becomes the latest solution in outer loop.

<img src="images/MLE-Agent Workflow.png" alt="image" width="600"/>


#### (二) Workflow in Langgraph Multi-Agent System
1. **Web Search**：Search for latest model 
2. **Deep Research**：Analyze model algorithm and example code
3. **Code Generation**：Initial solution version 1
4. **Model Evaluation**：
5. **Code Refinement**：
6. **Solution Refinement**：Refine model and codes
7. **Summary Report**：Summarize final solution


### 🔧 Tool ExplanationWeb 

#### (一) Model Configuration

#### (二) DuckDuckGo Search Engine

Due to the resource limitation, so choose duckduckgo search engine as web search method

<img src="images/duckduckgo-search-engine.png" alt="image" width="600"/>


## 2️⃣ Claude-Flow Service in Sales Prediction Workflow

### Basic Information
1. Branch Name：**claude-flow/dev_main**
2. Repo Architecture
```plaintext
/Users/yuchen/Google-MLE-Agent/
├── data/                                   ⭐ Rossmann Sales Predictioin Dataset
│   ├── data_cleaning.ipynb.                ⭐ Analyze Raw Dataset Myself
│   ├── train.csv                           
│   ├── test.csv
│   ├── store.csv
│   └── sample_submission.csv
├── images/                                 ⭐ All necessary images
├── ./models/                        ⭐ All outputs from mle-star workflow
├── src/
│   └── cli/
│       └── simple-commands/
│           └── templates/
│               └── mle-star-workflow.json  ⭐ Template to interact with claude-flow by CLI
├── call_claude_flow_tool.py                ⭐ Python interface
├── requirements.txt                        ⭐ Python dependencies
├── README.md                               ⭐ Documentation
└── finally_fix_the_issue.md                ⭐ This guide
```

### Some Issues I Found
1. claude-flow github repo is **v2.7.12**, but MLE_STAR workflow alpha version appears in **v2.7.26**

2. Claude-Flow doesn't support open model source to set into **src/cli/simple-commands/templates/mle-star-workflow.json** yet, so this practice still need to use claude-flow CLI to practice MLE_STAR workflow <2025/10/30>

### Testing Record

#### First Testing Record - mle_star_execution_20251030_141953.md
1. ✅ Goods News
    1. Both of the summaries by **ml-researcher-agent** and **data-analyst-agent** store in folder **models/v1-testing-record**
2. ❌ Bad News
    1. Only **ml-researcher-agent** and **data-analyst-agent** work, because agent in task2-1 doesn't get the specific file as input
3. ➡️ Next Optimization
    1. **Prompt engineering** on agents' scripts to make the agents work. Reference from [claude-flow github repo](https://github.com/ruvnet/claude-flow/tree/main/.claude/agents/analysis/code-review).
    2. **Check workflow input and output**

#### 