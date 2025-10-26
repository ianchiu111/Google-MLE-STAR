## Welcome 
Hi, I'm YuChen Chiu. 

This project will going to use the paper written by google cloud team to practice Machine Learning Engineering Agent via search and targeted refinement.

### Reference
* [Google Cloud - MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement](https://arxiv.org/abs/2506.15692v3) 
* [Web Search example - Langgraph with Ollama](https://github.com/john-adeojo/graph_websearch_agent)

### （一）Machine‑Learning AI Agent Framework and Concepts

#### System Workflow

Overview of MLE-STAR

(a) **AI Research stage**: MLE-STAR use **Web Search** to retrieves task-specific models to generate an initial solution.

(b) **Outer Loop: Target Issue stage**: Find out which target code block of ML components can be better via [**ablation study**](https://blog.csdn.net/flyfish1986/article/details/104812229).

(c) **Inner Loop: Code Block Refinement**: Iteratively refine the target code block until where the improved solution of inner loop becomes the latest solution in outer loop.

<img src="images/MLE-Agent Workflow.png" alt="image" width="600"/>


#### Workflow in this project
1. Web Search for Model Information
2. Python Code Generator
3. Ablation 
4. 


###  （二）Practice Dataset - Rossmann Store Sales

* [Resource: Kaggle Dataset](https://www.kaggle.com/competitions/rossmann-store-sales/)

#### Dataset Introduction

#### Dataset Cleaning

#### Feature Engineering

### （三）Web Search Apporoach - DuckDuckGo Search Engine

Due to the resource limitation, so choose duckduckgo search engine as web search method

<img src="images/duckduckgo-search-engine.png" alt="image" width="600"/>
