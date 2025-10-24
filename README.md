## Welcome 
Hi, I'm YuChen Chiu. 

This project will going to use the paper written by google cloud team to practice Machine Learning Engineering Agent via search and targeted refinement.

### Reference
* [Google Cloud - MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement](https://arxiv.org/abs/2506.15692v3) 
* [Web Search example - Langgraph with Ollama](https://github.com/john-adeojo/graph_websearch_agent)

### （一）Machine‑Learning AI Agent Framework and Concepts

#### System Workflow

Overview of MLE-STAR

(a) Using search as a tool, MLE-STAR retrieves task-specific models
and uses them to generate an initial solution. 

(b) In each refinement step, MLE-STAR performs an ablation study to extract the code block that have the greatest impact. Previously modified code blocks are also provided as feedback for diversity. 

(c) The extracted code block is iteratively refined based on
plans suggested by the LLM, which explores various plans using previous experiments as feedback
(i.e., inner loop), and the target code block is also selected repeatedly (i.e., outer loop, where the
improved solution of (c) becomes the previous solution in (b)).

<img src="images/MLE-Agent Workflow.png" alt="image" width="600"/>



###  （二）Practice Dataset - Rossmann Store Sales

* [Resource: Kaggle Dataset](https://www.kaggle.com/competitions/rossmann-store-sales/)

#### Dataset Introduction

#### Dataset Cleaning

#### Feature Engineering

