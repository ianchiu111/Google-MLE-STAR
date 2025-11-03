import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Annotated
from langchain_core.tools import tool


@tool
def call_claude_flow_mle_star(
    task_description: Annotated[ str, " description of the ML task to be performed. "],
    dataset: Annotated[ str, " it should be a file path end with .csv "],
    target: Annotated[ str, " it should be the target column name in the dataset for machine learning process. "],
    output: Annotated[ str, " it should be a folder path to store the result from claude-flow CLI "] = "models/",
    search_iterations: Annotated[ str, " how many times to search for the best model, please input with arabic numerals in string. "] = "3",
    refinement_iterations: Annotated[ str, " how many times to refine the model, please input with arabic numerals in string. "] = "3",
    nums_solutions: Annotated[ str, " how many machine learning models to generate, please input with arabic numerals in string. "] = "5",
):
    """
    Call Claude Flow MLE-STAR automation with --claude flag
    claude-flow automation mle-star 
    --dataset data/train.csv 
    --target Sales 
    --output models/
    --search_iterations 3 
    --nums_solutions 5 
    --task_description "please use the train.csv dataset and target column to predict sales revenue" 
    --refinement_iterations 3
    --claude
    """
    
    # Build command
    command = [
        "claude-flow",
        "automation", 
        "mle-star",
        "--dataset", str(dataset),
        "--target", str(target),
        "--output", str(output),
        "--search_iterations", str(search_iterations),
        "--nums_solutions", str(nums_solutions),
        "--task_description", str(task_description),
        "--refinement_iterations", str(refinement_iterations),
        "--claude"
    ]

    print("Double Check - Command to be executed:\n", command)
    
    # Execute command with real-time output streaming
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Stream output in real-time and collect it
    output_lines = []
    for line in process.stdout:
        print(line, end='')  # Print to terminal in real-time
        output_lines.append(line)

    process.wait()

    # Combine all output
    full_output = ''.join(output_lines)
    
    # Create result object similar to subprocess.run
    result = subprocess.CompletedProcess(
        command, 
        process.returncode, 
        full_output, 
        ""  # stderr is merged with stdout
    )


if __name__ == "__main__":
    # Test the tool independently
    output = call_claude_flow_mle_star(
        task_description="please use the train.csv dataset and target column to predict sales revenue",
        dataset="data/train.csv",
        target="Sales",
        output="models/",
        search_iterations=3,
        refinement_iterations=3,
        nums_solutions=5,
    )
    print("Tool Output:\n", output)