"""
MLE-STAR Tool Usage Examples
Practical examples of how to use the MLE-STAR agent tool in various scenarios
"""

import asyncio
import json
from pathlib import Path
from what_I_can_save_for_langgraph_practice.mle_star_agent_tool import (
    mle_star_process_tool,
    mle_star_process_tool_async,
    MLESTARWorkflow,
    MLESTARConfig
)


# ==================== Example 1: Basic Usage ====================

def example_1_basic_usage():
    """
    Example 1: Simple usage of MLE-STAR for Rossmann store sales prediction
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)

    results = mle_star_process_tool(
        task_description="Predict daily store sales for Rossmann stores using historical data "
                        "and store information",
        dataset_path="data/train.csv"
    )

    print(f"✓ Task: {results.get('task_description')}")
    print(f"✓ Dataset: {results.get('dataset_path')}")
    print(f"✓ Steps Completed: {len(results.get('workflow_messages', []))}")
    print(f"✓ Completion Time: {results.get('completion_timestamp')}")

    return results


# ==================== Example 2: Custom Configuration ====================

def example_2_custom_configuration():
    """
    Example 2: Using custom configuration for different LLM models
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)

    # Configure for higher accuracy but slower execution
    config = MLESTARConfig(
        llm_base_url="http://127.0.0.1:11434/v1",
        llm_api_key="ollama",
        llm_model="qwen2.5:7b-instruct",  # More capable model
        temperature=0.0,  # Deterministic output
        max_iterations=15,  # More iterations for better results
        timeout_seconds=7200,  # 2 hours timeout
        output_dir="data/information_from_agent"
    )

    workflow = MLESTARWorkflow(config)
    results = workflow.run(
        task_description="Advanced prediction with custom configuration",
        dataset_path="data/train.csv"
    )

    print(f"✓ Model: {config.llm_model}")
    print(f"✓ Temperature: {config.temperature}")
    print(f"✓ Output Directory: {config.output_dir}")
    print(f"✓ Completed: {'error' not in results}")

    return results


# ==================== Example 3: Different ML Tasks ====================

def example_3_different_tasks():
    """
    Example 3: Using MLE-STAR for different types of ML problems
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Different ML Tasks")
    print("="*60)

    tasks = [
        {
            "name": "Time Series Forecasting",
            "description": "Forecast daily sales 30 days into the future based on historical patterns",
            "dataset": "data/train.csv"
        },
        {
            "name": "Classification",
            "description": "Classify whether a store will perform above or below median sales",
            "dataset": "data/train.csv"
        },
        {
            "name": "Anomaly Detection",
            "description": "Identify unusual sales patterns that deviate from normal behavior",
            "dataset": "data/train.csv"
        },
        {
            "name": "Feature Analysis",
            "description": "Identify which factors most strongly influence store sales",
            "dataset": "data/train.csv"
        },
    ]

    results_dict = {}

    for task in tasks:
        print(f"\n  Running: {task['name']}")
        try:
            results = mle_star_process_tool(
                task_description=task['description'],
                dataset_path=task['dataset']
            )
            results_dict[task['name']] = results
            print(f"  ✓ {task['name']} completed successfully")
        except Exception as e:
            print(f"  ✗ {task['name']} failed: {e}")
            results_dict[task['name']] = {"error": str(e)}

    return results_dict


# ==================== Example 4: Async Execution ====================

async def example_4_async_execution():
    """
    Example 4: Running multiple workflows asynchronously
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Async Execution (Parallel Workflows)")
    print("="*60)

    # Create multiple tasks
    tasks = [
        ("Sales Prediction", "data/train.csv"),
        ("Customer Analysis", "data/train.csv"),
        ("Store Performance", "data/train.csv"),
    ]

    # Run all tasks in parallel
    print(f"  Starting {len(tasks)} workflows in parallel...")
    async_tasks = [
        mle_star_process_tool_async(desc, dataset)
        for desc, dataset in tasks
    ]

    results = await asyncio.gather(*async_tasks, return_exceptions=True)

    print(f"  ✓ All {len(results)} workflows completed")
    for i, (task_name, _) in enumerate(tasks):
        status = "✓ Success" if not isinstance(results[i], Exception) else "✗ Failed"
        print(f"  {status}: {task_name}")

    return results


# ==================== Example 5: Batch Processing ====================

def example_5_batch_processing():
    """
    Example 5: Processing multiple datasets in batch mode
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Batch Processing")
    print("="*60)

    # Simulate multiple datasets (in real scenario, these would be different files)
    batch_tasks = [
        {
            "id": "task_001",
            "description": "Store Sales Forecast - Q1 2025",
            "dataset": "data/train.csv"
        },
        {
            "id": "task_002",
            "description": "Store Sales Forecast - Q2 2025",
            "dataset": "data/train.csv"
        },
        {
            "id": "task_003",
            "description": "Store Sales Forecast - Q3 2025",
            "dataset": "data/train.csv"
        },
    ]

    batch_results = {
        "batch_id": "batch_rossmann_2025",
        "timestamp": datetime.now().isoformat(),
        "tasks": []
    }

    for task in batch_tasks:
        print(f"\n  Processing: {task['id']} - {task['description'][:40]}...")
        try:
            results = mle_star_process_tool(
                task_description=task['description'],
                dataset_path=task['dataset']
            )
            batch_results['tasks'].append({
                "task_id": task['id'],
                "status": "completed",
                "results": results
            })
            print(f"  ✓ {task['id']} completed")
        except Exception as e:
            batch_results['tasks'].append({
                "task_id": task['id'],
                "status": "failed",
                "error": str(e)
            })
            print(f"  ✗ {task['id']} failed: {e}")

    # Save batch results
    output_file = "batch_processing_results.json"
    with open(output_file, 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)
    print(f"\n  ✓ Batch results saved to {output_file}")

    return batch_results


# ==================== Example 6: Integration with Custom Logic ====================

def example_6_custom_integration():
    """
    Example 6: Integrating MLE-STAR with custom preprocessing and post-processing
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Integration")
    print("="*60)

    import pandas as pd

    # Step 1: Custom preprocessing
    print("\n  Step 1: Custom Data Preprocessing")
    df = pd.read_csv("data/train.csv")
    print(f"  ✓ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Create a temporary processed dataset
    processed_path = "data/preprocessed_temp.csv"
    df_processed = df.dropna().copy()
    df_processed.to_csv(processed_path, index=False)
    print(f"  ✓ Preprocessed data saved to {processed_path}")

    # Step 2: Run MLE-STAR workflow
    print("\n  Step 2: Running MLE-STAR Workflow")
    results = mle_star_process_tool(
        task_description="Sales prediction with custom preprocessing",
        dataset_path=processed_path
    )
    print(f"  ✓ Workflow completed")

    # Step 3: Custom post-processing
    print("\n  Step 3: Custom Result Analysis")
    if "error" not in results:
        print(f"  ✓ Task: {results['task_description'][:50]}...")
        print(f"  ✓ Messages: {len(results.get('workflow_messages', []))} agent interactions")
        print(f"  ✓ Timestamp: {results['completion_timestamp']}")

    # Cleanup
    Path(processed_path).unlink()
    print(f"  ✓ Temporary files cleaned up")

    return results


# ==================== Example 7: Error Handling ====================

def example_7_error_handling():
    """
    Example 7: Proper error handling and recovery
    """
    print("\n" + "="*60)
    print("EXAMPLE 7: Error Handling and Recovery")
    print("="*60)

    # Test with non-existent file
    print("\n  Testing error handling with invalid dataset...")
    try:
        results = mle_star_process_tool(
            task_description="Test with invalid dataset",
            dataset_path="data/nonexistent.csv"
        )

        if "error" in results:
            print(f"  ✓ Error caught: {results['error']}")
            print(f"  ✓ Error log entries: {len(results.get('error_log', []))}")
        else:
            print(f"  ✓ Workflow completed despite invalid dataset")

    except FileNotFoundError as e:
        print(f"  ✓ FileNotFoundError caught: {e}")
    except Exception as e:
        print(f"  ✓ Exception caught: {type(e).__name__}: {e}")

    # Test with valid file
    print("\n  Testing with valid dataset...")
    try:
        results = mle_star_process_tool(
            task_description="Sales prediction",
            dataset_path="data/train.csv"
        )

        if "error" not in results:
            print(f"  ✓ Workflow successful")
        else:
            print(f"  ✗ Workflow had errors: {results['error']}")

    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")


# ==================== Example 8: Monitoring and Logging ====================

def example_8_monitoring_logging():
    """
    Example 8: Monitoring workflow execution and accessing logs
    """
    print("\n" + "="*60)
    print("EXAMPLE 8: Monitoring and Logging")
    print("="*60)

    import logging
    from what_I_can_save_for_langgraph_practice.mle_star_agent_tool import logger

    # Configure custom logging
    print("\n  Setting up monitoring...")

    # Create file handler for results
    results_dir = Path("data/information_from_agent")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run workflow with monitoring
    print("  Running workflow...")
    results = mle_star_process_tool(
        task_description="Sales prediction with monitoring",
        dataset_path="data/train.csv"
    )

    # Check log files
    print("\n  Checking generated files...")
    log_files = list(results_dir.glob("mle_star_*.log"))
    result_files = list(results_dir.glob("mle_star_results_*.json"))

    print(f"  ✓ Log files generated: {len(log_files)}")
    if log_files:
        print(f"    Latest: {log_files[-1].name}")

    print(f"  ✓ Result files generated: {len(result_files)}")
    if result_files:
        print(f"    Latest: {result_files[-1].name}")
        # Load and display results summary
        with open(result_files[-1]) as f:
            result_data = json.load(f)
            print(f"    Completion time: {result_data.get('completion_timestamp')}")


# ==================== Example 9: Configuration Comparison ====================

def example_9_configuration_comparison():
    """
    Example 9: Comparing different configurations
    """
    print("\n" + "="*60)
    print("EXAMPLE 9: Configuration Comparison")
    print("="*60)

    configs = [
        {
            "name": "Fast (Minimal)",
            "config": MLESTARConfig(
                temperature=0.1,
                max_iterations=5,
                timeout_seconds=1800
            )
        },
        {
            "name": "Balanced",
            "config": MLESTARConfig(
                temperature=0.0,
                max_iterations=10,
                timeout_seconds=3600
            )
        },
        {
            "name": "Thorough (Comprehensive)",
            "config": MLESTARConfig(
                temperature=0.0,
                max_iterations=15,
                timeout_seconds=7200
            )
        },
    ]

    print("\n  Configuration Comparison:")
    print("  " + "-"*56)

    for cfg in configs:
        print(f"\n  {cfg['name']}:")
        print(f"    Temperature: {cfg['config'].temperature}")
        print(f"    Max Iterations: {cfg['config'].max_iterations}")
        print(f"    Timeout: {cfg['config'].timeout_seconds}s")
        print(f"    Output Dir: {cfg['config'].output_dir}")


# ==================== Main Runner ====================

def run_all_examples():
    """Run all examples"""
    from datetime import datetime

    print("\n" + "="*60)
    print("MLE-STAR TOOL - COMPREHENSIVE USAGE EXAMPLES")
    print("="*60)
    print(f"Started: {datetime.now().isoformat()}")

    try:
        # Run basic example (always safe)
        print("\n✓ Running Example 1 (Basic Usage)...")
        example_1_basic_usage()

        # Run custom configuration example
        print("\n✓ Running Example 2 (Custom Configuration)...")
        example_2_custom_configuration()

        # Run different tasks example
        print("\n✓ Running Example 3 (Different Tasks)...")
        example_3_different_tasks()

        # Run async example
        print("\n✓ Running Example 4 (Async Execution)...")
        asyncio.run(example_4_async_execution())

        # Run batch processing example
        print("\n✓ Running Example 5 (Batch Processing)...")
        example_5_batch_processing()

        # Run custom integration example
        print("\n✓ Running Example 6 (Custom Integration)...")
        example_6_custom_integration()

        # Run error handling example
        print("\n✓ Running Example 7 (Error Handling)...")
        example_7_error_handling()

        # Run monitoring example
        print("\n✓ Running Example 8 (Monitoring and Logging)...")
        example_8_monitoring_logging()

        # Run configuration comparison
        print("\n✓ Running Example 9 (Configuration Comparison)...")
        example_9_configuration_comparison()

        print("\n" + "="*60)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Finished: {datetime.now().isoformat()}")

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    from datetime import datetime

    # Run all examples
    run_all_examples()
