#!/usr/bin/env python3
"""
Call Claude Flow MLE-STAR automation service and capture outputs

https://github.com/ruvnet/claude-flow/wiki/MLE-STAR-Workflow

> python3 call_claude_flow_tool.py --dataset data/train.csv --target Sales --output ./models/
"""

import os
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

# Load environment variables from .env file for Ollama configuration
load_dotenv(".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClaudeFlowMLESTARService:
    """
    Wrapper for Claude Flow MLE-STAR automation service

    Calls: claude-flow automation mle-star

    Based on documentation:
    https://github.com/ruvnet/claude-flow/wiki/MLE-STAR-Workflow
    """

    def __init__(
        self,
        dataset: str,
        target: str,
        output_dir: str = "./models/",
        # name: str = "mle-star-workflow",
        search_iterations: int = 5,
        refinement_iterations: int = 8,
        max_agents: int = 8,
        use_claude: bool = False,  # Changed default to False to use Ollama
        interactive: bool = False
    ):
        """
        Initialize Claude Flow MLE-STAR Service wrapper

        Args:
            dataset: Path to input dataset CSV file
            target: Target column name for prediction
            output_dir: Output directory for results (default: ./models/)
            name: Workflow name (default: mle-star-workflow)
            search_iterations: Number of search iterations (default: 5)
            refinement_iterations: Number of refinement iterations (default: 8)
            max_agents: Maximum number of agents (default: 8)
            use_claude: Use Claude for automation (default: False, uses Ollama)
            interactive: Run in interactive mode (default: False)
        """
        self.dataset = dataset
        self.target = target
        self.output_dir = output_dir
        # self.name = name
        self.search_iterations = search_iterations
        self.refinement_iterations = refinement_iterations
        self.max_agents = max_agents
        self.use_claude = use_claude
        self.interactive = interactive

        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Output markdown file path
        self.markdown_output = Path(self.output_dir) / f"mle_star_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        # Set up Ollama environment variables from .env
        self._setup_ollama_env()

        logger.info(f"Initialized Claude Flow MLE-STAR Service")
        logger.info(f"  Dataset: {self.dataset}")
        logger.info(f"  Target: {self.target}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Markdown output: {self.markdown_output}")
        logger.info(f"  Using Claude API: {self.use_claude}")
        if not self.use_claude:
            logger.info(f"  Ollama Model: {os.getenv('llm_model', 'Not set')}")
            logger.info(f"  Ollama Base URL: {os.getenv('llm_base_url', 'Not set')}")

    def _setup_ollama_env(self) -> None:
        """
        Set up Ollama environment variables for claude-flow to use

        Sets claude-flow environment variables:
        - DEFAULT_LLM_PROVIDER: Set to 'ollama' to use Ollama instead of Claude
        - OLLAMA_API_URL: Ollama base URL
        """
        if not self.use_claude:
            # Read from .env file and strip quotes
            base_url = os.getenv('llm_base_url', 'http://127.0.0.1:11434/v1').strip('"').strip("'")
            model = os.getenv('llm_model', 'qwen2.5:7b-instruct').strip('"').strip("'")

            # Remove /v1 suffix if present (claude-flow adds it automatically)
            ollama_base_url = base_url.replace('/v1', '')

            # CRITICAL: Tell claude-flow to use Ollama provider instead of Anthropic
            os.environ['DEFAULT_LLM_PROVIDER'] = 'ollama'

            # Set Ollama-specific configuration for claude-flow
            os.environ['OLLAMA_API_URL'] = ollama_base_url

            # Unset ANTHROPIC_API_KEY to ensure it doesn't default to Claude
            if 'ANTHROPIC_API_KEY' in os.environ:
                del os.environ['ANTHROPIC_API_KEY']

            logger.info(f"Configured Ollama environment:")
            logger.info(f"  DEFAULT_LLM_PROVIDER: ollama")
            logger.info(f"  OLLAMA_API_URL: {ollama_base_url}")
            logger.info(f"  Model: {model}")

    def _build_command(self) -> list:
        """
        Build the claude-flow CLI command

        Command structure:
        claude-flow automation mle-star \
          --dataset <path> \
          --target <column> \
          --output <dir> \
          --name <name> \
          --search-iterations <n> \
          --refinement-iterations <n> \
          --max-agents <n> \
          --claude \
          [--interactive]

        Returns:
            list: Command arguments for subprocess
        """
        command = [
            "claude-flow",
            "automation",
            "mle-star",
            "--dataset", self.dataset,
            "--target", self.target,
            "--output", self.output_dir,
            # "--name", self.name,
            "--search-iterations", str(self.search_iterations),
            "--refinement-iterations", str(self.refinement_iterations),
            "--max-agents", str(self.max_agents),
        ]

        # Add optional flags
        # IMPORTANT: Do NOT add --claude flag when use_claude=False
        # The --claude flag forces claude-flow to use Claude CLI which costs money
        # When --claude is omitted, claude-flow will use the provider specified
        # in DEFAULT_LLM_PROVIDER environment variable (which we set to 'ollama')
        if self.use_claude:
            command.append("--claude")

        if self.interactive:
            command.append("--interactive")

        logger.info(f"Built command: {' '.join(command)}")
        return command

    def _save_to_markdown(self, content: str) -> None:
        """
        Save console output to markdown file

        Args:
            content: Output content to save
        """
        try:
            with open(self.markdown_output, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Output saved to: {self.markdown_output}")
        except Exception as e:
            logger.error(f"Failed to save markdown: {e}")

    def _create_markdown_header(self) -> str:
        """
        Create markdown header with execution metadata

        Returns:
            str: Markdown formatted header
        """
        header = f"""# MLE-STAR Execution Report

**Execution Time**: {datetime.now().isoformat()}

## Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | {self.dataset} |
| Target Column | {self.target} |
| Output Directory | {self.output_dir} |
| Workflow Name | !! no name here !! |
| Search Iterations | {self.search_iterations} |
| Refinement Iterations | {self.refinement_iterations} |
| Max Agents | {self.max_agents} |
| Claude Integration | {self.use_claude} |
| Interactive Mode | {self.interactive} |

## Command Executed

```bash
{' '.join(self._build_command())}
```

## Console Output

"""
        return header

    def run(self) -> Dict[str, Any]:
        """
        Execute the Claude Flow MLE-STAR automation

        Returns:
            dict: Execution results with keys:
                - success: bool - Whether execution succeeded
                - output: str - Console output
                - markdown_file: str - Path to markdown output file
                - return_code: int - Process return code
                - error: str - Error message if failed
        """
        logger.info("Starting Claude Flow MLE-STAR automation...")

        # Start markdown content
        markdown_content = self._create_markdown_header()

        try:
            # Build command
            command = self._build_command()

            logger.info(f"Executing: {' '.join(command)}")
            print("\n" + "="*70)
            print("🧠 MLE-STAR: Machine Learning Engineering via Search and Targeted Refinement")
            print("🎯 This is the flagship automation workflow for ML engineering tasks")
            print("="*70 + "\n")

            # Execute command with real-time output capture
            console_output = ""
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # Print and capture output in real-time
                for line in process.stdout:
                    print(line, end='')  # Print to console in real-time
                    console_output += line  # Capture for markdown

                return_code = process.wait()

            except FileNotFoundError:
                error_msg = "❌ Claude Flow CLI not found. Please install: npm install -g claude-flow"
                print(error_msg)
                console_output = error_msg
                return_code = 1

            # Add console output to markdown
            markdown_content += f"""```
{console_output}
```

## Execution Summary

"""

            if return_code == 0:
                markdown_content += """✅ **Status**: SUCCESS

The MLE-STAR workflow has completed successfully. Check the output directory for generated models and reports.

"""
                logger.info("✅ Claude Flow execution completed successfully")
            else:
                markdown_content += f"""❌ **Status**: FAILED (Return Code: {return_code})

The workflow encountered an error. Please check the console output above for details.

"""
                logger.error(f"❌ Claude Flow execution failed with return code: {return_code}")

            # Add footer
            markdown_content += f"""
## Output Directory

All generated models and reports are saved in: `{self.output_dir}`

**Markdown Report Generated**: {self.markdown_output}
"""

            # Save to markdown
            self._save_to_markdown(markdown_content)

            # Return results
            return {
                "success": return_code == 0,
                "output": console_output,
                "markdown_file": str(self.markdown_output),
                "return_code": return_code,
                "error": None if return_code == 0 else "Command failed"
            }

        except Exception as e:
            error_msg = f"Error executing Claude Flow: {str(e)}"
            logger.error(error_msg)

            # Add error to markdown
            markdown_content += f"""```
ERROR: {str(e)}
```

"""
            markdown_content += f"""## Error Details

The execution encountered an error:
```
{str(e)}
```

"""
            self._save_to_markdown(markdown_content)

            return {
                "success": False,
                "output": "",
                "markdown_file": str(self.markdown_output),
                "return_code": -1,
                "error": error_msg
            }


def call_claude_flow_service(
    dataset: str,
    target: str,
    output_dir: str = "./models/",
    # name: str = "mle-star-workflow",
    search_iterations: int = 5,
    refinement_iterations: int = 8,
    max_agents: int = 8,
    use_claude: bool = False,  # Default to False to use Ollama
    interactive: bool = False
) -> Dict[str, Any]:
    """
    Main function to call Claude Flow MLE-STAR automation service

    This is the single tool function that integrates with other Python code.

    Args:
        dataset: Path to input dataset CSV file
        target: Target column name for prediction
        output_dir: Output directory for results (default: ./models/)
        name: Workflow name (default: mle-star-workflow)
        search_iterations: Number of search iterations (default: 5)
        refinement_iterations: Number of refinement iterations (default: 8)
        max_agents: Maximum number of agents (default: 8)
        use_claude: Use Claude for automation (default: False, uses Ollama from .env)
        interactive: Run in interactive mode (default: False)

    Returns:
        dict: Execution results containing:
            - success (bool): Whether execution succeeded
            - output (str): Console output from Claude Flow
            - markdown_file (str): Path to saved markdown report
            - return_code (int): Process return code
            - error (str): Error message if failed

    Example:
        >>> results = call_claude_flow_service(
        ...     dataset="data/sales.csv",
        ...     target="revenue",
        ...     output_dir="./models/",
        ...     name="revenue-prediction",
        ...     search_iterations=5,
        ...     refinement_iterations=8,
        ...     max_agents=8,
        ...     use_claude=True
        ... )
        >>> print(f"Success: {results['success']}")
        >>> print(f"Report: {results['markdown_file']}")

    Documentation:
        https://github.com/ruvnet/claude-flow/wiki/MLE-STAR-Workflow
    """
    service = ClaudeFlowMLESTARService(
        dataset=dataset,
        target=target,
        output_dir=output_dir,
        # name=name,
        search_iterations=search_iterations,
        refinement_iterations=refinement_iterations,
        max_agents=max_agents,
        use_claude=use_claude,
        interactive=interactive
    )

    return service.run()


# ==================== CLI Interface ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Call Claude Flow MLE-STAR automation service"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to input dataset CSV file"
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target column name for prediction"
    )
    parser.add_argument(
        "--output",
        default="./models/",
        help="Output directory for results (default: ./models/)"
    )
    parser.add_argument(
        "--name",
        default="mle-star-workflow",
        help="Workflow name (default: mle-star-workflow)"
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=5,
        help="Number of search iterations (default: 5)"
    )
    parser.add_argument(
        "--refinement-iterations",
        type=int,
        default=8,
        help="Number of refinement iterations (default: 8)"
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=8,
        help="Maximum number of agents (default: 8)"
    )
    parser.add_argument(
        "--claude",
        action="store_true",
        help="Enable Claude integration (uses Ollama by default)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    # Call the service
    results = call_claude_flow_service(
        dataset=args.dataset,
        target=args.target,
        output_dir=args.output,
        # name=args.name,
        search_iterations=args.search_iterations,
        refinement_iterations=args.refinement_iterations,
        max_agents=args.max_agents,
        use_claude=args.claude,  # Now defaults to False (Ollama), unless --claude is passed
        interactive=args.interactive
    )

    # Print summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    print(f"✓ Success: {results['success']}")
    print(f"✓ Return Code: {results['return_code']}")
    print(f"✓ Report: {results['markdown_file']}")
    if results['error']:
        print(f"✗ Error: {results['error']}")
    print("="*70 + "\n")
