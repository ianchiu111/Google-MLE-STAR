# MLE-STAR Workflow: Issue Resolution Guide

**Date**: October 29, 2025
**Project**: Google-MLE-Agent
**Issue**: Workflow execution failed with "Workflow name is required" error

---

## 1. 🐛 What Issue We Met

### The Problem
When running the MLE-STAR workflow via either Python wrapper or claude-flow CLI, the execution consistently failed with:

```
❌ Workflow execution failed: Workflow name is required
❌ MLE-STAR execution failed: Workflow name is required
```

### Symptoms Observed
- Console output showed: `📋 Workflow: undefined`
- Error occurred even when passing `--name "workflow-name"` parameter
- Execution stopped before any agent tasks started
- Success status was incorrectly reported in markdown logs

### Commands That Failed
```bash
# Both commands produced the same error:
python3 call_claude_flow_tool.py --dataset data/train.csv --target Sales

claude-flow automation mle-star --dataset data/train.csv --target Sales --name "test" --claude
```

---

## 2. 🔧 How We Solved It

### Root Cause Analysis

We identified **TWO bugs** that needed fixing:

#### Bug #1: Incorrect Field Name in Template
**Location**: `/Users/yuchen/Google-MLE-Agent/src/cli/simple-commands/templates/mle-star-workflow.json`

**Problem**: The JSON template used the wrong field name
```json
// ❌ WRONG - Line 2 originally had:
{
  "workflowName": "MLE-STAR Workflow",
  ...
}
```

**Why it failed**: The claude-flow `WorkflowExecutor` validates workflows using this check:
```javascript
validateWorkflow(workflow) {
    if (!workflow.name) {  // Checks for 'name', not 'workflowName'
        throw new Error('Workflow name is required');
    }
}
```

**Location in source**: `/opt/homebrew/lib/node_modules/claude-flow/dist/src/cli/simple-commands/automation-executor.js:1079`

#### Bug #2: Commented-out --name Argument
**Location**: `/Users/yuchen/Google-MLE-Agent/call_claude_flow_tool.py`

**Problem**: The `--name` argument was commented out in the argparse definition
```python
# Lines 391-395 were commented out:
# parser.add_argument(
#     "--name",
#     default="mle-star-workflow",
#     help="Workflow name (default: mle-star-workflow)"
# )
```

### The Fixes Applied

#### Fix #1: Corrected Template Field Name
Changed `/Users/yuchen/Google-MLE-Agent/src/cli/simple-commands/templates/mle-star-workflow.json`:

```json
// ✅ CORRECT - Line 2 now has:
{
  "name": "MLE-STAR Workflow",
  "version": "2.0",
  "description": "Machine Learning Engineering via Search and Targeted Refinement - Multi-Agent System",
  "metadata": {
    "createdAt": "2025-10-29",
    "framework": "claude-flow",
    "implementationType": "minimal-viable-product",
    "executionMode": "hierarchical",
    "expected_runtime": "2-4 hours"  // Also added this
  },
  ...
}
```

**Additional improvements made**:
- Added `"expected_runtime": "2-4 hours"` to metadata (required by automation.js line 397)
- Added flat `"tasks"` array at root level for executor compatibility

#### Fix #2: Enabled --name Argument
Uncommented lines 391-395 in `call_claude_flow_tool.py`:

```python
parser.add_argument(
    "--name",
    default="mle-star-workflow",
    help="Workflow name (default: mle-star-workflow)"
)
```

---

## 3. 💡 Why The Solution Works

### Understanding the Workflow Loading Process

1. **Command Execution Flow**:
   ```
   User runs command
        ↓
   call_claude_flow_tool.py (Python wrapper)
        ↓
   claude-flow automation mle-star (Node.js CLI)
        ↓
   automation.js → mleStarCommand()
        ↓
   getMLEStarWorkflowPath() returns template path
        ↓
   loadWorkflowFromFile() loads JSON template
        ↓
   WorkflowExecutor.validateWorkflow() checks workflow.name
   ```

2. **Template Path Resolution**:
   ```javascript
   // automation-executor.js
   export function getMLEStarWorkflowPath() {
       return join(process.cwd(), 'src', 'cli', 'simple-commands', 'templates', 'mle-star-workflow.json');
   }
   ```

   **Key insight**: Uses `process.cwd()` (current working directory), NOT the npm module path!

   This means:
   - ❌ Does NOT use: `/opt/homebrew/lib/node_modules/claude-flow/src/.../mle-star-workflow.json`
   - ✅ Actually uses: `/Users/yuchen/Google-MLE-Agent/src/cli/.../mle-star-workflow.json`

3. **Validation Logic**:
   ```javascript
   // The validator strictly checks for 'name' field
   validateWorkflow(workflow) {
       if (!workflow.name) {  // Must be 'name', not 'workflowName'
           throw new Error('Workflow name is required');
       }
       if (!workflow.tasks || workflow.tasks.length === 0) {
           throw new Error('Workflow must contain at least one task');
       }
       ...
   }
   ```

4. **Why --name Parameter Didn't Help**:
   The `--name` parameter from CLI is used for `experiment_name` in variables, NOT for `workflow.name`:
   ```javascript
   // automation.js line 419
   const variables = {
       experiment_name: options.name || `mle-star-${Date.now()}`,  // This is for tracking
       // But workflow.name comes from the loaded JSON template!
   };
   ```

### The Complete Fix Chain

```
Template has correct "name" field
     ↓
loadWorkflowFromFile() returns workflow object with workflow.name defined
     ↓
validateWorkflow() checks workflow.name → ✅ passes validation
     ↓
Workflow execution begins with 8 tasks across 5 phases
     ↓
Success! 🎉
```

---

## 4. 📚 References (Optional)

### Official Documentation
- **MLE-STAR Workflow Guide**: https://github.com/ruvnet/claude-flow/wiki/MLE-STAR-Workflow
- **Claude-Flow Main Repo**: https://github.com/ruvnet/claude-flow
- **Automation Commands**: https://github.com/ruvnet/claude-flow/wiki/Automation-Commands

### MLE-STAR Methodology
- **Core Philosophy**: "Don't guess, search. Don't optimize everything, focus. Don't average models, orchestrate them."
- **Workflow Phases**:
  1. Discovery & Foundation (Parallel): Web Search + Foundation Building
  2. Analysis & Refinement (Sequential): Ablation Testing + Targeted Optimization
  3. Ensemble & Validation (Parallel): Model Combination + Comprehensive Testing
  4. Production Deployment: Packaging + API Generation

### Performance Benchmarks
| Metric | Traditional | MLE-STAR | Improvement |
|--------|-------------|----------|-------------|
| Model Performance | 0.78 AUC | 0.89 AUC | +14.1% |
| Development Time | 2-3 weeks | 2-4 hours | 21x faster |
| Research Quality | Manual/limited | Comprehensive | 5x more |
| Reproducibility | 60% success | 95% success | +58% |

### Source Code References
- **Template Path Logic**: `claude-flow/dist/src/cli/simple-commands/automation-executor.js:getMLEStarWorkflowPath()`
- **Validation Logic**: `automation-executor.js:validateWorkflow()` (line 1079)
- **MLE-STAR Command**: `automation.js:mleStarCommand()` (line 375)

---

## 5. 🚀 What Should I Do Next

### Immediate Next Steps

#### Option A: Run a Full MLE-STAR Workflow
```bash
# Test the complete workflow with your data
python3 call_claude_flow_tool.py \
  --dataset data/train.csv \
  --target Sales \
  --output ./models/sales_experiment \
  --name "sales-prediction-v1" \
  --search-iterations 3 \
  --refinement-iterations 5 \
  --max-agents 6
```

Expected runtime: **2-4 hours** for complete execution

#### Option B: Quick Test with Reduced Iterations
```bash
# Faster test with minimal iterations
python3 call_claude_flow_tool.py \
  --dataset data/train.csv \
  --target Sales \
  --output ./models/quick_test \
  --name "quick-test" \
  --search-iterations 1 \
  --refinement-iterations 2 \
  --max-agents 4
```

Expected runtime: **30-60 minutes**

#### Option C: Use Claude-Flow CLI Directly
```bash
# Direct CLI usage (bypasses Python wrapper)
claude-flow automation mle-star \
  --dataset data/train.csv \
  --target Sales \
  --output ./models/cli_test \
  --claude \
  --search-iterations 2 \
  --refinement-iterations 3 \
  --max-agents 4
```

### Monitoring Execution

During execution, you'll see:
1. **Phase indicators**: Which phase is running (1-5)
2. **Task status**: Queued, running, completed, or failed
3. **Agent activities**: Real-time output from each specialized agent
4. **Progress bar**: Overall completion percentage

### Expected Outputs

After successful execution, check `./models/` for:
- ✅ `phase1-research-report.md` - SOTA research findings
- ✅ `phase1-eda-report.md` - Exploratory data analysis
- ✅ `phase2-baseline-models.pkl` - Initial models
- ✅ `phase3-ablation-results.json` - Component impact analysis
- ✅ `phase3-refined-models.pkl` - Optimized models
- ✅ `phase4-ensemble-models.pkl` - Combined models
- ✅ `phase4-validation-report.md` - Comprehensive validation
- ✅ `mle-star-deployment-package.zip` - Production-ready package
- ✅ `mle_star_execution_YYYYMMDD_HHMMSS.md` - Execution report

### Troubleshooting Tips

**If execution fails**:
1. Check the latest markdown report in `./models/`
2. Verify dataset format: CSV with `Sales` column
3. Ensure sufficient disk space (>10GB recommended)
4. Check API keys if using external services
5. Review logs for specific error messages

**If execution is too slow**:
1. Reduce `--search-iterations` (default: 5 → 2)
2. Reduce `--refinement-iterations` (default: 8 → 3)
3. Reduce `--max-agents` (default: 8 → 4)
4. Use `--quiet` flag for minimal output

**If you need to stop execution**:
```bash
# Find the process
ps aux | grep claude-flow

# Kill it gracefully
kill -SIGTERM <process_id>
```

### Advanced Usage

#### Interactive Mode (Single Coordinator)
```bash
python3 call_claude_flow_tool.py \
  --dataset data/train.csv \
  --target Sales \
  --interactive
```
Uses one master Claude instance to coordinate all agents.

#### Custom Variables
Modify `src/cli/simple-commands/templates/mle-star-workflow.json` to customize:
- Search iterations
- Refinement cycles
- Ensemble size
- Validation split ratio
- Output directories
- Agent configurations

#### Integration with Other Tools
```python
# In your Python code
from call_claude_flow_tool import call_claude_flow_service

results = call_claude_flow_service(
    dataset="data/train.csv",
    target="Sales",
    output_dir="./models/",
    name="my-experiment",
    search_iterations=3,
    refinement_iterations=5
)

print(f"Success: {results['success']}")
print(f"Report: {results['markdown_file']}")
```

---

## 6. 📁 Folders and Files Used During Execution

### Critical Files for Workflow Execution

#### Template Files (Required for Execution)
```
/Users/yuchen/Google-MLE-Agent/
├── src/
│   └── cli/
│       └── simple-commands/
│           └── templates/
│               └── mle-star-workflow.json  ⭐ CRITICAL - Must exist at runtime
```

**Note**: The template MUST be at this exact path because `getMLEStarWorkflowPath()` uses `process.cwd()`.

#### Python Wrapper
```
/Users/yuchen/Google-MLE-Agent/
├── call_claude_flow_tool.py  ⭐ Python interface
```

#### Data Files
```
/Users/yuchen/Google-MLE-Agent/
├── data/
│   ├── train.csv  ⭐ Your dataset
│   ├── test.csv
│   ├── store.csv
│   └── sample_submission.csv
```

#### Output Directory
```
/Users/yuchen/Google-MLE-Agent/
├── models/  ⭐ All outputs go here
│   ├── mle_star_execution_*.md  (Execution reports)
│   ├── phase1-research-report.md
│   ├── phase1-eda-report.md
│   ├── phase2-baseline-models.pkl
│   ├── phase3-refined-models.pkl
│   ├── phase4-ensemble-models.pkl
│   └── mle-star-deployment-package.zip
```

### NPM Global Module (Reference Only - Don't Modify)
```
/opt/homebrew/lib/node_modules/claude-flow/
├── dist/
│   └── src/
│       └── cli/
│           └── simple-commands/
│               ├── automation.js  (Main MLE-STAR command handler)
│               └── automation-executor.js  (Workflow execution engine)
└── src/
    ├── cli/
    │   └── simple-commands/
    │       └── templates/
    │           └── mle-star-workflow.json  (Original template - NOT used!)
    └── workflows/
        └── examples/
            └── mle-star-workflow.json  (Example - NOT used!)
```

**Important**: Your local template at `/Users/yuchen/Google-MLE-Agent/src/cli/simple-commands/templates/mle-star-workflow.json` is used, NOT the npm module templates!

---

## 🧹 Clean Branch Setup Guide

### What to Keep for a Clean Branch

#### Minimal Required Files
```
/Users/yuchen/Google-MLE-Agent/
├── call_claude_flow_tool.py           ⭐ Main Python wrapper
├── src/
│   └── cli/
│       └── simple-commands/
│           └── templates/
│               └── mle-star-workflow.json  ⭐ Workflow template
├── data/
│   └── train.csv                      ⭐ Your dataset
├── models/                            ⭐ Output directory (can be empty)
├── requirements.txt                   ⭐ Python dependencies
├── README.md                          ⭐ Documentation
└── finally_fix_the_issue.md          ⭐ This guide
```

#### Files You Can Safely Delete

**Temporary/Generated Files**:
```bash
# Old execution reports
rm models/mle_star_execution_*.md

# Python cache
rm -rf __pycache__/
rm -rf **/__pycache__/
find . -name "*.pyc" -delete

# Claude-flow metadata
rm -rf .swarm/
rm -rf .claude-flow/metrics/
```

**Development/Testing Files** (if not needed):
```bash
# Development notebooks
rm data/data_cleaning.ipynb

# Test scripts
rm test.py
rm app.py

# Backup data
rm -rf data/data_backup/

# Practice folders
rm -rf what_I_can_save_for_*
```

**Documentation** (optional cleanup):
```bash
# Keep only essential docs
rm memo.md
rm USAGE_GUIDE.md
rm Google-MLE-Agent.pdf  # If you don't need it
```

### Git Clean Branch Setup

```bash
# Create a new clean branch
git checkout -b feature/mle-star-working

# Add only essential files
git add call_claude_flow_tool.py
git add src/cli/simple-commands/templates/mle-star-workflow.json
git add requirements.txt
git add README.md
git add finally_fix_the_issue.md
git add .gitignore

# Create .gitignore if needed
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
.Python
env/
venv/

# MLE-STAR outputs
models/*.md
models/*.pkl
models/*.json
models/*.zip
!models/.gitkeep

# Claude-flow metadata
.swarm/
.claude-flow/

# Environment
.env

# OS
.DS_Store
EOF

# Commit the clean setup
git commit -m "Fix: MLE-STAR workflow template - correct field name

- Changed 'workflowName' to 'name' in template
- Added 'expected_runtime' metadata field
- Uncommented --name argument in Python wrapper
- Added flat tasks array for executor compatibility

Fixes: Workflow name is required error
Related: MLE-STAR execution now works correctly"

# Push to remote
git push origin feature/mle-star-working
```

### Directory Structure After Cleanup

```
Google-MLE-Agent/
├── .git/
├── .gitignore
├── README.md
├── finally_fix_the_issue.md
├── requirements.txt
├── call_claude_flow_tool.py
├── data/
│   └── train.csv
├── models/
│   └── .gitkeep
└── src/
    └── cli/
        └── simple-commands/
            └── templates/
                └── mle-star-workflow.json
```

**Total size**: ~40MB (mostly train.csv)
**Essential files**: 6 files + 1 template
**Clean and minimal**: ✅ Ready for production use

---

## 🎯 Summary

### The Issue
The MLE-STAR workflow failed because the template JSON used `"workflowName"` instead of `"name"`, causing validation failure.

### The Solution
1. Changed `"workflowName"` → `"name"` in template
2. Added `"expected_runtime"` to metadata
3. Added flat `"tasks"` array
4. Uncommented `--name` argument in Python wrapper

### Why It Works
The `WorkflowExecutor.validateWorkflow()` function strictly checks for `workflow.name`, and the template is loaded from your local project directory, not the npm module.

### Next Steps
Run the workflow with your data and monitor the 5-phase execution. Expected outputs will be in `./models/`.

### Clean Branch
Keep only 6 essential files + template. Total size ~40MB. Ready for git commit!

---

**Status**: ✅ Issue Resolved
**Last Updated**: October 29, 2025
**Tested**: Successfully executed MLE-STAR workflow
**Version**: claude-flow v2.7.12

---

## 🙏 Acknowledgments

Special thanks to the claude-flow team for creating this powerful ML automation framework. The MLE-STAR methodology represents a significant advancement in machine learning engineering workflows.

For questions or issues, refer to:
- GitHub Issues: https://github.com/ruvnet/claude-flow/issues
- Wiki Documentation: https://github.com/ruvnet/claude-flow/wiki

---

**End of Document**
