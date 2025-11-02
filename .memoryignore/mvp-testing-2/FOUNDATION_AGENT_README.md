# Foundation Agent - Baseline Model Builder

**MLE-STAR Workflow - Foundation Phase**

## Overview

The Foundation Agent is responsible for the initial model building phase in the MLE-STAR workflow. It implements data preprocessing, feature engineering, and trains multiple baseline models based on research recommendations.

## Agent Information

- **Agent ID:** `foundation_agent`
- **Role:** CODER
- **Capabilities:** `data_preprocessing`, `initial_modeling`, `baseline_creation`
- **Frameworks:** `scikit-learn`, `pandas`, `numpy`, `lightgbm`, `xgboost`, `matplotlib`

## Files Created

### Core Implementation
- `foundation_agent_preprocessing.py` - Data preprocessing and feature engineering pipeline
- `foundation_agent_baseline_models.py` - Baseline model training and evaluation
- `foundation_agent_main.py` - Main execution orchestrator

### Configuration
- `requirements_foundation.txt` - Python dependencies
- `FOUNDATION_AGENT_README.md` - This file

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_foundation.txt
```

### 2. Run Full Pipeline

```bash
python foundation_agent_main.py
```

This will:
1. Preprocess data (clean, engineer features)
2. Train baseline models (LightGBM, XGBoost, RandomForest)
3. Generate submissions and reports
4. Store results for next agents

### 3. Run Individual Modules

**Preprocessing Only:**
```bash
python foundation_agent_preprocessing.py
```

**Model Training Only (requires preprocessed data):**
```bash
python foundation_agent_baseline_models.py
```

## Outputs

All outputs are saved to `models/` directory:

### Preprocessed Data
- `foundation_agent_X_train.csv` - Training features
- `foundation_agent_y_train.csv` - Training target (Sales)
- `foundation_agent_X_test.csv` - Test features
- `foundation_agent_test_ids.csv` - Test IDs for submission

### Trained Models
- `foundation_agent_lightgbm_model.txt` - LightGBM booster
- `foundation_agent_xgboost_model.json` - XGBoost booster
- `foundation_agent_random_forest_model.pkl` - RandomForest sklearn model

### Submissions
- `foundation_agent_submission_lightgbm.csv`
- `foundation_agent_submission_xgboost.csv`
- `foundation_agent_submission_random_forest.csv`
- `foundation_agent_submission_ensemble.csv`

### Reports
- `foundation_agent_report.md` - Comprehensive execution report
- `foundation_agent_results_summary.json` - Model performance metrics
- `foundation_agent_metadata.json` - Execution metadata

## Features Engineered

The preprocessing pipeline creates the following feature categories:

### Temporal Features
- Year, Month, Day, Quarter, WeekOfYear
- DayOfMonth, IsMonthStart, IsMonthEnd, DaysInMonth

### Competition Features
- CompetitionOpenMonths (duration since competition opened)
- CompetitionIntensity (duration / distance)

### Promotion Features
- Promo2OpenMonths (duration since Promo2 started)
- IsPromoMonth (whether current month is in PromoInterval)

### Store Features
- StoreAssortment (combined categorical: StoreType_Assortment)
- Store-level aggregations (mean, median, std of Sales and Customers)

### Holiday Features
- IsHoliday (either StateHoliday or SchoolHoliday)

## Baseline Models

### 1. LightGBM
**Purpose:** Fast baseline with localized approach

**Hyperparameters:**
- `num_leaves`: 31
- `learning_rate`: 0.05
- `max_depth`: -1 (no limit)
- `feature_fraction`: 0.8
- `bagging_fraction`: 0.8

**Validation:** 5-Fold Cross-Validation

### 2. XGBoost
**Purpose:** Robust baseline with global approach

**Hyperparameters:**
- `max_depth`: 6
- `learning_rate`: 0.05
- `subsample`: 0.8
- `colsample_bytree`: 0.8

**Validation:** 5-Fold Cross-Validation

### 3. Random Forest
**Purpose:** Ensemble component and feature importance

**Hyperparameters:**
- `n_estimators`: 100
- `max_depth`: 15
- `max_features`: 'sqrt'
- `min_samples_split`: 10

**Validation:** 20% Hold-out

## Evaluation Metric

**RMSPE (Root Mean Square Percentage Error)**

Formula: √(mean((y_true - y_pred) / y_true)²)

This is the official Kaggle competition metric and is scale-invariant.

## MLE-STAR Methodology Alignment

✓ **Feature Engineering Focus:** 50% time allocation as recommended
✓ **Tree-based Models:** LightGBM, XGBoost, RandomForest as suggested by research
✓ **Modular Design:** Separate preprocessing and modeling for easy refinement
✓ **Baseline Establishment:** Multiple robust baselines for comparison
✓ **Proper Validation:** Cross-validation with official metric

## Coordination with Other Agents

### Memory Storage

The foundation agent stores results in the claude-flow memory system:

```bash
npx claude-flow@alpha memory store 'agent/foundation_agent/status' 'completed'
npx claude-flow@alpha memory store 'agent/foundation_agent/best_model' 'lightgbm'
npx claude-flow@alpha memory store 'agent/foundation_agent/best_rmspe' '0.XXXXX'
```

### Hook Integration

Hooks are triggered at key points:

1. **Pre-task:** Before starting foundation phase
2. **Post-edit:** After each file creation/modification
3. **Post-task:** After completing all foundation work

### Next Agent Handoff

The foundation agent provides:
- Preprocessed datasets ready for refinement
- Baseline performance benchmarks
- Trained models that can be improved
- Feature names and engineering logic
- Recommendations for optimization

**Next Agent:** Refinement Agent (`refinement_agent`)

## Usage in MLE-STAR Workflow

```bash
# Via claude-flow automation
npx claude-flow@alpha automation mle-star \
  --dataset data/train.csv \
  --target Sales \
  --output models/ \
  --claude
```

The foundation agent will be automatically invoked as part of the workflow.

## Troubleshooting

### Common Issues

**1. Missing Dependencies**
```bash
pip install -r requirements_foundation.txt
```

**2. Data Not Found**
Ensure data files exist:
- `data/train.csv`
- `data/test.csv`
- `data/store.csv`

**3. Insufficient Memory**
Reduce dataset size or use sampling for testing.

**4. Slow Training**
Reduce `n_folds` in cross-validation or `n_estimators` in RandomForest.

## Performance Expectations

**Dataset Size:**
- Training: ~850K samples (after filtering closed stores)
- Test: ~41K samples
- Features: ~25 engineered features

**Execution Time (approximate):**
- Preprocessing: 30-60 seconds
- LightGBM (5-fold CV): 2-5 minutes
- XGBoost (5-fold CV): 5-10 minutes
- RandomForest: 2-3 minutes
- Total: 10-20 minutes (depending on hardware)

**Expected RMSPE:**
- LightGBM: 0.10-0.15
- XGBoost: 0.10-0.15
- RandomForest: 0.12-0.18

## Code Quality

All code follows best practices:
- ✓ Modular and reusable components
- ✓ Comprehensive documentation
- ✓ Error handling and logging
- ✓ Type hints where appropriate
- ✓ Reproducible (fixed random seeds)

## Contact & Support

For issues related to the Foundation Agent:
1. Check execution logs in console output
2. Review `foundation_agent_report.md` for detailed diagnostics
3. Check `foundation_agent_metadata.json` for execution details

---

*Part of the MLE-STAR Workflow - Foundation Phase*
*Agent: foundation_agent*
*Version: 1.0*
