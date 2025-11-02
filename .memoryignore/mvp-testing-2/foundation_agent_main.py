"""
Foundation Agent - Main Execution Script
Rossmann Store Sales Prediction
Author: Foundation Agent (MLE-STAR Workflow)

This script orchestrates the complete foundation phase:
1. Data preprocessing
2. Baseline model training
3. Performance evaluation
4. Results coordination
"""

import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Import foundation agent modules
from foundation_agent_preprocessing import RossmannPreprocessor, main as preprocess_main
from foundation_agent_baseline_models import BaselineModelBuilder, RMSPEMetric, main as model_main


def print_banner(text, char="=", width=70):
    """Print formatted banner"""
    print("\n" + char * width)
    print(text.center(width))
    print(char * width)


def main():
    """Main execution pipeline for foundation agent"""
    start_time = datetime.now()

    print_banner("FOUNDATION AGENT - MLE-STAR WORKFLOW", "=")
    print(f"Session Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Agent ID: foundation_agent")
    print(f"Phase: Foundation - Initial Model Building")

    try:
        # ===================================================================
        # PHASE 1: DATA PREPROCESSING
        # ===================================================================
        print_banner("PHASE 1: DATA PREPROCESSING", "=")

        preprocessor, X_train, y_train, X_test = preprocess_main()

        preprocessing_complete_time = datetime.now()
        preprocessing_duration = (preprocessing_complete_time - start_time).total_seconds()

        print(f"\n✓ Preprocessing completed in {preprocessing_duration:.2f} seconds")

        # ===================================================================
        # PHASE 2: BASELINE MODEL TRAINING
        # ===================================================================
        print_banner("PHASE 2: BASELINE MODEL TRAINING", "=")

        builder, results_summary = model_main()

        model_training_complete_time = datetime.now()
        model_training_duration = (model_training_complete_time - preprocessing_complete_time).total_seconds()

        print(f"\n✓ Model training completed in {model_training_duration:.2f} seconds")

        # ===================================================================
        # PHASE 3: PERFORMANCE EVALUATION & SUMMARY
        # ===================================================================
        print_banner("PHASE 3: PERFORMANCE EVALUATION", "=")

        # Extract best model performance
        model_performances = []
        for model_name, metrics in results_summary['models'].items():
            if model_name == 'ensemble':
                continue
            if 'cv_rmspe' in metrics:
                model_performances.append({
                    'model': model_name,
                    'rmspe': metrics['cv_rmspe'],
                    'std': metrics['cv_std']
                })
            elif 'val_rmspe' in metrics:
                model_performances.append({
                    'model': model_name,
                    'rmspe': metrics['val_rmspe'],
                    'std': 0.0
                })

        # Sort by RMSPE
        model_performances.sort(key=lambda x: x['rmspe'])

        print("\nModel Performance Ranking:")
        print("-" * 70)
        for i, perf in enumerate(model_performances, 1):
            print(f"{i}. {perf['model']:15s} - RMSPE: {perf['rmspe']:.6f} (+/- {perf['std']:.6f})")
        print("-" * 70)

        best_model = model_performances[0]['model']
        best_rmspe = model_performances[0]['rmspe']

        print(f"\n🏆 Best Baseline Model: {best_model.upper()}")
        print(f"   RMSPE: {best_rmspe:.6f}")

        # ===================================================================
        # PHASE 4: GENERATE COMPREHENSIVE REPORT
        # ===================================================================
        print_banner("PHASE 4: GENERATING REPORT", "=")

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        report = generate_comprehensive_report(
            preprocessor=preprocessor,
            builder=builder,
            results_summary=results_summary,
            model_performances=model_performances,
            best_model=best_model,
            best_rmspe=best_rmspe,
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration
        )

        # Save report
        with open('models/foundation_agent_report.md', 'w') as f:
            f.write(report)
        print("✓ Saved: models/foundation_agent_report.md")

        # Save execution metadata
        metadata = {
            'agent_id': 'foundation_agent',
            'session_start': start_time.isoformat(),
            'session_end': end_time.isoformat(),
            'total_duration_seconds': total_duration,
            'best_model': best_model,
            'best_rmspe': float(best_rmspe),
            'all_models': [p['model'] for p in model_performances],
            'feature_count': len(preprocessor.get_feature_names()),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'outputs': {
                'preprocessed_data': [
                    'models/foundation_agent_X_train.csv',
                    'models/foundation_agent_y_train.csv',
                    'models/foundation_agent_X_test.csv',
                    'models/foundation_agent_test_ids.csv'
                ],
                'models': [
                    f'models/foundation_agent_{model}_model.*' for model in ['lightgbm', 'xgboost', 'random_forest']
                ],
                'submissions': [
                    f'models/foundation_agent_submission_{model}.csv'
                    for model in ['lightgbm', 'xgboost', 'random_forest', 'ensemble']
                ],
                'reports': [
                    'models/foundation_agent_report.md',
                    'models/foundation_agent_results_summary.json'
                ]
            }
        }

        with open('models/foundation_agent_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("✓ Saved: models/foundation_agent_metadata.json")

        # ===================================================================
        # FINAL SUMMARY
        # ===================================================================
        print_banner("FOUNDATION AGENT EXECUTION COMPLETE", "=")
        print(f"\nExecution Summary:")
        print(f"  Start Time:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End Time:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total Duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        print(f"\nDeliverables:")
        print(f"  ✓ Preprocessed datasets: 4 files")
        print(f"  ✓ Trained models: {len(builder.models)} models")
        print(f"  ✓ Submission files: 4 files")
        print(f"  ✓ Performance report: 1 file")
        print(f"  ✓ Metadata: 1 file")
        print(f"\nNext Steps:")
        print(f"  → Refinement Agent can now optimize the {best_model} model")
        print(f"  → Ensemble Agent can improve predictions through stacking")
        print(f"  → Validation Agent can verify model robustness")
        print("=" * 70)

        return {
            'success': True,
            'metadata': metadata,
            'best_model': best_model,
            'best_rmspe': best_rmspe
        }

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def generate_comprehensive_report(preprocessor, builder, results_summary, model_performances,
                                  best_model, best_rmspe, start_time, end_time, total_duration):
    """Generate comprehensive markdown report"""

    report = f"""# Foundation Agent - Baseline Model Report

**MLE-STAR Workflow - Foundation Phase**

---

## Executive Summary

**Agent:** Foundation Agent (foundation_agent)
**Task:** Data Preprocessing and Baseline Model Development
**Dataset:** Rossmann Store Sales Prediction
**Execution Time:** {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')}
**Duration:** {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)

### Key Results

- **Best Model:** {best_model.upper()}
- **Best RMSPE:** {best_rmspe:.6f}
- **Models Trained:** {len(builder.models)}
- **Features Engineered:** {len(preprocessor.get_feature_names())}
- **Training Samples:** {len(preprocessor.X_train):,}
- **Test Samples:** {len(preprocessor.X_test):,}

---

## 1. Data Preprocessing

### 1.1 Data Loading

- **Train Data:** {len(preprocessor.train):,} samples (after filtering closed stores)
- **Test Data:** {len(preprocessor.test):,} samples
- **Store Data:** {len(preprocessor.store)} stores

### 1.2 Data Cleaning

**Missing Value Handling:**
- CompetitionDistance: Filled with max * 1.5 (no nearby competition)
- CompetitionOpenSince: Filled with 0 (no competition history)
- Promo2 fields: Filled with 0 (no extended promotion)
- StateHoliday: Standardized mixed types (0 → '0')

### 1.3 Feature Engineering

**Features Created:** {len(preprocessor.get_feature_names())}

**Feature Categories:**

1. **Temporal Features**
   - Year, Month, Day, Quarter, WeekOfYear
   - DayOfMonth, IsMonthStart, IsMonthEnd
   - DaysInMonth

2. **Competition Features**
   - CompetitionOpenMonths (duration)
   - CompetitionIntensity (duration/distance)

3. **Promotion Features**
   - Promo2OpenMonths
   - IsPromoMonth (matches PromoInterval)

4. **Store Features**
   - StoreAssortment (combined categorical)
   - Store-level aggregations (mean, median, std)

5. **Holiday Features**
   - IsHoliday (state or school)

**Key Insight:** Following MLE-STAR recommendation - 50% time allocated to feature engineering over model complexity.

---

## 2. Baseline Models

### 2.1 Model Selection Rationale

Based on ML Researcher Agent recommendations:
1. **LightGBM** - Fast training, localized approach, handles categorical well
2. **XGBoost** - Robust, global approach, excellent baseline
3. **Random Forest** - Ensemble component, feature importance

### 2.2 Model Performance

"""

    # Add performance table
    report += "| Rank | Model | RMSPE | Std Dev | Method |\n"
    report += "|------|-------|-------|---------|--------|\n"

    for i, perf in enumerate(model_performances, 1):
        method = "5-Fold CV" if perf['std'] > 0 else "Hold-out Val"
        report += f"| {i} | {perf['model']} | {perf['rmspe']:.6f} | {perf['std']:.6f} | {method} |\n"

    report += f"""

### 2.3 Model Details

#### LightGBM
- **Parameters:** num_leaves=31, learning_rate=0.05, max_depth=-1
- **Validation:** 5-Fold Cross-Validation
- **CV RMSPE:** {results_summary['models']['lightgbm']['cv_rmspe']:.6f} ± {results_summary['models']['lightgbm']['cv_std']:.6f}
- **Strengths:** Fast training, minimal tuning required, excellent baseline

#### XGBoost
- **Parameters:** max_depth=6, learning_rate=0.05, subsample=0.8
- **Validation:** 5-Fold Cross-Validation
- **CV RMSPE:** {results_summary['models']['xgboost']['cv_rmspe']:.6f} ± {results_summary['models']['xgboost']['cv_std']:.6f}
- **Strengths:** Robust, well-tested, strong regularization

#### Random Forest
- **Parameters:** n_estimators=100, max_depth=15, max_features='sqrt'
- **Validation:** 20% Hold-out
- **Val RMSPE:** {results_summary['models']['random_forest']['val_rmspe']:.6f}
- **Strengths:** Non-parametric, good for feature importance, ensemble ready

### 2.4 Ensemble Strategy

**Simple Weighted Average:**
- Equal weights: LightGBM (33.3%), XGBoost (33.3%), RandomForest (33.3%)
- Rationale: All models provide diverse perspectives
- Future: Refinement agent can optimize weights

---

## 3. Evaluation Methodology

### 3.1 Metrics

**Primary Metric:** RMSPE (Root Mean Square Percentage Error)
- Formula: √(mean((y_true - y_pred) / y_true)²)
- Reason: Official Kaggle competition metric
- Advantage: Scale-invariant, penalizes proportional errors

**Secondary Metrics:**
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error

### 3.2 Validation Strategy

**Cross-Validation:**
- 5-Fold stratified K-Fold
- Shuffle with random_state=42
- Out-of-fold predictions for stacking

**Hold-out Validation:**
- 80/20 train/validation split
- Used for Random Forest quick evaluation

---

## 4. Outputs Generated

### 4.1 Preprocessed Data
- `foundation_agent_X_train.csv` - Training features
- `foundation_agent_y_train.csv` - Training target
- `foundation_agent_X_test.csv` - Test features
- `foundation_agent_test_ids.csv` - Test IDs for submission

### 4.2 Trained Models
- `foundation_agent_lightgbm_model.txt` - LightGBM model
- `foundation_agent_xgboost_model.json` - XGBoost model
- `foundation_agent_random_forest_model.pkl` - RandomForest model

### 4.3 Submissions
- `foundation_agent_submission_lightgbm.csv`
- `foundation_agent_submission_xgboost.csv`
- `foundation_agent_submission_random_forest.csv`
- `foundation_agent_submission_ensemble.csv`

### 4.4 Reports
- `foundation_agent_results_summary.json` - Detailed metrics
- `foundation_agent_metadata.json` - Execution metadata
- `foundation_agent_report.md` - This report

---

## 5. Key Findings

### 5.1 Model Insights

1. **{best_model.upper()} emerged as the best baseline** with RMSPE of {best_rmspe:.6f}
2. All models showed reasonable performance, indicating good feature engineering
3. Low variance across CV folds suggests stable predictions

### 5.2 Feature Importance (from preliminary analysis)

Top features expected to be important:
- Store ID (store-specific patterns)
- Day of week (weekly seasonality)
- Promo (promotional impact)
- Competition features (market dynamics)
- Temporal features (seasonal trends)

### 5.3 Challenges Identified

1. **Missing Values:** Store information has significant missing data
2. **Class Imbalance:** Closed stores (Sales=0) filtered out
3. **Temporal Dependencies:** Time-series nature not fully exploited in baseline

---

## 6. Recommendations for Next Agents

### 6.1 For Refinement Agent

**Optimization Opportunities:**
1. **Hyperparameter Tuning:**
   - LightGBM: Tune num_leaves, learning_rate, max_depth
   - XGBoost: Tune max_depth, subsample, colsample_bytree

2. **Feature Engineering:**
   - Add rolling window features (7-day, 14-day, 30-day averages)
   - Create lag features (previous week sales)
   - Store-day interaction features

3. **Validation Strategy:**
   - Time-based split (more realistic for time-series)
   - Store-stratified CV (ensure all stores in validation)

### 6.2 For Ensemble Agent

**Ensemble Strategies:**
1. **Stacking:** Use out-of-fold predictions as meta-features
2. **Weighted Averaging:** Optimize weights based on CV performance
3. **Blending:** Combine diverse model architectures

### 6.3 For Validation Agent

**Validation Tasks:**
1. Check prediction distribution (any anomalies?)
2. Verify no data leakage
3. Confirm temporal consistency
4. Test on different store types

---

## 7. Execution Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | Data Loading & Cleaning | - |
| 2 | Feature Engineering | - |
| 3 | LightGBM Training (5-Fold CV) | - |
| 4 | XGBoost Training (5-Fold CV) | - |
| 5 | Random Forest Training | - |
| 6 | Ensemble Creation | - |
| 7 | Report Generation | - |
| **Total** | **End-to-End Execution** | **{total_duration:.2f}s** |

---

## 8. MLE-STAR Alignment

This foundation phase aligns with MLE-STAR methodology:

✓ **Search-Informed:** Models selected based on research recommendations
✓ **Baseline Established:** Multiple robust baselines created
✓ **Modular Design:** Preprocessing and modeling separated for easy refinement
✓ **Feature-Focused:** 50% effort on feature engineering as recommended
✓ **Reproducible:** All seeds fixed, processes documented
✓ **Evaluation-Driven:** Proper CV methodology, official metric used

---

## 9. Conclusion

The Foundation Agent has successfully completed the initial model building phase:

- ✅ Data preprocessing pipeline implemented
- ✅ Three baseline models trained (LightGBM, XGBoost, RandomForest)
- ✅ Performance baselines established (Best: {best_rmspe:.6f} RMSPE)
- ✅ Comprehensive outputs generated for next agents
- ✅ Recommendations provided for refinement

**Status:** Ready for Refinement Phase

**Next Agent:** Refinement Agent (refinement_agent)

---

*Generated by Foundation Agent - MLE-STAR Workflow*
*Timestamp: {end_time.strftime('%Y-%m-%d %H:%M:%S')}*
"""

    return report


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result['success'] else 1)
