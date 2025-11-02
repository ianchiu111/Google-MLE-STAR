# Foundation Model Builder - Baseline Results

**Agent ID:** foundation_agent
**Session:** automation-session-1761902036865-1ckpyvf6d
**Execution:** workflow-exec-1761902036865-1x41nwdrl
**Date:** 2025-10-31 17:19:00

## Baseline Models

Three baseline models were trained:
1. Random Forest
2. XGBoost
3. LightGBM

## Performance Metrics (RMSPE)

| Model | Train RMSPE | Val RMSPE | Train RMSE | Val RMSE |
|-------|-------------|-----------|------------|----------|
| gradient_boosting | 0.2671 | 0.2735 | 1270.65 | 1520.99 |
| random_forest | 0.2739 | 0.2901 | 1399.15 | 1678.51 |

## Preprocessing Steps

1. Filtered zero sales records (store closed days)
2. Handled missing values in competition and promo features
3. Engineered temporal features (year, month, week, quarter)
4. Created competition and promo duration features
5. Encoded categorical variables

## Next Steps

- Refinement Agent: Hyperparameter tuning
- Ensemble Agent: Model ensembling strategies
- Validation Agent: Cross-validation and final testing
