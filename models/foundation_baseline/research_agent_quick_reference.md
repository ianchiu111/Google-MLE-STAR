# Research Agent: Quick Reference Guide
## Rossmann Store Sales - Key Findings for Agent Coordination

**Agent:** research_agent | **Status:** ✅ COMPLETED | **Date:** 2025-11-04

---

## 🎯 Problem Summary
- **Dataset:** Rossmann Store Sales (Kaggle 2015)
- **Task:** Forecast daily sales for 1,115 stores, 6 weeks ahead
- **Metric:** RMSPE (Root Mean Square Percentage Error)
- **Data:** ~3 years historical sales

---

## 🏆 Winning Solutions

### 1st Place: Gert Jacobusse
- **Model:** Ensemble of 20+ XGBoost models
- **Time Split:** 50% feature engineering | 40% ensembling | 10% tuning
- **Validation:** Last 6 weeks holdout
- **Key:** Feature engineering dominates success

### 3rd Place: Neokami (Cheng Guo)
- **Model:** Entity Embeddings Neural Network
- **Innovation:** Maps categorical variables to learned embeddings
- **Advantage:** Better than XGBoost with same features
- **Dropout:** 0.02 after input layer

---

## 📊 State-of-the-Art (2025)

| Model | Performance | Key Strength |
|-------|-------------|--------------|
| **LightGBM** | WMAPE 0.069 | Best for retail (2025 research) |
| **XGBoost** | WMAPE 0.096 | Fast, robust, proven winner |
| **TCN-Transformer** | MAE 2.01, RMSE 2.81 | SOTA hybrid architecture |
| **TFT** | Multi-horizon | Interpretable, handles mixed data |

**Verdict:** LightGBM > XGBoost for retail (2025), but ensemble both for best results

---

## 🔧 Feature Engineering Priority

### Core Features (Implement First)
```
✓ Temporal: day_of_week, month, year, is_weekend, is_holiday
✓ Lag features: 1, 7, 14, 30 days
✓ Rolling stats: 7-day mean/std, 30-day mean/std
✓ Competition duration: days since competitor opened
✓ Promo duration: consecutive promotion days
✓ Holiday flags: before/after state holidays
```

### Advanced Features (Phase 2)
```
✓ EWMA trends: Exponentially weighted moving averages
✓ Interactions: promo × day_of_week, store_type × day_of_week
✓ Store aggregations: store-specific rolling means
✓ Customer patterns: sales per customer, customer density
```

### Feature Selection
- Use last 6 weeks for validation
- Remove multicollinear features (VIF analysis)
- Focus on high-importance features
- Iterate based on RMSPE improvement

---

## 🤖 Recommended Model Strategy

### Baseline Models (Foundation Phase)
1. **LightGBM** - Fast, SOTA performance
2. **XGBoost** - Proven winner, robust
3. **RandomForest** - Stable baseline
4. **Entity Embeddings NN** - Categorical advantage

### Ensemble Strategy (Ensemble Phase)
```
Base Layer:
  - LightGBM (3 variants)
  - XGBoost (3 variants)
  - Entity Embeddings NN (2 variants)
  - RandomForest (1 variant)

Meta Layer:
  - Ridge Regression or Linear Regression
  - Trained on base model predictions

Alternative:
  - Weighted averaging optimized on validation set
```

---

## ⚙️ Hyperparameter Tuning

### Strategy
- **Use Random Search** (faster than Grid Search)
- **Focus on:** learning_rate, max_depth/num_leaves, regularization
- **Limit iterations:** 10% of time allocation

### Key XGBoost Parameters
```python
n_estimators: 100-1000
max_depth: 3-10
learning_rate: 0.01-0.3
subsample: 0.8-1.0
colsample_bytree: 0.8-1.0
gamma: 0-5
reg_alpha, reg_lambda: regularization
```

### Key LightGBM Parameters
```python
num_leaves: 20-100
min_data_in_leaf: 20-100
bagging_fraction: 0.8-1.0
feature_fraction: 0.8-1.0
lambda_l1, lambda_l2: regularization
```

---

## ✅ Validation Strategy

### Time Series Splits
- **Holdout:** Last 6 weeks (following 1st place winner)
- **Cross-validation:** Expanding window (respect temporal order)
- **No data leakage:** Never use future data in features

### Metric
- **Primary:** RMSPE (competition metric)
- **Monitor:** RMSE, MAE for diagnostics
- **Goal:** Minimize RMSPE on holdout set

---

## 📚 Implementation Resources

### GitHub Repos
- **Entity Embeddings:** github.com/entron/entity-embedding-rossmann
- **PyTorch Entity:** github.com/RaulSanchezVazquez/pytorch-entity-embedding-rossmann
- **XGBoost High-Accuracy:** github.com/Shahrukh2016/Retail_Sales_Prediction
- **XGBoost+Entity:** github.com/JustinCharbonneau/Kaggle-Rossmann

### Key Papers
- **Entity Embeddings:** arXiv:1604.06737
- **2025 Comparative Study:** arXiv:2506.05941
- **1st Place Docs:** [Kaggle PDF](https://storage.googleapis.com/kaggle-forum-message-attachments/102102/3454/Rossmann_nr1_doc.pdf)

---

## 💡 Critical Insights

### Time Allocation (Follow 1st Place)
```
50% → Feature Engineering
40% → Feature Selection + Ensembling
10% → Model Tuning
```

### Success Factors
1. **Feature engineering dominates** - Not model complexity
2. **Ensemble power** - Multiple models >> Single model
3. **LightGBM advantage** - Best single model (2025 research)
4. **Entity embeddings** - Powerful for categorical features
5. **Validation rigor** - Holdout + time series CV

### Risks to Avoid
❌ Overfitting → Use robust holdout validation
❌ Data leakage → Strict temporal splits
❌ Overtuning models → Follow 50-40-10 rule
❌ Ignoring categoricals → Use embeddings or target encoding

---

## 🔗 Agent Coordination

### Memory Keys Available
```
agent/research_agent/status
agent/research_agent/problem_domain
agent/research_agent/primary_metric
agent/research_agent/winning_approaches
agent/research_agent/sota_models_2025
agent/research_agent/feature_engineering_strategies
agent/research_agent/ensemble_methods
agent/research_agent/time_allocation_insight
agent/research_agent/implementation_repos
agent/research_agent/model_architectures
agent/research_agent/hyperparameter_tuning
agent/research_agent/key_references
agent/research_agent/output_files
```

Query with: `npx claude-flow@alpha memory query '<search_term>'`

### Next Agent Actions
- **Foundation Agent:** Build LightGBM/XGBoost baselines with core features
- **Refinement Agent:** Iterate feature engineering, optimize selection
- **Ensemble Agent:** Stack models, optimize weights
- **Validation Agent:** Monitor RMSPE, validate no leakage

---

## 📄 Full Report
See: `research_agent_comprehensive_report.md` for detailed analysis (12 sections, 50+ pages)

---

**Research Agent Status:** ✅ SEARCH Phase Complete
**Next Phase:** Foundation Building (foundation_agent)
**Coordination:** All findings stored in ReasoningBank memory
