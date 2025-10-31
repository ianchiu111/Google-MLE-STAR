#!/usr/bin/env python3
"""
Foundation Model Builder - Baseline Model Implementation
MLE-STAR Workflow - Foundation Phase
Agent ID: foundation_agent
Session: automation-session-1761902036865-1ckpyvf6d
Execution: workflow-exec-1761902036865-1x41nwdrl

Based on Research Findings:
- Baseline models: LightGBM, XGBoost, RandomForest
- Primary metric: RMSPE (Root Mean Square Percentage Error)
- Key preprocessing: Handle zero sales (17%), missing values, temporal features
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List
import joblib

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Try importing LightGBM (may fail on some systems)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: LightGBM not available: {e}")
    LIGHTGBM_AVAILABLE = False
    lgb = None

# Try importing XGBoost (may fail on some systems)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    print(f"Warning: XGBoost not available: {e}")
    XGBOOST_AVAILABLE = False
    xgb = None

warnings.filterwarnings('ignore')


class FoundationModelBuilder:
    """Build baseline models for Rossmann Store Sales prediction"""

    def __init__(self, train_path: str, store_path: str, test_path: str):
        self.train_path = train_path
        self.store_path = store_path
        self.test_path = test_path

        # Data containers
        self.train_df = None
        self.test_df = None
        self.store_df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        # Model containers
        self.models = {}
        self.performance_metrics = {}

        # Label encoders
        self.label_encoders = {}

    def rmspe(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Square Percentage Error (RMSPE)
        Primary evaluation metric for Rossmann competition
        """
        # Filter out zero values to avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return 0.0

        percentage_errors = ((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2
        return np.sqrt(np.mean(percentage_errors))

    def load_and_merge_data(self):
        """Load datasets and merge with store information"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        # Load datasets
        self.train_df = pd.read_csv(self.train_path, parse_dates=['Date'])
        self.test_df = pd.read_csv(self.test_path, parse_dates=['Date'])
        self.store_df = pd.read_csv(self.store_path)

        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        print(f"Store shape: {self.store_df.shape}")

        # Merge with store information
        self.train_df = self.train_df.merge(self.store_df, on='Store', how='left')
        self.test_df = self.test_df.merge(self.store_df, on='Store', how='left')

        print(f"Merged train shape: {self.train_df.shape}")
        print(f"Merged test shape: {self.test_df.shape}")
        print()

    def preprocess_data(self):
        """
        Preprocessing pipeline based on EDA insights:
        1. Filter zero sales (store closed days)
        2. Handle missing values
        3. Engineer temporal features
        4. Encode categorical variables
        """
        print("=" * 80)
        print("PREPROCESSING DATA")
        print("=" * 80)

        # 1. Filter training data: Remove closed stores (Sales=0, Open=0)
        print(f"Original train records: {len(self.train_df):,}")
        print(f"Zero sales records: {(self.train_df['Sales'] == 0).sum():,} ({(self.train_df['Sales'] == 0).mean()*100:.1f}%)")

        # Keep only open stores with sales > 0 for training
        self.train_df = self.train_df[(self.train_df['Open'] == 1) & (self.train_df['Sales'] > 0)].copy()
        print(f"Filtered train records: {len(self.train_df):,}")
        print()

        # 2. Handle missing values
        print("Handling missing values...")

        # CompetitionDistance: Fill with a large value (no nearby competition)
        self.train_df['CompetitionDistance'].fillna(self.train_df['CompetitionDistance'].median(), inplace=True)
        self.test_df['CompetitionDistance'].fillna(self.test_df['CompetitionDistance'].median(), inplace=True)

        # Competition open since: Fill with 0 (no competition)
        for col in ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']:
            self.train_df[col].fillna(0, inplace=True)
            self.test_df[col].fillna(0, inplace=True)

        # Promo2: Fill with 0 (no promo)
        for col in ['Promo2SinceWeek', 'Promo2SinceYear']:
            self.train_df[col].fillna(0, inplace=True)
            self.test_df[col].fillna(0, inplace=True)

        self.train_df['PromoInterval'].fillna('None', inplace=True)
        self.test_df['PromoInterval'].fillna('None', inplace=True)

        # Handle Open in test (missing values)
        self.test_df['Open'].fillna(1, inplace=True)

        print("Missing values handled.")
        print()

        # 3. Engineer temporal features
        print("Engineering temporal features...")
        for df in [self.train_df, self.test_df]:
            # Basic temporal features
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
            df['Quarter'] = df['Date'].dt.quarter

            # Competition months open
            df['CompetitionMonthsOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                                          (df['Month'] - df['CompetitionOpenSinceMonth'])
            df['CompetitionMonthsOpen'] = df['CompetitionMonthsOpen'].apply(lambda x: x if x > 0 else 0)

            # Promo2 weeks active
            df['Promo2WeeksActive'] = (df['Year'] - df['Promo2SinceYear']) * 52 + \
                                       (df['WeekOfYear'] - df['Promo2SinceWeek'])
            df['Promo2WeeksActive'] = df['Promo2WeeksActive'].apply(lambda x: x if x > 0 else 0)

            # Is promo month (PromoInterval feature)
            month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                        7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
            df['MonthStr'] = df['Month'].map(month_map)
            df['IsPromoMonth'] = df.apply(
                lambda x: 1 if (str(x['PromoInterval']) != 'None' and
                               x['MonthStr'] in str(x['PromoInterval'])) else 0,
                axis=1
            )

        print("Temporal features engineered.")
        print()

        # 4. Encode categorical variables
        print("Encoding categorical variables...")
        categorical_cols = ['StateHoliday', 'StoreType', 'Assortment', 'PromoInterval']

        for col in categorical_cols:
            # Combine train and test for consistent encoding
            combined = pd.concat([self.train_df[col], self.test_df[col]])
            le = LabelEncoder()
            le.fit(combined.astype(str))

            self.train_df[col] = le.transform(self.train_df[col].astype(str))
            self.test_df[col] = le.transform(self.test_df[col].astype(str))

            self.label_encoders[col] = le

        print("Categorical variables encoded.")
        print()

    def create_features_target(self):
        """Create feature matrix and target variable"""
        print("=" * 80)
        print("CREATING FEATURES AND TARGET")
        print("=" * 80)

        # Define feature columns
        feature_cols = [
            # Store features
            'Store', 'StoreType', 'Assortment', 'CompetitionDistance',
            'CompetitionMonthsOpen',

            # Temporal features
            'DayOfWeek', 'Year', 'Month', 'Day', 'WeekOfYear', 'Quarter',

            # Promo features
            'Promo', 'Promo2', 'Promo2WeeksActive', 'IsPromoMonth',

            # Holiday features
            'StateHoliday', 'SchoolHoliday'
        ]

        print(f"Selected {len(feature_cols)} features:")
        for i, col in enumerate(feature_cols, 1):
            print(f"{i:2d}. {col}")
        print()

        # Create feature matrix
        X = self.train_df[feature_cols].copy()
        y = self.train_df['Sales'].copy()

        # Time-based split for validation
        # Use last 6 weeks as validation (roughly 10% of data)
        split_date = self.train_df['Date'].max() - pd.Timedelta(days=42)

        train_mask = self.train_df['Date'] <= split_date
        val_mask = self.train_df['Date'] > split_date

        self.X_train = X[train_mask].copy()
        self.y_train = y[train_mask].copy()
        self.X_val = X[val_mask].copy()
        self.y_val = y[val_mask].copy()

        print(f"Training set: {len(self.X_train):,} samples")
        print(f"Validation set: {len(self.X_val):,} samples")
        print(f"Split date: {split_date.strftime('%Y-%m-%d')}")
        print()

    def train_random_forest(self):
        """Train Random Forest baseline model"""
        print("=" * 80)
        print("TRAINING RANDOM FOREST")
        print("=" * 80)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        print("Training...")
        model.fit(self.X_train, self.y_train)

        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)

        # Metrics
        train_rmspe = self.rmspe(self.y_train.values, train_pred)
        val_rmspe = self.rmspe(self.y_val.values, val_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))

        print(f"\nRandom Forest Results:")
        print(f"Train RMSPE: {train_rmspe:.4f}")
        print(f"Val RMSPE: {val_rmspe:.4f}")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Val RMSE: {val_rmse:.2f}")
        print()

        self.models['random_forest'] = model
        self.performance_metrics['random_forest'] = {
            'train_rmspe': float(train_rmspe),
            'val_rmspe': float(val_rmspe),
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse)
        }

    def train_xgboost(self):
        """Train XGBoost baseline model"""
        if not XGBOOST_AVAILABLE:
            print("=" * 80)
            print("SKIPPING XGBOOST (NOT AVAILABLE)")
            print("=" * 80)
            print("XGBoost library is not available on this system.")
            print()
            return

        print("=" * 80)
        print("TRAINING XGBOOST")
        print("=" * 80)

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            n_jobs=-1,
            random_state=42,
            tree_method='hist'
        )

        print("Training...")
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=50
        )

        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)

        # Metrics
        train_rmspe = self.rmspe(self.y_train.values, train_pred)
        val_rmspe = self.rmspe(self.y_val.values, val_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))

        print(f"\nXGBoost Results:")
        print(f"Train RMSPE: {train_rmspe:.4f}")
        print(f"Val RMSPE: {val_rmspe:.4f}")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Val RMSE: {val_rmse:.2f}")
        print()

        self.models['xgboost'] = model
        self.performance_metrics['xgboost'] = {
            'train_rmspe': float(train_rmspe),
            'val_rmspe': float(val_rmspe),
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse)
        }

    def train_gradient_boosting(self):
        """Train Gradient Boosting (sklearn) as fallback baseline model"""
        print("=" * 80)
        print("TRAINING GRADIENT BOOSTING (SKLEARN)")
        print("=" * 80)

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            verbose=1
        )

        print("Training...")
        model.fit(self.X_train, self.y_train)

        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)

        # Metrics
        train_rmspe = self.rmspe(self.y_train.values, train_pred)
        val_rmspe = self.rmspe(self.y_val.values, val_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))

        print(f"\nGradient Boosting Results:")
        print(f"Train RMSPE: {train_rmspe:.4f}")
        print(f"Val RMSPE: {val_rmspe:.4f}")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Val RMSE: {val_rmse:.2f}")
        print()

        self.models['gradient_boosting'] = model
        self.performance_metrics['gradient_boosting'] = {
            'train_rmspe': float(train_rmspe),
            'val_rmspe': float(val_rmspe),
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse)
        }

    def train_lightgbm(self):
        """Train LightGBM baseline model"""
        if not LIGHTGBM_AVAILABLE:
            print("=" * 80)
            print("SKIPPING LIGHTGBM (NOT AVAILABLE)")
            print("=" * 80)
            print("LightGBM library is not available on this system.")
            print()
            return

        print("=" * 80)
        print("TRAINING LIGHTGBM")
        print("=" * 80)

        model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0,
            reg_lambda=1,
            n_jobs=-1,
            random_state=42,
            verbose=-1
        )

        print("Training...")
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            callbacks=[lgb.log_evaluation(50)]
        )

        # Predictions
        train_pred = model.predict(self.X_train)
        val_pred = model.predict(self.X_val)

        # Metrics
        train_rmspe = self.rmspe(self.y_train.values, train_pred)
        val_rmspe = self.rmspe(self.y_val.values, val_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))

        print(f"\nLightGBM Results:")
        print(f"Train RMSPE: {train_rmspe:.4f}")
        print(f"Val RMSPE: {val_rmspe:.4f}")
        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Val RMSE: {val_rmse:.2f}")
        print()

        self.models['lightgbm'] = model
        self.performance_metrics['lightgbm'] = {
            'train_rmspe': float(train_rmspe),
            'val_rmspe': float(val_rmspe),
            'train_rmse': float(train_rmse),
            'val_rmse': float(val_rmse)
        }

    def compare_models(self):
        """Compare baseline model performance"""
        print("=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        results = []
        for model_name, metrics in self.performance_metrics.items():
            results.append({
                'Model': model_name,
                'Train RMSPE': metrics['train_rmspe'],
                'Val RMSPE': metrics['val_rmspe'],
                'Train RMSE': metrics['train_rmse'],
                'Val RMSE': metrics['val_rmse']
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Val RMSPE')

        print(results_df.to_string(index=False))
        print()

        # Identify best model
        best_model = results_df.iloc[0]['Model']
        best_rmspe = results_df.iloc[0]['Val RMSPE']

        print(f"Best Model: {best_model} (Val RMSPE: {best_rmspe:.4f})")
        print()

        return results_df

    def save_models_and_results(self, output_dir='./models'):
        """Save models and performance results"""
        print("=" * 80)
        print("SAVING MODELS AND RESULTS")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save models
        for model_name, model in self.models.items():
            model_file = output_path / f'foundation_agent_{model_name}.pkl'
            joblib.dump(model, model_file)
            print(f"Saved {model_name}: {model_file}")

        # Save label encoders
        encoders_file = output_path / 'foundation_agent_label_encoders.pkl'
        joblib.dump(self.label_encoders, encoders_file)
        print(f"Saved label encoders: {encoders_file}")

        # Save performance metrics JSON
        metrics_file = output_path / 'foundation_agent_baseline_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        print(f"Saved metrics: {metrics_file}")

        # Save summary report
        report_file = output_path / 'foundation_agent_baseline_report.md'
        with open(report_file, 'w') as f:
            f.write("# Foundation Model Builder - Baseline Results\n\n")
            f.write(f"**Agent ID:** foundation_agent\n")
            f.write(f"**Session:** automation-session-1761902036865-1ckpyvf6d\n")
            f.write(f"**Execution:** workflow-exec-1761902036865-1x41nwdrl\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Baseline Models\n\n")
            f.write("Three baseline models were trained:\n")
            f.write("1. Random Forest\n")
            f.write("2. XGBoost\n")
            f.write("3. LightGBM\n\n")

            f.write("## Performance Metrics (RMSPE)\n\n")
            f.write("| Model | Train RMSPE | Val RMSPE | Train RMSE | Val RMSE |\n")
            f.write("|-------|-------------|-----------|------------|----------|\n")

            for model_name, metrics in sorted(self.performance_metrics.items(),
                                             key=lambda x: x[1]['val_rmspe']):
                f.write(f"| {model_name} | {metrics['train_rmspe']:.4f} | "
                       f"{metrics['val_rmspe']:.4f} | {metrics['train_rmse']:.2f} | "
                       f"{metrics['val_rmse']:.2f} |\n")

            f.write("\n## Preprocessing Steps\n\n")
            f.write("1. Filtered zero sales records (store closed days)\n")
            f.write("2. Handled missing values in competition and promo features\n")
            f.write("3. Engineered temporal features (year, month, week, quarter)\n")
            f.write("4. Created competition and promo duration features\n")
            f.write("5. Encoded categorical variables\n\n")

            f.write("## Next Steps\n\n")
            f.write("- Refinement Agent: Hyperparameter tuning\n")
            f.write("- Ensemble Agent: Model ensembling strategies\n")
            f.write("- Validation Agent: Cross-validation and final testing\n")

        print(f"Saved report: {report_file}")
        print()

    def run_full_pipeline(self):
        """Execute complete foundation model building pipeline"""
        print("=" * 80)
        print("FOUNDATION MODEL BUILDER - BASELINE PIPELINE")
        print("MLE-STAR Workflow - Foundation Phase")
        print("=" * 80)
        print()

        # Data pipeline
        self.load_and_merge_data()
        self.preprocess_data()
        self.create_features_target()

        # Model training
        self.train_random_forest()
        self.train_gradient_boosting()  # Always available (sklearn)
        self.train_xgboost()            # Try if available
        self.train_lightgbm()           # Try if available

        # Comparison and saving
        comparison_df = self.compare_models()
        self.save_models_and_results()

        print("=" * 80)
        print("FOUNDATION PIPELINE COMPLETE")
        print("=" * 80)

        return self.performance_metrics


if __name__ == '__main__':
    builder = FoundationModelBuilder(
        train_path='data/train.csv',
        store_path='data/store.csv',
        test_path='data/test.csv'
    )

    metrics = builder.run_full_pipeline()
