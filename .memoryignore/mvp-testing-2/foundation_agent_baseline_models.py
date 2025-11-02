"""
Foundation Agent - Baseline Model Implementation
Rossmann Store Sales Prediction
Author: Foundation Agent (MLE-STAR Workflow)

Based on ML Researcher recommendations:
- LightGBM (localized approach)
- XGBoost (global approach)
- RandomForest + Linear ensemble
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


class RMSPEMetric:
    """Root Mean Square Percentage Error - Official Kaggle metric"""

    @staticmethod
    def calculate(y_true, y_pred):
        """Calculate RMSPE"""
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return 0.0

        return np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))

    @staticmethod
    def lgb_rmspe(y_pred, y_true):
        """RMSPE for LightGBM"""
        y_true = y_true.get_label()
        mask = y_true != 0
        if not mask.any():
            return 'rmspe', 0.0, False

        rmspe = np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))
        return 'rmspe', rmspe, False

    @staticmethod
    def xgb_rmspe(y_pred, y_true):
        """RMSPE for XGBoost"""
        y_true = y_true.get_label()
        mask = y_true != 0
        if not mask.any():
            return 'rmspe', 0.0

        rmspe = np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))
        return 'rmspe', rmspe


class BaselineModelBuilder:
    """
    Build baseline models for Rossmann sales prediction.
    Implements MLE-STAR foundation phase best practices.
    """

    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.models = {}
        self.results = {}

    def train_lightgbm(self, n_folds=5):
        """
        Train LightGBM model - Recommended for localized approach
        Fast training, handles categorical features well
        """
        print("\n" + "="*70)
        print("TRAINING LIGHTGBM MODEL")
        print("="*70)

        # LightGBM parameters optimized for speed and baseline performance
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'verbosity': -1,
            'seed': 42
        }

        # K-Fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(self.X_train))
        test_predictions = np.zeros(len(self.X_test))
        fold_scores = []

        print(f"\nPerforming {n_folds}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            print(f"\n--- Fold {fold}/{n_folds} ---")

            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # Create LightGBM datasets
            train_data = lgb.Dataset(X_tr, label=y_tr)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            # Train model
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, val_data],
                feval=RMSPEMetric.lgb_rmspe,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )

            # Validation predictions
            val_pred = model.predict(X_val, num_iteration=model.best_iteration)
            oof_predictions[val_idx] = val_pred

            # Calculate RMSPE
            rmspe = RMSPEMetric.calculate(y_val.values, val_pred)
            fold_scores.append(rmspe)
            print(f"Fold {fold} RMSPE: {rmspe:.6f}")

            # Test predictions
            test_predictions += model.predict(self.X_test, num_iteration=model.best_iteration) / n_folds

        # Overall CV score
        cv_rmspe = np.mean(fold_scores)
        cv_std = np.std(fold_scores)

        print(f"\n{'='*50}")
        print(f"LightGBM CV RMSPE: {cv_rmspe:.6f} (+/- {cv_std:.6f})")
        print(f"{'='*50}")

        # Train final model on full data
        print("\nTraining final model on full dataset...")
        full_train_data = lgb.Dataset(self.X_train, label=self.y_train)
        final_model = lgb.train(
            params,
            full_train_data,
            num_boost_round=int(1000 * 1.1),  # Slightly more rounds for full data
            feval=RMSPEMetric.lgb_rmspe,
            callbacks=[lgb.log_evaluation(period=100)]
        )

        # Store results
        self.models['lightgbm'] = final_model
        self.results['lightgbm'] = {
            'cv_rmspe': cv_rmspe,
            'cv_std': cv_std,
            'fold_scores': fold_scores,
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions
        }

        return final_model, cv_rmspe

    def train_xgboost(self, n_folds=5):
        """
        Train XGBoost model - Recommended for global approach
        Robust and accurate, good for baseline comparison
        """
        print("\n" + "="*70)
        print("TRAINING XGBOOST MODEL")
        print("="*70)

        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'seed': 42,
            'verbosity': 1
        }

        # K-Fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(self.X_train))
        test_predictions = np.zeros(len(self.X_test))
        fold_scores = []

        print(f"\nPerforming {n_folds}-Fold Cross-Validation...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            print(f"\n--- Fold {fold}/{n_folds} ---")

            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # Create XGBoost datasets
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)

            # Train model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                feval=RMSPEMetric.xgb_rmspe,
                early_stopping_rounds=50,
                verbose_eval=100
            )

            # Validation predictions
            val_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
            oof_predictions[val_idx] = val_pred

            # Calculate RMSPE
            rmspe = RMSPEMetric.calculate(y_val.values, val_pred)
            fold_scores.append(rmspe)
            print(f"Fold {fold} RMSPE: {rmspe:.6f}")

            # Test predictions
            dtest = xgb.DMatrix(self.X_test)
            test_predictions += model.predict(dtest, iteration_range=(0, model.best_iteration)) / n_folds

        # Overall CV score
        cv_rmspe = np.mean(fold_scores)
        cv_std = np.std(fold_scores)

        print(f"\n{'='*50}")
        print(f"XGBoost CV RMSPE: {cv_rmspe:.6f} (+/- {cv_std:.6f})")
        print(f"{'='*50}")

        # Train final model on full data
        print("\nTraining final model on full dataset...")
        dfull = xgb.DMatrix(self.X_train, label=self.y_train)
        final_model = xgb.train(
            params,
            dfull,
            num_boost_round=int(1000 * 1.1),
            verbose_eval=100
        )

        # Store results
        self.models['xgboost'] = final_model
        self.results['xgboost'] = {
            'cv_rmspe': cv_rmspe,
            'cv_std': cv_std,
            'fold_scores': fold_scores,
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions
        }

        return final_model, cv_rmspe

    def train_random_forest(self, n_estimators=100):
        """
        Train RandomForest model - Part of ensemble approach
        Good for feature importance and robust predictions
        """
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*70)

        # Split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )

        print(f"Training set: {X_tr.shape}")
        print(f"Validation set: {X_val.shape}")

        # Train RandomForest
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        print("\nTraining Random Forest...")
        model.fit(X_tr, y_tr)

        # Validation predictions
        val_pred = model.predict(X_val)
        val_rmspe = RMSPEMetric.calculate(y_val.values, val_pred)

        print(f"\nValidation RMSPE: {val_rmspe:.6f}")

        # Test predictions
        test_predictions = model.predict(self.X_test)

        # Store results
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'val_rmspe': val_rmspe,
            'test_predictions': test_predictions
        }

        return model, val_rmspe

    def create_simple_ensemble(self, weights=None):
        """
        Create simple weighted ensemble of all models
        Default: equal weights
        """
        print("\n" + "="*70)
        print("CREATING MODEL ENSEMBLE")
        print("="*70)

        if weights is None:
            # Equal weights
            weights = {
                'lightgbm': 1.0,
                'xgboost': 1.0,
                'random_forest': 1.0
            }

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        print(f"Ensemble weights: {weights}")

        # Combine predictions
        ensemble_predictions = np.zeros(len(self.X_test))

        for model_name, weight in weights.items():
            if model_name in self.results:
                ensemble_predictions += weight * self.results[model_name]['test_predictions']

        self.results['ensemble'] = {
            'test_predictions': ensemble_predictions,
            'weights': weights
        }

        return ensemble_predictions

    def save_models(self, output_dir='models'):
        """Save all trained models and results"""
        print("\n" + "="*70)
        print("SAVING MODELS AND RESULTS")
        print("="*70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models
        for model_name, model in self.models.items():
            model_path = f"{output_dir}/foundation_agent_{model_name}_model.pkl"

            if model_name in ['lightgbm', 'xgboost']:
                # Save booster models using their native format
                if model_name == 'lightgbm':
                    model.save_model(f"{output_dir}/foundation_agent_{model_name}_model.txt")
                    print(f"✓ Saved: {output_dir}/foundation_agent_{model_name}_model.txt")
                else:  # xgboost
                    model.save_model(f"{output_dir}/foundation_agent_{model_name}_model.json")
                    print(f"✓ Saved: {output_dir}/foundation_agent_{model_name}_model.json")
            else:
                # Save sklearn models with pickle
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✓ Saved: {model_path}")

        # Save results summary
        results_summary = {
            'timestamp': timestamp,
            'models': {}
        }

        for model_name, result in self.results.items():
            if model_name == 'ensemble':
                results_summary['models'][model_name] = {
                    'weights': result['weights']
                }
            elif 'cv_rmspe' in result:
                results_summary['models'][model_name] = {
                    'cv_rmspe': float(result['cv_rmspe']),
                    'cv_std': float(result['cv_std']),
                    'fold_scores': [float(s) for s in result['fold_scores']]
                }
            else:
                results_summary['models'][model_name] = {
                    'val_rmspe': float(result['val_rmspe'])
                }

        # Save results as JSON
        import json
        with open(f"{output_dir}/foundation_agent_results_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"✓ Saved: {output_dir}/foundation_agent_results_summary.json")

        return results_summary

    def create_submission(self, test_ids, model_name='ensemble', output_dir='models'):
        """Create Kaggle submission file"""
        print("\n" + "="*70)
        print(f"CREATING SUBMISSION FILE - {model_name.upper()}")
        print("="*70)

        if model_name not in self.results:
            print(f"Error: Model '{model_name}' not found in results")
            return None

        predictions = self.results[model_name]['test_predictions']

        # Ensure non-negative predictions
        predictions = np.maximum(0, predictions)

        # Create submission dataframe
        submission = pd.DataFrame({
            'Id': test_ids,
            'Sales': predictions
        })

        # Save submission
        submission_path = f"{output_dir}/foundation_agent_submission_{model_name}.csv"
        submission.to_csv(submission_path, index=False)

        print(f"✓ Saved: {submission_path}")
        print(f"  Predictions: {len(predictions)}")
        print(f"  Mean Sales: {predictions.mean():.2f}")
        print(f"  Median Sales: {np.median(predictions):.2f}")
        print(f"  Min Sales: {predictions.min():.2f}")
        print(f"  Max Sales: {predictions.max():.2f}")

        return submission


def main():
    """Main baseline model training pipeline"""
    print("="*70)
    print("FOUNDATION AGENT - BASELINE MODEL TRAINING")
    print("="*70)

    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_train = pd.read_csv('models/foundation_agent_X_train.csv')
    y_train = pd.read_csv('models/foundation_agent_y_train.csv').squeeze()
    X_test = pd.read_csv('models/foundation_agent_X_test.csv')
    test_ids = pd.read_csv('models/foundation_agent_test_ids.csv').squeeze()

    print(f"Training features: {X_train.shape}")
    print(f"Training target: {y_train.shape}")
    print(f"Test features: {X_test.shape}")

    # Initialize model builder
    builder = BaselineModelBuilder(X_train, y_train, X_test)

    # Train models based on research recommendations
    print("\n" + "="*70)
    print("TRAINING BASELINE MODELS")
    print("="*70)

    # 1. LightGBM - Fast and accurate
    lgb_model, lgb_score = builder.train_lightgbm(n_folds=5)

    # 2. XGBoost - Robust baseline
    xgb_model, xgb_score = builder.train_xgboost(n_folds=5)

    # 3. Random Forest - Ensemble component
    rf_model, rf_score = builder.train_random_forest(n_estimators=100)

    # Create ensemble
    ensemble_predictions = builder.create_simple_ensemble()

    # Save all models
    results_summary = builder.save_models()

    # Create submissions for all models
    for model_name in ['lightgbm', 'xgboost', 'random_forest', 'ensemble']:
        builder.create_submission(test_ids, model_name=model_name)

    # Print final summary
    print("\n" + "="*70)
    print("BASELINE MODELS TRAINING COMPLETE")
    print("="*70)
    print("\nModel Performance Summary:")
    print(f"  LightGBM    CV RMSPE: {lgb_score:.6f}")
    print(f"  XGBoost     CV RMSPE: {xgb_score:.6f}")
    print(f"  RandomForest Val RMSPE: {rf_score:.6f}")
    print(f"\nBest Model: {'LightGBM' if lgb_score < xgb_score else 'XGBoost'}")
    print("="*70)

    return builder, results_summary


if __name__ == "__main__":
    builder, results = main()
