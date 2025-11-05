"""
Foundation Agent - Baseline Models
MLE-STAR Workflow - Rossmann Store Sales Prediction

This module implements baseline models for sales prediction:
1. Linear Regression - Simple baseline
2. Random Forest - Tree-based ensemble
3. Gradient Boosting - Advanced ensemble (XGBoost)

Each model is implemented with proper validation and performance metrics.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class BaselineModelBuilder:
    """
    Builder class for creating and evaluating baseline models.
    Follows MLE-STAR methodology for initial model development.
    """

    def __init__(self, random_state=42):
        """
        Initialize baseline model builder.

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

    def rmspe(self, y_true, y_pred):
        """
        Calculate Root Mean Squared Percentage Error (RMSPE).
        This is the evaluation metric for Rossmann Kaggle competition.

        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted values

        Returns:
        --------
        float : RMSPE score
        """
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return 0.0

        return np.sqrt(np.mean(np.square((y_true[mask] - y_pred[mask]) / y_true[mask])))

    def evaluate_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """
        Evaluate model performance on validation set.

        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_train : pd.DataFrame
            Training features
        y_train : np.array
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : np.array
            Validation target
        model_name : str
            Name of the model

        Returns:
        --------
        dict : Performance metrics
        """
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Clip negative predictions to zero (sales can't be negative)
        y_train_pred = np.maximum(y_train_pred, 0)
        y_val_pred = np.maximum(y_val_pred, 0)

        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'train_rmspe': self.rmspe(y_train, y_train_pred),
            'val_rmspe': self.rmspe(y_val, y_val_pred),
            'timestamp': datetime.now().isoformat()
        }

        return metrics

    def train_linear_regression(self, X_train, y_train, X_val, y_val):
        """
        Train Ridge Regression baseline model.

        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data

        Returns:
        --------
        tuple : (model, metrics)
        """
        print("\n--- Training Linear Regression (Ridge) ---")

        model = Ridge(alpha=1.0, random_state=self.random_state)
        model.fit(X_train, y_train)

        metrics = self.evaluate_model(model, X_train, y_train, X_val, y_val, 'Ridge Regression')

        print(f"Validation RMSE: {metrics['val_rmse']:.2f}")
        print(f"Validation RMSPE: {metrics['val_rmspe']:.4f}")
        print(f"Validation R²: {metrics['val_r2']:.4f}")

        return model, metrics

    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """
        Train Random Forest baseline model.

        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data

        Returns:
        --------
        tuple : (model, metrics)
        """
        print("\n--- Training Random Forest ---")

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train, y_train)

        metrics = self.evaluate_model(model, X_train, y_train, X_val, y_val, 'Random Forest')

        print(f"Validation RMSE: {metrics['val_rmse']:.2f}")
        print(f"Validation RMSPE: {metrics['val_rmspe']:.4f}")
        print(f"Validation R²: {metrics['val_r2']:.4f}")

        return model, metrics

    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """
        Train Gradient Boosting baseline model (sklearn).

        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data

        Returns:
        --------
        tuple : (model, metrics)
        """
        print("\n--- Training Gradient Boosting (sklearn) ---")

        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=self.random_state,
            verbose=0
        )
        model.fit(X_train, y_train)

        metrics = self.evaluate_model(model, X_train, y_train, X_val, y_val, 'Gradient Boosting')

        print(f"Validation RMSE: {metrics['val_rmse']:.2f}")
        print(f"Validation RMSPE: {metrics['val_rmspe']:.4f}")
        print(f"Validation R²: {metrics['val_r2']:.4f}")

        return model, metrics

    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost baseline model.

        Parameters:
        -----------
        X_train, y_train : Training data
        X_val, y_val : Validation data

        Returns:
        --------
        tuple : (model, metrics) or (None, None) if XGBoost unavailable
        """
        if not XGBOOST_AVAILABLE:
            print("\n--- XGBoost not available, skipping ---")
            return None, None

        print("\n--- Training XGBoost ---")

        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train)

        metrics = self.evaluate_model(model, X_train, y_train, X_val, y_val, 'XGBoost')

        print(f"Validation RMSE: {metrics['val_rmse']:.2f}")
        print(f"Validation RMSPE: {metrics['val_rmspe']:.4f}")
        print(f"Validation R²: {metrics['val_r2']:.4f}")

        return model, metrics

    def train_all_baselines(self, X, y, test_size=0.2):
        """
        Train all baseline models and compare performance.

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : np.array
            Target values
        test_size : float
            Proportion of data for validation

        Returns:
        --------
        dict : Results for all models
        """
        print("\n" + "="*60)
        print("BASELINE MODEL TRAINING - MLE-STAR Foundation Phase")
        print("="*60)

        # Split data
        print(f"\nSplitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% validation")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        print(f"Training samples: {X_train.shape[0]:,}")
        print(f"Validation samples: {X_val.shape[0]:,}")
        print(f"Number of features: {X_train.shape[1]}")

        # Train models
        models_to_train = [
            ('ridge', self.train_linear_regression),
            ('random_forest', self.train_random_forest),
            ('gradient_boosting', self.train_gradient_boosting),
            ('xgboost', self.train_xgboost)
        ]

        for model_key, train_func in models_to_train:
            model, metrics = train_func(X_train, y_train, X_val, y_val)
            if model is not None:
                self.models[model_key] = model
                self.results[model_key] = metrics

        # Identify best model based on validation RMSPE
        self._identify_best_model()

        return self.results

    def _identify_best_model(self):
        """Identify the best performing model based on validation RMSPE."""
        if not self.results:
            return

        best_rmspe = float('inf')
        for model_key, metrics in self.results.items():
            if metrics['val_rmspe'] < best_rmspe:
                best_rmspe = metrics['val_rmspe']
                self.best_model_name = model_key
                self.best_model = self.models[model_key]

        print("\n" + "="*60)
        print("BASELINE COMPARISON SUMMARY")
        print("="*60)

        # Create comparison dataframe
        comparison = pd.DataFrame(self.results).T
        comparison = comparison[['val_rmse', 'val_rmspe', 'val_r2', 'val_mae']]
        comparison = comparison.round(4)

        print("\n" + comparison.to_string())

        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Validation RMSPE: {self.results[self.best_model_name]['val_rmspe']:.4f}")

    def save_models(self, output_dir='models/'):
        """
        Save all trained models and results.

        Parameters:
        -----------
        output_dir : str
            Directory to save models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save models
        for model_name, model in self.models.items():
            filepath = os.path.join(output_dir, f'foundation_agent_{model_name}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved: {filepath}")

        # Save results as JSON
        results_path = os.path.join(output_dir, 'foundation_agent_baseline_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved: {results_path}")

        # Save best model separately
        if self.best_model is not None:
            best_path = os.path.join(output_dir, 'foundation_agent_best_model.pkl')
            with open(best_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            print(f"Saved best model: {best_path}")

    def load_model(self, model_path):
        """
        Load a saved model.

        Parameters:
        -----------
        model_path : str
            Path to saved model file

        Returns:
        --------
        model : Loaded model
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def predict(self, X, model=None):
        """
        Make predictions using specified or best model.

        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction
        model : sklearn model, optional
            Model to use (defaults to best model)

        Returns:
        --------
        np.array : Predictions
        """
        if model is None:
            model = self.best_model

        if model is None:
            raise ValueError("No model available for prediction")

        predictions = model.predict(X)
        predictions = np.maximum(predictions, 0)  # Clip negative values

        return predictions


if __name__ == "__main__":
    # Example usage
    print("Foundation Agent - Baseline Models")
    print("This module should be used with preprocessed data from foundation_agent_preprocessing.py")
    print("\nExample usage:")
    print("""
    from foundation_agent_preprocessing import RossmannPreprocessor
    from foundation_agent_baseline_models import BaselineModelBuilder

    # Load and preprocess data
    preprocessor = RossmannPreprocessor()
    train_df, store_df, test_df = preprocessor.load_data(...)
    X_train, y_train, X_test, _ = preprocessor.preprocess(...)

    # Train baselines
    builder = BaselineModelBuilder()
    results = builder.train_all_baselines(X_train, y_train)

    # Save models
    builder.save_models('models/')

    # Make predictions
    predictions = builder.predict(X_test)
    """)
