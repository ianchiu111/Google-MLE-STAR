"""
Foundation Agent - Complete Baseline Runner
MLE-STAR Workflow - Rossmann Store Sales Prediction

This script orchestrates the complete foundation phase:
1. Data loading and preprocessing
2. Baseline model training
3. Performance evaluation and comparison
4. Model persistence and reporting
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

from foundation_agent_preprocessing import RossmannPreprocessor
from foundation_agent_baseline_models import BaselineModelBuilder


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def run_foundation_baseline(
    train_path='data/train.csv',
    store_path='data/store.csv',
    test_path='data/test.csv',
    output_dir='models/',
    validation_split=0.2,
    random_state=42
):
    """
    Execute complete foundation baseline workflow.

    Parameters:
    -----------
    train_path : str
        Path to training data
    store_path : str
        Path to store metadata
    test_path : str
        Path to test data
    output_dir : str
        Directory for model outputs
    validation_split : float
        Validation set proportion
    random_state : int
        Random seed

    Returns:
    --------
    dict : Complete results including models, metrics, and predictions
    """
    start_time = datetime.now()

    print_header("MLE-STAR FOUNDATION AGENT - BASELINE EXECUTION")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {random_state}")

    # ========================================================================
    # PHASE 1: DATA PREPROCESSING
    # ========================================================================
    print_header("PHASE 1: DATA PREPROCESSING")

    preprocessor = RossmannPreprocessor()

    # Load data
    train_df, store_df, test_df = preprocessor.load_data(
        train_path=train_path,
        store_path=store_path,
        test_path=test_path
    )

    # Preprocess
    X_train, y_train, X_test, store_clean = preprocessor.preprocess(
        train_df, store_df, test_df
    )

    print(f"\n✓ Preprocessing complete")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Training samples: {X_train.shape[0]:,}")
    if X_test is not None:
        print(f"  Test samples: {X_test.shape[0]:,}")

    # ========================================================================
    # PHASE 2: BASELINE MODEL TRAINING
    # ========================================================================
    print_header("PHASE 2: BASELINE MODEL TRAINING")

    builder = BaselineModelBuilder(random_state=random_state)
    results = builder.train_all_baselines(X_train, y_train, test_size=validation_split)

    # ========================================================================
    # PHASE 3: MODEL PERSISTENCE
    # ========================================================================
    print_header("PHASE 3: MODEL PERSISTENCE")

    os.makedirs(output_dir, exist_ok=True)
    builder.save_models(output_dir=output_dir)

    # Save preprocessor
    import pickle
    preprocessor_path = os.path.join(output_dir, 'foundation_agent_preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Saved preprocessor: {preprocessor_path}")

    # ========================================================================
    # PHASE 4: PREDICTIONS (if test data available)
    # ========================================================================
    test_predictions = None
    if X_test is not None:
        print_header("PHASE 4: TEST PREDICTIONS")
        test_predictions = builder.predict(X_test)
        print(f"Generated {len(test_predictions):,} predictions")
        print(f"Prediction stats:")
        print(f"  Mean: {test_predictions.mean():.2f}")
        print(f"  Std: {test_predictions.std():.2f}")
        print(f"  Min: {test_predictions.min():.2f}")
        print(f"  Max: {test_predictions.max():.2f}")

        # Save predictions
        if 'Id' in test_df.columns:
            submission = pd.DataFrame({
                'Id': test_df['Id'],
                'Sales': test_predictions
            })
            submission_path = os.path.join(output_dir, 'foundation_agent_predictions.csv')
            submission.to_csv(submission_path, index=False)
            print(f"Saved predictions: {submission_path}")

    # ========================================================================
    # PHASE 5: SUMMARY REPORT
    # ========================================================================
    print_header("PHASE 5: EXECUTION SUMMARY")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n✓ Foundation baseline workflow complete")
    print(f"  Duration: {duration:.1f} seconds")
    print(f"  Best model: {builder.best_model_name}")
    print(f"  Best RMSPE: {results[builder.best_model_name]['val_rmspe']:.4f}")
    print(f"  Output directory: {output_dir}")

    # Generate summary report
    summary = {
        'execution_time': {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'duration_seconds': duration
        },
        'data_info': {
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]) if X_test is not None else 0,
            'num_features': int(X_train.shape[1]),
            'feature_names': preprocessor.feature_names,
            'target_stats': {
                'mean': float(y_train.mean()),
                'std': float(y_train.std()),
                'min': float(y_train.min()),
                'max': float(y_train.max())
            }
        },
        'models': results,
        'best_model': {
            'name': builder.best_model_name,
            'metrics': results[builder.best_model_name]
        },
        'random_state': random_state
    }

    # Save summary report
    import json
    report_path = os.path.join(output_dir, 'foundation_agent_summary.json')
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n📄 Summary report: {report_path}")

    print_header("FOUNDATION AGENT COMPLETE")

    return {
        'preprocessor': preprocessor,
        'builder': builder,
        'results': results,
        'summary': summary,
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'test_predictions': test_predictions
    }


if __name__ == "__main__":
    """
    Run foundation baseline workflow from command line.

    Usage:
        python foundation_agent_runner.py
        python foundation_agent_runner.py --output models/baseline_v1/
    """
    import argparse

    parser = argparse.ArgumentParser(description='Foundation Agent - Baseline Workflow')
    parser.add_argument('--train', default='data/train.csv', help='Training data path')
    parser.add_argument('--store', default='data/store.csv', help='Store data path')
    parser.add_argument('--test', default='data/test.csv', help='Test data path')
    parser.add_argument('--output', default='models/', help='Output directory')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Execute workflow
    results = run_foundation_baseline(
        train_path=args.train,
        store_path=args.store,
        test_path=args.test,
        output_dir=args.output,
        validation_split=args.val_split,
        random_state=args.random_state
    )

    print("\n✓ Workflow complete. Results available in:", args.output)
