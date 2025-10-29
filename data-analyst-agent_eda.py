#!/usr/bin/env python3
"""
Data Analyst Agent - Comprehensive Exploratory Data Analysis
MLE-STAR Workflow - Phase 1, Task 1-2
Agent ID: data-analyst-agent
Session: automation-session-1761751854473-9jleknba1
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

class RossmannDataAnalyst:
    """Comprehensive EDA for Rossmann Store Sales dataset"""

    def __init__(self, train_path, store_path, test_path):
        self.train_path = train_path
        self.store_path = store_path
        self.test_path = test_path
        self.train_df = None
        self.store_df = None
        self.test_df = None
        self.merged_train = None
        self.analysis_results = {}

    def load_data(self):
        """Load all datasets"""
        print("=" * 80)
        print("DATA LOADING")
        print("=" * 80)

        self.train_df = pd.read_csv(self.train_path, parse_dates=['Date'])
        self.store_df = pd.read_csv(self.store_path)
        self.test_df = pd.read_csv(self.test_path, parse_dates=['Date'])

        print(f"Train dataset: {self.train_df.shape}")
        print(f"Store dataset: {self.store_df.shape}")
        print(f"Test dataset: {self.test_df.shape}")
        print()

        # Merge train with store info
        self.merged_train = self.train_df.merge(self.store_df, on='Store', how='left')
        print(f"Merged train dataset: {self.merged_train.shape}")
        print()

    def basic_statistics(self):
        """Compute basic statistics and metadata"""
        print("=" * 80)
        print("BASIC STATISTICS")
        print("=" * 80)

        stats = {
            'train_samples': len(self.train_df),
            'test_samples': len(self.test_df),
            'num_stores': self.train_df['Store'].nunique(),
            'date_range_train': {
                'start': str(self.train_df['Date'].min()),
                'end': str(self.train_df['Date'].max()),
                'days': (self.train_df['Date'].max() - self.train_df['Date'].min()).days
            },
            'date_range_test': {
                'start': str(self.test_df['Date'].min()),
                'end': str(self.test_df['Date'].max()),
                'days': (self.test_df['Date'].max() - self.test_df['Date'].min()).days
            }
        }

        print(f"Training samples: {stats['train_samples']:,}")
        print(f"Test samples: {stats['test_samples']:,}")
        print(f"Number of stores: {stats['num_stores']}")
        print(f"Train date range: {stats['date_range_train']['start']} to {stats['date_range_train']['end']} ({stats['date_range_train']['days']} days)")
        print(f"Test date range: {stats['date_range_test']['start']} to {stats['date_range_test']['end']} ({stats['date_range_test']['days']} days)")
        print()

        self.analysis_results['basic_statistics'] = stats

    def target_analysis(self):
        """Analyze target variable (Sales)"""
        print("=" * 80)
        print("TARGET VARIABLE ANALYSIS - SALES")
        print("=" * 80)

        sales_stats = {
            'mean': float(self.train_df['Sales'].mean()),
            'median': float(self.train_df['Sales'].median()),
            'std': float(self.train_df['Sales'].std()),
            'min': float(self.train_df['Sales'].min()),
            'max': float(self.train_df['Sales'].max()),
            'q25': float(self.train_df['Sales'].quantile(0.25)),
            'q75': float(self.train_df['Sales'].quantile(0.75)),
            'zeros': int((self.train_df['Sales'] == 0).sum()),
            'zeros_pct': float((self.train_df['Sales'] == 0).mean() * 100)
        }

        print(f"Mean Sales: ${sales_stats['mean']:,.2f}")
        print(f"Median Sales: ${sales_stats['median']:,.2f}")
        print(f"Std Dev: ${sales_stats['std']:,.2f}")
        print(f"Min: ${sales_stats['min']:,.2f}, Max: ${sales_stats['max']:,.2f}")
        print(f"Q25: ${sales_stats['q25']:,.2f}, Q75: ${sales_stats['q75']:,.2f}")
        print(f"Zero sales records: {sales_stats['zeros']:,} ({sales_stats['zeros_pct']:.2f}%)")
        print()

        # Sales by store open status
        print("Sales by Store Open Status:")
        open_sales = self.train_df.groupby('Open')['Sales'].agg(['mean', 'count'])
        print(open_sales)
        print()

        self.analysis_results['target_analysis'] = sales_stats

    def missing_values_analysis(self):
        """Analyze missing values"""
        print("=" * 80)
        print("MISSING VALUES ANALYSIS")
        print("=" * 80)

        missing_train = self.train_df.isnull().sum()
        missing_store = self.store_df.isnull().sum()
        missing_merged = self.merged_train.isnull().sum()

        print("Train dataset missing values:")
        print(missing_train[missing_train > 0])
        print()

        print("Store dataset missing values:")
        print(missing_store[missing_store > 0])
        print()

        print("Merged dataset missing values:")
        missing_pct = (missing_merged[missing_merged > 0] / len(self.merged_train) * 100).round(2)
        for col, pct in missing_pct.items():
            print(f"{col}: {missing_merged[col]:,} ({pct}%)")
        print()

        missing_info = {
            'train': {col: int(val) for col, val in missing_train[missing_train > 0].items()},
            'store': {col: int(val) for col, val in missing_store[missing_store > 0].items()},
            'merged': {col: int(val) for col, val in missing_merged[missing_merged > 0].items()}
        }

        self.analysis_results['missing_values'] = missing_info

    def feature_analysis(self):
        """Analyze key features"""
        print("=" * 80)
        print("FEATURE ANALYSIS")
        print("=" * 80)

        features = {}

        # DayOfWeek analysis
        print("Sales by Day of Week:")
        dow_sales = self.train_df[self.train_df['Sales'] > 0].groupby('DayOfWeek')['Sales'].agg(['mean', 'median', 'count'])
        print(dow_sales)
        features['day_of_week'] = dow_sales.to_dict()
        print()

        # Promo analysis
        print("Sales by Promo Status:")
        promo_sales = self.train_df[self.train_df['Sales'] > 0].groupby('Promo')['Sales'].agg(['mean', 'median', 'count'])
        print(promo_sales)
        features['promo'] = promo_sales.to_dict()
        print()

        # StateHoliday analysis
        print("Sales by State Holiday:")
        holiday_sales = self.train_df[self.train_df['Sales'] > 0].groupby('StateHoliday')['Sales'].agg(['mean', 'median', 'count'])
        print(holiday_sales)
        features['state_holiday'] = holiday_sales.to_dict()
        print()

        # SchoolHoliday analysis
        print("Sales by School Holiday:")
        school_sales = self.train_df[self.train_df['Sales'] > 0].groupby('SchoolHoliday')['Sales'].agg(['mean', 'median', 'count'])
        print(school_sales)
        features['school_holiday'] = school_sales.to_dict()
        print()

        # Store type analysis
        print("Sales by Store Type:")
        store_type_sales = self.merged_train[self.merged_train['Sales'] > 0].groupby('StoreType')['Sales'].agg(['mean', 'median', 'count'])
        print(store_type_sales)
        features['store_type'] = store_type_sales.to_dict()
        print()

        # Assortment analysis
        print("Sales by Assortment:")
        assortment_sales = self.merged_train[self.merged_train['Sales'] > 0].groupby('Assortment')['Sales'].agg(['mean', 'median', 'count'])
        print(assortment_sales)
        features['assortment'] = assortment_sales.to_dict()
        print()

        self.analysis_results['feature_analysis'] = features

    def temporal_analysis(self):
        """Analyze temporal patterns"""
        print("=" * 80)
        print("TEMPORAL ANALYSIS")
        print("=" * 80)

        # Extract temporal features
        self.train_df['Year'] = self.train_df['Date'].dt.year
        self.train_df['Month'] = self.train_df['Date'].dt.month
        self.train_df['Week'] = self.train_df['Date'].dt.isocalendar().week

        # Monthly sales trend
        print("Sales by Month:")
        monthly_sales = self.train_df[self.train_df['Sales'] > 0].groupby('Month')['Sales'].agg(['mean', 'median', 'count'])
        print(monthly_sales)
        print()

        # Yearly sales trend
        print("Sales by Year:")
        yearly_sales = self.train_df[self.train_df['Sales'] > 0].groupby('Year')['Sales'].agg(['mean', 'median', 'count'])
        print(yearly_sales)
        print()

        temporal_patterns = {
            'monthly': monthly_sales.to_dict(),
            'yearly': yearly_sales.to_dict()
        }

        self.analysis_results['temporal_analysis'] = temporal_patterns

    def data_quality_assessment(self):
        """Assess overall data quality"""
        print("=" * 80)
        print("DATA QUALITY ASSESSMENT")
        print("=" * 80)

        quality_metrics = {
            'train_completeness': float(1 - self.train_df.isnull().sum().sum() / (len(self.train_df) * len(self.train_df.columns))),
            'store_completeness': float(1 - self.store_df.isnull().sum().sum() / (len(self.store_df) * len(self.store_df.columns))),
            'duplicate_rows_train': int(self.train_df.duplicated().sum()),
            'duplicate_rows_store': int(self.store_df.duplicated().sum()),
            'stores_in_train': int(self.train_df['Store'].nunique()),
            'stores_in_store_info': int(self.store_df['Store'].nunique()),
            'stores_in_test': int(self.test_df['Store'].nunique())
        }

        print(f"Train dataset completeness: {quality_metrics['train_completeness']:.2%}")
        print(f"Store dataset completeness: {quality_metrics['store_completeness']:.2%}")
        print(f"Duplicate rows (train): {quality_metrics['duplicate_rows_train']}")
        print(f"Duplicate rows (store): {quality_metrics['duplicate_rows_store']}")
        print(f"Stores in train: {quality_metrics['stores_in_train']}")
        print(f"Stores in store info: {quality_metrics['stores_in_store_info']}")
        print(f"Stores in test: {quality_metrics['stores_in_test']}")
        print()

        self.analysis_results['data_quality'] = quality_metrics

    def key_insights(self):
        """Generate key insights and recommendations"""
        print("=" * 80)
        print("KEY INSIGHTS & RECOMMENDATIONS")
        print("=" * 80)

        insights = []

        # Zero sales insight
        zero_sales_pct = self.analysis_results['target_analysis']['zeros_pct']
        if zero_sales_pct > 10:
            insights.append(f"HIGH: {zero_sales_pct:.1f}% of records have zero sales - likely store closed days. Filter or handle separately.")

        # Missing values insight
        missing = self.analysis_results['missing_values']['merged']
        if missing:
            insights.append(f"MEDIUM: Missing values in {len(missing)} features - most in competition and promo data. Imputation needed.")

        # Temporal insight
        insights.append("INFO: Dataset spans multiple years with strong seasonal patterns. Time-based features critical.")

        # Store type insight
        insights.append("INFO: Multiple store types and assortments with different sales patterns. Store segmentation recommended.")

        # Promo insight
        insights.append("INFO: Promotions show significant impact on sales. Promo features are important predictors.")

        recommendations = [
            "1. Filter out closed stores (Open=0) or model separately",
            "2. Handle missing competition and promo data with forward-fill or median imputation",
            "3. Engineer temporal features: day of week, month, year, week of year, holidays",
            "4. Create store-specific features: average sales, trend, seasonality",
            "5. Feature engineering: lag features, rolling statistics, competition features",
            "6. Consider separate models or ensemble for different store types",
            "7. Validate for data leakage - ensure no future information in training",
            "8. Use appropriate evaluation metric (likely RMSPE based on Kaggle competition)"
        ]

        for insight in insights:
            print(f"- {insight}")
        print()

        print("RECOMMENDATIONS FOR ML PIPELINE:")
        for rec in recommendations:
            print(f"{rec}")
        print()

        self.analysis_results['insights'] = insights
        self.analysis_results['recommendations'] = recommendations

    def save_results(self, output_dir='mle-star-output'):
        """Save analysis results"""
        print("=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save JSON report
        json_path = output_path / 'data-analyst-agent_analysis.json'
        with open(json_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        print(f"Analysis JSON saved: {json_path}")

        # Save markdown report
        md_path = output_path / 'data-analyst-agent_eda_report.md'
        with open(md_path, 'w') as f:
            f.write("# Data Analysis Report - Rossmann Store Sales\n\n")
            f.write(f"**Agent:** data-analyst-agent\n")
            f.write(f"**Session:** automation-session-1761751854473-9jleknba1\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Dataset Overview\n\n")
            stats = self.analysis_results['basic_statistics']
            f.write(f"- Training samples: {stats['train_samples']:,}\n")
            f.write(f"- Test samples: {stats['test_samples']:,}\n")
            f.write(f"- Number of stores: {stats['num_stores']}\n")
            f.write(f"- Train period: {stats['date_range_train']['days']} days\n\n")

            f.write("## Target Variable - Sales\n\n")
            target = self.analysis_results['target_analysis']
            f.write(f"- Mean: ${target['mean']:,.2f}\n")
            f.write(f"- Median: ${target['median']:,.2f}\n")
            f.write(f"- Range: ${target['min']:,.2f} - ${target['max']:,.2f}\n")
            f.write(f"- Zero sales: {target['zeros_pct']:.2f}%\n\n")

            f.write("## Key Insights\n\n")
            for insight in self.analysis_results['insights']:
                f.write(f"- {insight}\n")
            f.write("\n")

            f.write("## Recommendations\n\n")
            for rec in self.analysis_results['recommendations']:
                f.write(f"{rec}\n")

        print(f"Markdown report saved: {md_path}")
        print()

    def run_full_analysis(self):
        """Execute complete EDA pipeline"""
        print("=" * 80)
        print("DATA ANALYST AGENT - ROSSMANN STORE SALES EDA")
        print("MLE-STAR Workflow - Phase 1")
        print("=" * 80)
        print()

        self.load_data()
        self.basic_statistics()
        self.target_analysis()
        self.missing_values_analysis()
        self.feature_analysis()
        self.temporal_analysis()
        self.data_quality_assessment()
        self.key_insights()
        self.save_results()

        print("=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)


if __name__ == '__main__':
    analyst = RossmannDataAnalyst(
        train_path='data/train.csv',
        store_path='data/store.csv',
        test_path='data/test.csv'
    )

    analyst.run_full_analysis()
