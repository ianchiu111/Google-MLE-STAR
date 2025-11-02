"""
Foundation Agent - Data Preprocessing Pipeline
Rossmann Store Sales Prediction
Author: Foundation Agent (MLE-STAR Workflow)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder


class RossmannPreprocessor:
    """
    Preprocessing pipeline for Rossmann Store Sales dataset.
    Implements feature engineering based on MLE-STAR research insights.
    """

    def __init__(self):
        self.label_encoders = {}
        self.feature_stats = {}

    def load_data(self, train_path, test_path, store_path):
        """Load raw data files"""
        print("Loading data files...")
        self.train = pd.read_csv(train_path, low_memory=False)
        self.test = pd.read_csv(test_path, low_memory=False)
        self.store = pd.read_csv(store_path)

        # Store test IDs for submission
        self.test_ids = self.test['Id'].copy()

        print(f"Train shape: {self.train.shape}")
        print(f"Test shape: {self.test.shape}")
        print(f"Store shape: {self.store.shape}")

        return self

    def clean_data(self):
        """Clean and fix data type issues"""
        print("\nCleaning data...")

        # Fix StateHoliday mixed types (0 and '0')
        for df in [self.train, self.test]:
            df['StateHoliday'] = df['StateHoliday'].replace(0, '0')
            df['Date'] = pd.to_datetime(df['Date'])

        # Handle store data missing values
        # CompetitionDistance: fill with large value (no nearby competition)
        self.store['CompetitionDistance'].fillna(
            self.store['CompetitionDistance'].max() * 1.5, inplace=True
        )

        # Competition open since: fill with store opening (no competition history)
        self.store['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        self.store['CompetitionOpenSinceYear'].fillna(0, inplace=True)

        # Promo2 related: fill with 0 (no extended promotion)
        self.store['Promo2SinceWeek'].fillna(0, inplace=True)
        self.store['Promo2SinceYear'].fillna(0, inplace=True)
        self.store['PromoInterval'].fillna('None', inplace=True)

        print("Data cleaning complete")
        return self

    def engineer_features(self, df):
        """
        Create features based on domain knowledge and research insights.
        Focus: Feature engineering over model complexity (50% time allocation)
        """
        print("\nEngineering features...")

        # Merge with store information
        df = df.merge(self.store, on='Store', how='left')

        # === Temporal Features ===
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['Quarter'] = df['Date'].dt.quarter

        # Day of month features (for paydays, beginning/end of month effects)
        df['DayOfMonth'] = df['Date'].dt.day
        df['IsMonthStart'] = (df['Date'].dt.is_month_start).astype(int)
        df['IsMonthEnd'] = (df['Date'].dt.is_month_end).astype(int)
        df['DaysInMonth'] = df['Date'].dt.days_in_month

        # === Competition Features ===
        # How long has competition been open (in months)
        df['CompetitionOpenMonths'] = np.maximum(
            0,
            12 * (df['Year'] - df['CompetitionOpenSinceYear']) +
            (df['Month'] - df['CompetitionOpenSinceMonth'])
        )
        df['CompetitionOpenMonths'] = df['CompetitionOpenMonths'].fillna(0)

        # Competition intensity feature
        df['CompetitionIntensity'] = df['CompetitionOpenMonths'] / (df['CompetitionDistance'] + 1)

        # === Promo2 Features ===
        # Calculate months since Promo2 started
        df['Promo2OpenMonths'] = np.maximum(
            0,
            12 * (df['Year'] - df['Promo2SinceYear']) +
            (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
        )
        df['Promo2OpenMonths'] = df['Promo2OpenMonths'].fillna(0)

        # Is the current month in PromoInterval?
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df['MonthStr'] = df['Month'].map(month_map)
        df['IsPromoMonth'] = df.apply(
            lambda x: int(x['MonthStr'] in str(x['PromoInterval'])) if x['Promo2'] == 1 else 0,
            axis=1
        )

        # === Store Features ===
        # Combine store characteristics
        df['StoreAssortment'] = df['StoreType'].astype(str) + '_' + df['Assortment'].astype(str)

        # === Holiday Features ===
        df['IsHoliday'] = ((df['StateHoliday'] != '0') | (df['SchoolHoliday'] == 1)).astype(int)

        # === Rolling and Aggregation Features (only for train) ===
        if 'Sales' in df.columns:
            # Store-level aggregations
            df = self._add_store_aggregations(df)

        # Drop temporary columns
        df.drop(['MonthStr', 'Date'], axis=1, inplace=True, errors='ignore')

        print(f"Feature engineering complete. Total features: {df.shape[1]}")
        return df

    def _add_store_aggregations(self, df):
        """Add store-level historical aggregations (train only)"""
        # Sort by store and date
        df_sorted = df.sort_values(['Store', 'Year', 'Month', 'Day'])

        # Store average sales and customers
        store_stats = df_sorted.groupby('Store').agg({
            'Sales': ['mean', 'median', 'std'],
            'Customers': ['mean', 'median']
        }).reset_index()

        store_stats.columns = ['Store', 'Store_Sales_Mean', 'Store_Sales_Median',
                               'Store_Sales_Std', 'Store_Customers_Mean', 'Store_Customers_Median']

        # Store these stats for use in test set
        self.feature_stats['store_stats'] = store_stats

        df = df.merge(store_stats, on='Store', how='left')

        # Fill any NaN in std with 0
        df['Store_Sales_Std'].fillna(0, inplace=True)

        return df

    def encode_categorical(self, df):
        """Encode categorical variables"""
        print("\nEncoding categorical features...")

        categorical_cols = ['StoreType', 'Assortment', 'StateHoliday',
                           'StoreAssortment', 'PromoInterval']

        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # For test set, handle unseen labels
                    df[col] = df[col].astype(str).map(
                        lambda x: self.label_encoders[col].transform([x])[0]
                        if x in self.label_encoders[col].classes_ else -1
                    )

        print("Categorical encoding complete")
        return df

    def prepare_train_data(self):
        """Prepare training dataset"""
        print("\n" + "="*50)
        print("PREPARING TRAINING DATA")
        print("="*50)

        # Filter out closed stores (Sales = 0 when Open = 0)
        self.train = self.train[self.train['Open'] == 1].copy()
        print(f"After filtering closed stores: {self.train.shape}")

        # Engineer features
        self.train = self.engineer_features(self.train)

        # Encode categorical
        self.train = self.encode_categorical(self.train)

        # Separate features and target
        # Remove non-feature columns
        drop_cols = ['Sales', 'Customers', 'Open', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']

        self.X_train = self.train.drop(columns=drop_cols, errors='ignore')
        self.y_train = self.train['Sales']

        print(f"\nFinal training features: {self.X_train.shape}")
        print(f"Target shape: {self.y_train.shape}")

        return self.X_train, self.y_train

    def prepare_test_data(self):
        """Prepare test dataset"""
        print("\n" + "="*50)
        print("PREPARING TEST DATA")
        print("="*50)

        # Engineer features
        self.test = self.engineer_features(self.test)

        # Add store stats from training data if available
        if 'store_stats' in self.feature_stats:
            self.test = self.test.merge(
                self.feature_stats['store_stats'],
                on='Store',
                how='left'
            )
            # Fill missing with 0
            for col in ['Store_Sales_Mean', 'Store_Sales_Median', 'Store_Sales_Std',
                       'Store_Customers_Mean', 'Store_Customers_Median']:
                if col in self.test.columns:
                    self.test[col].fillna(0, inplace=True)

        # Encode categorical
        self.test = self.encode_categorical(self.test)

        # Remove non-feature columns
        drop_cols = ['Id', 'Open', 'Customers', 'CompetitionOpenSinceMonth',
                     'CompetitionOpenSinceYear', 'Promo2SinceWeek', 'Promo2SinceYear']

        self.X_test = self.test.drop(columns=drop_cols, errors='ignore')

        # Ensure test has same columns as train
        missing_cols = set(self.X_train.columns) - set(self.X_test.columns)
        for col in missing_cols:
            self.X_test[col] = 0

        # Reorder columns to match training
        self.X_test = self.X_test[self.X_train.columns]

        print(f"\nFinal test features: {self.X_test.shape}")

        return self.X_test

    def get_feature_names(self):
        """Return list of feature names"""
        return self.X_train.columns.tolist()


def main():
    """Main preprocessing pipeline execution"""
    print("="*70)
    print("FOUNDATION AGENT - DATA PREPROCESSING PIPELINE")
    print("="*70)

    # Initialize preprocessor
    preprocessor = RossmannPreprocessor()

    # Load data
    preprocessor.load_data(
        train_path='data/train.csv',
        test_path='data/test.csv',
        store_path='data/store.csv'
    )

    # Clean data
    preprocessor.clean_data()

    # Prepare datasets
    X_train, y_train = preprocessor.prepare_train_data()
    X_test = preprocessor.prepare_test_data()

    # Save processed data
    print("\n" + "="*50)
    print("SAVING PROCESSED DATA")
    print("="*50)

    X_train.to_csv('models/foundation_agent_X_train.csv', index=False)
    y_train.to_csv('models/foundation_agent_y_train.csv', index=False)
    X_test.to_csv('models/foundation_agent_X_test.csv', index=False)

    print(f"✓ Saved: models/foundation_agent_X_train.csv")
    print(f"✓ Saved: models/foundation_agent_y_train.csv")
    print(f"✓ Saved: models/foundation_agent_X_test.csv")

    # Save test IDs
    pd.DataFrame({'Id': preprocessor.test_ids}).to_csv(
        'models/foundation_agent_test_ids.csv', index=False
    )
    print(f"✓ Saved: models/foundation_agent_test_ids.csv")

    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)

    return preprocessor, X_train, y_train, X_test


if __name__ == "__main__":
    preprocessor, X_train, y_train, X_test = main()
