"""
Foundation Agent - Data Preprocessing Pipeline
MLE-STAR Workflow - Rossmann Store Sales Prediction

This module implements robust data preprocessing for the Rossmann sales prediction task.
It handles feature engineering, missing value imputation, and data transformation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class RossmannPreprocessor:
    """
    Preprocessing pipeline for Rossmann Store Sales data.
    Handles both training and test datasets with proper feature engineering.
    """

    def __init__(self):
        """Initialize preprocessor with encoders and feature parameters."""
        self.label_encoders = {}
        self.feature_names = None
        self.anchor_date = pd.Timestamp('2015-12-31')

        # Month mapping for promo intervals
        self.MONTH_MAP = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
        }

    def load_data(self, train_path, store_path, test_path=None):
        """
        Load training, store, and optionally test datasets.

        Parameters:
        -----------
        train_path : str
            Path to training CSV file
        store_path : str
            Path to store metadata CSV file
        test_path : str, optional
            Path to test CSV file

        Returns:
        --------
        tuple : (train_df, store_df, test_df or None)
        """
        print("Loading datasets...")
        train_df = pd.read_csv(train_path, low_memory=False)
        store_df = pd.read_csv(store_path)
        test_df = pd.read_csv(test_path) if test_path else None

        print(f"Train shape: {train_df.shape}")
        print(f"Store shape: {store_df.shape}")
        if test_df is not None:
            print(f"Test shape: {test_df.shape}")

        return train_df, store_df, test_df

    def clean_train_data(self, train_df):
        """
        Clean training data - handle data types and inconsistencies.

        Parameters:
        -----------
        train_df : pd.DataFrame
            Raw training data

        Returns:
        --------
        pd.DataFrame : Cleaned training data
        """
        df = train_df.copy()

        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Fix StateHoliday inconsistency (0 vs '0')
        df['StateHoliday'] = df['StateHoliday'].replace(0, '0')

        # Remove records where store is closed (Sales=0 when Open=0)
        # Keep only open stores for training
        df = df[df['Open'] == 1].copy()

        print(f"After removing closed stores: {df.shape[0]} records")

        return df

    def clean_store_data(self, store_df):
        """
        Clean store metadata - handle missing values and engineer Promo2 features.

        Parameters:
        -----------
        store_df : pd.DataFrame
            Raw store metadata

        Returns:
        --------
        pd.DataFrame : Cleaned store data with engineered features
        """
        df = store_df.copy()

        # Fill missing CompetitionDistance with a large value (no nearby competition)
        df['CompetitionDistance'].fillna(df['CompetitionDistance'].max() * 2, inplace=True)

        # Engineer Promo2Rounds feature
        df = self._engineer_promo2_rounds(df)

        # Drop original Promo2 timing columns (replaced by Promo2Rounds)
        cols_to_drop = ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']
        df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

        # Fill missing competition dates with store opening assumption
        df['CompetitionOpenSinceMonth'].fillna(1, inplace=True)
        df['CompetitionOpenSinceYear'].fillna(1900, inplace=True)

        return df

    def _engineer_promo2_rounds(self, store_df):
        """
        Engineer Promo2Rounds feature: count promotional cycles until anchor date.

        Parameters:
        -----------
        store_df : pd.DataFrame
            Store data with Promo2 information

        Returns:
        --------
        pd.DataFrame : Store data with Promo2Rounds feature
        """
        df = store_df.copy()

        # Create Promo2StartDate from year and week
        df['Promo2StartDate'] = [
            self._iso_week_monday(y, w)
            for y, w in zip(df.get('Promo2SinceYear', [np.nan]*len(df)),
                          df.get('Promo2SinceWeek', [np.nan]*len(df)))
        ]

        # Parse PromoInterval
        if 'PromoInterval' in df.columns:
            months_sets = df['PromoInterval'].apply(self._parse_interval)
        else:
            months_sets = [set()] * len(df)

        # Calculate rounds
        df['Promo2Rounds'] = 0
        if 'Promo2' in df.columns:
            mask = df['Promo2'] == 1
            df.loc[mask, 'Promo2Rounds'] = [
                self._count_rounds(start, mset)
                for start, mset in zip(df.loc[mask, 'Promo2StartDate'],
                                      [months_sets[i] for i in df[mask].index])
            ]

        df.drop(columns=['Promo2StartDate'], inplace=True, errors='ignore')

        return df

    def _iso_week_monday(self, year, week):
        """Convert ISO year and week to Monday date."""
        try:
            if pd.isna(year) or pd.isna(week) or int(week) <= 0:
                return pd.NaT
            return pd.Timestamp.fromisocalendar(int(year), int(week), 1)
        except:
            return pd.NaT

    def _parse_interval(self, interval):
        """Parse PromoInterval string to set of month numbers."""
        if pd.isna(interval):
            return set()
        if not isinstance(interval, str):
            interval = str(interval)
        interval = interval.strip()
        if not interval:
            return set()
        return {self.MONTH_MAP.get(m.strip()[:3], None) for m in interval.split(",")} - {None}

    def _count_rounds(self, start_ts, months_set):
        """Count promotional rounds from start_ts to anchor_date."""
        if pd.isna(start_ts) or not months_set:
            return 0
        if self.anchor_date < start_ts:
            return 0

        start_ms = pd.Timestamp(start_ts.year, start_ts.month, 1)
        end_ms = pd.Timestamp(self.anchor_date.year, self.anchor_date.month, 1)
        month_starts = pd.date_range(start=start_ms, end=end_ms, freq='MS')

        return sum((dt.month in months_set) and (dt >= start_ts) for dt in month_starts)

    def engineer_features(self, df, is_train=True):
        """
        Engineer temporal and interaction features.

        Parameters:
        -----------
        df : pd.DataFrame
            Merged dataset with sales and store data
        is_train : bool
            Whether this is training data (affects target handling)

        Returns:
        --------
        pd.DataFrame : Data with engineered features
        """
        df = df.copy()

        # Temporal features from Date
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)

        # Competition duration in months
        df['CompetitionOpenMonths'] = (
            (df['Year'] - df['CompetitionOpenSinceYear']) * 12 +
            (df['Month'] - df['CompetitionOpenSinceMonth'])
        )
        df['CompetitionOpenMonths'] = df['CompetitionOpenMonths'].clip(lower=0)

        # Promo interaction features
        df['PromoInterval'] = df['Promo'] * df['DayOfWeek']
        df['PromoStateHoliday'] = df['Promo'] * (df['StateHoliday'] != '0').astype(int)

        # Store-specific features
        df['SalesPerCustomer'] = 0  # Will be filled during training if available
        if is_train and 'Sales' in df.columns and 'Customers' in df.columns:
            mask = df['Customers'] > 0
            df.loc[mask, 'SalesPerCustomer'] = df.loc[mask, 'Sales'] / df.loc[mask, 'Customers']

        return df

    def encode_categorical(self, df, fit=True):
        """
        Encode categorical variables using label encoding.

        Parameters:
        -----------
        df : pd.DataFrame
            Data with categorical columns
        fit : bool
            If True, fit encoders; if False, use existing encoders

        Returns:
        --------
        pd.DataFrame : Data with encoded categorical variables
        """
        df = df.copy()

        categorical_cols = ['StateHoliday', 'StoreType', 'Assortment']

        for col in categorical_cols:
            if col not in df.columns:
                continue

            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        return df

    def prepare_features(self, df, drop_cols=None):
        """
        Prepare final feature matrix for modeling.

        Parameters:
        -----------
        df : pd.DataFrame
            Fully processed data
        drop_cols : list, optional
            Additional columns to drop

        Returns:
        --------
        pd.DataFrame : Feature matrix ready for modeling
        """
        df = df.copy()

        # Default columns to drop
        default_drop = ['Date', 'Customers']
        if drop_cols:
            default_drop.extend(drop_cols)

        # Drop columns that exist in the dataframe
        cols_to_drop = [col for col in default_drop if col in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)

        return df

    def preprocess(self, train_df, store_df, test_df=None):
        """
        Complete preprocessing pipeline for training and test data.

        Parameters:
        -----------
        train_df : pd.DataFrame
            Raw training data
        store_df : pd.DataFrame
            Raw store metadata
        test_df : pd.DataFrame, optional
            Raw test data

        Returns:
        --------
        tuple : (X_train, y_train, X_test or None, cleaned_store_df)
        """
        print("\n=== Starting Preprocessing Pipeline ===\n")

        # Clean datasets
        print("Step 1: Cleaning datasets...")
        train_clean = self.clean_train_data(train_df)
        store_clean = self.clean_store_data(store_df)

        # Merge train with store
        print("Step 2: Merging train with store data...")
        train_merged = train_clean.merge(store_clean, on='Store', how='left')

        # Engineer features
        print("Step 3: Engineering features...")
        train_featured = self.engineer_features(train_merged, is_train=True)

        # Encode categorical
        print("Step 4: Encoding categorical variables...")
        train_encoded = self.encode_categorical(train_featured, fit=True)

        # Extract target
        y_train = train_encoded['Sales'].values

        # Prepare feature matrix
        print("Step 5: Preparing feature matrix...")
        X_train = self.prepare_features(train_encoded, drop_cols=['Sales'])

        self.feature_names = X_train.columns.tolist()

        # Process test data if provided
        X_test = None
        if test_df is not None:
            print("\nStep 6: Processing test data...")
            test_df['Date'] = pd.to_datetime(test_df['Date'])
            test_df['StateHoliday'] = test_df['StateHoliday'].replace(0, '0')

            test_merged = test_df.merge(store_clean, on='Store', how='left')
            test_featured = self.engineer_features(test_merged, is_train=False)
            test_encoded = self.encode_categorical(test_featured, fit=False)
            X_test = self.prepare_features(test_encoded, drop_cols=['Id'])

            # Align test features with train features
            for col in self.feature_names:
                if col not in X_test.columns:
                    X_test[col] = 0
            X_test = X_test[self.feature_names]

        print(f"\n=== Preprocessing Complete ===")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Number of features: {X_train.shape[1]}")
        if X_test is not None:
            print(f"Test samples: {X_test.shape[0]}")

        return X_train, y_train, X_test, store_clean


if __name__ == "__main__":
    # Example usage
    preprocessor = RossmannPreprocessor()

    # Load data
    train_df, store_df, test_df = preprocessor.load_data(
        train_path='data/train.csv',
        store_path='data/store.csv',
        test_path='data/test.csv'
    )

    # Preprocess
    X_train, y_train, X_test, store_clean = preprocessor.preprocess(
        train_df, store_df, test_df
    )

    print("\nFeature names:")
    print(preprocessor.feature_names)
    print(f"\nTarget statistics:")
    print(f"Mean: {y_train.mean():.2f}")
    print(f"Std: {y_train.std():.2f}")
    print(f"Min: {y_train.min():.2f}")
    print(f"Max: {y_train.max():.2f}")
