"""
Data Preprocessor for the Insider Trading Anomaly Detection System.

This module handles loading, validating, cleaning, and preprocessing
raw insider trading transaction data from CSV files.

Classes:
    DataPreprocessor: Main class for data preprocessing operations
    DataLoadException: Custom exception for data loading errors
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import constants
from .utils import (
    setup_logging,
    validate_dataframe,
    DataQualityReport,
    parse_date
)


class DataLoadException(Exception):
    """Custom exception for data loading errors."""
    pass


class DataPreprocessor:
    """
    Preprocessor for raw insider trading transaction data.

    This class provides methods for:
    - Loading and validating CSV data
    - Cleaning and transforming data
    - Filtering by date range and executive
    - Generating summary statistics

    Attributes:
        filepath: Path to the loaded data file
        df: Cleaned pandas DataFrame
        raw_df: Original DataFrame before cleaning
        quality_report: Data quality report
    """

    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.logger = setup_logging(__name__)
        self.filepath: Optional[Path] = None
        self.df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.quality_report: Optional[DataQualityReport] = None
        self._cleaning_log: List[str] = []

    def load_raw_data(
        self,
        filepath: Union[str, Path],
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load CSV data and parse transaction dates.

        Args:
            filepath: Path to CSV file
            validate: Whether to validate required columns

        Returns:
            Cleaned DataFrame

        Raises:
            DataLoadException: If file cannot be loaded or validation fails
        """
        self.logger.info(f"Loading data from {filepath}")
        filepath = Path(filepath)

        if not filepath.exists():
            raise DataLoadException(f"File not found: {filepath}")

        try:
            # Load CSV
            self.raw_df = pd.read_csv(filepath)
            self.filepath = filepath
            self.logger.info(f"Loaded {len(self.raw_df)} rows from CSV")

            # Validate columns
            if validate:
                is_valid, missing = validate_dataframe(
                    self.raw_df,
                    constants.REQUIRED_COLUMNS,
                    raise_on_error=False
                )
                if not is_valid:
                    raise DataLoadException(
                        f"Missing required columns: {missing}"
                    )

            # Clean and process data
            self.df = self._clean_data(self.raw_df.copy())

            # Generate quality report
            self.quality_report = DataQualityReport(self.df)
            self.logger.info(f"Data cleaning complete. {len(self.df)} rows retained")

            return self.df

        except Exception as e:
            if isinstance(e, DataLoadException):
                raise
            raise DataLoadException(f"Error loading data: {str(e)}")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform raw data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        self._cleaning_log = []
        initial_count = len(df)

        # Parse transaction_date as datetime
        try:
            df['transaction_date'] = pd.to_datetime(
                df['transaction_date'],
                errors='coerce'
            )
            invalid_dates = df['transaction_date'].isna().sum()
            if invalid_dates > 0:
                self._cleaning_log.append(
                    f"Removed {invalid_dates} rows with invalid dates"
                )
                df = df.dropna(subset=['transaction_date'])
        except Exception as e:
            self.logger.error(f"Error parsing dates: {e}")
            raise DataLoadException(f"Error parsing transaction_date: {e}")

        # Validate shares > 0
        if 'shares' in df.columns:
            df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
            invalid_shares = df['shares'].isna().sum() + (df['shares'] <= 0).sum()
            if invalid_shares > 0:
                self._cleaning_log.append(
                    f"Removed {invalid_shares} rows with invalid shares"
                )
                df = df[df['shares'] > 0]

        # Standardize acquisition_or_disposal values
        if 'acquisition_or_disposal' in df.columns:
            df['acquisition_or_disposal'] = df['acquisition_or_disposal'].apply(
                self._standardize_transaction_type
            )
            invalid_types = df['acquisition_or_disposal'].isna().sum()
            if invalid_types > 0:
                self._cleaning_log.append(
                    f"Removed {invalid_types} rows with invalid transaction types"
                )
                df = df.dropna(subset=['acquisition_or_disposal'])

        # Clean string columns
        string_columns = ['executive', 'executive_title', 'security_type']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # Replace empty strings and 'nan' with np.nan
                df[col] = df[col].replace(['', 'nan', 'None'], np.nan)

        # Remove rows with missing critical values
        critical_columns = ['transaction_date', 'shares']
        missing_critical = df[critical_columns].isna().any(axis=1).sum()
        if missing_critical > 0:
            self._cleaning_log.append(
                f"Removed {missing_critical} rows with missing critical values"
            )
            df = df.dropna(subset=critical_columns)

        # Sort by transaction_date
        df = df.sort_values('transaction_date').reset_index(drop=True)

        # Log cleaning summary
        final_count = len(df)
        removed = initial_count - final_count
        if removed > 0:
            self.logger.info(
                f"Cleaned data: {removed} rows removed ({removed/initial_count*100:.1f}%)"
            )
        for log_entry in self._cleaning_log:
            self.logger.info(f"  - {log_entry}")

        return df

    def _standardize_transaction_type(self, value: str) -> Optional[str]:
        """
        Standardize transaction type to 'Buy' or 'Sell'.

        Args:
            value: Raw transaction type value

        Returns:
            'Buy', 'Sell', or None if invalid
        """
        if pd.isna(value):
            return None

        value_upper = str(value).strip().upper()

        if value_upper in ['BUY', 'A', 'ACQUISITION']:
            return 'Buy'
        elif value_upper in ['SELL', 'D', 'DISPOSAL']:
            return 'Sell'

        return None

    def get_date_range(self) -> Tuple[datetime, datetime]:
        """
        Get the date range of the loaded data.

        Returns:
            Tuple of (start_date, end_date)

        Raises:
            ValueError: If no data is loaded
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        start_date = self.df['transaction_date'].min()
        end_date = self.df['transaction_date'].max()

        return start_date.to_pydatetime(), end_date.to_pydatetime()

    def filter_by_date_range(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Filter data by date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            inplace: Whether to modify the internal DataFrame

        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        df = self.df.copy()

        if start_date is not None:
            start_date = parse_date(start_date)
            df = df[df['transaction_date'] >= start_date]

        if end_date is not None:
            end_date = parse_date(end_date)
            df = df[df['transaction_date'] <= end_date]

        self.logger.info(
            f"Filtered to {len(df)} rows (date range: {start_date} to {end_date})"
        )

        if inplace:
            self.df = df

        return df

    def filter_by_executive(
        self,
        executive_name: str,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Filter data for a specific executive.

        Args:
            executive_name: Name of the executive to filter for
            inplace: Whether to modify the internal DataFrame

        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        df = self.df[
            self.df['executive'].str.lower() == executive_name.lower()
        ]

        self.logger.info(
            f"Filtered to {len(df)} rows for executive: {executive_name}"
        )

        if inplace:
            self.df = df

        return df

    def filter_by_security(
        self,
        security_type: str,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Filter data for a specific security type.

        Args:
            security_type: Security type to filter for
            inplace: Whether to modify the internal DataFrame

        Returns:
            Filtered DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        df = self.df[
            self.df['security_type'].str.lower() == security_type.lower()
        ]

        self.logger.info(
            f"Filtered to {len(df)} rows for security type: {security_type}"
        )

        if inplace:
            self.df = df

        return df

    def get_executives_list(self) -> List[str]:
        """
        Get list of unique executives in the data.

        Returns:
            Sorted list of executive names
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        executives = self.df['executive'].dropna().unique().tolist()
        return sorted(executives)

    def get_security_types_list(self) -> List[str]:
        """
        Get list of unique security types in the data.

        Returns:
            Sorted list of security types
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        securities = self.df['security_type'].dropna().unique().tolist()
        return sorted(securities)

    def get_executive_titles_list(self) -> List[str]:
        """
        Get list of unique executive titles in the data.

        Returns:
            Sorted list of executive titles
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        titles = self.df['executive_title'].dropna().unique().tolist()
        return sorted(titles)

    def get_summary_stats(self) -> Dict:
        """
        Get comprehensive summary statistics for the loaded data.

        Returns:
            Dictionary with summary statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        df = self.df
        start_date, end_date = self.get_date_range()

        # Calculate buy/sell statistics
        buys = df[df['acquisition_or_disposal'] == 'Buy']
        sells = df[df['acquisition_or_disposal'] == 'Sell']

        buy_volume = buys['shares'].sum() if len(buys) > 0 else 0
        sell_volume = sells['shares'].sum() if len(sells) > 0 else 0
        total_volume = buy_volume + sell_volume

        buy_sell_ratio = (
            buy_volume / sell_volume if sell_volume > 0 else float('inf')
        )

        # Date distribution
        df['month'] = df['transaction_date'].dt.to_period('M')
        monthly_counts = df.groupby('month').size()

        # Executive statistics
        exec_counts = df.groupby('executive')['shares'].agg(['count', 'sum'])
        top_executives = exec_counts.nlargest(5, 'sum')

        summary = {
            'total_transactions': len(df),
            'date_range': {
                'start': start_date.strftime(constants.DATE_FORMAT),
                'end': end_date.strftime(constants.DATE_FORMAT),
                'days': (end_date - start_date).days
            },
            'executives': {
                'unique_count': df['executive'].nunique(),
                'list': self.get_executives_list()[:10],  # First 10
                'top_by_volume': top_executives.to_dict()
            },
            'security_types': {
                'unique_count': df['security_type'].nunique(),
                'list': self.get_security_types_list()
            },
            'transaction_types': {
                'buys': {
                    'count': len(buys),
                    'volume': float(buy_volume),
                    'percentage': len(buys) / len(df) * 100 if len(df) > 0 else 0
                },
                'sells': {
                    'count': len(sells),
                    'volume': float(sell_volume),
                    'percentage': len(sells) / len(df) * 100 if len(df) > 0 else 0
                },
                'buy_sell_ratio': float(buy_sell_ratio)
            },
            'volume': {
                'total': float(total_volume),
                'mean': float(df['shares'].mean()),
                'median': float(df['shares'].median()),
                'std': float(df['shares'].std()),
                'min': float(df['shares'].min()),
                'max': float(df['shares'].max())
            },
            'monthly_distribution': {
                'mean_per_month': float(monthly_counts.mean()),
                'max_month': str(monthly_counts.idxmax()),
                'max_count': int(monthly_counts.max()),
                'min_month': str(monthly_counts.idxmin()),
                'min_count': int(monthly_counts.min())
            },
            'data_quality': {
                'missing_values': self.quality_report.report['missing_values']
                    if self.quality_report else {},
                'issues': self.quality_report.get_issues()
                    if self.quality_report else []
            }
        }

        return summary

    def get_preview(self, n_rows: int = 20) -> pd.DataFrame:
        """
        Get a preview of the data.

        Args:
            n_rows: Number of rows to return

        Returns:
            DataFrame with first n_rows
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        return self.df.head(n_rows)

    def export_cleaned_data(
        self,
        filepath: Union[str, Path],
        include_index: bool = False
    ) -> Path:
        """
        Export cleaned data to CSV.

        Args:
            filepath: Path to save CSV
            include_index: Whether to include DataFrame index

        Returns:
            Path to saved file
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(filepath, index=include_index)
        self.logger.info(f"Exported cleaned data to {filepath}")

        return filepath

    def reset_filters(self) -> pd.DataFrame:
        """
        Reset all filters and return to original cleaned data.

        Returns:
            Original cleaned DataFrame
        """
        if self.raw_df is None:
            raise ValueError("No data loaded. Call load_raw_data first.")

        self.df = self._clean_data(self.raw_df.copy())
        self.logger.info("Filters reset to original cleaned data")

        return self.df

    def get_cleaning_log(self) -> List[str]:
        """
        Get the log of cleaning operations performed.

        Returns:
            List of cleaning log entries
        """
        return self._cleaning_log.copy()

    def __repr__(self) -> str:
        """String representation of the preprocessor."""
        if self.df is None:
            return "DataPreprocessor(no data loaded)"

        return (
            f"DataPreprocessor(file={self.filepath.name if self.filepath else None}, "
            f"rows={len(self.df)}, "
            f"date_range={self.get_date_range()})"
        )
