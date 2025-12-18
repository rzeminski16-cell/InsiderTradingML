"""
Utility functions for the Insider Trading Anomaly Detection System.

This module provides common utility functions used throughout the system including:
- Logging setup and configuration
- Date/time utilities
- Data validation helpers
- File I/O utilities
- Statistical helper functions
"""

import logging
import os
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import constants


def setup_logging(
    name: str,
    log_to_file: bool = True,
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Set up logging for a module.

    Args:
        name: Name of the logger (typically __name__)
        log_to_file: Whether to also log to a file
        log_dir: Directory for log files (defaults to constants.LOGS_DIR)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, constants.LOG_LEVEL_CONSOLE))
    console_formatter = logging.Formatter(
        constants.LOG_FORMAT,
        datefmt=constants.LOG_DATE_FORMAT
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_dir = log_dir or constants.LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

        log_filename = f"{datetime.now().strftime('%Y-%m-%d')}_insidertrading.log"
        log_filepath = log_dir / log_filename

        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(getattr(logging, constants.LOG_LEVEL_FILE))
        file_formatter = logging.Formatter(
            constants.LOG_FORMAT,
            datefmt=constants.LOG_DATE_FORMAT
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamp() -> str:
    """Get current timestamp in standard format."""
    return datetime.now().strftime(constants.TIMESTAMP_FORMAT)


def get_date_string() -> str:
    """Get current date as string."""
    return datetime.now().strftime(constants.DATE_FORMAT)


def parse_date(date_str: Union[str, datetime, pd.Timestamp]) -> datetime:
    """
    Parse a date string to datetime object.

    Args:
        date_str: Date string or datetime object

    Returns:
        Parsed datetime object
    """
    if isinstance(date_str, datetime):
        return date_str
    if isinstance(date_str, pd.Timestamp):
        return date_str.to_pydatetime()

    # Try common date formats
    formats = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%Y-%m-%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S'
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # Let pandas try to parse it
    return pd.to_datetime(date_str).to_pydatetime()


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    raise_on_error: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        raise_on_error: Whether to raise an exception on validation failure

    Returns:
        Tuple of (is_valid, list of missing columns)

    Raises:
        ValueError: If validation fails and raise_on_error is True
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0

    if not is_valid and raise_on_error:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return is_valid, missing_columns


def calculate_herfindahl_index(series: pd.Series) -> float:
    """
    Calculate the Herfindahl-Hirschman Index (HHI) for concentration.

    The HHI is a measure of concentration, ranging from 0 (perfectly distributed)
    to 1 (completely concentrated in one entity).

    Args:
        series: Pandas Series with values to analyze

    Returns:
        HHI value between 0 and 1
    """
    if series.empty or series.sum() == 0:
        return 0.0

    # Calculate market shares
    shares = series / series.sum()

    # HHI = sum of squared market shares
    hhi = (shares ** 2).sum()

    return float(hhi)


def normalize_scores(
    scores: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize anomaly scores to [0, 1] range.

    Args:
        scores: Array of anomaly scores
        method: Normalization method ('minmax', 'zscore', 'sigmoid')

    Returns:
        Normalized scores in [0, 1] range
    """
    scores = np.asarray(scores).astype(float)

    if len(scores) == 0:
        return scores

    if method == 'minmax':
        min_val = np.min(scores)
        max_val = np.max(scores)
        if max_val - min_val == 0:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    elif method == 'zscore':
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return np.zeros_like(scores)
        z_scores = (scores - mean) / std
        # Convert z-scores to [0, 1] using sigmoid
        return 1 / (1 + np.exp(-z_scores))

    elif method == 'sigmoid':
        return 1 / (1 + np.exp(-scores))

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_zscore(
    value: float,
    mean: float,
    std: float
) -> float:
    """
    Calculate z-score for a value.

    Args:
        value: Value to calculate z-score for
        mean: Mean of the distribution
        std: Standard deviation of the distribution

    Returns:
        Z-score
    """
    if std == 0:
        return 0.0
    return (value - mean) / std


def get_risk_level(
    anomaly_score: float,
    confidence: float = 1.0,
    model_agreement: Optional[float] = None
) -> str:
    """
    Determine risk level based on anomaly score and confidence.

    Args:
        anomaly_score: Anomaly score in [0, 1]
        confidence: Confidence score in [0, 1]
        model_agreement: Fraction of models agreeing on anomaly classification

    Returns:
        Risk level string ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL')
    """
    # Adjust score based on model agreement if provided
    if model_agreement is not None and model_agreement > 0.8:
        anomaly_score = min(1.0, anomaly_score * 1.1)

    for level, thresholds in constants.RISK_LEVEL_THRESHOLDS.items():
        if (anomaly_score >= thresholds['min_score'] and
            confidence >= thresholds['min_confidence']):
            return level

    return 'NORMAL'


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    default: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Safely divide two numbers, returning default value on division by zero.

    Args:
        numerator: Numerator value(s)
        denominator: Denominator value(s)
        default: Default value for division by zero

    Returns:
        Division result or default value
    """
    if isinstance(denominator, np.ndarray):
        result = np.where(denominator != 0, numerator / denominator, default)
        return result
    else:
        if denominator == 0:
            return default
        return numerator / denominator


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save an object to a pickle file.

    Args:
        obj: Object to save
        filepath: Path to save to
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load an object from a pickle file.

    Args:
        filepath: Path to load from

    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: Dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Save a dictionary to a JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save to
        indent: JSON indentation
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load a dictionary from a JSON file.

    Args:
        filepath: Path to load from

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def format_number(
    value: float,
    decimal_places: int = 2,
    as_percentage: bool = False
) -> str:
    """
    Format a number for display.

    Args:
        value: Number to format
        decimal_places: Number of decimal places
        as_percentage: Whether to format as percentage

    Returns:
        Formatted string
    """
    if pd.isna(value) or np.isinf(value):
        return 'N/A'

    if as_percentage:
        return f"{value * 100:.{decimal_places}f}%"

    return f"{value:,.{decimal_places}f}"


def get_percentile(
    value: float,
    distribution: np.ndarray
) -> float:
    """
    Get the percentile of a value within a distribution.

    Args:
        value: Value to find percentile for
        distribution: Array of values representing the distribution

    Returns:
        Percentile value (0-100)
    """
    if len(distribution) == 0:
        return 50.0

    percentile = (distribution < value).sum() / len(distribution) * 100
    return float(percentile)


def calculate_rolling_statistics(
    series: pd.Series,
    window: int
) -> Dict[str, pd.Series]:
    """
    Calculate rolling statistics for a series.

    Args:
        series: Pandas Series of values
        window: Rolling window size

    Returns:
        Dictionary with rolling mean, std, min, max
    """
    return {
        'mean': series.rolling(window=window, min_periods=1).mean(),
        'std': series.rolling(window=window, min_periods=1).std(),
        'min': series.rolling(window=window, min_periods=1).min(),
        'max': series.rolling(window=window, min_periods=1).max()
    }


def create_date_range(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    freq: str = 'D'
) -> pd.DatetimeIndex:
    """
    Create a date range.

    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency ('D' for daily, 'W' for weekly, etc.)

    Returns:
        DatetimeIndex
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def split_data_chronologically(
    df: pd.DataFrame,
    date_column: str,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (older data for training).

    Args:
        df: DataFrame to split
        date_column: Name of the date column
        train_ratio: Ratio of data to use for training

    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.sort_values(date_column)
    split_idx = int(len(df) * train_ratio)

    return df.iloc[:split_idx], df.iloc[split_idx:]


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns.

    Args:
        df: DataFrame with numeric columns

    Returns:
        Correlation matrix DataFrame
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5
) -> pd.Series:
    """
    Detect outliers using the IQR method.

    Args:
        series: Series to analyze
        multiplier: IQR multiplier (default 1.5)

    Returns:
        Boolean Series (True for outliers)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return (series < lower_bound) | (series > upper_bound)


class DataQualityReport:
    """Class for generating data quality reports."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.

        Args:
            df: DataFrame to analyze
        """
        self.df = df
        self.report = {}
        self._generate_report()

    def _generate_report(self) -> None:
        """Generate the data quality report."""
        self.report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicate_rows': self.df.duplicated().sum(),
            'column_types': self.df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024 / 1024
        }

        # Add numeric column statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.report['numeric_stats'] = self.df[numeric_cols].describe().to_dict()

    def get_report(self) -> Dict:
        """Get the quality report as a dictionary."""
        return self.report

    def get_issues(self) -> List[str]:
        """Get list of data quality issues."""
        issues = []

        # Check for missing values
        for col, pct in self.report['missing_percentage'].items():
            if pct > 0:
                issues.append(f"Column '{col}' has {pct:.1f}% missing values")

        # Check for duplicates
        if self.report['duplicate_rows'] > 0:
            issues.append(f"Found {self.report['duplicate_rows']} duplicate rows")

        return issues

    def __str__(self) -> str:
        """String representation of the report."""
        lines = [
            "Data Quality Report",
            "=" * 40,
            f"Total Rows: {self.report['total_rows']:,}",
            f"Total Columns: {self.report['total_columns']}",
            f"Duplicate Rows: {self.report['duplicate_rows']}",
            f"Memory Usage: {self.report['memory_usage_mb']:.2f} MB",
            "",
            "Issues Found:",
        ]

        issues = self.get_issues()
        if issues:
            for issue in issues:
                lines.append(f"  - {issue}")
        else:
            lines.append("  No issues found")

        return "\n".join(lines)
