"""
Constants and configuration values for the Insider Trading Anomaly Detection System.

This module contains all configuration constants including:
- Color schemes for GUI and reports
- Font configurations
- File paths
- Model hyperparameter defaults
- Feature categories
- Risk level thresholds
"""

from typing import Dict, List, Tuple
from pathlib import Path
import os

# ==============================================================================
# PROJECT PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
NEW_TRANSACTIONS_DIR = DATA_DIR / "new_transactions"
MODELS_DIR = PROJECT_ROOT / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
TRAINING_LOGS_DIR = MODELS_DIR / "training_logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_ARCHIVE_DIR = REPORTS_DIR / "archive"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"

# ==============================================================================
# DATA COLUMN SPECIFICATIONS
# ==============================================================================
REQUIRED_COLUMNS = [
    'transaction_date',
    'executive',
    'executive_title',
    'security_type',
    'acquisition_or_disposal',
    'shares'
]

COLUMN_TYPES = {
    'transaction_date': 'datetime64[ns]',
    'executive': 'str',
    'executive_title': 'str',
    'security_type': 'str',
    'acquisition_or_disposal': 'str',
    'shares': 'float64'
}

# Valid values for categorical columns
VALID_ACQUISITION_DISPOSAL = ['Buy', 'Sell', 'A', 'D', 'Acquisition', 'Disposal']
BUY_VALUES = ['Buy', 'A', 'Acquisition']
SELL_VALUES = ['Sell', 'D', 'Disposal']

# ==============================================================================
# AGGREGATION WINDOWS
# ==============================================================================
DEFAULT_WINDOW_DAYS = 14
AVAILABLE_WINDOW_SIZES = [7, 14, 21, 30, 60, 90]
BASELINE_WINDOW_DAYS = 90  # For calculating baseline statistics

# ==============================================================================
# FEATURE CATEGORIES
# ==============================================================================
VOLUME_FEATURES = [
    'total_transaction_volume',
    'transaction_count',
    'avg_transaction_size',
    'volume_volatility',
    'volume_std_deviation'
]

BEHAVIORAL_FEATURES = [
    'buy_sell_ratio',
    'buy_sell_asymmetry',
    'executive_concentration',
    'security_concentration',
    'title_diversity'
]

TEMPORAL_FEATURES = [
    'days_since_last_transaction',
    'trading_consistency',
    'day_of_week_concentration'
]

COMPARATIVE_FEATURES = [
    'volume_change_yoy',
    'volume_change_prev_period',
    'buy_ratio_change'
]

NETWORK_FEATURES = [
    'insider_network_density',
    'transaction_timing_correlation'
]

ACQUISITION_DISPOSAL_FEATURES = [
    'net_share_change',
    'acquisition_disposal_ratio'
]

ALL_FEATURES = (
    VOLUME_FEATURES +
    BEHAVIORAL_FEATURES +
    TEMPORAL_FEATURES +
    COMPARATIVE_FEATURES +
    NETWORK_FEATURES +
    ACQUISITION_DISPOSAL_FEATURES
)

FEATURE_CATEGORIES = {
    'Volume-Based Features': VOLUME_FEATURES,
    'Behavioral Features': BEHAVIORAL_FEATURES,
    'Temporal Features': TEMPORAL_FEATURES,
    'Comparative Features': COMPARATIVE_FEATURES,
    'Network Features': NETWORK_FEATURES,
    'Acquisition/Disposal Features': ACQUISITION_DISPOSAL_FEATURES
}

# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================
SUPPORTED_MODELS = [
    'IsolationForest',
    'DBSCAN',
    'LocalOutlierFactor',
    'NNAutoencoder',
    'KMeans',
    'OneClassSVM'
]

# Alias for backwards compatibility
SUPPORTED_MODELS_ALIASES = {
    'LSTMAutoencoder': 'NNAutoencoder'
}

DEFAULT_HYPERPARAMETERS: Dict[str, Dict] = {
    'IsolationForest': {
        'contamination': 0.05,
        'n_estimators': 100,
        'max_samples': 'auto',
        'random_state': 42
    },
    'DBSCAN': {
        'eps': 0.5,
        'min_samples': 5,
        'metric': 'euclidean'
    },
    'LocalOutlierFactor': {
        'n_neighbors': 20,
        'contamination': 0.05,
        'novelty': True
    },
    'NNAutoencoder': {
        'encoding_dim': 8,
        'hidden_layers': (32, 16, 8, 16, 32),
        'max_iter': 500,
        'learning_rate_init': 0.001,
        'random_state': 42
    },
    'KMeans': {
        'n_clusters': 5,
        'init': 'k-means++',
        'random_state': 42
    },
    'OneClassSVM': {
        'kernel': 'rbf',
        'nu': 0.05,
        'gamma': 'scale'
    }
}

HYPERPARAMETER_RANGES: Dict[str, Dict] = {
    'IsolationForest': {
        'contamination': {'min': 0.01, 'max': 0.2, 'step': 0.01, 'type': 'float'},
        'n_estimators': {'min': 50, 'max': 300, 'step': 25, 'type': 'int'},
        'max_samples': {'options': ['auto', 128, 256, 512], 'type': 'choice'}
    },
    'DBSCAN': {
        'eps': {'min': 0.1, 'max': 3.0, 'step': 0.1, 'type': 'float'},
        'min_samples': {'min': 2, 'max': 20, 'step': 1, 'type': 'int'},
        'metric': {'options': ['euclidean', 'manhattan', 'cosine'], 'type': 'choice'}
    },
    'LocalOutlierFactor': {
        'n_neighbors': {'min': 5, 'max': 50, 'step': 5, 'type': 'int'},
        'contamination': {'min': 0.01, 'max': 0.2, 'step': 0.01, 'type': 'float'}
    },
    'NNAutoencoder': {
        'encoding_dim': {'min': 4, 'max': 32, 'step': 4, 'type': 'int'},
        'max_iter': {'min': 100, 'max': 1000, 'step': 100, 'type': 'int'},
        'learning_rate_init': {'min': 0.0001, 'max': 0.01, 'step': 0.001, 'type': 'float'}
    },
    'KMeans': {
        'n_clusters': {'min': 2, 'max': 15, 'step': 1, 'type': 'int'},
        'init': {'options': ['k-means++', 'random'], 'type': 'choice'}
    },
    'OneClassSVM': {
        'nu': {'min': 0.01, 'max': 0.5, 'step': 0.01, 'type': 'float'},
        'kernel': {'options': ['rbf', 'linear', 'poly', 'sigmoid'], 'type': 'choice'},
        'gamma': {'options': ['scale', 'auto'], 'type': 'choice'}
    }
}

# ==============================================================================
# ANOMALY SCORING CONFIGURATION
# ==============================================================================
DEFAULT_ANOMALY_THRESHOLD_METHOD = 'percentile'
DEFAULT_ANOMALY_PERCENTILE = 95
DEFAULT_ANOMALY_STD_MULTIPLIER = 3.0

# Risk level thresholds
RISK_LEVEL_THRESHOLDS = {
    'CRITICAL': {'min_score': 0.85, 'min_confidence': 0.7},
    'HIGH': {'min_score': 0.70, 'min_confidence': 0.6},
    'MEDIUM': {'min_score': 0.50, 'min_confidence': 0.5},
    'LOW': {'min_score': 0.30, 'min_confidence': 0.4},
    'NORMAL': {'min_score': 0.0, 'min_confidence': 0.0}
}

RISK_LEVEL_COLORS = {
    'CRITICAL': '#DC143C',  # Crimson
    'HIGH': '#FF4500',      # Orange Red
    'MEDIUM': '#FFA500',    # Orange
    'LOW': '#FFD700',       # Gold
    'NORMAL': '#32CD32'     # Lime Green
}

# ==============================================================================
# GUI CONFIGURATION
# ==============================================================================
# Color scheme
GUI_COLORS = {
    'primary': '#007BFF',
    'secondary': '#6C757D',
    'success': '#28A745',
    'danger': '#DC3545',
    'warning': '#FFC107',
    'info': '#17A2B8',
    'light': '#F8F9FA',
    'dark': '#343A40',
    'background': '#FFFFFF',
    'text': '#333333',
    'border': '#DEE2E6'
}

# Font configuration
GUI_FONTS = {
    'header': ('Arial', 14, 'bold'),
    'section_title': ('Arial', 12, 'bold'),
    'body': ('Arial', 11, 'normal'),
    'small': ('Arial', 9, 'normal'),
    'monospace': ('Consolas', 10, 'normal')
}

# Window dimensions
GUI_WINDOW_SIZE = (1400, 900)
GUI_MIN_SIZE = (1200, 700)

# Table configurations
TABLE_ROW_HEIGHT = 25
TABLE_HEADER_HEIGHT = 30
TABLE_MAX_ROWS_PREVIEW = 20

# ==============================================================================
# EXCEL REPORT CONFIGURATION
# ==============================================================================
EXCEL_COLORS = {
    'header_bg': '#1E88E5',
    'header_font': '#FFFFFF',
    'anomaly_high': '#DC3545',
    'anomaly_medium': '#FFC107',
    'anomaly_low': '#28A745',
    'border': '#DEE2E6',
    'alternate_row': '#F8F9FA',
    'text_dark': '#333333',
    'text_light': '#6C757D'
}

EXCEL_FONTS = {
    'title': {'name': 'Arial', 'size': 16, 'bold': True},
    'header': {'name': 'Arial', 'size': 14, 'bold': True},
    'section': {'name': 'Arial', 'size': 12, 'bold': True},
    'body': {'name': 'Arial', 'size': 11, 'bold': False},
    'small': {'name': 'Arial', 'size': 9, 'bold': False}
}

EXCEL_NUMBER_FORMATS = {
    'percentage': '0.00%',
    'decimal_2': '0.00',
    'decimal_3': '0.000',
    'integer': '#,##0',
    'currency': '$#,##0.00',
    'date': 'YYYY-MM-DD',
    'datetime': 'YYYY-MM-DD HH:MM:SS'
}

EXCEL_SHEETS = [
    'Executive Summary',
    'Historical Anomalies',
    'Feature Engineering',
    'Model Performance',
    'New Transaction Scoring',
    'Anomaly Explanations',
    'Technical Specifications',
    'Appendix'
]

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL_CONSOLE = 'INFO'
LOG_LEVEL_FILE = 'DEBUG'
LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5

# ==============================================================================
# PERFORMANCE SETTINGS
# ==============================================================================
MAX_TRANSACTIONS_WARNING = 100000
CHUNK_SIZE = 10000
MEMORY_LIMIT_MB = 1024

# ==============================================================================
# TRAIN/TEST SPLIT
# ==============================================================================
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VALIDATION_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1

# ==============================================================================
# MISCELLANEOUS
# ==============================================================================
RANDOM_SEED = 42
FLOAT_PRECISION = 6
DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
