"""
Feature Engineering for the Insider Trading Anomaly Detection System.

This module provides the FeatureEngineer class that creates 20+ time-series
features from raw insider trading transaction data. Features are grouped into:
- Volume-based features (activity spikes)
- Behavioral features (trading patterns)
- Temporal features (timing behavior)
- Comparative features (change vs past)
- Network features (coordination between insiders)
- Acquisition/Disposal features

Classes:
    FeatureEngineer: Main class for feature engineering operations
"""

import logging
from datetime import datetime, timedelta
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from . import constants
from .utils import (
    setup_logging,
    calculate_herfindahl_index,
    safe_divide,
    calculate_zscore
)


class FeatureEngineer:
    """
    Engineer time-series features from raw insider trading transactions.

    This class aggregates transactions by time window and computes various
    features designed to detect anomalous trading patterns.

    Attributes:
        df: Source DataFrame with transaction data
        aggregation_window_days: Number of days per aggregation window
        aggregated_df: DataFrame with time-windowed aggregations
        feature_matrix: Final feature matrix for modeling
    """

    def __init__(
        self,
        df: pd.DataFrame,
        aggregation_window_days: int = constants.DEFAULT_WINDOW_DAYS
    ):
        """
        Initialize the FeatureEngineer.

        Args:
            df: DataFrame with cleaned transaction data
            aggregation_window_days: Number of days per aggregation window
        """
        self.logger = setup_logging(__name__)
        self.df = df.copy()
        self.aggregation_window_days = aggregation_window_days
        self.aggregated_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self._feature_stats: Dict = {}

        # Ensure transaction_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df['transaction_date']):
            self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])

        self.logger.info(
            f"FeatureEngineer initialized with {len(df)} transactions, "
            f"{aggregation_window_days}-day windows"
        )

    def aggregate_by_window(
        self,
        window_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Aggregate transactions into time windows.

        Creates time bins of specified length and calculates base aggregations
        for each window.

        Args:
            window_days: Number of days per window (uses default if not specified)

        Returns:
            DataFrame with window_start_date, window_end_date as columns
        """
        if window_days is not None:
            self.aggregation_window_days = window_days

        self.logger.info(
            f"Aggregating by {self.aggregation_window_days}-day windows"
        )

        df = self.df.copy()

        # Get date range
        min_date = df['transaction_date'].min()
        max_date = df['transaction_date'].max()

        # Create window bins
        windows = []
        current_start = min_date

        while current_start <= max_date:
            window_end = current_start + timedelta(days=self.aggregation_window_days - 1)
            windows.append({
                'window_start': current_start,
                'window_end': min(window_end, max_date)
            })
            current_start = window_end + timedelta(days=1)

        # Assign transactions to windows
        df['window_idx'] = None
        for idx, window in enumerate(windows):
            mask = (
                (df['transaction_date'] >= window['window_start']) &
                (df['transaction_date'] <= window['window_end'])
            )
            df.loc[mask, 'window_idx'] = idx

        # Create aggregated DataFrame
        agg_data = []
        for idx, window in enumerate(windows):
            window_df = df[df['window_idx'] == idx]

            if len(window_df) == 0:
                # Empty window - still include with zeros
                agg_data.append({
                    'window_idx': idx,
                    'window_start': window['window_start'],
                    'window_end': window['window_end'],
                    'transaction_count': 0,
                    'total_volume': 0,
                    'unique_executives': 0,
                    'unique_securities': 0,
                    'unique_titles': 0,
                    'buy_count': 0,
                    'sell_count': 0,
                    'buy_volume': 0,
                    'sell_volume': 0,
                    'executives': [],
                    'securities': [],
                    'titles': []
                })
            else:
                buys = window_df[window_df['acquisition_or_disposal'] == 'Buy']
                sells = window_df[window_df['acquisition_or_disposal'] == 'Sell']

                agg_data.append({
                    'window_idx': idx,
                    'window_start': window['window_start'],
                    'window_end': window['window_end'],
                    'transaction_count': len(window_df),
                    'total_volume': window_df['shares'].sum(),
                    'unique_executives': window_df['executive'].nunique(),
                    'unique_securities': window_df['security_type'].nunique(),
                    'unique_titles': window_df['executive_title'].nunique(),
                    'buy_count': len(buys),
                    'sell_count': len(sells),
                    'buy_volume': buys['shares'].sum(),
                    'sell_volume': sells['shares'].sum(),
                    'executives': window_df['executive'].tolist(),
                    'securities': window_df['security_type'].tolist(),
                    'titles': window_df['executive_title'].tolist(),
                    'transaction_dates': window_df['transaction_date'].tolist(),
                    'shares_list': window_df['shares'].tolist()
                })

        self.aggregated_df = pd.DataFrame(agg_data)
        self._window_df = df  # Store for later use

        self.logger.info(
            f"Created {len(self.aggregated_df)} windows from "
            f"{min_date.date()} to {max_date.date()}"
        )

        return self.aggregated_df

    # =========================================================================
    # VOLUME-BASED FEATURES
    # =========================================================================

    def total_transaction_volume(self) -> pd.Series:
        """
        Calculate total shares traded per window.

        High volume indicates potential coordinated trading.

        Returns:
            Series with total volume per window
        """
        return self.aggregated_df['total_volume'].rename('total_transaction_volume')

    def transaction_count(self) -> pd.Series:
        """
        Calculate number of transactions per window.

        Spike in transaction frequency indicates unusual activity.

        Returns:
            Series with transaction count per window
        """
        return self.aggregated_df['transaction_count'].rename('transaction_count')

    def avg_transaction_size(self) -> pd.Series:
        """
        Calculate mean shares per transaction in window.

        Larger than normal average size may indicate unusual trades.

        Returns:
            Series with average transaction size per window
        """
        result = safe_divide(
            self.aggregated_df['total_volume'],
            self.aggregated_df['transaction_count'],
            default=0.0
        )
        return pd.Series(result, name='avg_transaction_size')

    def volume_volatility(self) -> pd.Series:
        """
        Calculate standard deviation of transaction sizes in window.

        Low volatility (uniform sizes) may indicate coordinated trading.

        Returns:
            Series with volume volatility per window
        """
        volatilities = []
        for _, row in self.aggregated_df.iterrows():
            shares_list = row.get('shares_list', [])
            if len(shares_list) > 1:
                volatilities.append(np.std(shares_list))
            else:
                volatilities.append(0.0)

        return pd.Series(volatilities, name='volume_volatility')

    def volume_std_deviation(
        self,
        baseline_window_days: int = constants.BASELINE_WINDOW_DAYS
    ) -> pd.Series:
        """
        Calculate z-score of volume vs rolling baseline.

        Compares current window volume to historical baseline.

        Args:
            baseline_window_days: Number of days for baseline calculation

        Returns:
            Series with z-score per window
        """
        # Calculate rolling baseline statistics
        baseline_windows = baseline_window_days // self.aggregation_window_days

        volume = self.aggregated_df['total_volume']
        rolling_mean = volume.rolling(
            window=baseline_windows,
            min_periods=1
        ).mean().shift(1)
        rolling_std = volume.rolling(
            window=baseline_windows,
            min_periods=1
        ).std().shift(1)

        # Calculate z-score
        z_scores = (volume - rolling_mean) / rolling_std.replace(0, np.nan)
        z_scores = z_scores.fillna(0)

        return z_scores.rename('volume_std_deviation')

    # =========================================================================
    # BEHAVIORAL FEATURES
    # =========================================================================

    def buy_sell_ratio(self) -> pd.Series:
        """
        Calculate ratio of buy volume to sell volume per window.

        Extreme ratios indicate unusual behavior.

        Returns:
            Series with buy/sell ratio per window
        """
        result = safe_divide(
            self.aggregated_df['buy_volume'],
            self.aggregated_df['sell_volume'],
            default=1.0  # Default to balanced if no sells
        )
        # Cap extreme values
        result = np.clip(result, 0.01, 100.0)
        return pd.Series(result, name='buy_sell_ratio')

    def buy_sell_asymmetry(self) -> pd.Series:
        """
        Calculate buy/sell asymmetry: (Buy Count - Sell Count) / Total.

        Range [-1, 1]: 1 = all buys, -1 = all sells.

        Returns:
            Series with asymmetry per window
        """
        total = self.aggregated_df['buy_count'] + self.aggregated_df['sell_count']
        result = safe_divide(
            self.aggregated_df['buy_count'] - self.aggregated_df['sell_count'],
            total,
            default=0.0
        )
        return pd.Series(result, name='buy_sell_asymmetry')

    def executive_concentration(self) -> pd.Series:
        """
        Calculate Herfindahl index of executives trading per window.

        Range [0, 1]: High values indicate few executives dominate volume.

        Returns:
            Series with executive concentration per window
        """
        concentrations = []
        for _, row in self.aggregated_df.iterrows():
            if row['transaction_count'] == 0:
                concentrations.append(0.0)
                continue

            # Get volume per executive in this window
            window_df = self._window_df[
                self._window_df['window_idx'] == row['window_idx']
            ]
            exec_volumes = window_df.groupby('executive')['shares'].sum()
            concentrations.append(calculate_herfindahl_index(exec_volumes))

        return pd.Series(concentrations, name='executive_concentration')

    def security_concentration(self) -> pd.Series:
        """
        Calculate Herfindahl index of securities traded per window.

        High values indicate trading focused on few securities.

        Returns:
            Series with security concentration per window
        """
        concentrations = []
        for _, row in self.aggregated_df.iterrows():
            if row['transaction_count'] == 0:
                concentrations.append(0.0)
                continue

            window_df = self._window_df[
                self._window_df['window_idx'] == row['window_idx']
            ]
            sec_volumes = window_df.groupby('security_type')['shares'].sum()
            concentrations.append(calculate_herfindahl_index(sec_volumes))

        return pd.Series(concentrations, name='security_concentration')

    def title_diversity(self) -> pd.Series:
        """
        Calculate count of unique executive titles per window.

        Cross-level coordination (multiple titles) may be unusual.

        Returns:
            Series with title diversity per window
        """
        return self.aggregated_df['unique_titles'].rename('title_diversity')

    # =========================================================================
    # TEMPORAL FEATURES
    # =========================================================================

    def days_since_last_transaction(self) -> pd.Series:
        """
        Calculate days since last transaction in previous window.

        Activity bursts after dormancy may be suspicious.

        Returns:
            Series with days since last transaction per window
        """
        days_since = [0.0]  # First window has no previous

        for i in range(1, len(self.aggregated_df)):
            current_start = self.aggregated_df.iloc[i]['window_start']

            # Find last transaction date in all previous windows
            prev_transactions = self._window_df[
                self._window_df['window_idx'] < i
            ]

            if len(prev_transactions) > 0:
                last_date = prev_transactions['transaction_date'].max()
                days = (current_start - last_date).days
                days_since.append(max(0, days))
            else:
                days_since.append(0)

        return pd.Series(days_since, name='days_since_last_transaction')

    def trading_consistency(self) -> pd.Series:
        """
        Calculate ratio of days with trades to total days in window.

        High consistency = intensive trading period.

        Returns:
            Series with trading consistency per window
        """
        consistencies = []
        for _, row in self.aggregated_df.iterrows():
            if row['transaction_count'] == 0:
                consistencies.append(0.0)
                continue

            window_df = self._window_df[
                self._window_df['window_idx'] == row['window_idx']
            ]

            unique_days = window_df['transaction_date'].dt.date.nunique()
            total_days = (row['window_end'] - row['window_start']).days + 1

            consistencies.append(safe_divide(unique_days, total_days, 0.0))

        return pd.Series(consistencies, name='trading_consistency')

    def day_of_week_concentration(self) -> pd.Series:
        """
        Calculate concentration of trades on specific weekdays.

        Trades clustered on specific days may be anomalous.

        Returns:
            Series with day-of-week concentration per window
        """
        concentrations = []
        for _, row in self.aggregated_df.iterrows():
            if row['transaction_count'] == 0:
                concentrations.append(0.0)
                continue

            window_df = self._window_df[
                self._window_df['window_idx'] == row['window_idx']
            ]

            # Count transactions by day of week
            dow_counts = window_df['transaction_date'].dt.dayofweek.value_counts()
            concentrations.append(calculate_herfindahl_index(dow_counts))

        return pd.Series(concentrations, name='day_of_week_concentration')

    # =========================================================================
    # COMPARATIVE FEATURES
    # =========================================================================

    def volume_change_yoy(self) -> pd.Series:
        """
        Calculate year-over-year volume change.

        Significant YoY changes may indicate unusual activity.

        Returns:
            Series with YoY volume change per window
        """
        volumes = self.aggregated_df['total_volume'].values
        windows_per_year = 365 // self.aggregation_window_days

        changes = []
        for i in range(len(volumes)):
            if i < windows_per_year:
                changes.append(0.0)
            else:
                prev_volume = volumes[i - windows_per_year]
                if prev_volume > 0:
                    change = (volumes[i] - prev_volume) / prev_volume
                    changes.append(change)
                else:
                    changes.append(0.0 if volumes[i] == 0 else 1.0)

        return pd.Series(changes, name='volume_change_yoy')

    def volume_change_prev_period(self) -> pd.Series:
        """
        Calculate volume change from previous period.

        Rapid acceleration indicates potential anomaly.

        Returns:
            Series with period-over-period volume change
        """
        volumes = self.aggregated_df['total_volume']
        prev_volumes = volumes.shift(1).fillna(0)

        changes = safe_divide(volumes - prev_volumes, prev_volumes, 0.0)
        # Cap extreme values
        changes = np.clip(changes, -10.0, 10.0)

        return pd.Series(changes, name='volume_change_prev_period')

    def buy_ratio_change(self) -> pd.Series:
        """
        Calculate change in buy/sell ratio from previous period.

        Sudden shifts in direction may be significant.

        Returns:
            Series with buy ratio change per window
        """
        current_ratio = self.buy_sell_ratio()
        prev_ratio = current_ratio.shift(1).fillna(1.0)

        changes = current_ratio - prev_ratio

        return changes.rename('buy_ratio_change')

    # =========================================================================
    # NETWORK FEATURES
    # =========================================================================

    def insider_network_density(self) -> pd.Series:
        """
        Calculate count of unique executive pairs co-trading in window.

        High density indicates coordinated activity.

        Returns:
            Series with network density per window
        """
        densities = []
        for _, row in self.aggregated_df.iterrows():
            executives = row.get('executives', [])
            unique_execs = set(executives)

            if len(unique_execs) < 2:
                densities.append(0.0)
                continue

            # Count pairs that traded in same window
            pairs = list(combinations(unique_execs, 2))
            densities.append(len(pairs))

        return pd.Series(densities, name='insider_network_density')

    def transaction_timing_correlation(self) -> pd.Series:
        """
        Calculate correlation of transaction timing between executives.

        If transactions are clustered within same hours = high correlation.

        Returns:
            Series with timing correlation per window
        """
        correlations = []
        for _, row in self.aggregated_df.iterrows():
            if row['transaction_count'] < 2:
                correlations.append(0.0)
                continue

            window_df = self._window_df[
                self._window_df['window_idx'] == row['window_idx']
            ]

            # Get transaction dates as timestamps
            dates = window_df['transaction_date'].values

            if len(dates) < 2:
                correlations.append(0.0)
                continue

            # Calculate coefficient of variation of time gaps
            time_diffs = np.diff(dates.astype('datetime64[s]').astype('int64'))
            if len(time_diffs) > 0 and np.std(time_diffs) > 0:
                # Low CV = high correlation (transactions evenly spaced)
                cv = np.std(time_diffs) / (np.mean(time_diffs) + 1)
                # Invert so higher = more clustered
                correlation = 1 / (1 + cv)
            else:
                correlation = 0.0

            correlations.append(correlation)

        return pd.Series(correlations, name='transaction_timing_correlation')

    # =========================================================================
    # ACQUISITION/DISPOSAL FEATURES
    # =========================================================================

    def net_share_change(self) -> pd.Series:
        """
        Calculate net share change: sum(buys) - sum(sells).

        Large net accumulation before events may be suspicious.

        Returns:
            Series with net share change per window
        """
        net_change = (
            self.aggregated_df['buy_volume'] -
            self.aggregated_df['sell_volume']
        )
        return net_change.rename('net_share_change')

    def acquisition_disposal_ratio(self) -> pd.Series:
        """
        Calculate acquisition count / disposal count per window.

        Directional bias in transaction counts.

        Returns:
            Series with acquisition/disposal ratio per window
        """
        result = safe_divide(
            self.aggregated_df['buy_count'],
            self.aggregated_df['sell_count'],
            default=1.0
        )
        # Cap extreme values
        result = np.clip(result, 0.01, 100.0)
        return pd.Series(result, name='acquisition_disposal_ratio')

    # =========================================================================
    # FEATURE MATRIX CREATION
    # =========================================================================

    def get_available_features(self) -> List[str]:
        """
        Return list of all available feature methods.

        Returns:
            List of feature names
        """
        return constants.ALL_FEATURES.copy()

    def get_feature_categories(self) -> Dict[str, List[str]]:
        """
        Return feature categories and their features.

        Returns:
            Dictionary mapping category names to feature lists
        """
        return constants.FEATURE_CATEGORIES.copy()

    def create_feature_matrix(
        self,
        selected_features: Optional[List[str]] = None,
        handle_missing: str = 'fill'
    ) -> pd.DataFrame:
        """
        Create feature matrix with selected features.

        Args:
            selected_features: List of feature names to include
                             (uses all features if None)
            handle_missing: How to handle NaN/inf values
                           ('fill' = forward fill then drop,
                            'drop' = drop rows with missing values)

        Returns:
            Feature matrix DataFrame with window dates as index
        """
        if self.aggregated_df is None:
            self.aggregate_by_window()

        if selected_features is None:
            selected_features = self.get_available_features()

        self.logger.info(f"Creating feature matrix with {len(selected_features)} features")

        # Map feature names to methods
        feature_methods = {
            'total_transaction_volume': self.total_transaction_volume,
            'transaction_count': self.transaction_count,
            'avg_transaction_size': self.avg_transaction_size,
            'volume_volatility': self.volume_volatility,
            'volume_std_deviation': self.volume_std_deviation,
            'buy_sell_ratio': self.buy_sell_ratio,
            'buy_sell_asymmetry': self.buy_sell_asymmetry,
            'executive_concentration': self.executive_concentration,
            'security_concentration': self.security_concentration,
            'title_diversity': self.title_diversity,
            'days_since_last_transaction': self.days_since_last_transaction,
            'trading_consistency': self.trading_consistency,
            'day_of_week_concentration': self.day_of_week_concentration,
            'volume_change_yoy': self.volume_change_yoy,
            'volume_change_prev_period': self.volume_change_prev_period,
            'buy_ratio_change': self.buy_ratio_change,
            'insider_network_density': self.insider_network_density,
            'transaction_timing_correlation': self.transaction_timing_correlation,
            'net_share_change': self.net_share_change,
            'acquisition_disposal_ratio': self.acquisition_disposal_ratio
        }

        # Create feature DataFrame
        features_dict = {}
        for feature_name in selected_features:
            if feature_name in feature_methods:
                try:
                    feature_series = feature_methods[feature_name]()
                    features_dict[feature_name] = feature_series.values
                    self.logger.debug(f"Computed feature: {feature_name}")
                except Exception as e:
                    self.logger.error(f"Error computing {feature_name}: {e}")
                    features_dict[feature_name] = np.zeros(len(self.aggregated_df))
            else:
                self.logger.warning(f"Unknown feature: {feature_name}")

        # Create DataFrame with window dates as index
        feature_matrix = pd.DataFrame(features_dict)
        feature_matrix.index = self.aggregated_df['window_end']
        feature_matrix.index.name = 'window_end_date'

        # Handle missing values
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)

        if handle_missing == 'fill':
            # Forward fill, then backward fill for any remaining
            feature_matrix = feature_matrix.ffill().bfill()
            # If still any NaN, fill with 0
            feature_matrix = feature_matrix.fillna(0)
        elif handle_missing == 'drop':
            initial_len = len(feature_matrix)
            feature_matrix = feature_matrix.dropna()
            dropped = initial_len - len(feature_matrix)
            if dropped > 0:
                self.logger.info(f"Dropped {dropped} rows with missing values")

        # Calculate and store feature statistics
        self._feature_stats = {
            col: {
                'mean': feature_matrix[col].mean(),
                'std': feature_matrix[col].std(),
                'min': feature_matrix[col].min(),
                'max': feature_matrix[col].max(),
                'median': feature_matrix[col].median()
            }
            for col in feature_matrix.columns
        }

        self.feature_matrix = feature_matrix
        self.logger.info(
            f"Feature matrix created: {feature_matrix.shape[0]} windows x "
            f"{feature_matrix.shape[1]} features"
        )

        return feature_matrix

    def get_feature_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for all computed features.

        Returns:
            Dictionary of feature statistics
        """
        return self._feature_stats.copy()

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix for feature matrix.

        Returns:
            Correlation matrix DataFrame
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not created. Call create_feature_matrix first.")

        return self.feature_matrix.corr()

    def export_feature_matrix(
        self,
        filepath: str,
        include_aggregated: bool = False
    ) -> str:
        """
        Export feature matrix to CSV.

        Args:
            filepath: Path to save CSV
            include_aggregated: Whether to also export aggregated data

        Returns:
            Path to saved file
        """
        if self.feature_matrix is None:
            raise ValueError("Feature matrix not created. Call create_feature_matrix first.")

        from pathlib import Path
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        self.feature_matrix.to_csv(filepath)
        self.logger.info(f"Exported feature matrix to {filepath}")

        if include_aggregated:
            agg_path = filepath.with_suffix('.aggregated.csv')
            export_cols = [
                'window_idx', 'window_start', 'window_end',
                'transaction_count', 'total_volume', 'unique_executives',
                'unique_securities', 'buy_count', 'sell_count',
                'buy_volume', 'sell_volume'
            ]
            self.aggregated_df[export_cols].to_csv(agg_path, index=False)
            self.logger.info(f"Exported aggregated data to {agg_path}")

        return str(filepath)

    def get_window_details(self, window_idx: int) -> Dict:
        """
        Get detailed information about a specific window.

        Args:
            window_idx: Index of the window

        Returns:
            Dictionary with window details
        """
        if self.aggregated_df is None:
            raise ValueError("No aggregated data. Call aggregate_by_window first.")

        if window_idx < 0 or window_idx >= len(self.aggregated_df):
            raise ValueError(f"Invalid window index: {window_idx}")

        row = self.aggregated_df.iloc[window_idx]
        window_transactions = self._window_df[
            self._window_df['window_idx'] == window_idx
        ]

        return {
            'window_idx': window_idx,
            'window_start': row['window_start'],
            'window_end': row['window_end'],
            'transaction_count': row['transaction_count'],
            'total_volume': row['total_volume'],
            'unique_executives': row['unique_executives'],
            'unique_securities': row['unique_securities'],
            'buy_count': row['buy_count'],
            'sell_count': row['sell_count'],
            'buy_volume': row['buy_volume'],
            'sell_volume': row['sell_volume'],
            'transactions': window_transactions.to_dict('records')
        }

    def __repr__(self) -> str:
        """String representation of the feature engineer."""
        n_windows = len(self.aggregated_df) if self.aggregated_df is not None else 0
        n_features = (
            self.feature_matrix.shape[1] if self.feature_matrix is not None else 0
        )
        return (
            f"FeatureEngineer(transactions={len(self.df)}, "
            f"window_days={self.aggregation_window_days}, "
            f"windows={n_windows}, features={n_features})"
        )
