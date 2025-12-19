"""
Alpha Vantage API Integration for the Insider Trading Anomaly Detection System.

This module provides functionality to:
- Fetch stock price data (daily, weekly, monthly, intraday)
- Fetch insider transactions data
- Track locally cached data
- Manage API rate limits

Classes:
    AlphaVantageClient: Main client for API interactions
    DataInventory: Tracks locally downloaded data
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests

from . import constants
from .utils import setup_logging


class AlphaVantageClient:
    """
    Client for Alpha Vantage API interactions.

    Supports fetching stock data and insider transactions with
    rate limiting and local caching.

    Attributes:
        api_key: Alpha Vantage API key
        base_url: API base URL
        data_dir: Directory for cached data
        rate_limit: Maximum requests per minute
    """

    BASE_URL = "https://www.alphavantage.co/query"

    # API function mappings
    FUNCTIONS = {
        'daily': 'TIME_SERIES_DAILY',
        'daily_adjusted': 'TIME_SERIES_DAILY_ADJUSTED',
        'weekly': 'TIME_SERIES_WEEKLY',
        'weekly_adjusted': 'TIME_SERIES_WEEKLY_ADJUSTED',
        'monthly': 'TIME_SERIES_MONTHLY',
        'monthly_adjusted': 'TIME_SERIES_MONTHLY_ADJUSTED',
        'intraday': 'TIME_SERIES_INTRADAY',
        'quote': 'GLOBAL_QUOTE',
        'search': 'SYMBOL_SEARCH',
        'insider_transactions': 'INSIDER_TRANSACTIONS',
        'company_overview': 'OVERVIEW',
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: Optional[Path] = None,
        rate_limit: int = 5
    ):
        """
        Initialize Alpha Vantage client.

        Args:
            api_key: API key (or set ALPHA_VANTAGE_API_KEY env var)
            data_dir: Directory for storing downloaded data
            rate_limit: Max requests per minute (free tier: 5/min, 25/day)
        """
        self.logger = setup_logging(__name__)

        self.api_key = api_key or os.environ.get('ALPHA_VANTAGE_API_KEY', '')
        if not self.api_key:
            self.logger.warning(
                "No API key provided. Set ALPHA_VANTAGE_API_KEY environment "
                "variable or pass api_key parameter."
            )

        self.data_dir = data_dir or constants.RAW_DATA_DIR / "alpha_vantage"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limit = rate_limit
        self._last_request_time = 0
        self._request_count = 0
        self._request_count_reset = time.time()

        # Initialize data inventory
        self.inventory = DataInventory(self.data_dir)

    def _rate_limit_wait(self) -> None:
        """Enforce rate limiting between API calls."""
        # Minimum wait between requests (based on requests per minute)
        min_interval = 60.0 / self.rate_limit

        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            self.logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
            time.sleep(wait_time)

        self._last_request_time = time.time()

    def _make_request(self, params: Dict) -> Dict:
        """
        Make API request with error handling.

        Args:
            params: Query parameters

        Returns:
            JSON response as dictionary

        Raises:
            ValueError: If API returns an error
            requests.RequestException: On network errors
        """
        if not self.api_key:
            raise ValueError("API key is required. Please set your Alpha Vantage API key.")

        self._rate_limit_wait()

        params['apikey'] = self.api_key
        params['datatype'] = 'json'

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            if 'Note' in data:
                self.logger.warning(f"API Note: {data['Note']}")
                if 'rate limit' in data['Note'].lower():
                    raise ValueError("Rate limit exceeded. Please wait and try again.")
            if 'Information' in data:
                self.logger.info(f"API Info: {data['Information']}")

            return data

        except requests.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise

    def get_stock_data(
        self,
        symbol: str,
        interval: str = 'daily',
        outputsize: str = 'full',
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Fetch stock price data.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
            interval: 'daily', 'weekly', 'monthly', or intraday intervals
            outputsize: 'compact' (100 points) or 'full' (20+ years)
            adjusted: Whether to get adjusted prices

        Returns:
            DataFrame with OHLCV data
        """
        # Determine function
        if interval in ['1min', '5min', '15min', '30min', '60min']:
            function = self.FUNCTIONS['intraday']
            params = {
                'function': function,
                'symbol': symbol,
                'interval': interval,
                'outputsize': outputsize
            }
        else:
            func_key = f"{interval}_adjusted" if adjusted else interval
            function = self.FUNCTIONS.get(func_key, self.FUNCTIONS['daily'])
            params = {
                'function': function,
                'symbol': symbol,
                'outputsize': outputsize
            }

        self.logger.info(f"Fetching {interval} data for {symbol}...")
        data = self._make_request(params)

        # Parse response
        df = self._parse_time_series(data, symbol)

        if len(df) > 0:
            # Cache the data
            self._save_data(symbol, interval, df)
            self.inventory.update(symbol, interval, df)

        return df

    def _parse_time_series(self, data: Dict, symbol: str) -> pd.DataFrame:
        """Parse time series response into DataFrame."""
        # Find the time series key
        ts_key = None
        for key in data.keys():
            if 'Time Series' in key or 'Weekly' in key or 'Monthly' in key:
                ts_key = key
                break

        if not ts_key:
            self.logger.warning(f"No time series data found for {symbol}")
            return pd.DataFrame()

        ts_data = data[ts_key]

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df.index.name = 'date'

        # Standardize column names
        col_mapping = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. adjusted close': 'adjusted_close',
            '5. volume': 'volume',
            '6. volume': 'volume',
            '7. dividend amount': 'dividend',
            '8. split coefficient': 'split_coefficient'
        }

        df.columns = [col_mapping.get(c.lower(), c) for c in df.columns]

        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_index()
        df['symbol'] = symbol

        return df

    def get_insider_transactions(self, symbol: str) -> pd.DataFrame:
        """
        Fetch insider transactions data.

        Note: This is a premium endpoint and may not be available
        on free tier.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with insider transaction data
        """
        params = {
            'function': self.FUNCTIONS['insider_transactions'],
            'symbol': symbol
        }

        try:
            self.logger.info(f"Fetching insider transactions for {symbol}...")
            data = self._make_request(params)

            if 'data' not in data:
                self.logger.warning(
                    f"No insider transactions data for {symbol}. "
                    "This may require a premium API subscription."
                )
                return pd.DataFrame()

            df = pd.DataFrame(data['data'])

            if len(df) > 0:
                # Standardize column names and types
                if 'transaction_date' in df.columns:
                    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
                if 'shares' in df.columns:
                    df['shares'] = pd.to_numeric(df['shares'], errors='coerce')

                # Cache the data
                self._save_data(symbol, 'insider', df)
                self.inventory.update(symbol, 'insider', df)

            return df

        except ValueError as e:
            if 'premium' in str(e).lower():
                self.logger.warning(
                    f"Insider transactions requires premium API access. "
                    f"Error: {e}"
                )
            raise

    def get_company_overview(self, symbol: str) -> Dict:
        """
        Fetch company fundamental data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with company information
        """
        params = {
            'function': self.FUNCTIONS['company_overview'],
            'symbol': symbol
        }

        self.logger.info(f"Fetching company overview for {symbol}...")
        return self._make_request(params)

    def search_symbols(self, keywords: str) -> pd.DataFrame:
        """
        Search for stock symbols by keywords.

        Args:
            keywords: Search keywords

        Returns:
            DataFrame with matching symbols
        """
        params = {
            'function': self.FUNCTIONS['search'],
            'keywords': keywords
        }

        data = self._make_request(params)

        if 'bestMatches' not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data['bestMatches'])

        # Standardize column names
        col_mapping = {
            '1. symbol': 'symbol',
            '2. name': 'name',
            '3. type': 'type',
            '4. region': 'region',
            '5. marketOpen': 'market_open',
            '6. marketClose': 'market_close',
            '7. timezone': 'timezone',
            '8. currency': 'currency',
            '9. matchScore': 'match_score'
        }

        df.columns = [col_mapping.get(c, c) for c in df.columns]

        return df

    def get_quote(self, symbol: str) -> Dict:
        """
        Get current quote for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with current price data
        """
        params = {
            'function': self.FUNCTIONS['quote'],
            'symbol': symbol
        }

        data = self._make_request(params)

        if 'Global Quote' not in data:
            return {}

        quote = data['Global Quote']

        # Standardize keys
        return {
            'symbol': quote.get('01. symbol', symbol),
            'open': float(quote.get('02. open', 0)),
            'high': float(quote.get('03. high', 0)),
            'low': float(quote.get('04. low', 0)),
            'price': float(quote.get('05. price', 0)),
            'volume': int(float(quote.get('06. volume', 0))),
            'latest_trading_day': quote.get('07. latest trading day'),
            'previous_close': float(quote.get('08. previous close', 0)),
            'change': float(quote.get('09. change', 0)),
            'change_percent': quote.get('10. change percent', '0%')
        }

    def _save_data(
        self,
        symbol: str,
        data_type: str,
        df: pd.DataFrame
    ) -> Path:
        """Save data to local cache."""
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{symbol}_{data_type}_{timestamp}.csv"
        filepath = self.data_dir / filename

        df.to_csv(filepath)
        self.logger.info(f"Saved data to {filepath}")

        return filepath

    def load_cached_data(
        self,
        symbol: str,
        data_type: str = 'daily'
    ) -> Optional[pd.DataFrame]:
        """
        Load previously cached data.

        Args:
            symbol: Stock ticker symbol
            data_type: Type of data ('daily', 'weekly', 'monthly', 'insider')

        Returns:
            DataFrame if found, None otherwise
        """
        return self.inventory.get_cached_data(symbol, data_type)


class DataInventory:
    """
    Tracks locally downloaded data from Alpha Vantage.

    Maintains an index of what data has been downloaded,
    date ranges available, and last update times.
    """

    INVENTORY_FILE = "data_inventory.json"

    def __init__(self, data_dir: Path):
        """
        Initialize data inventory.

        Args:
            data_dir: Directory containing cached data
        """
        self.data_dir = Path(data_dir)
        self.inventory_path = self.data_dir / self.INVENTORY_FILE
        self.logger = setup_logging(__name__)

        self._inventory = self._load_inventory()

    def _load_inventory(self) -> Dict:
        """Load inventory from file."""
        if self.inventory_path.exists():
            try:
                with open(self.inventory_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load inventory: {e}")

        return {'symbols': {}, 'last_updated': None}

    def _save_inventory(self) -> None:
        """Save inventory to file."""
        self._inventory['last_updated'] = datetime.now().isoformat()
        with open(self.inventory_path, 'w') as f:
            json.dump(self._inventory, f, indent=2, default=str)

    def update(
        self,
        symbol: str,
        data_type: str,
        df: pd.DataFrame
    ) -> None:
        """
        Update inventory with new data.

        Args:
            symbol: Stock ticker symbol
            data_type: Type of data
            df: DataFrame with the data
        """
        if symbol not in self._inventory['symbols']:
            self._inventory['symbols'][symbol] = {}

        if len(df) == 0:
            return

        # Get date range
        if df.index.name == 'date' or 'date' in str(df.index.dtype):
            date_col = df.index
        elif 'date' in df.columns:
            date_col = pd.to_datetime(df['date'])
        elif 'transaction_date' in df.columns:
            date_col = pd.to_datetime(df['transaction_date'])
        else:
            date_col = None

        entry = {
            'last_updated': datetime.now().isoformat(),
            'row_count': len(df),
            'columns': list(df.columns)
        }

        if date_col is not None:
            entry['start_date'] = str(date_col.min())[:10]
            entry['end_date'] = str(date_col.max())[:10]

        self._inventory['symbols'][symbol][data_type] = entry
        self._save_inventory()

    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get information about cached data for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with available data types and date ranges
        """
        return self._inventory['symbols'].get(symbol, {})

    def get_all_symbols(self) -> List[str]:
        """Get list of all symbols with cached data."""
        return list(self._inventory['symbols'].keys())

    def get_cached_data(
        self,
        symbol: str,
        data_type: str
    ) -> Optional[pd.DataFrame]:
        """
        Load cached data for a symbol.

        Args:
            symbol: Stock ticker symbol
            data_type: Type of data

        Returns:
            DataFrame if found, None otherwise
        """
        # Find the most recent file
        pattern = f"{symbol}_{data_type}_*.csv"
        files = list(self.data_dir.glob(pattern))

        if not files:
            return None

        # Get most recent file
        latest = max(files, key=lambda f: f.stat().st_mtime)

        try:
            df = pd.read_csv(latest, index_col=0, parse_dates=True)
            self.logger.info(f"Loaded cached data from {latest}")
            return df
        except Exception as e:
            self.logger.warning(f"Could not load cached data: {e}")
            return None

    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all cached data.

        Returns:
            DataFrame with symbol, data types, date ranges
        """
        rows = []
        for symbol, data_types in self._inventory['symbols'].items():
            for data_type, info in data_types.items():
                rows.append({
                    'symbol': symbol,
                    'data_type': data_type,
                    'start_date': info.get('start_date', 'N/A'),
                    'end_date': info.get('end_date', 'N/A'),
                    'rows': info.get('row_count', 0),
                    'last_updated': info.get('last_updated', 'N/A')[:10]
                })

        return pd.DataFrame(rows)

    def has_data(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if data exists for a symbol within date range.

        Args:
            symbol: Stock ticker symbol
            data_type: Type of data
            start_date: Required start date
            end_date: Required end date

        Returns:
            Tuple of (has_data, cached_start, cached_end)
        """
        info = self.get_symbol_info(symbol).get(data_type)

        if not info:
            return False, None, None

        cached_start = info.get('start_date')
        cached_end = info.get('end_date')

        has_data = True

        if start_date and cached_start:
            if start_date < cached_start:
                has_data = False

        if end_date and cached_end:
            if end_date > cached_end:
                has_data = False

        return has_data, cached_start, cached_end
