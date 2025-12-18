"""
Sample Data Generator for the Insider Trading Anomaly Detection System.

This module generates realistic sample insider trading transaction data
for testing and demonstration purposes.

Functions:
    generate_sample_data: Generate sample transaction CSV file
"""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from . import constants


def generate_sample_data(
    output_path: Optional[str] = None,
    n_transactions: int = 1000,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    n_executives: int = 25,
    inject_anomalies: bool = True,
    random_seed: int = 42
) -> str:
    """
    Generate sample insider trading transaction data.

    Creates a realistic dataset with normal trading patterns and
    optionally injected anomalous periods for testing.

    Args:
        output_path: Path to save CSV (auto-generated if None)
        n_transactions: Number of transactions to generate
        start_date: Start date for transactions
        end_date: End date for transactions
        n_executives: Number of unique executives
        inject_anomalies: Whether to inject anomalous trading patterns
        random_seed: Random seed for reproducibility

    Returns:
        Path to saved CSV file
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Generate executives
    first_names = [
        "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert",
        "Lisa", "William", "Jennifer", "James", "Amanda", "Charles",
        "Jessica", "Thomas", "Ashley", "Christopher", "Michelle", "Daniel",
        "Stephanie", "Matthew", "Nicole", "Andrew", "Elizabeth", "Joseph"
    ]
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
        "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez",
        "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
        "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris"
    ]
    titles = [
        "CEO", "CFO", "COO", "CTO", "VP of Sales", "VP of Marketing",
        "VP of Operations", "Director", "Senior Director", "EVP",
        "General Counsel", "Controller", "Treasurer", "Secretary"
    ]

    executives = []
    exec_titles = {}
    for i in range(n_executives):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        while name in exec_titles:
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
        exec_titles[name] = random.choice(titles)
        executives.append(name)

    security_types = ["Common Stock", "Options", "Preferred Stock", "Restricted Stock"]

    # Generate date range
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days

    # Generate normal transactions
    transactions = []
    for _ in range(n_transactions):
        # Random date with slight weekday bias
        day_offset = np.random.randint(0, total_days)
        date = start + timedelta(days=day_offset)

        # Skip weekends with high probability
        if date.weekday() >= 5 and random.random() < 0.8:
            date = date - timedelta(days=date.weekday() - 4)

        executive = random.choice(executives)
        title = exec_titles[executive]
        security = random.choices(
            security_types,
            weights=[0.6, 0.25, 0.1, 0.05]
        )[0]

        # Normal transaction sizes
        if security == "Options":
            shares = int(np.random.lognormal(7, 1))  # ~1000 avg
        else:
            shares = int(np.random.lognormal(6.5, 1.2))  # ~700 avg

        # Buy/sell ratio (slightly more buys)
        action = random.choices(["Buy", "Sell"], weights=[0.55, 0.45])[0]

        transactions.append({
            'transaction_date': date.strftime("%Y-%m-%d"),
            'executive': executive,
            'executive_title': title,
            'security_type': security,
            'acquisition_or_disposal': action,
            'shares': shares
        })

    # Inject anomalies if requested
    if inject_anomalies:
        # Create 3-5 anomalous periods
        n_anomaly_periods = random.randint(3, 5)

        for _ in range(n_anomaly_periods):
            # Random anomaly start date
            anomaly_start_offset = np.random.randint(30, total_days - 30)
            anomaly_start = start + timedelta(days=anomaly_start_offset)
            anomaly_duration = random.randint(5, 14)  # 5-14 days

            # Select a subset of executives for coordinated trading
            n_involved = random.randint(3, 8)
            involved_execs = random.sample(executives, n_involved)

            # Generate anomalous transactions
            n_anomaly_transactions = random.randint(15, 40)

            # Determine if this is a buying or selling spree
            direction = random.choice(["Buy", "Sell"])
            direction_weight = 0.9 if random.random() < 0.7 else 0.6

            for _ in range(n_anomaly_transactions):
                day_offset = random.randint(0, anomaly_duration)
                date = anomaly_start + timedelta(days=day_offset)

                executive = random.choice(involved_execs)
                title = exec_titles[executive]
                security = "Common Stock"  # Focus on common stock

                # Larger transaction sizes
                shares = int(np.random.lognormal(8, 0.5))  # ~3000 avg

                # Biased direction
                action = random.choices(
                    ["Buy", "Sell"],
                    weights=[direction_weight, 1 - direction_weight]
                    if direction == "Buy"
                    else [1 - direction_weight, direction_weight]
                )[0]

                transactions.append({
                    'transaction_date': date.strftime("%Y-%m-%d"),
                    'executive': executive,
                    'executive_title': title,
                    'security_type': security,
                    'acquisition_or_disposal': action,
                    'shares': shares
                })

    # Create DataFrame and sort by date
    df = pd.DataFrame(transactions)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df = df.sort_values('transaction_date').reset_index(drop=True)
    df['transaction_date'] = df['transaction_date'].dt.strftime('%Y-%m-%d')

    # Save to file
    if output_path is None:
        output_path = constants.RAW_DATA_DIR / "sample_insider_trading.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} transactions")
    print(f"  Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print(f"  Executives: {df['executive'].nunique()}")
    print(f"  Anomaly periods injected: {inject_anomalies}")
    print(f"  Saved to: {output_path}")

    return str(output_path)


def generate_new_transactions(
    output_path: Optional[str] = None,
    n_transactions: int = 30,
    base_date: Optional[str] = None,
    executives: Optional[List[str]] = None,
    random_seed: int = 123
) -> str:
    """
    Generate new transactions for scoring.

    Args:
        output_path: Path to save CSV
        n_transactions: Number of transactions
        base_date: Base date for transactions (default: today)
        executives: List of executives (generates new if None)
        random_seed: Random seed

    Returns:
        Path to saved file
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    if base_date is None:
        base_date = datetime.now().strftime("%Y-%m-%d")

    base = datetime.strptime(base_date, "%Y-%m-%d")

    if executives is None:
        executives = [
            "John Smith", "Jane Doe", "Michael Johnson", "Sarah Williams",
            "David Brown", "Emily Davis", "Robert Wilson", "Lisa Anderson"
        ]

    titles = ["CEO", "CFO", "COO", "Director", "VP of Sales"]
    security_types = ["Common Stock", "Options"]

    transactions = []
    for _ in range(n_transactions):
        day_offset = random.randint(-30, 0)
        date = base + timedelta(days=day_offset)

        executive = random.choice(executives)
        title = random.choice(titles)
        security = random.choice(security_types)
        shares = int(np.random.lognormal(6.5, 1))
        action = random.choice(["Buy", "Sell"])

        transactions.append({
            'transaction_date': date.strftime("%Y-%m-%d"),
            'executive': executive,
            'executive_title': title,
            'security_type': security,
            'acquisition_or_disposal': action,
            'shares': shares
        })

    df = pd.DataFrame(transactions)
    df = df.sort_values('transaction_date').reset_index(drop=True)

    if output_path is None:
        output_path = constants.NEW_TRANSACTIONS_DIR / "new_transactions.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    return str(output_path)


if __name__ == '__main__':
    # Generate sample data when run directly
    generate_sample_data()
