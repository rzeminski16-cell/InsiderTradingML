# Insider Trading Anomaly Detection System

A production-ready Python system for detecting suspicious insider trading patterns using unsupervised machine learning. The system processes transaction data, engineers time-series features, trains multiple anomaly detection models, and generates professional Excel reports.

## Features

- **6 Anomaly Detection Models**: Isolation Forest, DBSCAN, Local Outlier Factor, Neural Network Autoencoder, K-Means, One-Class SVM
- **20+ Engineered Features**: Volume-based, behavioral, temporal, comparative, and network features
- **Professional GUI**: 6-tab PyQt5 interface for complete workflow management
- **Excel Reports**: 9-sheet reports with charts, formatting, and anomaly timeline visualization
- **Alpha Vantage Integration**: Fetch stock data and insider transactions directly from the API
- **CLI Support**: Run analysis from command line for automation

## Installation

### Requirements

- Python 3.9+ (tested with Python 3.13)
- No TensorFlow required (uses sklearn for all models)

### Setup

```bash
# Clone or navigate to the project
cd InsiderTradingML

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- pandas, numpy - Data processing
- scikit-learn, scipy - Machine learning
- PyQt5 - GUI framework
- openpyxl, xlsxwriter - Excel reports
- matplotlib, seaborn - Visualization
- requests - Alpha Vantage API

## Quick Start

### Launch the GUI

```bash
python main.py
```

### Run from Command Line

```bash
# Process data and generate report
python main.py --cli --input data/raw/transactions.csv --output reports/

# With custom window size
python main.py --cli --input data.csv --window 21
```

## Using the GUI

The application has 6 tabs that guide you through the analysis workflow:

### Tab 1: Data Management

Load and preview your transaction data.

1. Click **"Load Raw Data (CSV)"** to select your CSV file
2. View summary statistics (total transactions, unique executives, date range)
3. Use filters to focus on specific executives or date ranges
4. Review data quality report

**Required CSV Columns:**
| Column | Description |
|--------|-------------|
| `transaction_date` | Date of transaction |
| `executive` | Name of the insider |
| `executive_title` | Position/title |
| `security_type` | Type of security traded |
| `acquisition_or_disposal` | Buy/Sell indicator (A/D or Buy/Sell) |
| `shares` | Number of shares |

### Tab 2: Feature Engineering

Configure and create features from your transaction data.

1. Set **Window Size** (7-90 days) - how transactions are aggregated
2. Select features to compute (default: all 20+ features)
3. Click **"Run Feature Engineering"**
4. Preview or export the feature matrix

**Feature Categories:**
- **Volume-Based**: transaction volume, count, average size, volatility
- **Behavioral**: buy/sell ratio, executive concentration, diversity metrics
- **Temporal**: trading frequency, consistency, day-of-week patterns
- **Comparative**: year-over-year changes, period comparisons
- **Network**: insider correlation, timing patterns

### Tab 3: Model Training

Train anomaly detection models on the feature matrix.

1. Select models to train (checkboxes)
2. Click **"Configure Selected Models"** to adjust hyperparameters
3. Set train/test split ratio
4. Click **"Train All Selected Models"**
5. Review results in the comparison table

**Available Models:**
| Model | Best For |
|-------|----------|
| Isolation Forest | General anomaly detection, fast |
| DBSCAN | Density-based clustering |
| Local Outlier Factor | Local density deviations |
| NNAutoencoder | Complex pattern reconstruction |
| K-Means | Cluster-based outliers |
| One-Class SVM | Boundary-based detection |

### Tab 4: Anomaly Detection

Detect anomalies and score transactions.

1. Select the trained model to use
2. Configure threshold:
   - **Percentile**: Flag top X% as anomalies (e.g., 95th percentile)
   - **Fixed threshold**: Set exact score cutoff (0-1)
3. Click **"Detect Anomalies"** to find historical anomalies
4. Load and score new transactions

**Risk Levels:**
| Level | Score Range | Description |
|-------|-------------|-------------|
| CRITICAL | >= 0.85 | Requires immediate review |
| HIGH | >= 0.70 | Significant deviation |
| MEDIUM | >= 0.50 | Notable unusual activity |
| LOW | >= 0.30 | Minor deviation |
| NORMAL | < 0.30 | Within expected range |

### Tab 5: Reporting

Generate professional Excel reports.

1. Select which sections to include
2. Enter report title and description
3. Click **"Generate Excel Report"**
4. Open the report directly from the app

**Report Sheets:**
1. **Executive Summary** - Key metrics and findings
2. **Historical Anomalies** - Flagged periods with scores
3. **Feature Engineering** - Feature statistics and importance
4. **Model Performance** - Training metrics and comparison
5. **New Transaction Scoring** - Scored new transactions
6. **Anomaly Explanations** - Why each anomaly was flagged
7. **Technical Specifications** - Methods and parameters
8. **Appendix** - Glossary and methodology
9. **Anomaly Timeline** - Time series charts of anomaly activity

### Tab 6: Data Fetcher (Alpha Vantage)

Fetch stock data directly from Alpha Vantage API.

1. Enter your API key (get free at [alphavantage.co](https://www.alphavantage.co/))
2. Click **"Connect"** to test the connection
3. Search for stock symbols
4. Select data type and output size
5. View cached data for the symbol (shows what you already have)
6. Click **"Fetch Data from Alpha Vantage"**
7. Export to CSV for analysis

**Data Types Available:**
- Daily/Weekly/Monthly price data (adjusted or unadjusted)
- Insider Transactions (premium API required)

**Rate Limits:**
- Free tier: 25 requests/day, 5 requests/minute
- Data is cached locally to avoid redundant API calls

## Configuration Files

Located in `config/`:

### hyperparameter_config.json
Default hyperparameters for each model. Customize before training.

```json
{
  "IsolationForest": {
    "contamination": 0.05,
    "n_estimators": 100
  },
  "NNAutoencoder": {
    "encoding_dim": 8,
    "hidden_layers": [32, 16, 8, 16, 32],
    "max_iter": 500
  }
}
```

### feature_config.json
Enable/disable specific features.

### system_config.json
General system settings (paths, logging, etc.)

## Project Structure

```
InsiderTradingML/
├── main.py                 # Entry point (GUI and CLI)
├── requirements.txt        # Dependencies
├── config/                 # Configuration files
│   ├── hyperparameter_config.json
│   ├── feature_config.json
│   └── system_config.json
├── src/
│   ├── constants.py        # All constants and defaults
│   ├── utils.py            # Utility functions
│   ├── data_preprocessor.py    # Data loading and validation
│   ├── feature_engineer.py     # Feature computation
│   ├── model_factory.py        # Model creation and training
│   ├── anomaly_scorer.py       # Scoring and thresholds
│   ├── excel_reporter.py       # Report generation
│   ├── alpha_vantage.py        # Alpha Vantage API client
│   ├── gui_main.py             # Main GUI application
│   ├── gui_components.py       # Reusable GUI widgets
│   └── sample_data.py          # Sample data generator
├── data/
│   ├── raw/                # Input CSV files
│   ├── processed/          # Processed data
│   └── new_transactions/   # New data to score
├── models/
│   ├── saved_models/       # Trained model files
│   └── training_logs/      # Training history
└── reports/                # Generated Excel reports
```

## Sample Data

Generate sample transaction data for testing:

```python
from src.sample_data import generate_sample_transactions

# Generate 1000 sample transactions
df = generate_sample_transactions(n_transactions=1000)
df.to_csv('data/raw/sample_transactions.csv', index=False)
```

## Programmatic Usage

```python
from src.data_preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.model_factory import ModelFactory
from src.excel_reporter import ExcelReporter

# 1. Load data
preprocessor = DataPreprocessor()
df = preprocessor.load_raw_data('transactions.csv')

# 2. Engineer features
engineer = FeatureEngineer(df, window_days=14)
engineer.aggregate_by_window()
features = engineer.create_feature_matrix()

# 3. Train models
factory = ModelFactory()
factory.create_model('IsolationForest', {'contamination': 0.05})
predictions, metrics = factory.train_model('IsolationForest', features.values)

# 4. Detect anomalies
anomalies = factory.detect_historical_anomalies(
    'IsolationForest',
    features.values,
    features.index,
    percentile=95
)

# 5. Generate report
reporter = ExcelReporter('report.xlsx')
reporter.generate_full_report(
    summary_data={'total_transactions': len(df)},
    anomalies_df=anomalies,
    feature_stats=engineer.get_feature_statistics(),
    training_metrics={'IsolationForest': metrics}
)
```

## Understanding Results

### Anomaly Scores

Scores are normalized to 0-1 range:
- **0.0**: Completely normal behavior
- **0.5**: Moderately unusual
- **1.0**: Highly anomalous

### What Makes a Transaction Anomalous?

The system looks for deviations from normal patterns:
- Unusual trading volume compared to historical baseline
- Abnormal buy/sell ratios
- Trading activity spikes
- Concentrated activity by few executives
- Timing patterns that differ from norm
- Correlated trading among insiders

### False Positives

Not all flagged transactions are insider trading. High scores may indicate:
- Legitimate large transactions (M&A, stock buybacks)
- New executive activity
- Market-wide events
- Seasonal patterns

Always review flagged transactions with domain expertise.

## Troubleshooting

### "No module named 'src'"
Run from the project root directory, not from inside `src/`.

### GUI doesn't start
Ensure PyQt5 is installed: `pip install PyQt5`

### Alpha Vantage "Rate limit exceeded"
Free tier allows 25 requests/day. Wait 24 hours or upgrade to premium.

### Model training is slow
- Reduce feature count
- Use smaller window sizes
- Use 'compact' output size when fetching data

## License

This project is provided for educational and research purposes.

## Support

For issues and questions, please open an issue on the repository.
