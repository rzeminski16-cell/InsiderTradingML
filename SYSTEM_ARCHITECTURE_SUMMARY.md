# Insider Trading Anomaly Detection System - Architecture Summary

## Quick Overview

A **production-ready Python system** that identifies suspicious insider trading patterns through machine learning. The system processes raw transaction data, engineers time-series features, trains multiple anomaly detection models, detects historical suspicious periods, scores new transactions, and generates comprehensive Excel reports.

**End-to-end time**: ~60 seconds for typical dataset  
**Main deliverable**: Interactive GUI + automated Excel reporting

---

## System Architecture

┌─────────────────────────────────────────────────────────────────┐
│ GUI APPLICATION (PyQt5) │
│ │
│ ┌──────────────┬──────────────┬──────────────┬─────────────┐ │
│ │ Tab 1: │ Tab 2: │ Tab 3: │ Tab 4: │ │
│ │ Data Mgmt │ Features │ Models │ Anomalies │ │
│ │ │ │ │ │ │
│ │ - Load CSV │ - Window cfg │ - Select │ - Detect │ │
│ │ - Preview │ - Choose │ models │ anomalies │ │
│ │ - Filter │ features │ - Configure │ - New trans │ │
│ │ - Summary │ - Run eng │ params │ - Visualize │ │
│ └──────────────┴──────────────┴──────────────┴─────────────┘ │
│ │
│ Tab 5: Reporting │
│ [Report Config] [Generate] [Export] │
└─────────────────────────────────────────────────────────────────┘
↓ ↓ ↓
┌────────────────┐ ┌──────────────┐ ┌─────────────────┐
│ Data Layer │ │ Modeling │ │ Reporting │
│ │ │ Layer │ │ Layer │
│ CSV Files ←→ │ │ │ │ │
│ Preprocessor │ │ Model │ │ Excel Workbook │
│ │ │ Factory ←→ │ │ - 8 sheets │
│ ✓ Load │ │ │ │ - Charts │
│ ✓ Validate │ │ 6 Models: │ │ - Formatting │
│ ✓ Clean │ │ - IForest │ │ - Heatmaps │
│ │ │ - DBSCAN │ │ │
└────────────────┘ │ - LOF │ └─────────────────┘
│ - LSTM AE │
│ - KMeans │
│ - One-SVM │
│ │
│ ✓ Train │
│ ✓ Predict │
│ ✓ Score │
│ ✓ Ensemble │
└──────────────┘
↓
┌──────────────┐
│ Feature │
│ Engineering │
│ │
│ 20+ Features │
│ - Volume │
│ - Behavior │
│ - Temporal │
│ - Network │
│ │
│ ✓ Extract │
│ ✓ Aggregate │
│ ✓ Normalize │
└──────────────┘

text

---

## Data Flow Pipeline

Raw CSV Data
│
├─ transaction_date
├─ executive
├─ executive_title
├─ security_type
├─ acquisition_or_disposal
└─ shares
│
↓
DataPreprocessor
│
├─ Load & validate
├─ Parse dates
├─ Handle missing values
└─ Data quality report
│
↓
Cleaned DataFrame (1000+ transactions)
│
↓
FeatureEngineer.aggregate_by_window()
│
├─ Group by time window (14 days)
├─ Create temporal bins
└─ Generate base aggregations
│
↓
120–130 time windows (observations)
│
↓
FeatureEngineer.create_feature_matrix()
│
├─ Volume-based features (5)
├─ Behavioral features (5)
├─ Temporal features (3)
├─ Comparative features (3)
├─ Network features (2)
└─ Other features (2+)
│
↓
Feature Matrix: (120 windows × 20 features)
│
├──→ Preview in GUI Tab 2
├──→ Export as CSV
└──→ Pass to models
│
↓
ModelFactory.train_model() [×6 models]
│
├─ IsolationForest
├─ DBSCAN
├─ Local Outlier Factor
├─ LSTM Autoencoder
├─ K-Means
└─ One-Class SVM
│
↓
6 Trained Models + Predictions
│
├─ Historical anomaly scores (0–1 per window)
├─ Model-specific hyperparameters
├─ Training metrics (time, convergence)
└─ Flagged anomalies (binary)
│
↓
ModelFactory.detect_historical_anomalies()
│
├─ Rank windows by anomaly score
├─ Apply threshold (95th percentile)
├─ Flag top N anomalies
└─ Get model consensus (if ensemble)
│
├──→ Display in GUI Tab 4 (Historical Anomalies table)
└──→ Pass to reporter
│
↓
New Transaction Data (30 transactions, May 2024)
│
├─ Same format as raw data
├─ Create latest window features
└─ Score with trained model
│
↓
ModelFactory.predict_on_new_data()
│
├─ Anomaly scores for each transaction
├─ Risk level classification
├─ Confidence scores
└─ Percentile rank vs. historical
│
├──→ Display in GUI Tab 4 (New Transactions table)
└──→ Pass to reporter
│
↓
ExcelReporter.generate_full_report()
│
├─ Sheet 1: Executive Summary (KPIs, findings)
├─ Sheet 2: Historical Anomalies (table + charts)
├─ Sheet 3: Feature Engineering Details
├─ Sheet 4: Model Performance & Comparison
├─ Sheet 5: New Transaction Scoring
├─ Sheet 6: Detailed Anomaly Explanations
├─ Sheet 7: Technical Specifications
└─ Sheet 8: Appendix & Methodology
│
↓
Excel Report (.xlsx file)
│
├──→ 8 professionally formatted sheets
├──→ Embedded charts and heatmaps
├──→ Color-coded risk levels
└──→ Ready for stakeholder distribution

text

---

## Key Components

### 1. DataPreprocessor

**Purpose**: Load and validate raw insider trading data.

Methods:

load_raw_data(filepath) → DataFrame

get_date_range() → (start_date, end_date)

filter_by_date_range(df, start, end) → DataFrame

filter_by_executive(df, name) → DataFrame

get_executives_list() → List[str]

get_summary_stats() → Dict

text

**Input**: CSV file with 6 columns  
**Output**: Cleaned pandas DataFrame

---

### 2. FeatureEngineer

**Purpose**: Create 20+ time-series features from raw transactions.

Features Categories:

Volume-Based (5):

total_transaction_volume

transaction_count

avg_transaction_size

volume_volatility

volume_std_deviation

Behavioral (5):

buy_sell_ratio

buy_sell_asymmetry

executive_concentration

security_concentration

title_diversity

Temporal (3):

days_since_last_transaction

trading_consistency

day_of_week_concentration

Comparative (3):

volume_change_yoy

volume_change_prev_period

buy_ratio_change

Network (2):

insider_network_density

transaction_timing_correlation

Key Methods:

init(df, aggregation_window_days=)​

aggregate_by_window(window_days) → time-windowed data

create_feature_matrix(selected_features) → (n_windows, n_features)

get_available_features() → List[str]

text

**Input**: Cleaned DataFrame from DataPreprocessor  
**Output**: Feature matrix (e.g., 120 windows × 20 features), CSV export

---

### 3. ModelFactory

**Purpose**: Train and manage 6 different anomaly detection models.

Supported Models:

Isolation Forest
├─ contamination: controls % anomalies
├─ Fast, good for high dimensions
└─ Explicitly isolates anomalies

DBSCAN
├─ Density-based clustering
├─ Finds outliers as noise points
└─ No parameter K required

Local Outlier Factor (LOF)
├─ Detects local density deviations
├─ Good for varying density regions
└─ Novelty mode for new data

LSTM Autoencoder
├─ Deep learning: Encoder-Decoder
├─ Trains on clean data
└─ Flags high reconstruction error

K-Means
├─ Cluster-based approach
├─ Points far from centroids = anomalies
└─ Simple but effective

One-Class SVM
├─ Learn normal data boundary
├─ Flag data outside boundary
└─ Non-linear kernel

Key Methods:

train_model(config, X_train, feature_names) → trained_model

predict_anomaly(X, threshold) → binary predictions​

get_anomaly_scores(X) → continuous scores​

detect_historical_anomalies(X, percentile=95) → anomaly table

classify_new_transaction(df_raw, feature_engineer) → scored table

ensemble_predict(models, X, voting_method) → consensus predictions

text

**Input**: Feature matrix from FeatureEngineer  
**Output**: Trained models, predictions, anomaly rankings

---

### 4. GUI (PyQt5)

**Purpose**: Interactive interface for full workflow.

5 Tabs:

TAB 1: Data Management

Load raw CSV data

Preview first 20 rows

Filter by executive, date range

Data quality summary

TAB 2: Feature Engineering

Configure aggregation window (7/14/30 days)

Select features to engineer (checkboxes)

Run feature engineering

Preview feature matrix

View correlations heatmap

Export feature matrix

TAB 3: Model Training & Configuration

Select models (checkboxes for 6 available)

Configure hyperparameters per model (modal dialogs)

Specify train/test split method

Train all models

Compare results side-by-side

View training logs

TAB 4: Anomaly Detection & Scoring

Select active model from dropdown

Configure anomaly threshold (percentile, std, etc)

Detect historical anomalies

Browse anomaly table (sorted by score)

Load new transaction data

Score new transactions

View visualizations (time series, heatmaps)

Cross-model comparison

TAB 5: Reporting

Select report sections to include

Choose format (Excel, PDF, or both)

Configure report title/description

Generate report

Open in default application

text

**Input**: User interactions via GUI  
**Output**: Visual feedback, data tables, charts, Excel export

---

### 5. ExcelReporter

**Purpose**: Generate professional 8-sheet Excel reports.

8 Sheets Generated:

SHEET 1: Executive Summary
├─ Key metrics (KPIs)
├─ Methodology overview
├─ Key findings (3–5 bullets)
├─ Recommendations
└─ Table of contents with hyperlinks

SHEET 2: Historical Anomalies
├─ Anomalies table (sorted by score)
├─ Columns: date, scores by model, risk level, volume, execs
├─ Color formatting (red=high, yellow=medium, green=low)
├─ Charts: scatter, time series, bar, heatmap
└─ Summary statistics

SHEET 3: Feature Engineering Details
├─ Feature statistics table (mean, std, min, max)
├─ Feature importance for each model
├─ Correlation matrix (heatmap)
├─ Distributions of top features
└─ Feature engineering methodology

SHEET 4: Model Performance & Comparison
├─ Model training summary table
├─ Prediction comparison (all models)
├─ Model divergence analysis
├─ Venn diagram / upset plot for overlaps
├─ ROC/PR curves (if ground truth)
└─ Score distributions

SHEET 5: New Transaction Scoring
├─ Table: new transactions with scores
├─ Color-coded risk levels
├─ Pie chart of risk categories
├─ Time series of risk scores
└─ Summary and flagged transactions

SHEET 6: Detailed Anomaly Explanations
├─ Per anomaly explanation
├─ Raw transactions view
├─ Executive network view
├─ Feature deviations chart
└─ Investigator notes

SHEET 7: Technical Specifications
├─ Data specifications
├─ Feature engineering details
├─ Model specifications
├─ Threshold specifications
└─ Report generation details

SHEET 8: Appendix & Methodology
├─ Glossary of terms
├─ Methodology overview
├─ Limitations & considerations
└─ Contact & questions

Formatting:

Color scheme: Red/Yellow/Green for risk

Fonts: Headers 14pt bold, data 11pt

Numbers: 2–3 decimals for scores, currency for volume

Charts: Professional styling with legends

Conditional formatting: Color scales for scores

Frozen headers & filters for all tables

Print-ready layout

text

**Input**: Model outputs, feature matrix, new transactions, metadata  
**Output**: Excel workbook `.xlsx`

---

## Feature Engineering Deep Dive

### 20+ Features Explained

VOLUME-BASED FEATURES (activity spikes):

total_transaction_volume

Sum of all shares traded in window

High = activity spike

transaction_count

Number of transactions in window

High = rapid trading, potential coordination

avg_transaction_size

Mean shares per transaction

High = larger than normal trades

volume_volatility

Std dev of transaction sizes

Low = uniform sizing (coordinated)

volume_std_deviation

Z-score vs. 90-day baseline

Captures deviation from typical

BEHAVIORAL FEATURES (patterns of trading):
6. buy_sell_ratio
- Total buys / Total sells
- Extreme ratios (e.g., 0.9 or 0.1) = unusual

buy_sell_asymmetry

(Buy Count - Sell Count) / Total

Range [-1,1]: 1=all buys, -1=all sells

executive_concentration

Herfindahl index of executives

High (≈0.9) = few execs dominate

security_concentration

Herfindahl index of securities

High = focus on one/few securities

title_diversity

Count of unique titles

Cross-level coordination unusual

TEMPORAL FEATURES (timing behavior):
11. days_since_last_transaction
- Days since previous trade
- Activity after long quiet period = suspicious

trading_consistency

Days with trades / days in window

High = intensive trading period

day_of_week_concentration

Concentration of trade activity on specific weekdays

COMPARATIVE FEATURES (change vs past):
14. volume_change_yoy
- Year-over-year volume change

volume_change_prev_period

Change vs previous window

buy_ratio_change

Change in buy/sell ratio vs prior period

NETWORK FEATURES (coordination between insiders):
17. insider_network_density
- Count of unique executive pairs co-trading
- High = many insiders trading together

transaction_timing_correlation

Correlation of transaction timings between execs

High = coordinated timing

ACQUISITION/DISPOSAL FEATURES:
19. net_share_change
- Sum(buys) - Sum(sells)
- Net accumulation vs liquidation

acquisition_disposal_ratio

Count acquisitions / disposals

Directional bias

text

### How Features Detect Insider Trading

Red Flags Captured:

🚩 ACTIVITY SPIKE

total_transaction_volume (e.g., 4× baseline)

transaction_count (many trades in short period)

volume_change_prev_period (abrupt spike)

🚩 COORDINATED BEHAVIOR

executive_concentration (few execs dominate volume)

transaction_timing_correlation (aligned timing)

insider_network_density (many execs trading together)

title_diversity (multiple hierarchy levels)

🚩 UNUSUAL PATTERNS

buy_sell_ratio (extreme bias to buy or sell)

volume_volatility (uniform trade sizes)

security_concentration (one security targeted)

buy_sell_asymmetry (all buys or all sells)

🚩 TEMPORAL ANOMALIES

days_since_last_transaction (sudden activity after silence)

trading_consistency (sustained flurry of trades)

day_of_week_concentration (clustered on specific days)

🚩 DIRECTIONAL SHIFTS

buy_ratio_change (sudden shift from balanced)

volume_change_yoy (unusual vs previous year)

net_share_change (large net accumulation pre-event)

text

---

## Model Selection Guide

WHEN TO USE EACH MODEL:

IsolationForest ★ RECOMMENDED

Use for: Fast baseline, high-dimensional features

Pros: Efficient, intuitive contamination parameter

Cons: Less sensitive to very subtle, local anomalies

DBSCAN ★ RECOMMENDED

Use for: Cluster-based anomalies, natural grouping

Pros: No need to pre-specify clusters, finds noise

Cons: Requires tuning eps/min_samples

LocalOutlierFactor (LOF) ✓ GOOD

Use for: Local density anomalies

Pros: Good when density varies

Cons: Less straightforward scoring for new data

LSTM Autoencoder ✓ GOOD

Use for: Temporal patterns and sequences

Pros: Captures complex time dependencies

Cons: Slowest, needs clean training data

K-Means ✓ SIMPLE

Use for: Quick clustering baseline

Pros: Very fast, easy to interpret

Cons: Assumes spherical clusters

One-Class SVM ✓ ALTERNATIVE

Use for: Non-linear anomaly boundaries

Pros: Flexible decision boundary

Cons: Sensitive to hyperparameters

ENSEMBLE APPROACH (BEST):

Combine all 6 models

Majority vote or average scores

High-confidence anomalies = flagged by multiple models

text

---

## GUI Workflow Example

USER WORKFLOW:

Launch Application

Opens on Tab 1: Data Management

Load Data (Tab 1)

Click "Load Data"

Choose insider_trading_raw.csv

See summary: #transactions, date range, #executives, buy/sell ratio

Preview top rows, filter if needed

Configure Features (Tab 2)

Select window size: 14 days

Select all feature groups

Run feature engineering

Review feature stats and correlations

Export feature matrix if desired

Train Models (Tab 3)

Select IsolationForest, DBSCAN, LOF, LSTM, K-Means, One-Class SVM

Configure hyperparameters via dialogs

Choose train/test split

Train all models

View model comparison table (anomalies detected, training time)

Detect Historical Anomalies (Tab 4)

Select ensemble as active model

Choose threshold: 95th percentile

Run anomaly detection

Review top anomaly windows, sorted by score

Inspect raw trades in each anomalous window

Score New Transactions (Tab 4)

Load new_transactions.csv

Score using trained models

See transaction-level scores and risk levels

Export results

Generate Report (Tab 5)

Choose sections and format (Excel)

Generate report

Open Excel and review all 8 sheets

Save Project

Save project file with all data, models, settings

text

---

## Performance Metrics

EXPECTED RUNTIMES (typical dataset ~1,000–5,000 transactions):

Component Time
──────────────────────────────────────────
Load raw data < 1 s
Feature engineering 5–15 s
IsolationForest training ~0.3 s
DBSCAN training ~0.4 s
LOF training ~0.6 s
LSTM Autoencoder training 5–10 s
K-Means training ~0.2 s
One-Class SVM training ~0.5 s
All models training 8–15 s
Historical anomalies scoring 1–2 s
Excel report generation 5–10 s
──────────────────────────────────────────
End-to-end run ~60 s

Memory (approx):

Raw data (1000+ rows): ~50 MB

Feature matrix: ~2 MB

Models: ~5 MB

Report: ~3 MB
Total working set: ~60 MB

text

---

## Key Differentiators

WHY THIS SYSTEM IS PRODUCTION-READY:

✓ Multi-model ensemble

6 algorithms → robust, diverse perspectives

Ensemble voting → improved reliability

✓ Rich feature engineering

20+ engineered features

Modular, extensible framework

✓ Professional reporting

8-sheet Excel report

Clear visuals and explanations

✓ User-friendly GUI

Full workflow: data → features → models → anomalies → report

Minimal coding required for daily use

✓ Solid architecture

Clear separation of concerns

Logging, error handling, configs, persistence

✓ Explainability

Feature importance analysis

Detailed anomaly explanations

Model comparison views

✓ Flexibility

Configurable windows, features, thresholds

Swappable models and hyperparameters

Extendable to new data sources/markets

text

---

## Next Steps After Implementation

AFTER THE SYSTEM IS BUILT:

Validate on real data

Run on historical insider trading data

Compare anomalies with known corporate events

Calibrate thresholds and contamination

Adjust sensitivity to match risk appetite

Ensure anomaly volume is actionable

Create labeled evaluation sets (if possible)

Use any known cases to evaluate precision/recall

Operationalize

Schedule regular scoring (daily/weekly)

Integrate with email alerts or ticketing

Feed results to compliance team

Iterate and extend

Add new features (e.g., price reactions, options data)

Explore new models (e.g., transformers)

Integrate NLP on news/filings for richer context

text

---

## Summary

This architecture summary describes a **complete end-to-end system** for insider trading anomaly detection:

- Multiple unsupervised ML models working together.  
- Rich, domain-relevant feature engineering.  
- A structured, user-friendly GUI.  
- Automated, polished Excel reporting.  
- Strong focus on interpretability and operational use.

Use this file alongside `insider_trading_system_spec.md` and `SUBMISSION_GUIDE.md` to guide both LLM-based code generation and implementation planning.
