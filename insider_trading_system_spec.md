# Insider Trading Anomaly Detection System - Complete Specification

## System Overview

Build a Python-based machine learning system that:
1. Processes raw insider trading transaction data
2. Engineers time-series features from raw data
3. Trains and runs multiple anomaly detection models via interactive GUI
4. Detects historical abnormal periods and classifies new transactions
5. Generates automated Excel reports with visualizations

**Architecture**: Data Processing Layer → Feature Engineering → Model Training → GUI Interface → Excel Reporting

---

## Part 1: Data Preparation & Feature Engineering Template

### 1.1 Raw Data Format

Input CSV structure:
transaction_date,executive,executive_title,security_type,acquisition_or_disposal,shares
2023-01-15,John Smith,CEO,Common Stock,Buy,5000
2023-01-16,Jane Doe,CFO,Options,Sell,1000

text

### 1.2 Data Preparation Pipeline

Create a `DataPreprocessor` class with these methods:

Class DataPreprocessor:

load_raw_data(filepath) -> DataFrame

Load CSV and parse transaction_date as datetime

Validate required columns present

Handle missing values (log and exclude if transaction_date or shares missing)

Validate shares > 0

Return cleaned DataFrame with data quality report

get_date_range() -> (start_date, end_date)

Return min and max transaction dates

filter_by_date_range(df, start_date, end_date) -> DataFrame

Allow users to subset data by date range for analysis

filter_by_executive(df, executive_name) -> DataFrame

Allow filtering for individual executive analysis

get_executives_list() -> List[str]

Return unique executives for filtering

get_summary_stats() -> Dict

Return data summary: transaction count, date range, unique executives, securities,
buy/sell ratio, total volume, date distribution

text

### 1.3 Feature Engineering Framework

Create a `FeatureEngineer` class with modular methods (user will call these sequentially):

**Aggregation Window Method**: All features computed over rolling time windows

Class FeatureEngineer:

init(df, aggregation_window_days=)​

Store DataFrame and window sizes for feature computation

TIME SERIES AGGREGATION:

aggregate_by_window(window_days) -> DataFrame

Create time bins of length window_days

For each window, calculate aggregate statistics

Return DataFrame with window_start_date, window_end_date as index

Each window becomes ONE observation for ML models

VOLUME-BASED FEATURES:

total_transaction_volume(agg_df) -> Series

Sum of shares per window

Indicator: High volume = potentially coordinated trading

transaction_count(agg_df) -> Series

Count of transactions per window

Indicator: Spike in transaction frequency

avg_transaction_size(agg_df) -> Series

Mean shares per transaction in window

Indicator: Larger than normal buys/sells

volume_volatility(agg_df) -> Series

Std dev of transaction sizes in window

Indicator: Coordinated trading shows lower volatility (uniform size)

volume_std_deviation(agg_df, baseline_window_days=90) -> Series

Z-score: (current_volume - mean_baseline) / std_baseline

Compare current window to rolling baseline

BEHAVIORAL FEATURES:

buy_sell_ratio(agg_df) -> Series

(Total Buy Shares / Total Sell Shares) per window

Indicator: Extreme ratios = unusual behavior

buy_sell_asymmetry(agg_df) -> Series

(Buy Count - Sell Count) / Total Transactions

Range [-1, 1]: 1 = all buys, -1 = all sells

executive_concentration(agg_df) -> Series

Herfindahl index of executives trading per window

Range: 1 = single executive, 0 = distributed​

Indicator: Coordinated trading shows high concentration

security_concentration(agg_df) -> Series

Herfindahl index of securities traded per window

Indicator: Focused trading on specific securities

title_diversity(agg_df) -> Series

Count of unique executive titles per window

Indicator: Cross-level coordination unusual

TEMPORAL FEATURES:

days_since_last_transaction(agg_df) -> Series

Days between current window start and last transaction

Indicator: Activity bursts after dormancy

trading_consistency(agg_df, window_days) -> Series

Ratio of days with trades / total days in window

Range: 1 = trading every day​

day_of_week_concentration(agg_df) -> Series

Concentration index of trading on specific days

Indicator: Trading clustered on specific days = anomalous

COMPARATIVE FEATURES (Year-over-Year / Period-over-Period):

volume_change_yoy(agg_df) -> Series

(Current volume - Same period last year) / Last year volume

Indicator: Significant YoY change

volume_change_prev_period(agg_df) -> Series

(Current volume - Previous period volume) / Previous period

Indicator: Rapid acceleration

buy_ratio_change(agg_df) -> Series

Change in buy/sell ratio from previous period

MULTI-EXECUTIVE COORDINATION:

insider_network_density(agg_df) -> Series

Count unique executive pairs co-trading in window

Indicator: Network clustering = coordinated activity

transaction_timing_correlation(agg_df) -> Series

Correlation of transaction timing between executives

If transactions clustered within same hours = high correlation

ACQUISITION DISPOSAL FEATURES:

net_share_change(agg_df) -> Series

Sum of buys - sum of sells (in shares)

acquisition_disposal_ratio(agg_df) -> Series

Acquisition count / Disposal count per window

COMBINE ALL FEATURES:

create_feature_matrix(selected_features_list) -> DataFrame

Takes list of feature method names

Calls each method sequentially

Combines into single DataFrame with index = window_date

Handle NaN/inf values: forward fill, then drop rows

Return feature matrix ready for modeling

Include feature statistics: mean, std, min, max per feature

get_available_features() -> List[str]

Return list of all available feature methods for user selection

text

### 1.4 Feature Engineering Output

Output: feature_matrix.csv
Index: window_end_date (datetime)
Columns: [total_volume, transaction_count, avg_transaction_size, volume_volatility,
buy_sell_ratio, executive_concentration, days_since_last_transaction,
volume_change_yoy, insider_network_density, ...]

text

---

## Part 2: Modeling System Specification

### 2.1 Model Architecture

Create a `ModelFactory` class that instantiates and manages multiple models:

Class ModelFactory:

init(feature_matrix_df, test_split_ratio=0.2)

Load feature matrix

Split into train (historical data up to test_split_ratio cutoff) and test (recent data)

Standardize features: (X - mean) / std using train set statistics

SUPPORTED MODELS:
Each model inherits from BaseAnomalyModel abstract class

a) IsolationForestModel(random_state=42):

contamination: float [0.01 to 0.1]

n_estimators: int [50 to 200]

max_samples: 'auto' or int

fit() -> self

predict(X) -> array of [-1 (anomaly), 1 (normal)]

score_samples(X) -> array of anomaly scores [-1 to 1]

get_hyperparameters() -> dict

b) DBSCANModel():

eps: float [0.3 to 2.0]

min_samples: int [3 to 10]

metric: 'euclidean'

fit() -> self

predict(X) -> array of [-1 (noise/anomaly), 0-K (cluster labels)]

score_samples(X) -> anomaly probability [0 to 1]

get_hyperparameters() -> dict

c) LocalOutlierFactorModel():

n_neighbors: int [5 to 20]

contamination: float [0.01 to 0.1]

novelty: True (can predict on new data)

fit() -> self

predict(X) -> array of [-1 (anomaly), 1 (normal)]

score_samples(X) -> array of anomaly scores

get_hyperparameters() -> dict

d) LSTMAutoencoderModel(sequence_length=7):

ARCHITECTURE:

Input: (batch, sequence_length, n_features)

Encoder: [Dense(n_features) -> LSTM(64) -> LSTM(32) -> Dense(16)]

Decoder: [Dense(32) -> LSTM(64) -> Dense(n_features)]

Output: reconstructed time series

TRAINING:

Train ONLY on normal periods (user-specified clean subset)

Loss: MSE between input and reconstruction

Epochs: 50, Batch size: 16

Early stopping: patience=5 on validation loss

fit(X_clean, validation_split=0.2) -> self

predict(X) -> reconstructed X

score_samples(X) -> MSE reconstruction error per sequence

Higher error = anomaly (inverse scaling applied)

get_hyperparameters() -> dict

e) KMeansModel():

n_clusters: int [2 to 10]

init: 'k-means++'

random_state: 42

fit() -> self

predict(X) -> cluster labels [0 to K-1]

score_samples(X) -> distance to nearest centroid (normalized)

Distance > threshold = anomaly

anomaly_threshold: float [calculated as 95th percentile of distances]

get_hyperparameters() -> dict

f) OneClassSVMModel():

kernel: 'rbf'

nu: float [0.01 to 0.1] (fraction of outliers)

gamma: 'scale'

fit() -> self

predict(X) -> array of [-1 (anomaly), 1 (normal)]

score_samples(X) -> array of decision function values

get_hyperparameters() -> dict

UNIFIED MODEL INTERFACE:

fit(X_train) -> self

Trains model on feature matrix

Stores model state

predict_anomaly(X, threshold=None) -> array

Returns binary prediction [0 (normal), 1 (anomaly)]

Applies threshold to continuous anomaly scores

get_anomaly_scores(X) -> array

Returns continuous anomaly scores [0 to 1]

Normalized across models for comparison

detect_historical_anomalies(X_train, percentile=95) -> DataFrame

Returns rows from X_train marked as anomalies

anomaly_threshold = percentile of anomaly score

Output: DataFrame with index (dates), scores, binary prediction

classify_new_transaction(df_raw, feature_engineer) -> DataFrame

Takes raw transaction data and creates latest window features

Scores using trained model

Returns: transaction_date, executive, anomaly_score, is_anomaly, confidence

text

### 2.2 Model Configuration & Hyperparameter Management

Class HyperparameterConfig:

Store hyperparameter ranges for each model

Format: hyperparameter_config.json

Example:
{
"IsolationForest": {
"contamination": [0.01, 0.02, 0.05, 0.1],
"n_estimators": ,
"max_samples": ["auto"]
},
"DBSCAN": {
"eps": [0.5, 0.8, 1.0, 1.5],
"min_samples":​
},
"LSTM_Autoencoder": {
"sequence_length":,​
"epochs": ,
"batch_size":​
}
}

Allow users to modify before model runs

Validate ranges before training

text

### 2.3 Model Persistence

Class ModelManager:

save_model(model, model_name, model_type, hyperparameters, feature_names)

Pickle model and metadata to /models/[timestamp]_[model_name].pkl

Save hyperparameters and training date

Save feature names used for consistency

load_model(model_path) -> (model, metadata)

Unpickle and validate feature consistency

list_saved_models() -> List[Dict]

Return all saved models with metadata

delete_model(model_path)

Remove saved model file

text

---

## Part 3: GUI Specification

### 3.1 GUI Framework

Use **PyQt5** or **PySimpleGUI** (whichever preferred). Structure:

Application: InsiderTradingAnomalyDetector

MAIN WINDOW LAYOUT:
├── Menu Bar
│ ├── File
│ │ ├── Load Data
│ │ ├── Save Project
│ │ ├── Load Project
│ │ └── Exit
│ ├── Models
│ │ ├── Manage Saved Models
│ │ └── Model Comparison Report
│ └── Help
│
├── TAB 1: DATA MANAGEMENT
│ ├── [Button] Load Raw Data
│ │ └─ File dialog → load CSV
│ │ └─ Display: Row count, date range, executives
│ │
│ ├── Data Preview Table
│ │ └─ Show first 20 rows of raw data
│ │ └─ Sortable columns
│ │
│ ├── Filtering Options
│ │ ├─ [Dropdown] Filter by Executive
│ │ ├─ [Date Range Picker] From / To
│ │ ├─ [Button] Apply Filter
│ │ └─ [Button] Clear Filter
│ │
│ ├── Data Summary Panel
│ │ ├─ Total Transactions
│ │ ├─ Unique Executives
│ │ ├─ Date Range
│ │ ├─ Buy/Sell Ratio
│ │ └─ Data Quality Issues (count)
│ │
│ └─ [Button] Proceed to Feature Engineering
│
├── TAB 2: FEATURE ENGINEERING
│ ├── [Label] Step 1: Configure Time Window Aggregation
│ │ ├─ [Input] Window size (days): or custom​
│ │ ├─ [Checkbox] Use multiple windows simultaneously
│ │ └─ [Display] Generated windows: X windows from date1 to date2
│ │
│ ├─ [Label] Step 2: Select Features to Engineer
│ │ ├─ [Tree View] Feature Categories
│ │ │ ├─ [Expand] Volume-Based Features (checkbox each)
│ │ │ │ ├─ ☑ Total Transaction Volume
│ │ │ │ ├─ ☑ Transaction Count
│ │ │ │ ├─ ☑ Avg Transaction Size
│ │ │ │ ├─ ☑ Volume Volatility
│ │ │ │ └─ ☑ Volume Std Deviation
│ │ │ ├─ [Expand] Behavioral Features (checkbox each)
│ │ │ │ ├─ ☑ Buy/Sell Ratio
│ │ │ │ ├─ ☑ Buy/Sell Asymmetry
│ │ │ │ ├─ ☑ Executive Concentration
│ │ │ │ ├─ ☑ Security Concentration
│ │ │ │ └─ ☑ Title Diversity
│ │ │ ├─ [Expand] Temporal Features (checkbox each)
│ │ │ │ ├─ ☑ Days Since Last Transaction
│ │ │ │ ├─ ☑ Trading Consistency
│ │ │ │ └─ ☑ Day of Week Concentration
│ │ │ ├─ [Expand] Comparative Features (checkbox each)
│ │ │ │ ├─ ☑ Volume Change YoY
│ │ │ │ ├─ ☑ Volume Change Prev Period
│ │ │ │ └─ ☑ Buy Ratio Change
│ │ │ └─ [Expand] Network Features (checkbox each)
│ │ │ ├─ ☑ Insider Network Density
│ │ │ └─ ☑ Transaction Timing Correlation
│ │ │
│ │ ├─ [Button] Select All / [Button] Deselect All
│ │ ├─ [Display] X features selected
│ │
│ ├─ [Label] Step 3: Feature Engineering Execution
│ │ ├─ [Button] Run Feature Engineering
│ │ ├─ [Progress Bar] Shows execution status
│ │ └─ [Text Output] Logs feature creation steps
│ │
│ ├─ [Label] Feature Matrix Output
│ │ ├─ [Display] Shape: M windows × N features
│ │ ├─ [Table] Feature statistics: Name, Mean, Std, Min, Max
│ │ ├─ [Button] Preview Feature Matrix (table view)
│ │ ├─ [Button] Export Feature Matrix (CSV)
│ │ └─ [Button] View Feature Correlations (heatmap visualization)
│ │
│ └─ [Button] Proceed to Modeling
│
├── TAB 3: MODEL TRAINING & CONFIGURATION
│ ├─ [Label] Step 1: Model Selection & Configuration
│ │ ├─ [Tree Widget] Available Models
│ │ │ ├─ ☐ Isolation Forest
│ │ │ ├─ ☐ DBSCAN
│ │ │ ├─ ☐ Local Outlier Factor
│ │ │ ├─ ☐ LSTM Autoencoder
│ │ │ ├─ ☐ K-Means
│ │ │ └─ ☐ One-Class SVM
│ │ │
│ │ ├─ [Button] Configure Selected Models
│ │ │ └─ Opens Modal Dialog per selected model:
│ │ │
│ │ │ MODAL: Configure [Model Name]
│ │ │ ├─ [For each hyperparameter]
│ │ │ │ ├─ Label: "Contamination (fraction of anomalies)"
│ │ │ │ ├─ Input type (dropdown for categorical, spinner for numeric)
│ │ │ │ ├─ Value: [0.05] with slider [0 ---|--- 0.2]
│ │ │ │ ├─ [Info Icon] Tooltip: "Percentage of data to label as anomalies"
│ │ │ │ └─ [Button] Reset to Default
│ │ │ │
│ │ │ ├─ [Button] Save Configuration / [Button] Cancel
│ │ │
│ │ └─ [Display] X models configured, ready to train
│ │
│ ├─ [Label] Step 2: Training Data Configuration
│ │ ├─ [Radio] Train/Test Split
│ │ │ ├─ ○ Chronological (older = train, recent = test) [80/20]
│ │ │ ├─ ○ Random split [80/20]
│ │ │ └─ [Input] Customize ratio: []%
│ │ │
│ │ ├─ [Checkbox] Feature Standardization
│ │ │ ├─ ☑ Standardize features (mean=0, std=1) [default]
│ │ │ └─ [Checkbox] Use robust scaler (less sensitive to outliers)
│ │ │
│ │ └─ [Display] Training set: X windows, Test set: Y windows
│ │
│ ├─ [Label] Step 3: Training Execution
│ │ ├─ [Button] Train All Models
│ │ ├─ [Progress Bar] Overall progress
│ │ ├─ [List Widget] Training log
│ │ │ ├─ "2025-12-18 14:30:05 - Training IsolationForest..."
│ │ │ ├─ "2025-12-18 14:30:08 - IsolationForest complete. Anomalies detected: 12"
│ │ │ ├─ "2025-12-18 14:30:09 - Training DBSCAN..."
│ │ │ └─ [Auto-scroll to bottom]
│ │ │
│ │ └─ [Estimated Time Remaining Display]
│ │
│ ├─ [Label] Step 4: Model Comparison
│ │ ├─ [Table Widget] Model Results Summary
│ │ │ Columns: Model Name | Hyperparameters | Anomalies Detected |
│ │ │ Training Time (s) | Prediction Type
│ │ │ Rows: [One per trained model]
│ │ │ ├─ Sortable by clicking column header
│ │ │ ├─ [Button] Export Summary (CSV)
│ │ │ ├─ [Button] View Model Details
│ │ │ │ └─ Opens detail view: hyperparameters, training stats, anomaly distribution
│ │ │ │
│ │ │ └─ [Right-click Context Menu]
│ │ │ ├─ Save Model
│ │ │ ├─ Delete Model
│ │ │ ├─ Set as Active Model
│ │ │ └─ Generate Report
│ │
│ └─ [Button] Proceed to Reporting
│
├── TAB 4: ANOMALY DETECTION & SCORING
│ ├─ [Label] Step 1: Select Active Model
│ │ ├─ [Dropdown] Choose model: [IsolationForest_20251218_143008 ▼]
│ │ ├─ [Display] Model info: Type, Training date, Anomalies in train set
│ │ └─ [Button] Compare Multiple Models (cross-model scoring)
│ │
│ ├─ [Label] Step 2: Historical Anomaly Detection
│ │ ├─ [Label] Anomaly Threshold Configuration
│ │ │ ├─ [Radio] ○ Percentile-based: th percentile ← recommended
│ │ │ ├─ [Radio] ○ Fixed threshold: [0.75]
│ │ │ ├─ [Radio] ○ Statistical: [3 σ] standard deviations
│ │ │ └─ [Display] Using threshold, X anomalies detected (Y% of data)
│ │ │
│ │ ├─ [Button] Detect Historical Anomalies
│ │ │ └─ Runs prediction on full feature matrix
│ │ │ └─ Sorts by anomaly score (descending)
│ │ │
│ │ ├─ [Table Widget] Historical Anomalies
│ │ │ Columns: Window Date | Anomaly Score | Confidence |
│ │ │ Transaction Count | Avg Volume | Top Executive |
│ │ │ Top Security
│ │ │ ├─ Rows: Filtered to only anomalies (can toggle to show all)
│ │ │ ├─ [Checkbox] Show all windows (toggle anomalies only)
│ │ │ ├─ [Input] Filter by anomaly score: ≥ []
│ │ │ ├─ [Button] Export Anomalies (Excel with formatting)
│ │ │ ├─ [Right-click Row Context Menu]
│ │ │ │ ├─ View Raw Transactions in Period
│ │ │ │ │ └─ Shows popup with all raw transactions in that window
│ │ │ │ ├─ Add Note
│ │ │ │ └─ Mark as Verified (for manual review)
│ │ │ │
│ │ │ └─ [Heatmap Visualization Panel]
│ │ │ ├─ X-axis: Window dates
│ │ │ ├─ Y-axis: Model score normalized [0-1]
│ │ │ ├─ Color: Red (anomaly) to Green (normal)
│ │ │ └─ [Checkbox] Overlay actual events (user can mark)
│ │
│ ├─ [Label] Step 3: Classify New Transactions
│ │ ├─ [Button] Load New Transaction Data (CSV)
│ │ │ └─ Format: same as raw data (transaction_date, executive, ...)
│ │ │
│ │ ├─ [Button] Score New Transactions
│ │ │ └─ Creates latest window features, applies model
│ │ │ └─ Compares to historical baseline
│ │ │
│ │ ├─ [Table Widget] New Transaction Scores
│ │ │ Columns: Transaction Date | Executive | Title |
│ │ │ Security | Action (Buy/Sell) | Shares |
│ │ │ Anomaly Score | Risk Level | vs Baseline
│ │ │ ├─ Color-coded by anomaly score (red=high risk, yellow=medium, green=normal)
│ │ │ ├─ [Dropdown] Risk level filter: [All ▼] [High Only] [Medium+]
│ │ │ ├─ [Export Button] Export Scores
│ │ │ └─ [Alert Badge] X transactions flagged as anomalous
│ │
│ └─ [Label] Step 4: Time Series Visualization
│ ├─ [Chart] Anomaly Scores Over Time
│ │ ├─ X-axis: Date
│ │ ├─ Y-axis: Anomaly Score
│ │ ├─ Line plot: Score trend
│ │ ├─ Scatter: Anomalies highlighted (red points)
│ │ ├─ Shaded regions: Detected anomaly periods
│ │ └─ [Legend] Show/hide model scores
│ │
│ └─ [Chart] Feature Behavior During Anomalies
│ ├─ [Dropdown] Select feature: [Total Volume ▼]
│ ├─ Line plot: Feature value over time
│ ├─ Red shading: Anomaly periods
│ └─ Statistics box: Mean, Std in anomaly vs normal periods
│
└── TAB 5: REPORTING
├─ [Label] Report Configuration
│ ├─ [Checkbox] ☑ Summary Statistics
│ ├─ [Checkbox] ☑ Model Performance Metrics
│ ├─ [Checkbox] ☑ Historical Anomalies
│ ├─ [Checkbox] ☑ Feature Analysis
│ ├─ [Checkbox] ☑ Visualizations (charts, heatmaps)
│ ├─ [Checkbox] ☑ New Transaction Scores
│ └─ [Checkbox] ☑ Recommendations
│
├─ [Label] Report Format Selection
│ ├─ [Radio] ○ Excel (.xlsx) ← default
│ ├─ [Radio] ○ PDF
│ └─ [Radio] ○ Both
│
├─ [Input] Report Title: [Insider Trading Anomaly Analysis - Q4 2025]
├─ [Input] Report Description: [Multi-model analysis of insider trading...]
│
├─ [Button] Generate Report
│ ├─ [Progress Bar]
│ ├─ [Text Log] "Creating Excel workbook..."
│ └─ [On Complete] "Report saved to: /reports/[timestamp]_report.xlsx"
│
└─ [Button] Open Report (launches default Excel application)

FLOATING PANELS:
├─ Model Comparison Tool (floating window)
│ ├ Select 2+ models
│ ├ Compare predictions on same test set
│ ├ Venn diagram: Overlapping anomalies between models
│ ├ Confusion matrices
│ └ Consensus scoring (vote-based anomaly detection)
│
└─ Interactive Feature Inspector (right sidebar, collapsible)
├ Select feature from dropdown
├ Display distribution (histogram)
├ Show correlation with anomaly score
└ Anomaly threshold slider for that feature

text

### 3.2 GUI Behavior & Workflow

WORKFLOW FLOW:

User loads raw data (Tab 1)

Reviews data, applies filters if needed

Navigates to Tab 2, selects features, runs engineering

Views feature matrix, exports if needed

Goes to Tab 3, selects models, configures hyperparameters

Trains all models, compares results

Navigates to Tab 4, detects historical anomalies with active model

Scores new transactions (if available)

Reviews visualizations

Goes to Tab 5, configures report, generates Excel output

PERSISTENCE:

Save/load "projects" that include: raw data, feature config, trained models

Each project is a .pkl file containing all state

User can reload and re-run analyses

text

---

## Part 4: Modeling System Detailed Specification

### 4.1 Model Training Pipeline

Class ModelTrainingPipeline:

def train_model(model_config, X_train, feature_names, y_true=None):
"""
Unified training interface for all model types

text
Args:
    model_config: Dict with keys:
      - 'type': 'IsolationForest' | 'DBSCAN' | 'LOF' | 'LSTM' | 'KMeans' | 'OneClassSVM'
      - 'hyperparameters': Dict of model-specific params
      - 'name': Custom model name for tracking
      
    X_train: np.ndarray, shape (n_samples, n_features)
    feature_names: List[str], length n_features
    y_true: np.ndarray (optional), ground truth labels for validation

Returns:
    trained_model: Fitted model object
    training_metrics: Dict containing:
      - 'model_type': str
      - 'hyperparameters': dict
      - 'n_samples': int
      - 'n_features': int
      - 'feature_names': list
      - 'training_time': float (seconds)
      - 'timestamp': datetime
      - 'anomalies_detected': int (count where pred == 1)
      - 'anomaly_percentage': float
      - 'validation_score': float (if y_true provided, use precision/recall)
"""

Steps:
1. Validate model_config
2. Instantiate model based on type
3. Record start time
4. Call model.fit(X_train)
5. Generate predictions and anomaly scores
6. Compute metrics
7. Return model and metrics dict
def predict_on_new_data(trained_model, X_new, feature_names_expected):
"""
Score new data using trained model

text
Args:
    trained_model: Fitted model from train_model()
    X_new: np.ndarray, shape (m_samples, n_features)
    feature_names_expected: List[str]

Returns:
    predictions: Dict with keys:
      - 'anomaly_labels': np.ndarray, shape (m_samples,), values in {0, 1}
      - 'anomaly_scores': np.ndarray, shape (m_samples,), values in[4]
      - 'confidence': np.ndarray, shape (m_samples,), confidence in prediction
      - 'percentile_rank': np.ndarray, rank in historical distribution
"""

Steps:
1. Validate feature count matches training
2. Call model.score_samples(X_new)
3. Normalize scores to[4]
4. Compute confidence (distance to decision boundary)
5. Rank against historical scores
6. Return predictions dict
def ensemble_predict(trained_models_list, X_data, voting_method='majority'):
"""
Combine predictions from multiple models

text
Args:
    trained_models_list: List of (model, metadata) tuples
    X_data: np.ndarray
    voting_method: 'majority' | 'consensus' | 'average_score'

Returns:
    ensemble_predictions: Dict with keys:
      - 'individual_predictions': List[Dict] (one per model)
      - 'ensemble_label': np.ndarray (consensus prediction)
      - 'ensemble_score': np.ndarray (average anomaly score)
      - 'model_agreement': np.ndarray (fraction of models predicting anomaly)
      - 'recommendation': str ('ANOMALY', 'BORDERLINE', 'NORMAL')
"""

Steps:
1. Get predictions from each model individually
2. Combine using voting method
3. Compute inter-model agreement
4. Generate risk recommendation
text

### 4.2 Anomaly Scoring & Thresholding

Class AnomalyScorer:

def normalize_scores(raw_scores, model_type):
"""
Convert model-specific scores to range​

text
IsolationForest: offset_path / path_length →[4]
DBSCAN: density-based score
LOF: Local outlier factor → normalized
LSTM: reconstruction error → sigmoid normalization
KMeans: distance to centroid (normalized by max distance)
OneClassSVM: decision function → sigmoid

Returns: np.ndarray in, where 1 = most anomalous[4]
"""
def adaptive_threshold(scores_historical, method='percentile', param=95):
"""
Compute anomaly threshold based on historical data

text
Methods:
- 'percentile': threshold = nth percentile (recommended)
- 'std': threshold = mean + n*std
- 'elbow': identify knee in sorted scores
- 'silhouette': kmeans on scores, use cluster boundary

Returns: float threshold value
"""
def get_confidence_score(anomaly_score, threshold):
"""
Confidence = how far from threshold

text
If anomaly_score > threshold:
    confidence = (anomaly_score - threshold) / (1 - threshold)  #[4]
Else:
    confidence = (threshold - anomaly_score) / threshold  #[4]

Higher confidence = higher certainty in classification
"""
def get_risk_level(anomaly_score, confidence, model_agreement=None):
"""
Convert score/confidence to categorical risk

text
Returns: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' | 'NORMAL'

Rules:
- anomaly_score > 0.8 AND confidence > 0.7: CRITICAL
- anomaly_score > 0.65 AND confidence > 0.6: HIGH
- 0.5 < anomaly_score < 0.65: MEDIUM
- etc.

If ensemble (model_agreement > 0.8): elevate risk by 1 level
"""
text

### 4.3 Model Validation & Diagnostics

Class ModelDiagnostics:

def generate_diagnostics(trained_model, X_train, X_test):
"""
Run comprehensive model validation

text
Returns: diagnostics_dict with:
  - 'n_anomalies_train': int
  - 'n_anomalies_test': int
  - 'anomaly_rate_train': float
  - 'anomaly_rate_test': float
  - 'feature_importance': Dict (for interpretable models)
  - 'prediction_stability': float (consistency on similar points)
  - 'model_summary': str (interpretation of results)
  - 'recommendations': List[str] (guidance on model usage)
  - 'warnings': List[str] (alerts, e.g. "High anomaly rate suggests low threshold")
"""
def feature_importance_analysis(trained_model, X_data, feature_names):
"""
Identify which features drive anomaly detection

text
For tree-based (IsolationForest, KMeans):
  - Use model's built-in feature importance

For LSTM:
  - Permutation importance: shuffle each feature, measure MSE increase

For DBSCAN, LOF:
  - Correlation analysis: which features correlate with anomaly score

Returns: Dict[feature_name] = importance_score[4]
"""
def model_interpretability_report(trained_model, anomaly_indices, feature_names, X_data):
"""
For each anomaly, explain WHY it was flagged

text
Returns: List[Dict] with:
  - 'date': str
  - 'anomaly_score': float
  - 'top_reasons': List[str]
    * "Total volume (5000) is 3.2σ above mean (1500)"
    * "Executive concentration (Herfindahl=0.95) unusually high"
    * "Volume spike correlated with no public announcements"
  - 'feature_deviations': Dict[feature] = (value, z_score, percentile)
"""
text

---

## Part 5: Excel Reporting Specification

### 5.1 Report Structure & Content

FILE: [TIMESTAMP]_InsiderTrading_AnomalyReport.xlsx

SHEET 1: EXECUTIVE SUMMARY (1 page)
├─ Title: "Insider Trading Anomaly Detection Report"
├─ Report Date: [TIMESTAMP]
├─ Data Period: [DATE_START to DATE_END]
├─ Key Metrics (formatted as KPI boxes)
│ ├─ Total Transactions Analyzed: [X]
│ ├─ Historical Anomalies Detected: [Y] (Z%)
│ ├─ New Transactions Scored: [W]
│ ├─ Critical Risk Transactions: [V]
│ └─ Models Used: [Model1, Model2, ...]
│
├─ Methodology Overview
│ ├─ Data Aggregation Window: [X days]
│ ├─ Features Engineered: [N] total
│ ├─ Feature Engineering: Brief bullet points
│ ├─ Models Deployed: List with brief descriptions
│ └─ Anomaly Threshold: [Percentile or fixed value]
│
├─ Key Findings (3-5 bullet points)
│ ├─ "Model consensus identifies X anomaly periods"
│ ├─ "Highest risk cluster in [date range]: Y% above baseline"
│ ├─ "Executive concentration unusually high in top anomalies"
│ └─ "New transaction on [date] scores 0.92/1.0 risk"
│
├─ Recommendations
│ ├─ Priority 1: [Action based on CRITICAL findings]
│ ├─ Priority 2: [Follow-up investigations]
│ └─ Priority 3: [Monitoring recommendations]
│
└─ Table of Contents with hyperlinks to all sheets

SHEET 2: HISTORICAL ANOMALIES (sorted by anomaly score)
├─ Table:
│ Columns:
│ - Window Start Date
│ - Window End Date
│ - Anomaly Score (%) [IsolationForest]
│ - Anomaly Score (%) [DBSCAN]
│ - Anomaly Score (%) [Ensemble Average]
│ - Risk Level (categorical color: RED/YELLOW/GREEN)
│ - Transaction Count
│ - Total Volume (shares)
│ - Buy/Sell Ratio
│ - Top 3 Executives (by volume)
│ - Top Security Traded
│ - Notes (manual entry field)
│
├─ Formatting:
│ - Rows with anomaly score > threshold: RED background
│ - Rows with score 75-95 percentile: YELLOW background
│ - Freeze header row
│ - AutoFilter enabled on all columns
│ - Column width: auto-fit
│ - Number format: 2 decimal places for scores
│
├─ Charts:
│ - Scatter plot: X=Date, Y=Anomaly Score, sized by transaction count
│ - Time series line: Anomaly score over time with flagged periods highlighted
│ - Bar chart: Anomalies by month (frequency)
│ - Heatmap: Executives × Months (volume concentration)
│
└─ Summary statistics at bottom:
- Total anomaly windows: X
- Anomaly percentage: Y%
- Average anomaly score: Z
- Highest score window: [date, score]

SHEET 3: FEATURE ENGINEERING DETAILS
├─ Table 1: Feature Matrix Summary
│ - Feature Name
│ - Data Type
│ - Unit / Scale
│ - Mean
│ - Median
│ - Std Dev
│ - Min
│ - Max
│ - Skewness
│ - Correlation with Anomaly Score
│ - Distribution chart (mini sparkline or embedded chart)
│
├─ Table 2: Feature Importance (for interpretable models)
│ - Feature Name
│ - Importance Score (IsolationForest)
│ - Rank
│ - Impact on Anomaly Detection
│
├─ Table 3: Feature Correlations
│ - Correlation matrix (heatmap)
│ - Highlight strong correlations (>0.7)
│ - Note: High correlation features may be redundant
│
├─ Chart: Distribution of key features
│ - Histogram for each top-5 important features
│ - Show distribution in anomaly vs. normal periods (side-by-side)
│
└─ Notes on feature engineering choices

SHEET 4: MODEL PERFORMANCE & COMPARISON
├─ Table 1: Model Training Summary
│ Columns:
│ - Model Type
│ - Hyperparameters (formatted as key=value)
│ - Training Data Size
│ - Training Time (seconds)
│ - Anomalies Detected (count)
│ - Anomaly Rate (%)
│ - Validation Score (if applicable)
│ - Status (Success/Error)
│
├─ Table 2: Model Prediction Comparison (on test set)
│ Columns:
│ - Window Date
│ - Model1 Prediction | Model2 Prediction | Model3 Prediction | ...
│ - Ensemble Consensus
│ - Model Agreement (% models agreeing)
│
├─ Table 3: Model Divergence Analysis
│ - Dates where models DISAGREE on anomaly classification
│ - Useful for identifying borderline cases
│
├─ Chart 1: Anomaly Detection by Model
│ - Venn diagram (if ≤3 models) showing overlapping anomalies
│ - Or upset plot for >3 models
│ - Shows which anomalies are caught by which models
│
├─ Chart 2: ROC/Threshold Analysis (if ground truth available)
│ - ROC curves for each model
│ - Precision-Recall curves
│ - AUC scores
│
├─ Chart 3: Model Score Distribution
│ - Histogram of anomaly scores for each model
│ - Show threshold line(s)
│
└─ Recommendation: Best model(s) based on performance

SHEET 5: NEW TRANSACTION SCORING
├─ Table: New Transactions with Anomaly Scores
│ Columns:
│ - Transaction Date
│ - Executive Name
│ - Executive Title
│ - Company/Security
│ - Transaction Type (Buy/Sell)
│ - Shares
│ - Dollar Value (if available)
│ - Anomaly Score (%) [Active Model]
│ - Risk Level (categorical)
│ - Confidence Score
│ - Percentile Rank (vs. historical baseline)
│ - Flag for Review (checkbox)
│
├─ Formatting:
│ - Color code by risk level (RED/YELLOW/GREEN)
│ - High-risk transactions highlighted
│ - Sortable/filterable
│
├─ Chart: New Transactions Risk Distribution
│ - Pie chart: % in each risk category
│ - Time series: Risk scores over new transaction dates
│
├─ Summary:
│ - Total new transactions: X
│ - Critical risk: Y (%)
│ - High risk: Z (%)
│ - Requires further investigation: [Count]
│
└─ Flagged Transactions Detail
- If any transactions flagged, show detailed breakdown
- Feature values for that transaction vs. baseline

SHEET 6: DETAILED ANOMALY EXPLANATIONS
├─ For each historical anomaly (if not too many):
│
│ SECTION: Anomaly ID #N - [Window Date]
│ ├─ Anomaly Score: [0.87/1.0] - HIGH RISK
│ ├─ Time Period: [Start Date to End Date]
│ ├─ Anomaly Explanation:
│ │ "This period flagged by [Model1, Model2] shows unusual characteristics:"
│ │ - Volume 4.2x baseline (8500 vs. mean 2000)
│ │ - Executive concentration 0.92 (typically 0.45) → coordinated trading
│ │ - Buy ratio 0.89 → overwhelmingly purchases (typically 0.50)
│ │ - 12 executives trading vs. typical 3-4
│ │ - Top security: [TICKER], accounting for 65% of volume
│ │
│ ├─ Raw Transactions in Period:
│ │ - Table: All individual transactions during window
│ │ - Columns: Date | Executive | Title | Security | Action | Shares
│ │ - Sorted by date
│ │
│ ├─ Executive Network Analysis:
│ │ - Co-trading pairs: [Exec1 + Exec2], [Exec1 + Exec3], etc.
│ │ - Hierarchical breakdown (CEO, CFO, Directors)
│ │ - Unusual coordinations highlighted
│ │
│ ├─ Feature Deviations (vs. baseline):
│ │ - Bar chart: Each key feature showing deviation from mean
│ │ - Z-scores for statistical significance
│ │ - Highlighted features: those most unusual
│ │
│ ├─ Timeline:
│ │ - Were there relevant public events (earnings, announcements)?
│ │ - [Data source: User can add notes]
│ │
│ └─ Investigator Notes: [Free-text field for manual review]
│
│
├─ If >10 anomalies: Expand only top 5, put rest in summary

SHEET 7: TECHNICAL SPECIFICATIONS
├─ Data Specifications
│ - Raw data source: [Filepath/Date]
│ - Data quality: [X% complete, Y missing values]
│ - Date range: [START to END]
│ - Record count: [N transactions, M windows]
│
├─ Feature Engineering Specifications
│ - Aggregation window: [X days]
│ - Features created: [List all 20+ features with formulas]
│ - Standardization method: Z-score (mean=0, std=1)
│ - Missing value handling: [Forward fill, drop, impute]
│
├─ Model Specifications
│ - For each model:
│ * Model type: [Algorithm name]
│ * Hyperparameters: [All values]
│ * Validation method: [Train/test split, cross-validation]
│ * Random seed: [For reproducibility]
│ * Training environment: Python [X.X], scikit-learn [X.X], etc.
│
├─ Anomaly Threshold Specifications
│ - Method: [Percentile, std, etc.]
│ - Parameters: [e.g., 95th percentile]
│ - Threshold value(s): [Numeric values for each model]
│
└─ Report Generation Details
- Report created: [Timestamp]
- Generated by: [User name/system]
- Data sources: [List all input files]
- Next review date: [Recommended]

SHEET 8: APPENDIX - GLOSSARY & METHODOLOGY
├─ Term Definitions
│ - Anomaly Score: Normalized [0-1] measure of how unusual a period is
│ - Anomaly Rate: Percentage of windows/transactions flagged as anomalous
│ - Executive Concentration: Herfindahl index measuring trading dominance
│ - [etc. for all technical terms]
│
├─ Methodology Overview
│ - Why unsupervised learning: Insider trading patterns unknown/evolving
│ - How features capture insider behavior: Descriptions of each feature
│ - Why multiple models: Ensemble approach increases robustness
│ - Anomaly detection approach: What constitutes "abnormal"
│
├─ Limitations & Considerations
│ - Model assumes historical data representative of normal behavior
│ - Cannot directly prove insider trading, only flags suspicious patterns
│ - New market regime (e.g., post-M&A) may have different baseline
│ - [Other limitations specific to dataset]
│
└─ Contact & Questions
- For questions about methodology: [Contact]
- For data updates: [Process]

text

### 5.2 Excel Formatting & Styling

FORMATTING STANDARDS:

Color Scheme:

Anomaly/High Risk: Red (#FF0000 or #DC3545)

Medium/Borderline: Yellow (#FFC107 or #FDD835)

Normal/Low Risk: Green (#28A745 or #66BB6A)

Headers: Blue (#007BFF or #1E88E5)

Text: Dark Gray (#333333 or #424242)

Font:

Headers: Arial, 14pt, Bold

Section titles: Arial, 12pt, Bold, Blue

Data tables: Arial, 11pt

Footnotes: Arial, 9pt, Gray

Borders & Spacing:

Header rows: Bold borders, background color

Data cells: Light borders (0.5pt)

Row height: 20pt for data, 25pt for headers

Column padding: 8pt left/right

Number Formatting:

Percentages: 0.00%

Decimals (scores): 0.000 (3 places)

Integers: #,##0

Currency: $#,##0.00

Dates: YYYY-MM-DD or MM/DD/YYYY (user preference)

Charts:

Chart type: As specified per sheet

Legend: Auto, positioned best fit

Gridlines: Light gray

Data labels: As applicable

Colors: Use color scheme above

Font size: 10pt for labels

Conditional Formatting:

Anomaly Score column: Color scale (Red-Yellow-Green)

Risk Level column: Three-color scale matching risk categories

Volume deviations: Three-color scale (below baseline-baseline-above)

Page Setup (all sheets):

Orientation: Portrait (sheets 1-3), Landscape (sheets 4-5 if wide tables)

Margins: 0.5" all sides

Paper size: Letter (8.5" × 11")

Scale: Fit to 1 page wide (if needed)

Headers & Footers:

Header: [Report Title] | [Page 1 of N]

Footer: Generated [Timestamp] | Confidential

Print area: Define for each sheet

Interactive Features:

Freeze panes: Freeze header rows on all data sheets

AutoFilter: Enable on all data tables

Sorting: Users can sort by any column

Pivot tables: Optional, for additional analysis views

Export Properties:

Title: "Insider Trading Anomaly Detection Report"

Subject: "Analysis Period: [Dates]"

Author: [System/User name]

Comments: [Summary of findings]

text

### 5.3 Report Generation Code Structure

Class ExcelReporter:

def init(self, output_filepath):
"""Initialize Excel workbook and styles"""

def add_executive_summary(self, summary_data_dict):
"""Sheet 1: Executive summary with KPIs and key findings"""

def add_historical_anomalies(self, anomalies_df, models_list):
"""Sheet 2: Detailed anomaly table with formatting and charts"""

def add_feature_engineering_details(self, feature_matrix_df, feature_importance_dict):
"""Sheet 3: Feature statistics, importance, correlation heatmap"""

def add_model_performance(self, training_metrics_list, predictions_df):
"""Sheet 4: Model comparison, metrics, charts"""

def add_new_transaction_scoring(self, new_transactions_df, scores_dict):
"""Sheet 5: New transaction scores with risk highlighting"""

def add_anomaly_explanations(self, anomalies_list, feature_matrix_df, raw_data_df):
"""Sheet 6: Detailed explanations for each anomaly"""

def add_technical_specs(self, metadata_dict):
"""Sheet 7: Technical specifications and reproducibility info"""

def add_appendix(self, methodology_dict):
"""Sheet 8: Glossary, methodology, limitations"""

def save_and_validate():
"""Save workbook, validate structure, verify all sheets present"""

def generate_full_report(all_required_data):
"""Main method: Call all add_* methods in sequence with full dataset"""

text

### 5.4 Visualization Specifications

CHART SPECIFICATIONS:

Anomaly Scores Time Series

Type: Line chart with scatter overlay

X-axis: Date

Y-axis: Anomaly Score [0-1]

Line: Multiple lines (one per model)

Markers: Red dots for flagged anomalies

Shading: Red background for anomaly periods

Legend: Model names

Tooltip: Date, score, confidence

Feature Heatmap (Executives × Time)

Type: Heatmap (conditional formatting)

Rows: Executive names

Columns: Time periods (months/quarters)

Cell values: Total volume traded in that period

Color scale: Light (low volume) to Dark Red (high volume)

Highlights unusual concentration

Venn Diagram (Model Comparison)

Type: Venn diagram or UpSet plot

Circles/Sets: One per model

Intersection: Anomalies caught by multiple models

Size: Proportional to count

Legend: Model names and colors

Feature Deviation Chart (Top Anomaly)

Type: Bar chart (horizontal)

Bars: Each feature

Length: Z-score deviation from baseline

Color: Red (positive deviation), Blue (negative)

Labels: Feature name, z-score value

Shows what made the period anomalous

Buy/Sell Ratio Distribution

Type: Histogram

X-axis: Buy/Sell ratio

Y-axis: Frequency (count of windows)

Overlay: Red line for anomaly windows, Blue for normal

Shows how much anomalies differ in ratio

Volume Spike Timeline

Type: Bar chart

X-axis: Date

Y-axis: Transaction volume

Color: Green (normal), Yellow (elevated), Red (anomaly)

Trend line: Moving average

text

---

## Part 6: System Integration & Workflow

### 6.1 File Structure

project_root/
├── data/
│ ├── raw/
│ │ └── insider_trading_raw.csv
│ ├── processed/
│ │ ├── feature_matrix.csv
│ │ └── anomaly_labels.csv
│ └── new_transactions/
│ └── new_transactions.csv
│
├── models/
│ ├── saved_models/
│ │ ├── 20251218_143008_IsolationForest_v1.pkl
│ │ ├── 20251218_143508_DBSCAN_v1.pkl
│ │ └── metadata.json
│ └── training_logs/
│ └── 20251218_model_training.log
│
├── reports/
│ ├── 20251218_143000_report.xlsx
│ ├── 20251218_143000_report.pdf
│ └── archive/
│
├── config/
│ ├── hyperparameter_config.json
│ ├── feature_config.json
│ └── system_config.json
│
├── notebooks/
│ └── exploratory_analysis.ipynb (optional)
│
├── src/
│ ├── data_preprocessor.py
│ ├── feature_engineer.py
│ ├── model_factory.py
│ ├── model_trainer.py
│ ├── anomaly_scorer.py
│ ├── gui_main.py (PyQt5 or PySimpleGUI)
│ ├── gui_components.py
│ ├── excel_reporter.py
│ ├── utils.py
│ └── constants.py
│
└── README.md

text

### 6.2 Data Flow Diagram

Raw CSV Data
↓
DataPreprocessor.load_raw_data()
↓ (Cleaned DataFrame)
FeatureEngineer.aggregate_by_window()
↓ (Time-windowed aggregations)
FeatureEngineer.create_feature_matrix()
↓ (Full feature matrix with X features)
Feature Matrix CSV
↓
GUI Tab 2: Display feature stats, preview
↓
ModelFactory.train_model() [for each selected model]
↓ (X trained models with predictions)
GUI Tab 3: Compare model results
↓
ModelFactory.detect_historical_anomalies()
↓ (Ranked list of anomaly periods)
GUI Tab 4: Display historical anomalies
↓
User provides new transaction data
↓
ModelFactory.predict_on_new_data()
↓ (Scored new transactions)
GUI Tab 4: Display new scores
↓
ExcelReporter.generate_full_report()
↓ (Formatted Excel workbook)
GUI Tab 5: Export to Excel

text

### 6.3 Key Design Patterns

DESIGN PATTERNS USED:

Factory Pattern (ModelFactory)

Instantiates different model types based on configuration

Provides unified interface for all models

Easy to add new model types

Strategy Pattern (Different Feature Calculation Methods)

Each feature method is independent strategy

Can mix/match features via FeatureEngineer.create_feature_matrix()

Easily extended with new features

Template Method Pattern (BaseAnomalyModel)

Abstract base class defines training/prediction interface

Each model subclass implements specifics

Ensures consistency across models

Observer Pattern (GUI Updates)

Training progress updates GUI status

Model completion triggers report generation

Loosely coupled updates

Builder Pattern (ExcelReporter)

Builds complex report sheet by sheet

Each add_*() method constructs piece

.save_and_validate() finalizes

Singleton Pattern (Configuration Manager)

Single instance manages all configuration

Accessed globally for hyperparameters, file paths

Ensures consistency across application

text

### 6.4 Error Handling & Logging

LOGGING:

Create logger for each module

Log levels: DEBUG, INFO, WARNING, ERROR

Log file: /logs/[YYYY-MM-DD]_insidertrading.log

Console output: INFO level

File output: DEBUG level (verbose)

ERROR HANDLING:

DataPreprocessor: Catch missing files, format errors, validate columns

Raise DataLoadException with user-friendly message

FeatureEngineer: Catch NaN values, infinite values, division by zero

Log warning, use fallback value or skip feature

ModelFactory: Catch training errors, memory issues

Catch and log, display error in GUI with recovery option

ExcelReporter: Catch file I/O, formatting errors

Notify user, offer to retry or save alternate format

RECOVERY:

Autosave intermediate results (feature matrix, trained models)

Allow user to reload from checkpoint

Graceful degradation if optional features fail

text

---

## Part 7: Example Workflows

### 7.1 Typical Analysis Workflow

Day 1: Initial Setup

User loads 2 years of insider trading transaction data (1000+ records)

Sees 47 unique executives, data from 2022-2024

Goes to Feature Engineering tab

Selects 14-day aggregation window

Selects all 20+ available features

Runs feature engineering → creates 120 windows

Reviews feature matrix: correlations, distributions

Exports feature matrix for backup

Day 1-2: Model Training
9. Goes to Model Training tab
10. Selects: IsolationForest, DBSCAN, LSTM Autoencoder
11. Configures hyperparameters:
- IsolationForest: contamination=0.08 (expect 10% anomalies)
- DBSCAN: eps=0.8, min_samples=5
- LSTM: sequence_length=7 (two windows of history)
12. Trains all models: takes 2 minutes total
13. Reviews model comparison:
- IsolationForest flags 12 anomalies
- DBSCAN flags 8 anomalies
- LSTM flags 10 anomalies
- Ensemble (all three agree): 5 anomalies

Day 2-3: Anomaly Review
14. Goes to Anomaly Detection tab
15. Selects Ensemble model as "active"
16. Runs historical anomaly detection
17. Gets 5 flagged periods, sorted by score
18. For top anomaly (Jan 2024):
- Score: 0.89/1.0
- Period: Jan 15-21
- Key findings displayed:
* Volume 5x baseline
* 12 executives trading (vs. typical 3)
* Executive concentration 0.92 (very high)
* Buy ratio 0.87 (mostly purchases)
- Views raw transactions in that period: 47 trades
- Sees 4 executives each trading 5000+ shares
- Notes correlation with upcoming earnings announcement (Jan 24)

Day 3: External Review
19. Sends top 5 anomalies list to compliance team
20. Compliance reviews and flags 3 for investigation
21. Returns with notes on 2 (now fully explained by public events)
22. 1 remains suspicious → escalates for SEC review

Day 4-5: New Transaction Monitoring
23. User loads new transactions (May 2024, 30 days of data)
24. Scores new transactions against trained model
25. 3 transactions flagged as high-risk:
- Exec A buys 2000 shares (percentile rank 92, 0.78 risk score)
- Exec B sells 1000 shares (normal pattern, rank 65, 0.35 risk score)
- Network of 5 execs trades heavily in same security (0.82 risk)
26. Reviews details, adds notes
27. Exports scores to Excel for stakeholder review

Day 5: Report Generation
28. Configures report: selects all sections
29. Generates comprehensive Excel report
30. Report includes:
- Executive summary with KPIs
- 5 historical anomalies with detailed breakdowns
- Feature importance analysis
- Model performance comparison
- 3 flagged new transactions
- Technical specifications for reproducibility
31. Exports report to stakeholders
32. Saves project for future updates

text

### 7.2 Sensitivity Analysis / What-If Scenario

User wants to explore: "What if we use 30-day windows instead of 14-day?"

Steps:

Go to Feature Engineering tab

Change window size to 30 days

Re-run feature engineering

Creates 60 windows (vs. 120 before)

Re-train all models with new windows

Compare results:

Fewer total anomalies (fewer windows = fewer outliers)

Possibly different anomaly periods (earlier events might merge)

Top anomalies same? Different?

Export both feature matrices, both model results

Generate comparison report: Which window size better?

text

---

## Submission Format for LLM

This specification should be submitted to an LLM with the instruction:

"Build a complete Python insider trading anomaly detection system with the following specifications. Implement each component exactly as specified. Ensure all code is production-ready with proper error handling, logging, and documentation. Code should be modular, allowing independent development of each component. Provide clean separation between data layer, modeling layer, and GUI layer. Use type hints throughout. Include docstrings for all public methods. Make the GUI responsive and intuitive based on the layout specifications provided."

text

---

## Success Criteria

The completed system should:
- ✓ Load and validate raw insider trading data
- ✓ Engineer 20+ features via modular methods
- ✓ Train 6 different anomaly detection models
- ✓ Compare models and ensemble predictions
- ✓ Detect historical anomalies with confidence scores
- ✓ Score new transactions in real-time
- ✓ Generate comprehensive Excel reports with 8 sheets
- ✓ Provide interactive GUI with all specified tabs
- ✓ Handle errors gracefully with informative messages
- ✓ Enable reproducibility (save/load projects, log all settings)
- ✓ Allow customization (hyperparameters, thresholds, features)
- ✓ Produce publication-ready Excel reports
- ✓ Run end-to-end in < 10 minutes on typical dataset
