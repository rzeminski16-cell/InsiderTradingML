"""
Main GUI Application for the Insider Trading Anomaly Detection System.

This module provides the main application window with 5 tabs for a streamlined workflow:
- Tab 1: Data Fetcher (Alpha Vantage) - Fetch stock/insider data from API
- Tab 2: Data Management (CSV) - Load data from local CSV files
- Tab 3: Feature Engineering - Compute features from transaction data
- Tab 4: Model Training - Train and save anomaly detection models
- Tab 5: Detection & Reporting - Detect anomalies and generate Excel reports

Data Flow:
1. Data can come from either Alpha Vantage (Tab 1) or CSV files (Tab 2)
2. Alpha Vantage data is auto-split into historical and new transactions
3. Feature engineering runs on both datasets together for consistency
4. Models are trained on historical data and can be saved/loaded
5. Detection & Reporting scores both historical and new data, generates reports

Classes:
    MainWindow: Main application window
    DataFetcherTab: Tab for fetching data from Alpha Vantage API
    DataManagementTab: Tab for loading CSV data
    FeatureEngineeringTab: Tab for feature engineering
    ModelTrainingTab: Tab for model training and configuration
    AnomalyReportTab: Combined tab for anomaly detection and report generation
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QTabWidget, QLabel, QPushButton,
    QGroupBox, QFrame, QSplitter,
    QFileDialog, QMessageBox, QInputDialog,
    QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup,
    QProgressBar, QTextEdit, QLineEdit,
    QTableWidget, QTableWidgetItem,
    QListWidget, QListWidgetItem,
    QMenuBar, QMenu, QAction, QStatusBar,
    QSizePolicy
)

from . import constants
from .utils import setup_logging, get_timestamp
from .data_preprocessor import DataPreprocessor
from .feature_engineer import FeatureEngineer
from .model_factory import ModelFactory
from .anomaly_scorer import AnomalyScorer, ModelDiagnostics
from .excel_reporter import ExcelReporter
from .alpha_vantage import AlphaVantageClient, DataInventory
from .gui_components import (
    DataPreviewTable, FeatureTreeWidget,
    ModelConfigDialog, ProgressDialog,
    LogWidget, ChartWidget, SummaryCard,
    WorkerThread, FilterWidget, StatusBar
)


class DataManagementTab(QWidget):
    """Tab 1: Data Management - Load and preview data."""

    dataLoaded = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = setup_logging(__name__)
        self.preprocessor: Optional[DataPreprocessor] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Load data section
        load_group = QGroupBox("Load Data")
        load_layout = QHBoxLayout(load_group)

        self.load_btn = QPushButton("Load Raw Data (CSV)")
        self.load_btn.clicked.connect(self._load_data)
        load_layout.addWidget(self.load_btn)

        self.filepath_label = QLabel("No file loaded")
        load_layout.addWidget(self.filepath_label, stretch=1)

        layout.addWidget(load_group)

        # Summary cards
        cards_layout = QHBoxLayout()
        self.cards = {
            'transactions': SummaryCard("Total Transactions", "0", "#007BFF"),
            'executives': SummaryCard("Unique Executives", "0", "#28A745"),
            'date_range': SummaryCard("Date Range", "N/A", "#6C757D"),
            'buy_sell': SummaryCard("Buy/Sell Ratio", "N/A", "#FFC107")
        }
        for card in self.cards.values():
            cards_layout.addWidget(card)
        layout.addLayout(cards_layout)

        # Filter section
        self.filter_widget = FilterWidget()
        self.filter_widget.filterChanged.connect(self._apply_filter)
        layout.addWidget(self.filter_widget)

        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = DataPreviewTable()
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_group, stretch=1)

        # Data quality section
        quality_group = QGroupBox("Data Quality")
        quality_layout = QVBoxLayout(quality_group)
        self.quality_text = QTextEdit()
        self.quality_text.setReadOnly(True)
        self.quality_text.setMaximumHeight(100)
        quality_layout.addWidget(self.quality_text)
        layout.addWidget(quality_group)

    def _load_data(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not filepath:
            return

        try:
            self.preprocessor = DataPreprocessor()
            df = self.preprocessor.load_raw_data(filepath)

            self.filepath_label.setText(filepath)
            self._update_summary()
            self._update_preview()
            self._update_quality()

            self.dataLoaded.emit(self.preprocessor)
            self.logger.info(f"Loaded data from {filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.logger.error(f"Error loading data: {e}")

    def _update_summary(self) -> None:
        if self.preprocessor is None:
            return

        stats = self.preprocessor.get_summary_stats()

        self.cards['transactions'].set_value(f"{stats['total_transactions']:,}")
        self.cards['executives'].set_value(str(stats['executives']['unique_count']))
        self.cards['date_range'].set_value(
            f"{stats['date_range']['start'][:10]} to {stats['date_range']['end'][:10]}"
        )
        self.cards['buy_sell'].set_value(
            f"{stats['transaction_types']['buy_sell_ratio']:.2f}"
        )

        # Update filter widget
        self.filter_widget.set_executives(stats['executives']['list'])

    def _update_preview(self) -> None:
        if self.preprocessor is None:
            return

        df = self.preprocessor.get_preview(20)
        self.preview_table.set_dataframe(df)

    def _update_quality(self) -> None:
        if self.preprocessor is None or self.preprocessor.quality_report is None:
            return

        self.quality_text.setText(str(self.preprocessor.quality_report))

    def _apply_filter(self, filters: Dict) -> None:
        if self.preprocessor is None:
            return

        self.preprocessor.reset_filters()

        if filters.get('executive') and filters['executive'] != 'All':
            self.preprocessor.filter_by_executive(filters['executive'], inplace=True)

        if filters.get('start_date') or filters.get('end_date'):
            self.preprocessor.filter_by_date_range(
                filters.get('start_date'),
                filters.get('end_date'),
                inplace=True
            )

        self._update_preview()

    def get_data(self) -> Optional[pd.DataFrame]:
        if self.preprocessor is None:
            return None
        return self.preprocessor.df


class FeatureEngineeringTab(QWidget):
    """Tab 2: Feature Engineering - Configure and create features for both datasets."""

    featuresReady = pyqtSignal(object, object, object)  # (engineer, historical_features, new_features)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = setup_logging(__name__)
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.historical_features: Optional[pd.DataFrame] = None
        self.new_features: Optional[pd.DataFrame] = None
        self.historical_df: Optional[pd.DataFrame] = None
        self.new_transactions_df: Optional[pd.DataFrame] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Data status section
        data_group = QGroupBox("Data Status")
        data_layout = QGridLayout(data_group)

        data_layout.addWidget(QLabel("Historical Data:"), 0, 0)
        self.historical_status = QLabel("Not loaded")
        self.historical_status.setStyleSheet("color: gray;")
        data_layout.addWidget(self.historical_status, 0, 1)

        data_layout.addWidget(QLabel("New Transactions:"), 0, 2)
        self.new_status = QLabel("Not loaded")
        self.new_status.setStyleSheet("color: gray;")
        data_layout.addWidget(self.new_status, 0, 3)

        layout.addWidget(data_group)

        # Window configuration
        window_group = QGroupBox("Step 1: Configure Time Window")
        window_layout = QHBoxLayout(window_group)

        window_layout.addWidget(QLabel("Window Size (days):"))
        self.window_spin = QSpinBox()
        self.window_spin.setMinimum(7)
        self.window_spin.setMaximum(90)
        self.window_spin.setValue(14)
        window_layout.addWidget(self.window_spin)

        window_layout.addStretch()
        layout.addWidget(window_group)

        # Feature selection
        feature_group = QGroupBox("Step 2: Select Features")
        feature_layout = QVBoxLayout(feature_group)

        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(lambda: self.feature_tree.select_all())
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(lambda: self.feature_tree.deselect_all())
        button_layout.addWidget(self.select_all_btn)
        button_layout.addWidget(self.deselect_all_btn)
        button_layout.addStretch()
        feature_layout.addLayout(button_layout)

        self.feature_tree = FeatureTreeWidget()
        feature_layout.addWidget(self.feature_tree)

        self.selected_label = QLabel("20 features selected")
        feature_layout.addWidget(self.selected_label)
        self.feature_tree.selectionChanged.connect(
            lambda f: self.selected_label.setText(f"{len(f)} features selected")
        )

        layout.addWidget(feature_group)

        # Run section
        run_group = QGroupBox("Step 3: Run Feature Engineering")
        run_layout = QVBoxLayout(run_group)

        self.run_btn = QPushButton("Run Feature Engineering on Both Datasets")
        self.run_btn.clicked.connect(self._run_engineering)
        run_layout.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        run_layout.addWidget(self.progress)

        self.log_widget = LogWidget()
        run_layout.addWidget(self.log_widget)

        layout.addWidget(run_group)

        # Results section
        results_group = QGroupBox("Feature Matrices")
        results_layout = QGridLayout(results_group)

        results_layout.addWidget(QLabel("Historical Features:"), 0, 0)
        self.historical_result = QLabel("Not generated")
        results_layout.addWidget(self.historical_result, 0, 1)

        results_layout.addWidget(QLabel("New Transaction Features:"), 1, 0)
        self.new_result = QLabel("Not generated")
        results_layout.addWidget(self.new_result, 1, 1)

        button_row = QHBoxLayout()
        self.save_btn = QPushButton("Save Feature Matrices")
        self.save_btn.clicked.connect(self._save_matrices)
        self.save_btn.setEnabled(False)
        button_row.addWidget(self.save_btn)
        button_row.addStretch()
        results_layout.addLayout(button_row, 2, 0, 1, 2)

        layout.addWidget(results_group)

    def set_data(self, preprocessor: DataPreprocessor) -> None:
        """Set data from DataPreprocessor (legacy support)."""
        self.preprocessor = preprocessor
        self.historical_df = preprocessor.df
        self.new_transactions_df = None
        self._update_data_status()

    def set_split_data(self, historical_df: pd.DataFrame, new_df: pd.DataFrame) -> None:
        """Set split data from Alpha Vantage fetcher."""
        self.historical_df = historical_df
        self.new_transactions_df = new_df
        self._update_data_status()

    def _update_data_status(self) -> None:
        if self.historical_df is not None and len(self.historical_df) > 0:
            self.historical_status.setText(f"{len(self.historical_df)} rows")
            self.historical_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.historical_status.setText("Not loaded")
            self.historical_status.setStyleSheet("color: gray;")

        if self.new_transactions_df is not None and len(self.new_transactions_df) > 0:
            self.new_status.setText(f"{len(self.new_transactions_df)} rows")
            self.new_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.new_status.setText("None")
            self.new_status.setStyleSheet("color: gray;")

    def _run_engineering(self) -> None:
        if self.historical_df is None or len(self.historical_df) == 0:
            QMessageBox.warning(self, "Warning", "Please load historical data first.")
            return

        selected_features = self.feature_tree.get_selected_features()
        if not selected_features:
            QMessageBox.warning(self, "Warning", "Please select at least one feature.")
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)

        try:
            window_days = self.window_spin.value()
            self.log_widget.log("Starting feature engineering...", "INFO")

            # Combine datasets for consistent feature calculation
            if self.new_transactions_df is not None and len(self.new_transactions_df) > 0:
                # Mark the split point
                historical_end_idx = len(self.historical_df)
                combined_df = pd.concat([self.historical_df, self.new_transactions_df], ignore_index=False)
                self.log_widget.log(f"Combined {len(self.historical_df)} historical + {len(self.new_transactions_df)} new rows", "INFO")
            else:
                combined_df = self.historical_df
                historical_end_idx = len(combined_df)

            self.progress.setValue(20)

            # Create feature engineer on combined data
            self.feature_engineer = FeatureEngineer(combined_df, window_days)
            self.log_widget.log(f"Aggregating by {window_days}-day windows...", "INFO")

            self.feature_engineer.aggregate_by_window()
            self.progress.setValue(50)

            self.log_widget.log(f"Creating {len(selected_features)} features...", "INFO")

            # Create combined feature matrix
            combined_features = self.feature_engineer.create_feature_matrix(selected_features)
            self.progress.setValue(80)

            # Split back into historical and new
            if self.new_transactions_df is not None and len(self.new_transactions_df) > 0:
                # Find split point based on dates
                historical_max_date = self.historical_df.index.max()
                self.historical_features = combined_features[combined_features.index <= historical_max_date]
                self.new_features = combined_features[combined_features.index > historical_max_date]

                self.log_widget.log(
                    f"Split features: {len(self.historical_features)} historical, {len(self.new_features)} new",
                    "INFO"
                )
            else:
                self.historical_features = combined_features
                self.new_features = pd.DataFrame()

            self.progress.setValue(100)

            # Update UI
            self.historical_result.setText(
                f"{self.historical_features.shape[0]} windows × {self.historical_features.shape[1]} features"
            )
            if len(self.new_features) > 0:
                self.new_result.setText(
                    f"{self.new_features.shape[0]} windows × {self.new_features.shape[1]} features"
                )
            else:
                self.new_result.setText("No new data")

            self.save_btn.setEnabled(True)
            self.log_widget.log("Feature engineering complete!", "INFO")

            # Emit signal with both feature matrices
            self.featuresReady.emit(self.feature_engineer, self.historical_features, self.new_features)

        except Exception as e:
            self.log_widget.log(f"Error: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", str(e))

        finally:
            self.run_btn.setEnabled(True)
            self.progress.setVisible(False)

    def _save_matrices(self) -> None:
        """Save feature matrices to CSV files."""
        if self.historical_features is None:
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Features")
        if not save_dir:
            return

        try:
            # Save historical features
            hist_path = Path(save_dir) / "historical_features.csv"
            self.historical_features.to_csv(hist_path)
            self.log_widget.log(f"Saved historical features to {hist_path}", "INFO")

            # Save new features if available
            if self.new_features is not None and len(self.new_features) > 0:
                new_path = Path(save_dir) / "new_transaction_features.csv"
                self.new_features.to_csv(new_path)
                self.log_widget.log(f"Saved new transaction features to {new_path}", "INFO")

            QMessageBox.information(self, "Success", f"Feature matrices saved to {save_dir}")

        except Exception as e:
            self.log_widget.log(f"Error saving: {e}", "ERROR")
            QMessageBox.critical(self, "Error", str(e))


class ModelTrainingTab(QWidget):
    """Tab 3: Model Training & Configuration with model saving."""

    modelsReady = pyqtSignal(object, object, object)  # (model_factory, training_metrics, new_features)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = setup_logging(__name__)
        self.model_factory: Optional[ModelFactory] = None
        self.historical_features: Optional[pd.DataFrame] = None
        self.new_features: Optional[pd.DataFrame] = None
        self.training_metrics: Dict = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Model selection
        select_group = QGroupBox("Step 1: Select Models")
        select_layout = QVBoxLayout(select_group)

        self.model_checks: Dict[str, QCheckBox] = {}
        for model_type in constants.SUPPORTED_MODELS:
            cb = QCheckBox(model_type)
            cb.setChecked(model_type in ['IsolationForest', 'DBSCAN', 'LocalOutlierFactor'])
            self.model_checks[model_type] = cb
            select_layout.addWidget(cb)

        config_btn = QPushButton("Configure Selected Models")
        config_btn.clicked.connect(self._configure_models)
        select_layout.addWidget(config_btn)

        layout.addWidget(select_group)

        # Training configuration
        config_group = QGroupBox("Step 2: Training Configuration")
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel("Train/Test Split:"), 0, 0)
        self.split_spin = QSpinBox()
        self.split_spin.setMinimum(50)
        self.split_spin.setMaximum(95)
        self.split_spin.setValue(80)
        self.split_spin.setSuffix("%")
        config_layout.addWidget(self.split_spin, 0, 1)

        self.standardize_check = QCheckBox("Standardize features (recommended)")
        self.standardize_check.setChecked(True)
        config_layout.addWidget(self.standardize_check, 1, 0, 1, 2)

        layout.addWidget(config_group)

        # Training execution
        train_group = QGroupBox("Step 3: Train Models")
        train_layout = QVBoxLayout(train_group)

        self.train_btn = QPushButton("Train All Selected Models")
        self.train_btn.clicked.connect(self._train_models)
        train_layout.addWidget(self.train_btn)

        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        train_layout.addWidget(self.train_progress)

        self.train_log = LogWidget()
        train_layout.addWidget(self.train_log)

        layout.addWidget(train_group)

        # Results and Save
        results_group = QGroupBox("Step 4: Model Results & Save")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Model", "Type", "Time (s)", "Anomalies", "Rate %"
        ])
        results_layout.addWidget(self.results_table)

        save_layout = QHBoxLayout()
        self.save_models_btn = QPushButton("Save Trained Models")
        self.save_models_btn.clicked.connect(self._save_models)
        self.save_models_btn.setEnabled(False)
        save_layout.addWidget(self.save_models_btn)

        self.load_models_btn = QPushButton("Load Saved Models")
        self.load_models_btn.clicked.connect(self._load_models)
        save_layout.addWidget(self.load_models_btn)

        save_layout.addStretch()
        results_layout.addLayout(save_layout)

        self.model_path_label = QLabel("")
        results_layout.addWidget(self.model_path_label)

        layout.addWidget(results_group)

    def set_feature_matrix(self, feature_engineer, historical_features: pd.DataFrame, new_features: pd.DataFrame) -> None:
        """Set feature matrices from feature engineering tab."""
        self.feature_engineer = feature_engineer
        self.historical_features = historical_features
        self.new_features = new_features
        self.train_log.log(f"Received features: {len(historical_features)} historical, {len(new_features) if new_features is not None else 0} new", "INFO")

    def _configure_models(self) -> None:
        selected = [m for m, cb in self.model_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one model.")
            return

        for model_type in selected:
            current_params = constants.DEFAULT_HYPERPARAMETERS.get(model_type, {})
            dialog = ModelConfigDialog(model_type, current_params, self)
            if dialog.exec_():
                params = dialog.get_parameters()
                self.train_log.log(f"Configured {model_type}: {params}", "INFO")

    def _train_models(self) -> None:
        if self.historical_features is None:
            QMessageBox.warning(self, "Warning", "Please generate features first.")
            return

        selected = [m for m, cb in self.model_checks.items() if cb.isChecked()]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one model.")
            return

        self.train_btn.setEnabled(False)
        self.train_progress.setVisible(True)
        self.train_progress.setValue(0)

        try:
            self.model_factory = ModelFactory()
            X = self.historical_features.values
            feature_names = list(self.historical_features.columns)

            for i, model_type in enumerate(selected):
                self.train_log.log(f"Training {model_type}...", "INFO")
                self.train_progress.setValue(int((i / len(selected)) * 100))

                params = constants.DEFAULT_HYPERPARAMETERS.get(model_type, {})
                model_name = f"{model_type}_{get_timestamp()}"

                self.model_factory.create_model(model_type, params, model_name)
                _, metrics = self.model_factory.train_model(model_name, X, feature_names)
                self.training_metrics[model_name] = metrics

                self.train_log.log(
                    f"  {model_type}: {metrics['anomalies_detected']} anomalies "
                    f"({metrics['training_time']:.2f}s)", "INFO"
                )

            self.train_progress.setValue(100)
            self._update_results_table()

            self.save_models_btn.setEnabled(True)
            self.modelsReady.emit(self.model_factory, self.training_metrics, self.new_features)
            self.train_log.log("All models trained successfully!", "INFO")

        except Exception as e:
            self.train_log.log(f"Error: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", str(e))

        finally:
            self.train_btn.setEnabled(True)
            self.train_progress.setVisible(False)

    def _update_results_table(self) -> None:
        self.results_table.setRowCount(len(self.training_metrics))

        for row, (name, metrics) in enumerate(self.training_metrics.items()):
            self.results_table.setItem(row, 0, QTableWidgetItem(name))
            self.results_table.setItem(row, 1, QTableWidgetItem(metrics.get('model_type', '')))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{metrics.get('training_time', 0):.2f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(str(metrics.get('anomalies_detected', 0))))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{metrics.get('anomaly_rate', 0)*100:.1f}"))

        self.results_table.resizeColumnsToContents()

    def _save_models(self) -> None:
        """Save trained models to disk."""
        if self.model_factory is None:
            QMessageBox.warning(self, "Warning", "No trained models to save.")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Models")
        if not save_dir:
            return

        try:
            import joblib
            save_path = Path(save_dir)

            for model_name, model_info in self.model_factory.models.items():
                model_file = save_path / f"{model_name}.joblib"
                joblib.dump({
                    'model': model_info['model'],
                    'model_type': model_info['model_type'],
                    'hyperparameters': model_info['hyperparameters'],
                    'feature_names': model_info.get('feature_names', []),
                    'training_metrics': self.training_metrics.get(model_name, {})
                }, model_file)
                self.train_log.log(f"Saved {model_name} to {model_file}", "INFO")

            self.model_path_label.setText(f"Models saved to: {save_dir}")
            QMessageBox.information(self, "Success", f"Saved {len(self.model_factory.models)} models to {save_dir}")

        except Exception as e:
            self.train_log.log(f"Error saving models: {e}", "ERROR")
            QMessageBox.critical(self, "Error", str(e))

    def _load_models(self) -> None:
        """Load previously saved models."""
        load_dir = QFileDialog.getExistingDirectory(self, "Select Directory with Saved Models")
        if not load_dir:
            return

        try:
            import joblib
            load_path = Path(load_dir)
            model_files = list(load_path.glob("*.joblib"))

            if not model_files:
                QMessageBox.warning(self, "Warning", "No model files found in directory.")
                return

            self.model_factory = ModelFactory()

            for model_file in model_files:
                data = joblib.load(model_file)
                model_name = model_file.stem

                self.model_factory.models[model_name] = {
                    'model': data['model'],
                    'model_type': data['model_type'],
                    'hyperparameters': data['hyperparameters'],
                    'feature_names': data.get('feature_names', []),
                    'is_trained': True
                }
                self.training_metrics[model_name] = data.get('training_metrics', {})
                self.train_log.log(f"Loaded {model_name}", "INFO")

            self._update_results_table()
            self.save_models_btn.setEnabled(True)
            self.model_path_label.setText(f"Models loaded from: {load_dir}")
            self.modelsReady.emit(self.model_factory, self.training_metrics, self.new_features)

            QMessageBox.information(self, "Success", f"Loaded {len(model_files)} models")

        except Exception as e:
            self.train_log.log(f"Error loading models: {e}", "ERROR")
            QMessageBox.critical(self, "Error", str(e))


class AnomalyReportTab(QWidget):
    """Tab 4: Anomaly Detection & Report Generation (Combined)."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = setup_logging(__name__)
        self.model_factory: Optional[ModelFactory] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.new_features: Optional[pd.DataFrame] = None
        self.training_metrics: Dict = {}
        self.scorer = AnomalyScorer()
        self.anomalies_df: Optional[pd.DataFrame] = None
        self.new_scored_df: Optional[pd.DataFrame] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Create scrollable area for all content
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Model selection and threshold in same row
        config_group = QGroupBox("Step 1: Configure Anomaly Detection")
        config_layout = QGridLayout(config_group)

        config_layout.addWidget(QLabel("Active Model:"), 0, 0)
        self.model_combo = QComboBox()
        config_layout.addWidget(self.model_combo, 0, 1)

        self.ensemble_check = QCheckBox("Use Ensemble (all models)")
        config_layout.addWidget(self.ensemble_check, 0, 2, 1, 2)

        # Threshold configuration
        self.threshold_group = QButtonGroup()
        self.percentile_radio = QRadioButton("Percentile threshold:")
        self.percentile_radio.setChecked(True)
        self.threshold_group.addButton(self.percentile_radio)
        config_layout.addWidget(self.percentile_radio, 1, 0)

        self.percentile_spin = QSpinBox()
        self.percentile_spin.setMinimum(80)
        self.percentile_spin.setMaximum(99)
        self.percentile_spin.setValue(95)
        self.percentile_spin.setSuffix(" %ile")
        config_layout.addWidget(self.percentile_spin, 1, 1)

        self.fixed_radio = QRadioButton("Fixed threshold:")
        self.threshold_group.addButton(self.fixed_radio)
        config_layout.addWidget(self.fixed_radio, 1, 2)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMinimum(0.0)
        self.threshold_spin.setMaximum(1.0)
        self.threshold_spin.setValue(0.75)
        self.threshold_spin.setSingleStep(0.05)
        config_layout.addWidget(self.threshold_spin, 1, 3)

        scroll_layout.addWidget(config_group)

        # Detection section
        detect_group = QGroupBox("Step 2: Detect Anomalies")
        detect_layout = QVBoxLayout(detect_group)

        btn_layout = QHBoxLayout()
        self.detect_btn = QPushButton("Detect Anomalies on Historical Data")
        self.detect_btn.clicked.connect(self._detect_anomalies)
        btn_layout.addWidget(self.detect_btn)

        self.score_new_btn = QPushButton("Score New Transactions")
        self.score_new_btn.clicked.connect(self._score_new_transactions)
        self.score_new_btn.setEnabled(False)
        btn_layout.addWidget(self.score_new_btn)
        detect_layout.addLayout(btn_layout)

        # Status labels
        status_layout = QHBoxLayout()
        self.historical_status = QLabel("Historical: Not analyzed")
        self.historical_status.setStyleSheet("color: gray;")
        status_layout.addWidget(self.historical_status)

        self.new_status = QLabel("New Transactions: Not scored")
        self.new_status.setStyleSheet("color: gray;")
        status_layout.addWidget(self.new_status)
        detect_layout.addLayout(status_layout)

        # Anomaly results table
        self.anomaly_table = DataPreviewTable()
        self.anomaly_table.setMaximumHeight(200)
        detect_layout.addWidget(self.anomaly_table)

        scroll_layout.addWidget(detect_group)

        # Report configuration
        report_group = QGroupBox("Step 3: Configure Report")
        report_layout = QGridLayout(report_group)

        # Report sections (compact layout)
        sections_label = QLabel("Include sections:")
        report_layout.addWidget(sections_label, 0, 0)

        self.section_checks = {}
        sections = [
            ("summary", "Executive Summary"),
            ("anomalies", "Historical Anomalies"),
            ("features", "Feature Details"),
            ("models", "Model Performance"),
            ("new_scores", "New Transaction Scores"),
            ("timeline", "Anomaly Timeline Chart"),
            ("specs", "Technical Specs"),
            ("appendix", "Appendix")
        ]

        # Arrange in 2 columns
        for i, (key, label) in enumerate(sections):
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.section_checks[key] = cb
            row = (i // 4) + 1
            col = i % 4
            report_layout.addWidget(cb, row, col)

        # Report title
        report_layout.addWidget(QLabel("Report Title:"), 3, 0)
        self.title_edit = QLineEdit("Insider Trading Anomaly Analysis Report")
        report_layout.addWidget(self.title_edit, 3, 1, 1, 3)

        scroll_layout.addWidget(report_group)

        # Generate section
        generate_group = QGroupBox("Step 4: Generate Report")
        generate_layout = QVBoxLayout(generate_group)

        gen_btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Excel Report")
        self.generate_btn.clicked.connect(self._generate_report)
        self.generate_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        gen_btn_layout.addWidget(self.generate_btn)

        self.open_btn = QPushButton("Open Report")
        self.open_btn.clicked.connect(self._open_report)
        self.open_btn.setEnabled(False)
        gen_btn_layout.addWidget(self.open_btn)
        generate_layout.addLayout(gen_btn_layout)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        generate_layout.addWidget(self.progress)

        self.output_label = QLabel("")
        self.output_label.setStyleSheet("color: green;")
        generate_layout.addWidget(self.output_label)

        self.log_widget = LogWidget()
        self.log_widget.setMaximumHeight(120)
        generate_layout.addWidget(self.log_widget)

        scroll_layout.addWidget(generate_group)

        layout.addWidget(scroll_content)

    def set_models(self, model_factory: ModelFactory, training_metrics: Dict, new_features: pd.DataFrame) -> None:
        """Set models and new features from training tab."""
        self.model_factory = model_factory
        self.training_metrics = training_metrics
        self.new_features = new_features

        self.model_combo.clear()
        for name in model_factory.models.keys():
            self.model_combo.addItem(name)

        if new_features is not None and len(new_features) > 0:
            self.score_new_btn.setEnabled(True)
            self.new_status.setText(f"New Transactions: {len(new_features)} windows ready")
            self.new_status.setStyleSheet("color: blue;")

        self.log_widget.log(f"Loaded {len(model_factory.models)} models", "INFO")

    def set_feature_matrix(self, feature_engineer, historical_features: pd.DataFrame, new_features: pd.DataFrame) -> None:
        """Set feature matrices from feature engineering tab."""
        self.feature_engineer = feature_engineer
        self.feature_matrix = historical_features
        self.new_features = new_features

        if new_features is not None and len(new_features) > 0:
            self.score_new_btn.setEnabled(True)
            self.new_status.setText(f"New Transactions: {len(new_features)} windows ready")
            self.new_status.setStyleSheet("color: blue;")

    def _detect_anomalies(self) -> None:
        """Detect anomalies in historical data."""
        if self.model_factory is None or self.feature_matrix is None:
            QMessageBox.warning(self, "Warning", "Please train models and generate features first.")
            return

        try:
            model_name = self.model_combo.currentText()
            if not model_name:
                QMessageBox.warning(self, "Warning", "Please select a model.")
                return

            X = self.feature_matrix.values
            window_dates = self.feature_matrix.index

            self.log_widget.log(f"Detecting anomalies using {model_name}...", "INFO")

            if self.percentile_radio.isChecked():
                percentile = self.percentile_spin.value()
                results = self.model_factory.detect_historical_anomalies(
                    model_name, X, window_dates, percentile
                )
                self.log_widget.log(f"Using {percentile}th percentile threshold", "INFO")
            else:
                scores = self.model_factory.score(model_name, X)
                threshold = self.threshold_spin.value()
                results = pd.DataFrame({
                    'window_date': window_dates,
                    'anomaly_score': scores,
                    'is_anomaly': (scores >= threshold).astype(int)
                })
                self.log_widget.log(f"Using fixed threshold: {threshold}", "INFO")

            # Add risk levels
            results['risk_level'] = results['anomaly_score'].apply(
                lambda s: self.scorer.get_risk_level(s)
            )

            self.anomaly_table.set_dataframe(results, risk_column='risk_level')
            self.anomalies_df = results

            n_anomalies = len(results[results['is_anomaly'] == 1])
            self.historical_status.setText(f"Historical: {n_anomalies} anomalies found")
            self.historical_status.setStyleSheet("color: green; font-weight: bold;")

            self.log_widget.log(f"Found {n_anomalies} anomalies in {len(results)} windows", "INFO")

        except Exception as e:
            self.log_widget.log(f"Error: {e}", "ERROR")
            QMessageBox.critical(self, "Error", str(e))

    def _score_new_transactions(self) -> None:
        """Score new transactions using trained model."""
        if self.model_factory is None or self.new_features is None or len(self.new_features) == 0:
            QMessageBox.warning(self, "Warning", "No new transaction features available.")
            return

        try:
            model_name = self.model_combo.currentText()
            if not model_name:
                QMessageBox.warning(self, "Warning", "Please select a model.")
                return

            X = self.new_features.values
            window_dates = self.new_features.index

            self.log_widget.log(f"Scoring {len(X)} new windows using {model_name}...", "INFO")

            scores = self.model_factory.score(model_name, X)

            # Determine threshold
            if self.percentile_radio.isChecked():
                percentile = self.percentile_spin.value()
                threshold = np.percentile(scores, percentile)
            else:
                threshold = self.threshold_spin.value()

            self.new_scored_df = pd.DataFrame({
                'window_date': window_dates,
                'anomaly_score': scores,
                'is_anomaly': (scores >= threshold).astype(int),
                'risk_level': [self.scorer.get_risk_level(s) for s in scores]
            })

            n_anomalies = len(self.new_scored_df[self.new_scored_df['is_anomaly'] == 1])
            self.new_status.setText(f"New Transactions: {n_anomalies} anomalies in {len(self.new_scored_df)} windows")
            self.new_status.setStyleSheet("color: green; font-weight: bold;")

            self.log_widget.log(f"Scored {len(self.new_scored_df)} new windows, {n_anomalies} anomalies", "INFO")

        except Exception as e:
            self.log_widget.log(f"Error scoring: {e}", "ERROR")
            QMessageBox.critical(self, "Error", str(e))

    def _generate_report(self) -> None:
        """Generate Excel report with all data."""
        if self.anomalies_df is None:
            QMessageBox.warning(self, "Warning", "Please detect anomalies first.")
            return

        try:
            self.progress.setVisible(True)
            self.progress.setValue(0)
            self.log_widget.log("Generating report...", "INFO")

            filepath = constants.REPORTS_DIR / f"{get_timestamp()}_report.xlsx"

            reporter = ExcelReporter(
                filepath,
                self.title_edit.text(),
                ""
            )

            self.progress.setValue(10)

            # Prepare summary data
            summary_data = {
                'total_transactions': len(self.feature_matrix) if self.feature_matrix is not None else 0,
                'n_windows': len(self.feature_matrix) if self.feature_matrix is not None else 0,
                'n_features': self.feature_matrix.shape[1] if self.feature_matrix is not None else 0,
                'n_models': len(self.training_metrics),
                'n_anomalies': len(self.anomalies_df[self.anomalies_df['is_anomaly'] == 1]) if self.anomalies_df is not None else 0,
                'window_days': 14,
                'date_range': {}
            }

            feature_stats = {}
            if hasattr(self, 'feature_engineer') and self.feature_engineer is not None:
                feature_stats = self.feature_engineer.get_feature_statistics()

            self.progress.setValue(30)
            self.log_widget.log("Adding report sections...", "INFO")

            # Generate selected sections
            output_path = reporter.generate_full_report(
                summary_data=summary_data,
                anomalies_df=self.anomalies_df if self.section_checks['anomalies'].isChecked() else pd.DataFrame(),
                feature_stats=feature_stats if self.section_checks['features'].isChecked() else {},
                training_metrics=self.training_metrics if self.section_checks['models'].isChecked() else {},
                new_transactions_df=self.new_scored_df if self.section_checks['new_scores'].isChecked() else None,
                explanations=[],
                specs_data={'n_windows': len(self.feature_matrix) if self.feature_matrix is not None else 0}
            )

            self.progress.setValue(70)

            # Add anomaly timeline if selected
            if self.section_checks['timeline'].isChecked() and self.anomalies_df is not None:
                try:
                    reporter.add_anomaly_timeline(
                        self.anomalies_df,
                        time_column='window_date',
                        score_column='anomaly_score'
                    )
                    self.log_widget.log("Added anomaly timeline chart", "INFO")
                except Exception as e:
                    self.log_widget.log(f"Could not add timeline: {e}", "WARNING")

            self.progress.setValue(100)
            self.output_path = output_path
            self.output_label.setText(f"Report saved: {output_path}")
            self.open_btn.setEnabled(True)

            self.log_widget.log(f"Report generated: {output_path}", "INFO")
            QMessageBox.information(self, "Success", f"Report generated: {output_path}")

        except Exception as e:
            self.log_widget.log(f"Error: {e}", "ERROR")
            QMessageBox.critical(self, "Error", str(e))
            self.logger.error(f"Report generation error: {e}")

        finally:
            self.progress.setVisible(False)

    def _open_report(self) -> None:
        """Open the generated report."""
        if hasattr(self, 'output_path'):
            import subprocess
            import platform

            if platform.system() == 'Darwin':
                subprocess.run(['open', str(self.output_path)])
            elif platform.system() == 'Windows':
                os.startfile(str(self.output_path))
            else:
                subprocess.run(['xdg-open', str(self.output_path)])


class DataFetcherTab(QWidget):
    """Tab for fetching data from Alpha Vantage API with auto-split for new transactions."""

    dataFetched = pyqtSignal(object, object)  # (historical_df, new_transactions_df)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = setup_logging(__name__)
        self.client: Optional[AlphaVantageClient] = None
        self.historical_df: Optional[pd.DataFrame] = None
        self.new_transactions_df: Optional[pd.DataFrame] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # API Key Section
        api_group = QGroupBox("API Configuration")
        api_layout = QGridLayout(api_group)

        api_layout.addWidget(QLabel("Alpha Vantage API Key:"), 0, 0)
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setPlaceholderText("Enter your API key (get free at alphavantage.co)")
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self.api_key_edit, 0, 1)

        self.show_key_check = QCheckBox("Show")
        self.show_key_check.toggled.connect(
            lambda checked: self.api_key_edit.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )
        api_layout.addWidget(self.show_key_check, 0, 2)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._connect_api)
        api_layout.addWidget(self.connect_btn, 0, 3)

        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet("color: gray;")
        api_layout.addWidget(self.status_label, 1, 0, 1, 4)

        layout.addWidget(api_group)

        # Symbol Search Section
        search_group = QGroupBox("Symbol Search")
        search_layout = QHBoxLayout(search_group)

        search_layout.addWidget(QLabel("Symbol:"))
        self.symbol_edit = QLineEdit()
        self.symbol_edit.setPlaceholderText("Enter stock symbol (e.g., AAPL, MSFT)")
        self.symbol_edit.returnPressed.connect(self._search_symbol)
        search_layout.addWidget(self.symbol_edit)

        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self._search_symbol)
        search_layout.addWidget(self.search_btn)

        layout.addWidget(search_group)

        # Search results
        self.search_results = QTableWidget()
        self.search_results.setColumnCount(4)
        self.search_results.setHorizontalHeaderLabels(['Symbol', 'Name', 'Type', 'Region'])
        self.search_results.setMaximumHeight(120)
        self.search_results.itemDoubleClicked.connect(self._select_from_search)
        layout.addWidget(self.search_results)

        # Data Parameters Section
        params_group = QGroupBox("Data Parameters")
        params_layout = QGridLayout(params_group)

        params_layout.addWidget(QLabel("Data Type:"), 0, 0)
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems([
            'Daily (Adjusted)',
            'Daily',
            'Weekly (Adjusted)',
            'Weekly',
            'Monthly (Adjusted)',
            'Monthly',
            'Insider Transactions'
        ])
        params_layout.addWidget(self.data_type_combo, 0, 1)

        params_layout.addWidget(QLabel("Output Size:"), 0, 2)
        self.output_size_combo = QComboBox()
        self.output_size_combo.addItems(['Full (20+ years)', 'Compact (100 days)'])
        params_layout.addWidget(self.output_size_combo, 0, 3)

        # New transactions split configuration
        params_layout.addWidget(QLabel("New Transactions Period:"), 1, 0)
        self.new_period_spin = QSpinBox()
        self.new_period_spin.setMinimum(1)
        self.new_period_spin.setMaximum(24)
        self.new_period_spin.setValue(6)
        self.new_period_spin.setSuffix(" months")
        params_layout.addWidget(self.new_period_spin, 1, 1)

        self.auto_split_check = QCheckBox("Auto-split recent data as 'New Transactions'")
        self.auto_split_check.setChecked(True)
        params_layout.addWidget(self.auto_split_check, 1, 2, 1, 2)

        layout.addWidget(params_group)

        # Cached Data Info Section
        cache_group = QGroupBox("Cached Data for Selected Symbol")
        cache_layout = QVBoxLayout(cache_group)

        self.cache_info_label = QLabel("Select a symbol to see cached data info")
        self.cache_info_label.setStyleSheet("color: gray;")
        cache_layout.addWidget(self.cache_info_label)

        self.cache_table = QTableWidget()
        self.cache_table.setColumnCount(5)
        self.cache_table.setHorizontalHeaderLabels([
            'Data Type', 'Start Date', 'End Date', 'Rows', 'Last Updated'
        ])
        self.cache_table.setMaximumHeight(100)
        cache_layout.addWidget(self.cache_table)

        layout.addWidget(cache_group)

        # Fetch Section
        fetch_group = QGroupBox("Fetch Data")
        fetch_layout = QVBoxLayout(fetch_group)

        btn_layout = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch Data from Alpha Vantage")
        self.fetch_btn.clicked.connect(self._fetch_data)
        self.fetch_btn.setEnabled(False)
        btn_layout.addWidget(self.fetch_btn)

        self.load_cached_btn = QPushButton("Load Cached Data")
        self.load_cached_btn.clicked.connect(self._load_cached)
        self.load_cached_btn.setEnabled(False)
        btn_layout.addWidget(self.load_cached_btn)

        fetch_layout.addLayout(btn_layout)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        fetch_layout.addWidget(self.progress)

        self.log_widget = LogWidget()
        self.log_widget.setMaximumHeight(100)
        fetch_layout.addWidget(self.log_widget)

        layout.addWidget(fetch_group)

        # Data Split Summary
        split_group = QGroupBox("Data Split Summary")
        split_layout = QGridLayout(split_group)

        split_layout.addWidget(QLabel("Historical Data:"), 0, 0)
        self.historical_label = QLabel("Not loaded")
        self.historical_label.setStyleSheet("font-weight: bold;")
        split_layout.addWidget(self.historical_label, 0, 1)

        split_layout.addWidget(QLabel("New Transactions:"), 0, 2)
        self.new_trans_label = QLabel("Not loaded")
        self.new_trans_label.setStyleSheet("font-weight: bold;")
        split_layout.addWidget(self.new_trans_label, 0, 3)

        self.use_data_btn = QPushButton("Use This Data for Analysis")
        self.use_data_btn.clicked.connect(self._use_data)
        self.use_data_btn.setEnabled(False)
        split_layout.addWidget(self.use_data_btn, 1, 0, 1, 4)

        layout.addWidget(split_group)

        # Preview Section
        preview_group = QGroupBox("Data Preview (Historical)")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = DataPreviewTable()
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_group)

    def _connect_api(self) -> None:
        """Connect to Alpha Vantage API."""
        api_key = self.api_key_edit.text().strip()
        if not api_key:
            QMessageBox.warning(self, "Warning", "Please enter an API key.")
            return

        try:
            self.client = AlphaVantageClient(api_key=api_key)

            # Test connection with a simple search
            self.log_widget.log("Testing API connection...", "INFO")
            results = self.client.search_symbols("IBM")

            if len(results) > 0:
                self.status_label.setText("Connected successfully")
                self.status_label.setStyleSheet("color: green;")
                self.fetch_btn.setEnabled(True)
                self.log_widget.log("API connection successful!", "INFO")

                # Update cached data view
                self._update_all_cached_symbols()
            else:
                self.status_label.setText("Connected (no test results)")
                self.status_label.setStyleSheet("color: orange;")
                self.fetch_btn.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"Connection failed: {str(e)[:50]}")
            self.status_label.setStyleSheet("color: red;")
            self.log_widget.log(f"Connection failed: {e}", "ERROR")
            QMessageBox.critical(self, "Error", f"Failed to connect: {e}")

    def _search_symbol(self) -> None:
        """Search for stock symbols."""
        if not self.client:
            QMessageBox.warning(self, "Warning", "Please connect to API first.")
            return

        keywords = self.symbol_edit.text().strip()
        if not keywords:
            return

        try:
            self.log_widget.log(f"Searching for '{keywords}'...", "INFO")
            results = self.client.search_symbols(keywords)

            self.search_results.setRowCount(len(results))
            for row, (_, data) in enumerate(results.iterrows()):
                self.search_results.setItem(row, 0, QTableWidgetItem(str(data.get('symbol', ''))))
                self.search_results.setItem(row, 1, QTableWidgetItem(str(data.get('name', ''))[:50]))
                self.search_results.setItem(row, 2, QTableWidgetItem(str(data.get('type', ''))))
                self.search_results.setItem(row, 3, QTableWidgetItem(str(data.get('region', ''))))

            self.search_results.resizeColumnsToContents()
            self.log_widget.log(f"Found {len(results)} results", "INFO")

            # Also check cached data
            self._update_cache_info(keywords.upper())

        except Exception as e:
            self.log_widget.log(f"Search failed: {e}", "ERROR")

    def _select_from_search(self, item: QTableWidgetItem) -> None:
        """Select symbol from search results."""
        row = item.row()
        symbol = self.search_results.item(row, 0).text()
        self.symbol_edit.setText(symbol)
        self._update_cache_info(symbol)

    def _update_cache_info(self, symbol: str) -> None:
        """Update cached data information for symbol."""
        if not self.client:
            return

        info = self.client.inventory.get_symbol_info(symbol)

        if not info:
            self.cache_info_label.setText(f"No cached data found for {symbol}")
            self.cache_info_label.setStyleSheet("color: gray;")
            self.cache_table.setRowCount(0)
            self.load_cached_btn.setEnabled(False)
            return

        self.cache_info_label.setText(f"Cached data available for {symbol}:")
        self.cache_info_label.setStyleSheet("color: green;")
        self.load_cached_btn.setEnabled(True)

        self.cache_table.setRowCount(len(info))
        for row, (data_type, details) in enumerate(info.items()):
            self.cache_table.setItem(row, 0, QTableWidgetItem(data_type))
            self.cache_table.setItem(row, 1, QTableWidgetItem(str(details.get('start_date', 'N/A'))))
            self.cache_table.setItem(row, 2, QTableWidgetItem(str(details.get('end_date', 'N/A'))))
            self.cache_table.setItem(row, 3, QTableWidgetItem(str(details.get('row_count', 0))))
            self.cache_table.setItem(row, 4, QTableWidgetItem(str(details.get('last_updated', 'N/A'))[:10]))

        self.cache_table.resizeColumnsToContents()

    def _update_all_cached_symbols(self) -> None:
        """Update log with all cached symbols."""
        if not self.client:
            return

        symbols = self.client.inventory.get_all_symbols()
        if symbols:
            self.log_widget.log(f"Cached symbols: {', '.join(symbols)}", "INFO")

    def _get_data_type_params(self) -> tuple:
        """Get data type parameters from UI selection."""
        selection = self.data_type_combo.currentText()

        type_mapping = {
            'Daily (Adjusted)': ('daily', True),
            'Daily': ('daily', False),
            'Weekly (Adjusted)': ('weekly', True),
            'Weekly': ('weekly', False),
            'Monthly (Adjusted)': ('monthly', True),
            'Monthly': ('monthly', False),
            'Insider Transactions': ('insider', False)
        }

        return type_mapping.get(selection, ('daily', True))

    def _fetch_data(self) -> None:
        """Fetch data from Alpha Vantage."""
        if not self.client:
            QMessageBox.warning(self, "Warning", "Please connect to API first.")
            return

        symbol = self.symbol_edit.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Warning", "Please enter a symbol.")
            return

        data_type, adjusted = self._get_data_type_params()
        output_size = 'full' if 'Full' in self.output_size_combo.currentText() else 'compact'

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.fetch_btn.setEnabled(False)

        try:
            self.log_widget.log(f"Fetching {data_type} data for {symbol}...", "INFO")
            self.progress.setValue(30)

            if data_type == 'insider':
                df = self.client.get_insider_transactions(symbol)
            else:
                df = self.client.get_stock_data(
                    symbol,
                    interval=data_type,
                    outputsize=output_size,
                    adjusted=adjusted
                )

            self.progress.setValue(70)

            if len(df) > 0:
                self.log_widget.log(
                    f"Fetched {len(df)} rows from {df.index.min()} to {df.index.max()}",
                    "INFO"
                )

                # Split data if auto-split is enabled
                self._split_data(df)
                self.progress.setValue(90)

                self._update_cache_info(symbol)

            else:
                self.log_widget.log("No data returned", "WARNING")

            self.progress.setValue(100)

        except Exception as e:
            self.log_widget.log(f"Fetch failed: {e}", "ERROR")
            QMessageBox.critical(self, "Error", f"Failed to fetch data: {e}")

        finally:
            self.fetch_btn.setEnabled(True)
            self.progress.setVisible(False)

    def _split_data(self, df: pd.DataFrame) -> None:
        """Split data into historical and new transactions."""
        from datetime import timedelta

        if self.auto_split_check.isChecked():
            # Get the split date (N months ago from the most recent date)
            months = self.new_period_spin.value()

            if df.index.name == 'date' or hasattr(df.index, 'date'):
                max_date = df.index.max()
                split_date = max_date - timedelta(days=months * 30)

                self.historical_df = df[df.index < split_date].copy()
                self.new_transactions_df = df[df.index >= split_date].copy()

                self.log_widget.log(
                    f"Split at {split_date.strftime('%Y-%m-%d')}: "
                    f"{len(self.historical_df)} historical, {len(self.new_transactions_df)} new",
                    "INFO"
                )
            else:
                # No date index, use all as historical
                self.historical_df = df.copy()
                self.new_transactions_df = pd.DataFrame()
                self.log_widget.log("No date index found, using all data as historical", "WARNING")
        else:
            # No split, all data is historical
            self.historical_df = df.copy()
            self.new_transactions_df = pd.DataFrame()

        # Update labels
        if len(self.historical_df) > 0:
            hist_start = str(self.historical_df.index.min())[:10]
            hist_end = str(self.historical_df.index.max())[:10]
            self.historical_label.setText(f"{len(self.historical_df)} rows ({hist_start} to {hist_end})")
        else:
            self.historical_label.setText("No data")

        if len(self.new_transactions_df) > 0:
            new_start = str(self.new_transactions_df.index.min())[:10]
            new_end = str(self.new_transactions_df.index.max())[:10]
            self.new_trans_label.setText(f"{len(self.new_transactions_df)} rows ({new_start} to {new_end})")
        else:
            self.new_trans_label.setText("No new transactions")

        # Update preview
        self.preview_table.set_dataframe(self.historical_df.head(50))
        self.use_data_btn.setEnabled(True)

    def _load_cached(self) -> None:
        """Load cached data for symbol."""
        if not self.client:
            return

        symbol = self.symbol_edit.text().strip().upper()
        if not symbol:
            return

        data_type, _ = self._get_data_type_params()

        try:
            df = self.client.load_cached_data(symbol, data_type)

            if df is not None and len(df) > 0:
                self.log_widget.log(f"Loaded {len(df)} rows from cache", "INFO")
                self._split_data(df)
                self._update_cache_info(symbol)
            else:
                self.log_widget.log("No cached data found", "WARNING")

        except Exception as e:
            self.log_widget.log(f"Failed to load cached data: {e}", "ERROR")

    def _use_data(self) -> None:
        """Emit the split data for use in analysis pipeline."""
        if self.historical_df is None or len(self.historical_df) == 0:
            QMessageBox.warning(self, "Warning", "No historical data available.")
            return

        self.dataFetched.emit(self.historical_df, self.new_transactions_df)
        self.log_widget.log("Data sent to analysis pipeline", "INFO")
        QMessageBox.information(
            self, "Success",
            f"Data loaded:\n"
            f"- Historical: {len(self.historical_df)} rows\n"
            f"- New Transactions: {len(self.new_transactions_df) if self.new_transactions_df is not None else 0} rows\n\n"
            f"Switch to Feature Engineering tab to continue."
        )


class MainWindow(QMainWindow):
    """Main application window with 5 tabs for streamlined workflow."""

    def __init__(self):
        super().__init__()
        self.logger = setup_logging(__name__)
        self.setWindowTitle("Insider Trading Anomaly Detection System")
        self.resize(*constants.GUI_WINDOW_SIZE)
        self.setMinimumSize(*constants.GUI_MIN_SIZE)

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

    def _setup_ui(self) -> None:
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Tab widget
        self.tabs = QTabWidget()

        # Create tabs (streamlined 5-tab workflow)
        self.fetch_tab = DataFetcherTab()
        self.data_tab = DataManagementTab()
        self.feature_tab = FeatureEngineeringTab()
        self.model_tab = ModelTrainingTab()
        self.anomaly_report_tab = AnomalyReportTab()

        # Tab order: Fetch -> Data Mgmt -> Features -> Training -> Detection & Report
        self.tabs.addTab(self.fetch_tab, "1. Data Fetcher (Alpha Vantage)")
        self.tabs.addTab(self.data_tab, "2. Data Management (CSV)")
        self.tabs.addTab(self.feature_tab, "3. Feature Engineering")
        self.tabs.addTab(self.model_tab, "4. Model Training")
        self.tabs.addTab(self.anomaly_report_tab, "5. Detection & Reporting")

        layout.addWidget(self.tabs)

        # Status bar
        self.status_bar = StatusBar()
        layout.addWidget(self.status_bar)

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        fetch_action = QAction("Fetch from Alpha Vantage", self)
        fetch_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        file_menu.addAction(fetch_action)

        load_action = QAction("Load CSV Data", self)
        load_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self) -> None:
        # Alpha Vantage fetched data -> Feature Engineering (with split)
        self.fetch_tab.dataFetched.connect(self.feature_tab.set_split_data)
        self.fetch_tab.dataFetched.connect(
            lambda hist, new: self.status_bar.set_status(
                f"Data fetched: {len(hist)} historical, {len(new) if new is not None else 0} new"
            )
        )

        # CSV Data loaded -> Feature Engineering (legacy path)
        self.data_tab.dataLoaded.connect(self.feature_tab.set_data)
        self.data_tab.dataLoaded.connect(
            lambda _: self.status_bar.set_status("CSV data loaded successfully")
        )

        # Features ready -> Model Training and Anomaly/Report tab
        self.feature_tab.featuresReady.connect(self.model_tab.set_feature_matrix)
        self.feature_tab.featuresReady.connect(self.anomaly_report_tab.set_feature_matrix)
        self.feature_tab.featuresReady.connect(
            lambda eng, hist, new: self.status_bar.set_status(
                f"Features computed: {len(hist)} historical, {len(new) if new is not None else 0} new"
            )
        )

        # Models ready -> Anomaly/Report tab
        self.model_tab.modelsReady.connect(self.anomaly_report_tab.set_models)
        self.model_tab.modelsReady.connect(
            lambda factory, metrics, new: self.status_bar.set_status(
                f"Models trained: {len(factory.models)} models ready"
            )
        )

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            "Insider Trading Anomaly Detection System\n\n"
            "Version 1.1.0\n\n"
            "A production-ready Python system for detecting "
            "suspicious insider trading patterns using machine learning.\n\n"
            "Workflow:\n"
            "1. Fetch data from Alpha Vantage or load CSV\n"
            "2. Engineer features from transaction data\n"
            "3. Train anomaly detection models\n"
            "4. Detect anomalies and generate reports"
        )


def run_gui():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    run_gui()
