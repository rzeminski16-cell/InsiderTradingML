"""
Main GUI Application for the Insider Trading Anomaly Detection System.

This module provides the main application window with 5 tabs:
- Tab 1: Data Management
- Tab 2: Feature Engineering
- Tab 3: Model Training & Configuration
- Tab 4: Anomaly Detection & Scoring
- Tab 5: Reporting

Classes:
    MainWindow: Main application window
    DataManagementTab: Tab for loading and previewing data
    FeatureEngineeringTab: Tab for feature engineering
    ModelTrainingTab: Tab for model training and configuration
    AnomalyDetectionTab: Tab for anomaly detection and scoring
    ReportingTab: Tab for report generation
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
    """Tab 2: Feature Engineering - Configure and create features."""

    featuresReady = pyqtSignal(object, object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = setup_logging(__name__)
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

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

        self.run_btn = QPushButton("Run Feature Engineering")
        self.run_btn.clicked.connect(self._run_engineering)
        run_layout.addWidget(self.run_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        run_layout.addWidget(self.progress)

        self.log_widget = LogWidget()
        run_layout.addWidget(self.log_widget)

        layout.addWidget(run_group)

        # Results section
        results_group = QGroupBox("Feature Matrix")
        results_layout = QVBoxLayout(results_group)

        self.results_label = QLabel("No feature matrix generated yet")
        results_layout.addWidget(self.results_label)

        button_row = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Matrix")
        self.preview_btn.clicked.connect(self._preview_matrix)
        self.preview_btn.setEnabled(False)
        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.clicked.connect(self._export_matrix)
        self.export_btn.setEnabled(False)
        button_row.addWidget(self.preview_btn)
        button_row.addWidget(self.export_btn)
        button_row.addStretch()
        results_layout.addLayout(button_row)

        layout.addWidget(results_group)

    def set_data(self, preprocessor: DataPreprocessor) -> None:
        self.preprocessor = preprocessor

    def _run_engineering(self) -> None:
        if not hasattr(self, 'preprocessor') or self.preprocessor is None:
            QMessageBox.warning(self, "Warning", "Please load data first.")
            return

        selected_features = self.feature_tree.get_selected_features()
        if not selected_features:
            QMessageBox.warning(self, "Warning", "Please select at least one feature.")
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)

        try:
            self.log_widget.log("Starting feature engineering...", "INFO")

            window_days = self.window_spin.value()
            self.feature_engineer = FeatureEngineer(
                self.preprocessor.df,
                window_days
            )

            self.log_widget.log(f"Aggregating by {window_days}-day windows...", "INFO")
            self.progress.setValue(25)

            self.feature_engineer.aggregate_by_window()
            self.progress.setValue(50)

            self.log_widget.log(f"Creating {len(selected_features)} features...", "INFO")

            self.feature_matrix = self.feature_engineer.create_feature_matrix(
                selected_features
            )
            self.progress.setValue(100)

            self.log_widget.log(
                f"Feature matrix created: {self.feature_matrix.shape}", "INFO"
            )

            self.results_label.setText(
                f"Feature Matrix: {self.feature_matrix.shape[0]} windows × "
                f"{self.feature_matrix.shape[1]} features"
            )
            self.preview_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

            self.featuresReady.emit(self.feature_engineer, self.feature_matrix)

        except Exception as e:
            self.log_widget.log(f"Error: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Error", str(e))

        finally:
            self.run_btn.setEnabled(True)
            self.progress.setVisible(False)

    def _preview_matrix(self) -> None:
        if self.feature_matrix is None:
            return

        dialog = QMessageBox(self)
        dialog.setWindowTitle("Feature Matrix Preview")
        dialog.setText(str(self.feature_matrix.head(10)))
        dialog.exec_()

    def _export_matrix(self) -> None:
        if self.feature_matrix is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Feature Matrix", "feature_matrix.csv",
            "CSV Files (*.csv)"
        )
        if filepath:
            self.feature_matrix.to_csv(filepath)
            QMessageBox.information(self, "Success", f"Exported to {filepath}")


class ModelTrainingTab(QWidget):
    """Tab 3: Model Training & Configuration."""

    modelsReady = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = setup_logging(__name__)
        self.model_factory: Optional[ModelFactory] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
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

        # Results
        results_group = QGroupBox("Step 4: Model Comparison")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Model", "Type", "Time (s)", "Anomalies", "Rate %"
        ])
        results_layout.addWidget(self.results_table)

        layout.addWidget(results_group)

    def set_feature_matrix(self, feature_engineer, feature_matrix: pd.DataFrame) -> None:
        self.feature_engineer = feature_engineer
        self.feature_matrix = feature_matrix

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
        if self.feature_matrix is None:
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
            X = self.feature_matrix.values
            feature_names = list(self.feature_matrix.columns)

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

            self.modelsReady.emit(self.model_factory)
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


class AnomalyDetectionTab(QWidget):
    """Tab 4: Anomaly Detection & Scoring."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = setup_logging(__name__)
        self.model_factory: Optional[ModelFactory] = None
        self.feature_matrix: Optional[pd.DataFrame] = None
        self.scorer = AnomalyScorer()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Model selection
        model_group = QGroupBox("Step 1: Select Model")
        model_layout = QHBoxLayout(model_group)

        model_layout.addWidget(QLabel("Active Model:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)

        self.ensemble_check = QCheckBox("Use Ensemble (all models)")
        model_layout.addWidget(self.ensemble_check)

        layout.addWidget(model_group)

        # Threshold configuration
        threshold_group = QGroupBox("Step 2: Configure Threshold")
        threshold_layout = QGridLayout(threshold_group)

        self.threshold_group = QButtonGroup()
        self.percentile_radio = QRadioButton("Percentile:")
        self.percentile_radio.setChecked(True)
        self.threshold_group.addButton(self.percentile_radio)
        threshold_layout.addWidget(self.percentile_radio, 0, 0)

        self.percentile_spin = QSpinBox()
        self.percentile_spin.setMinimum(80)
        self.percentile_spin.setMaximum(99)
        self.percentile_spin.setValue(95)
        threshold_layout.addWidget(self.percentile_spin, 0, 1)

        self.fixed_radio = QRadioButton("Fixed threshold:")
        self.threshold_group.addButton(self.fixed_radio)
        threshold_layout.addWidget(self.fixed_radio, 1, 0)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMinimum(0.0)
        self.threshold_spin.setMaximum(1.0)
        self.threshold_spin.setValue(0.75)
        self.threshold_spin.setSingleStep(0.05)
        threshold_layout.addWidget(self.threshold_spin, 1, 1)

        layout.addWidget(threshold_group)

        # Detection
        detect_group = QGroupBox("Step 3: Detect Historical Anomalies")
        detect_layout = QVBoxLayout(detect_group)

        self.detect_btn = QPushButton("Detect Anomalies")
        self.detect_btn.clicked.connect(self._detect_anomalies)
        detect_layout.addWidget(self.detect_btn)

        self.anomaly_table = DataPreviewTable()
        detect_layout.addWidget(self.anomaly_table)

        layout.addWidget(detect_group)

        # New transactions
        new_group = QGroupBox("Step 4: Score New Transactions")
        new_layout = QVBoxLayout(new_group)

        btn_layout = QHBoxLayout()
        self.load_new_btn = QPushButton("Load New Transactions")
        self.load_new_btn.clicked.connect(self._load_new_transactions)
        self.score_btn = QPushButton("Score Transactions")
        self.score_btn.clicked.connect(self._score_new)
        self.score_btn.setEnabled(False)
        btn_layout.addWidget(self.load_new_btn)
        btn_layout.addWidget(self.score_btn)
        new_layout.addLayout(btn_layout)

        self.new_table = DataPreviewTable()
        new_layout.addWidget(self.new_table)

        layout.addWidget(new_group)

    def set_models(self, model_factory: ModelFactory) -> None:
        self.model_factory = model_factory
        self.model_combo.clear()
        for name in model_factory.models.keys():
            self.model_combo.addItem(name)

    def set_feature_matrix(self, feature_engineer, feature_matrix: pd.DataFrame) -> None:
        self.feature_engineer = feature_engineer
        self.feature_matrix = feature_matrix

    def _detect_anomalies(self) -> None:
        if self.model_factory is None or self.feature_matrix is None:
            QMessageBox.warning(self, "Warning", "Please train models first.")
            return

        try:
            model_name = self.model_combo.currentText()
            X = self.feature_matrix.values
            window_dates = self.feature_matrix.index

            if self.percentile_radio.isChecked():
                percentile = self.percentile_spin.value()
                results = self.model_factory.detect_historical_anomalies(
                    model_name, X, window_dates, percentile
                )
            else:
                scores = self.model_factory.score(model_name, X)
                threshold = self.threshold_spin.value()
                results = pd.DataFrame({
                    'window_date': window_dates,
                    'anomaly_score': scores,
                    'is_anomaly': (scores >= threshold).astype(int)
                })

            # Add risk levels
            results['risk_level'] = results['anomaly_score'].apply(
                lambda s: self.scorer.get_risk_level(s)
            )

            self.anomaly_table.set_dataframe(results, risk_column='risk_level')
            self.anomalies_df = results

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _load_new_transactions(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open New Transactions CSV", "",
            "CSV Files (*.csv)"
        )
        if filepath:
            try:
                self.new_df = pd.read_csv(filepath)
                self.new_table.set_dataframe(self.new_df)
                self.score_btn.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def _score_new(self) -> None:
        QMessageBox.information(
            self, "Info",
            "New transaction scoring requires feature engineering on new data. "
            "This feature aggregates the new transactions and scores them."
        )


class ReportingTab(QWidget):
    """Tab 5: Reporting - Generate Excel reports."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = setup_logging(__name__)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Report sections
        sections_group = QGroupBox("Report Sections")
        sections_layout = QVBoxLayout(sections_group)

        self.section_checks = {}
        sections = [
            ("summary", "Executive Summary"),
            ("anomalies", "Historical Anomalies"),
            ("features", "Feature Engineering Details"),
            ("models", "Model Performance"),
            ("new_scores", "New Transaction Scoring"),
            ("explanations", "Anomaly Explanations"),
            ("specs", "Technical Specifications"),
            ("appendix", "Appendix & Methodology")
        ]

        for key, label in sections:
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.section_checks[key] = cb
            sections_layout.addWidget(cb)

        layout.addWidget(sections_group)

        # Report info
        info_group = QGroupBox("Report Information")
        info_layout = QGridLayout(info_group)

        info_layout.addWidget(QLabel("Report Title:"), 0, 0)
        self.title_edit = QLineEdit("Insider Trading Anomaly Analysis Report")
        info_layout.addWidget(self.title_edit, 0, 1)

        info_layout.addWidget(QLabel("Description:"), 1, 0)
        self.desc_edit = QLineEdit()
        info_layout.addWidget(self.desc_edit, 1, 1)

        layout.addWidget(info_group)

        # Generate
        generate_group = QGroupBox("Generate Report")
        generate_layout = QVBoxLayout(generate_group)

        self.generate_btn = QPushButton("Generate Excel Report")
        self.generate_btn.clicked.connect(self._generate_report)
        generate_layout.addWidget(self.generate_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        generate_layout.addWidget(self.progress)

        self.output_label = QLabel("")
        generate_layout.addWidget(self.output_label)

        self.open_btn = QPushButton("Open Report")
        self.open_btn.clicked.connect(self._open_report)
        self.open_btn.setEnabled(False)
        generate_layout.addWidget(self.open_btn)

        layout.addWidget(generate_group)

        layout.addStretch()

    def set_data(
        self,
        preprocessor,
        feature_engineer,
        feature_matrix,
        model_factory,
        training_metrics,
        anomalies_df
    ) -> None:
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.feature_matrix = feature_matrix
        self.model_factory = model_factory
        self.training_metrics = training_metrics
        self.anomalies_df = anomalies_df

    def _generate_report(self) -> None:
        if not hasattr(self, 'feature_matrix') or self.feature_matrix is None:
            QMessageBox.warning(self, "Warning", "No data available for report.")
            return

        try:
            self.progress.setVisible(True)
            self.progress.setValue(0)

            filepath = constants.REPORTS_DIR / f"{get_timestamp()}_report.xlsx"

            reporter = ExcelReporter(
                filepath,
                self.title_edit.text(),
                self.desc_edit.text()
            )

            self.progress.setValue(20)

            # Prepare summary data
            summary_data = {
                'total_transactions': len(self.preprocessor.df) if hasattr(self, 'preprocessor') else 0,
                'n_windows': len(self.feature_matrix),
                'n_features': self.feature_matrix.shape[1],
                'n_models': len(self.training_metrics) if hasattr(self, 'training_metrics') else 0,
                'n_anomalies': len(self.anomalies_df[self.anomalies_df['is_anomaly'] == 1]) if hasattr(self, 'anomalies_df') else 0,
                'window_days': 14,
                'date_range': self.preprocessor.get_summary_stats().get('date_range', {}) if hasattr(self, 'preprocessor') else {}
            }

            feature_stats = self.feature_engineer.get_feature_statistics() if hasattr(self, 'feature_engineer') else {}

            self.progress.setValue(50)

            output_path = reporter.generate_full_report(
                summary_data=summary_data,
                anomalies_df=getattr(self, 'anomalies_df', pd.DataFrame()),
                feature_stats=feature_stats,
                training_metrics=getattr(self, 'training_metrics', {}),
                new_transactions_df=None,
                explanations=[],
                specs_data={'n_windows': len(self.feature_matrix)}
            )

            self.progress.setValue(100)
            self.output_path = output_path
            self.output_label.setText(f"Report saved: {output_path}")
            self.open_btn.setEnabled(True)

            QMessageBox.information(self, "Success", f"Report generated: {output_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.logger.error(f"Report generation error: {e}")

        finally:
            self.progress.setVisible(False)

    def _open_report(self) -> None:
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
    """Tab for fetching data from Alpha Vantage API."""

    dataFetched = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.logger = setup_logging(__name__)
        self.client: Optional[AlphaVantageClient] = None
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
        self.search_results.setMaximumHeight(150)
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
        self.cache_table.setMaximumHeight(150)
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
        fetch_layout.addWidget(self.log_widget)

        layout.addWidget(fetch_group)

        # Preview Section
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = DataPreviewTable()
        preview_layout.addWidget(self.preview_table)

        export_btn = QPushButton("Export to CSV for Analysis")
        export_btn.clicked.connect(self._export_for_analysis)
        preview_layout.addWidget(export_btn)

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

            self.progress.setValue(80)

            if len(df) > 0:
                self.log_widget.log(
                    f"Fetched {len(df)} rows from {df.index.min()} to {df.index.max()}",
                    "INFO"
                )
                self.preview_table.set_dataframe(df.head(50))
                self._update_cache_info(symbol)
                self.fetched_df = df
                self.dataFetched.emit(df)
            else:
                self.log_widget.log("No data returned", "WARNING")

            self.progress.setValue(100)

        except Exception as e:
            self.log_widget.log(f"Fetch failed: {e}", "ERROR")
            QMessageBox.critical(self, "Error", f"Failed to fetch data: {e}")

        finally:
            self.fetch_btn.setEnabled(True)
            self.progress.setVisible(False)

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
                self.preview_table.set_dataframe(df.head(50))
                self.fetched_df = df
                self.dataFetched.emit(df)
            else:
                self.log_widget.log("No cached data found", "WARNING")

        except Exception as e:
            self.log_widget.log(f"Failed to load cached data: {e}", "ERROR")

    def _export_for_analysis(self) -> None:
        """Export fetched data to CSV for use in analysis."""
        if not hasattr(self, 'fetched_df') or self.fetched_df is None:
            QMessageBox.warning(self, "Warning", "No data to export. Fetch data first.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "stock_data.csv",
            "CSV Files (*.csv)"
        )

        if filepath:
            self.fetched_df.to_csv(filepath)
            self.log_widget.log(f"Exported to {filepath}", "INFO")
            QMessageBox.information(self, "Success", f"Data exported to {filepath}")


class MainWindow(QMainWindow):
    """Main application window with 6 tabs."""

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

        # Create tabs
        self.data_tab = DataManagementTab()
        self.feature_tab = FeatureEngineeringTab()
        self.model_tab = ModelTrainingTab()
        self.anomaly_tab = AnomalyDetectionTab()
        self.report_tab = ReportingTab()
        self.fetch_tab = DataFetcherTab()

        self.tabs.addTab(self.data_tab, "1. Data Management")
        self.tabs.addTab(self.feature_tab, "2. Feature Engineering")
        self.tabs.addTab(self.model_tab, "3. Model Training")
        self.tabs.addTab(self.anomaly_tab, "4. Anomaly Detection")
        self.tabs.addTab(self.report_tab, "5. Reporting")
        self.tabs.addTab(self.fetch_tab, "6. Data Fetcher (Alpha Vantage)")

        layout.addWidget(self.tabs)

        # Status bar
        self.status_bar = StatusBar()
        layout.addWidget(self.status_bar)

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        load_action = QAction("Load Data", self)
        load_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
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
        # Data loaded -> enable feature engineering
        self.data_tab.dataLoaded.connect(self.feature_tab.set_data)
        self.data_tab.dataLoaded.connect(
            lambda _: self.status_bar.set_status("Data loaded successfully")
        )

        # Features ready -> enable model training
        self.feature_tab.featuresReady.connect(self.model_tab.set_feature_matrix)
        self.feature_tab.featuresReady.connect(self.anomaly_tab.set_feature_matrix)
        self.feature_tab.featuresReady.connect(
            lambda _, __: self.status_bar.set_status("Features computed")
        )

        # Models ready -> enable anomaly detection
        self.model_tab.modelsReady.connect(self.anomaly_tab.set_models)
        self.model_tab.modelsReady.connect(
            lambda _: self.status_bar.set_status("Models trained")
        )

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            "Insider Trading Anomaly Detection System\n\n"
            "Version 1.0.0\n\n"
            "A production-ready Python system for detecting "
            "suspicious insider trading patterns using machine learning."
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
