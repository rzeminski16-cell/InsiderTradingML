"""
GUI Components for the Insider Trading Anomaly Detection System.

This module provides reusable PyQt5 widgets and dialogs used throughout
the application GUI.

Classes:
    DataPreviewTable: Table widget for displaying DataFrames
    FeatureTreeWidget: Tree widget for feature selection
    ModelConfigDialog: Dialog for configuring model hyperparameters
    ProgressDialog: Dialog for showing progress during long operations
    ChartWidget: Widget for displaying matplotlib charts
    LogWidget: Widget for displaying log messages
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QFont, QPalette, QIcon
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QTreeWidget, QTreeWidgetItem,
    QDialog, QDialogButtonBox,
    QProgressBar, QProgressDialog,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QFrame, QSplitter,
    QFileDialog, QMessageBox,
    QTabWidget, QScrollArea,
    QSizePolicy, QSlider
)

from . import constants
from .utils import setup_logging

# Try to import matplotlib for charts
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class DataPreviewTable(QTableWidget):
    """
    Table widget for displaying pandas DataFrames.

    Provides sortable columns, color-coded cells, and selection handling.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the data preview table."""
        super().__init__(parent)
        self.logger = setup_logging(__name__)

        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSortingEnabled(True)

        # Styling
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.verticalHeader().setVisible(False)

        self._df: Optional[pd.DataFrame] = None

    def set_dataframe(
        self,
        df: pd.DataFrame,
        max_rows: int = 100,
        risk_column: Optional[str] = None
    ) -> None:
        """
        Set the DataFrame to display.

        Args:
            df: DataFrame to display
            max_rows: Maximum rows to show
            risk_column: Column name for risk-based coloring
        """
        self._df = df.head(max_rows).copy()

        self.clear()
        self.setRowCount(len(self._df))
        self.setColumnCount(len(self._df.columns))
        self.setHorizontalHeaderLabels(list(self._df.columns))

        for row_idx, (_, row) in enumerate(self._df.iterrows()):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                # Apply risk coloring
                if risk_column and self._df.columns[col_idx] == risk_column:
                    color = self._get_risk_color(str(value))
                    if color:
                        item.setBackground(color)

                self.setItem(row_idx, col_idx, item)

        # Resize columns to content
        self.resizeColumnsToContents()

    def _get_risk_color(self, risk_level: str) -> Optional[QColor]:
        """Get color for risk level."""
        colors = {
            'CRITICAL': QColor(220, 53, 69),
            'HIGH': QColor(255, 69, 0),
            'MEDIUM': QColor(255, 193, 7),
            'LOW': QColor(255, 215, 0),
            'NORMAL': QColor(40, 167, 69)
        }
        return colors.get(risk_level.upper() if risk_level else '')

    def get_selected_rows(self) -> List[int]:
        """Get indices of selected rows."""
        return list(set(item.row() for item in self.selectedItems()))

    def export_to_csv(self, filepath: str) -> None:
        """Export displayed data to CSV."""
        if self._df is not None:
            self._df.to_csv(filepath, index=False)


class FeatureTreeWidget(QTreeWidget):
    """
    Tree widget for hierarchical feature selection.

    Displays features organized by category with checkboxes.
    """

    selectionChanged = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the feature tree widget."""
        super().__init__(parent)
        self.setHeaderLabels(["Features"])
        self.setHeaderHidden(False)

        self._feature_items: Dict[str, QTreeWidgetItem] = {}
        self._setup_features()

        self.itemChanged.connect(self._on_item_changed)

    def _setup_features(self) -> None:
        """Set up the feature tree structure."""
        for category, features in constants.FEATURE_CATEGORIES.items():
            category_item = QTreeWidgetItem([category])
            category_item.setFlags(
                category_item.flags() | Qt.ItemIsTristate | Qt.ItemIsUserCheckable
            )
            category_item.setCheckState(0, Qt.Checked)

            for feature in features:
                feature_item = QTreeWidgetItem([feature])
                feature_item.setFlags(
                    feature_item.flags() | Qt.ItemIsUserCheckable
                )
                feature_item.setCheckState(0, Qt.Checked)
                category_item.addChild(feature_item)
                self._feature_items[feature] = feature_item

            self.addTopLevelItem(category_item)
            category_item.setExpanded(True)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle item check state changes."""
        self.selectionChanged.emit(self.get_selected_features())

    def get_selected_features(self) -> List[str]:
        """Get list of selected feature names."""
        selected = []
        for feature, item in self._feature_items.items():
            if item.checkState(0) == Qt.Checked:
                selected.append(feature)
        return selected

    def select_all(self) -> None:
        """Select all features."""
        for item in self._feature_items.values():
            item.setCheckState(0, Qt.Checked)

    def deselect_all(self) -> None:
        """Deselect all features."""
        for item in self._feature_items.values():
            item.setCheckState(0, Qt.Unchecked)


class ModelConfigDialog(QDialog):
    """
    Dialog for configuring model hyperparameters.

    Provides input widgets for each hyperparameter based on its type.
    """

    def __init__(
        self,
        model_type: str,
        current_params: Dict,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the model configuration dialog.

        Args:
            model_type: Type of model to configure
            current_params: Current hyperparameter values
            parent: Parent widget
        """
        super().__init__(parent)
        self.model_type = model_type
        self.current_params = current_params.copy()
        self.param_widgets: Dict[str, QWidget] = {}

        self.setWindowTitle(f"Configure {model_type}")
        self.setMinimumWidth(400)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel(f"Configure {self.model_type} Hyperparameters")
        title.setFont(QFont('Arial', 12, QFont.Bold))
        layout.addWidget(title)

        # Parameter widgets
        params_group = QGroupBox("Hyperparameters")
        params_layout = QGridLayout(params_group)

        param_ranges = constants.HYPERPARAMETER_RANGES.get(self.model_type, {})
        row = 0

        for param_name, param_range in param_ranges.items():
            label = QLabel(param_name.replace('_', ' ').title())
            params_layout.addWidget(label, row, 0)

            current_value = self.current_params.get(param_name)
            widget = self._create_param_widget(param_name, param_range, current_value)
            params_layout.addWidget(widget, row, 1)
            self.param_widgets[param_name] = widget

            row += 1

        layout.addWidget(params_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        layout.addWidget(button_box)

    def _create_param_widget(
        self,
        param_name: str,
        param_range: Dict,
        current_value: Any
    ) -> QWidget:
        """Create appropriate widget for parameter type."""
        param_type = param_range.get('type', 'float')

        if param_type == 'choice':
            widget = QComboBox()
            options = param_range.get('options', [])
            widget.addItems([str(opt) for opt in options])
            if current_value is not None:
                idx = widget.findText(str(current_value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            return widget

        elif param_type == 'int':
            widget = QSpinBox()
            widget.setMinimum(param_range.get('min', 1))
            widget.setMaximum(param_range.get('max', 1000))
            widget.setSingleStep(param_range.get('step', 1))
            if current_value is not None:
                widget.setValue(int(current_value))
            return widget

        else:  # float
            widget = QDoubleSpinBox()
            widget.setMinimum(param_range.get('min', 0.0))
            widget.setMaximum(param_range.get('max', 1.0))
            widget.setSingleStep(param_range.get('step', 0.01))
            widget.setDecimals(3)
            if current_value is not None:
                widget.setValue(float(current_value))
            return widget

    def _restore_defaults(self) -> None:
        """Restore default hyperparameter values."""
        defaults = constants.DEFAULT_HYPERPARAMETERS.get(self.model_type, {})
        for param_name, widget in self.param_widgets.items():
            default_value = defaults.get(param_name)
            if default_value is None:
                continue

            if isinstance(widget, QComboBox):
                idx = widget.findText(str(default_value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(default_value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(default_value))

    def get_parameters(self) -> Dict:
        """Get configured parameter values."""
        params = {}
        param_ranges = constants.HYPERPARAMETER_RANGES.get(self.model_type, {})

        for param_name, widget in self.param_widgets.items():
            param_type = param_ranges.get(param_name, {}).get('type', 'float')

            if isinstance(widget, QComboBox):
                value = widget.currentText()
                # Try to convert to appropriate type
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
            elif isinstance(widget, QSpinBox):
                value = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                value = widget.value()
            else:
                value = None

            params[param_name] = value

        return params


class ProgressDialog(QProgressDialog):
    """
    Progress dialog for long-running operations.

    Shows progress bar and allows cancellation.
    """

    def __init__(
        self,
        title: str = "Processing...",
        message: str = "Please wait...",
        parent: Optional[QWidget] = None
    ):
        """Initialize the progress dialog."""
        super().__init__(message, "Cancel", 0, 100, parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setAutoClose(True)
        self.setAutoReset(True)
        self.setMinimumDuration(500)

    def update_progress(self, value: int, message: Optional[str] = None) -> None:
        """Update progress value and message."""
        self.setValue(value)
        if message:
            self.setLabelText(message)


class LogWidget(QTextEdit):
    """
    Widget for displaying log messages.

    Provides color-coded log levels and auto-scrolling.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the log widget."""
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont('Consolas', 9))
        self.setMaximumHeight(150)

    def log(self, message: str, level: str = 'INFO') -> None:
        """
        Add a log message.

        Args:
            message: Message to log
            level: Log level (DEBUG, INFO, WARNING, ERROR)
        """
        colors = {
            'DEBUG': 'gray',
            'INFO': 'black',
            'WARNING': 'orange',
            'ERROR': 'red'
        }
        color = colors.get(level.upper(), 'black')
        timestamp = pd.Timestamp.now().strftime('%H:%M:%S')

        html = f'<span style="color:{color}">[{timestamp}] {level}: {message}</span>'
        self.append(html)

        # Auto-scroll to bottom
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_log(self) -> None:
        """Clear all log messages."""
        self.clear()


class ChartWidget(QWidget):
    """
    Widget for displaying matplotlib charts.

    Wraps matplotlib figure in a Qt widget.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the chart widget."""
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(6, 4), dpi=100)
            self.canvas = FigureCanvas(self.figure)
            self.layout.addWidget(self.canvas)
        else:
            self.figure = None
            self.canvas = None
            label = QLabel("Matplotlib not available for charts")
            self.layout.addWidget(label)

    def plot_line(
        self,
        x: np.ndarray,
        y: np.ndarray,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        highlight_indices: Optional[List[int]] = None
    ) -> None:
        """
        Create a line plot.

        Args:
            x: X values
            y: Y values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            highlight_indices: Indices to highlight as anomalies
        """
        if self.figure is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        ax.plot(x, y, 'b-', linewidth=1.5)

        if highlight_indices:
            ax.scatter(
                x[highlight_indices],
                y[highlight_indices],
                color='red',
                s=50,
                zorder=5,
                label='Anomalies'
            )
            ax.legend()

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_bar(
        self,
        labels: List[str],
        values: List[float],
        title: str = "",
        xlabel: str = "",
        ylabel: str = "",
        color: str = 'steelblue'
    ) -> None:
        """Create a bar chart."""
        if self.figure is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        x = np.arange(len(labels))
        ax.bar(x, values, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_heatmap(
        self,
        data: np.ndarray,
        labels: List[str],
        title: str = ""
    ) -> None:
        """Create a heatmap."""
        if self.figure is None:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')
        self.figure.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)

        ax.set_title(title)

        self.figure.tight_layout()
        self.canvas.draw()

    def clear_chart(self) -> None:
        """Clear the chart."""
        if self.figure:
            self.figure.clear()
            self.canvas.draw()


class SummaryCard(QFrame):
    """
    Card widget for displaying summary statistics.

    Styled card with title, value, and optional icon.
    """

    def __init__(
        self,
        title: str,
        value: str = "0",
        color: str = "#007BFF",
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the summary card.

        Args:
            title: Card title
            value: Display value
            color: Accent color
            parent: Parent widget
        """
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setStyleSheet(f"""
            SummaryCard {{
                background-color: white;
                border-radius: 8px;
                border: 1px solid #DEE2E6;
                padding: 10px;
            }}
        """)

        layout = QVBoxLayout(self)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setFont(QFont('Arial', 10))
        self.title_label.setStyleSheet("color: #6C757D;")
        layout.addWidget(self.title_label)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setFont(QFont('Arial', 24, QFont.Bold))
        self.value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(self.value_label)

        self.setMinimumWidth(150)

    def set_value(self, value: str) -> None:
        """Update the displayed value."""
        self.value_label.setText(value)


class WorkerThread(QThread):
    """
    Worker thread for running long operations.

    Emits signals for progress updates and completion.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        task: Callable,
        *args,
        **kwargs
    ):
        """
        Initialize the worker thread.

        Args:
            task: Callable to execute
            *args: Positional arguments for task
            **kwargs: Keyword arguments for task
        """
        super().__init__()
        self.task = task
        self.args = args
        self.kwargs = kwargs
        self._result = None

    def run(self) -> None:
        """Execute the task."""
        try:
            self._result = self.task(*self.args, **self.kwargs)
            self.finished.emit(self._result)
        except Exception as e:
            self.error.emit(str(e))


class FilterWidget(QWidget):
    """
    Widget for data filtering controls.

    Provides date range, executive, and security type filters.
    """

    filterChanged = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the filter widget."""
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the filter UI."""
        layout = QHBoxLayout(self)

        # Executive filter
        exec_group = QGroupBox("Executive")
        exec_layout = QVBoxLayout(exec_group)
        self.executive_combo = QComboBox()
        self.executive_combo.addItem("All")
        exec_layout.addWidget(self.executive_combo)
        layout.addWidget(exec_group)

        # Date range
        date_group = QGroupBox("Date Range")
        date_layout = QHBoxLayout(date_group)
        self.start_date = QLineEdit()
        self.start_date.setPlaceholderText("Start (YYYY-MM-DD)")
        self.end_date = QLineEdit()
        self.end_date.setPlaceholderText("End (YYYY-MM-DD)")
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(QLabel("to"))
        date_layout.addWidget(self.end_date)
        layout.addWidget(date_group)

        # Buttons
        button_layout = QVBoxLayout()
        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.clicked.connect(self._on_apply)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._on_clear)
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.clear_btn)
        layout.addLayout(button_layout)

    def set_executives(self, executives: List[str]) -> None:
        """Set available executives for filtering."""
        self.executive_combo.clear()
        self.executive_combo.addItem("All")
        self.executive_combo.addItems(executives)

    def _on_apply(self) -> None:
        """Apply current filters."""
        filters = {
            'executive': self.executive_combo.currentText(),
            'start_date': self.start_date.text(),
            'end_date': self.end_date.text()
        }
        self.filterChanged.emit(filters)

    def _on_clear(self) -> None:
        """Clear all filters."""
        self.executive_combo.setCurrentIndex(0)
        self.start_date.clear()
        self.end_date.clear()
        self.filterChanged.emit({})


class StatusBar(QWidget):
    """
    Custom status bar widget.

    Shows status message, progress, and data statistics.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the status bar."""
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        layout.addStretch()

        self.data_label = QLabel("")
        layout.addWidget(self.data_label)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

    def set_status(self, message: str) -> None:
        """Set status message."""
        self.status_label.setText(message)

    def set_data_info(self, info: str) -> None:
        """Set data info message."""
        self.data_label.setText(info)

    def show_progress(self, value: int = 0, maximum: int = 100) -> None:
        """Show and update progress bar."""
        self.progress.setVisible(True)
        self.progress.setMaximum(maximum)
        self.progress.setValue(value)

    def hide_progress(self) -> None:
        """Hide progress bar."""
        self.progress.setVisible(False)
