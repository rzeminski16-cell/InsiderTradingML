"""
Model Factory for the Insider Trading Anomaly Detection System.

This module provides the ModelFactory class and individual model implementations
for 6 different anomaly detection algorithms:
- Isolation Forest
- DBSCAN
- Local Outlier Factor (LOF)
- Neural Network Autoencoder (sklearn-based, no TensorFlow required)
- K-Means
- One-Class SVM

All models inherit from BaseAnomalyModel and provide a unified interface.

Classes:
    BaseAnomalyModel: Abstract base class for all anomaly models
    IsolationForestModel: Isolation Forest implementation
    DBSCANModel: DBSCAN clustering implementation
    LocalOutlierFactorModel: LOF implementation
    NNAutoencoderModel: Neural Network Autoencoder (sklearn MLPRegressor)
    KMeansModel: K-Means clustering implementation
    OneClassSVMModel: One-Class SVM implementation
    ModelFactory: Factory class for creating and managing models
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from . import constants
from .utils import (
    setup_logging,
    normalize_scores,
    save_pickle,
    load_pickle,
    get_timestamp
)


class BaseAnomalyModel(ABC):
    """
    Abstract base class for all anomaly detection models.

    Provides a unified interface for training, prediction, and scoring.
    All model implementations must inherit from this class.

    Attributes:
        model_type: Type of the model
        hyperparameters: Model hyperparameters
        is_fitted: Whether the model has been trained
        training_time: Time taken to train the model
        feature_names: Names of features used for training
    """

    def __init__(self, model_type: str, hyperparameters: Dict):
        """
        Initialize the base model.

        Args:
            model_type: Type of the model
            hyperparameters: Model hyperparameters
        """
        self.logger = setup_logging(__name__)
        self.model_type = model_type
        self.hyperparameters = hyperparameters.copy()
        self.is_fitted = False
        self.training_time: float = 0.0
        self.feature_names: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self._model = None
        self._training_timestamp: Optional[datetime] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseAnomalyModel':
        """
        Train the model on feature matrix.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Optional labels (not used for unsupervised models)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.

        Args:
            X: Data to predict on, shape (n_samples, n_features)

        Returns:
            Array of labels: 1 for anomaly, 0 for normal
        """
        pass

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for samples.

        Args:
            X: Data to score, shape (n_samples, n_features)

        Returns:
            Array of anomaly scores in [0, 1], where 1 = most anomalous
        """
        pass

    def get_hyperparameters(self) -> Dict:
        """Get model hyperparameters."""
        return self.hyperparameters.copy()

    def get_metadata(self) -> Dict:
        """Get model metadata."""
        return {
            'model_type': self.model_type,
            'hyperparameters': self.hyperparameters,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'training_timestamp': self._training_timestamp,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names)
        }


class IsolationForestModel(BaseAnomalyModel):
    """
    Isolation Forest anomaly detection model.

    Isolation Forest isolates anomalies by randomly selecting a feature
    and then randomly selecting a split value. Anomalies require fewer
    splits to be isolated.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        max_samples: Union[str, int] = 'auto',
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest model.

        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of isolation trees
            max_samples: Number of samples for each tree
            random_state: Random seed for reproducibility
        """
        hyperparameters = {
            'contamination': contamination,
            'n_estimators': n_estimators,
            'max_samples': max_samples,
            'random_state': random_state
        }
        super().__init__('IsolationForest', hyperparameters)

        self._model = IsolationForest(**hyperparameters)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'IsolationForestModel':
        """Train the Isolation Forest model."""
        start_time = time.time()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self._model.fit(X_scaled)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        self._training_timestamp = datetime.now()

        self.logger.info(
            f"IsolationForest trained in {self.training_time:.2f}s"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1 = anomaly, 0 = normal)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        predictions = self._model.predict(X_scaled)
        # Convert from sklearn format (-1 = anomaly, 1 = normal)
        return (predictions == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (higher = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        # score_samples returns negative values (more negative = more anomalous)
        raw_scores = -self._model.score_samples(X_scaled)
        # Normalize to [0, 1]
        return normalize_scores(raw_scores, method='minmax')


class DBSCANModel(BaseAnomalyModel):
    """
    DBSCAN clustering-based anomaly detection model.

    DBSCAN groups points based on density. Points that cannot be assigned
    to any cluster are labeled as noise (anomalies).
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean'
    ):
        """
        Initialize DBSCAN model.

        Args:
            eps: Maximum distance for points to be considered neighbors
            min_samples: Minimum points to form a dense region
            metric: Distance metric
        """
        hyperparameters = {
            'eps': eps,
            'min_samples': min_samples,
            'metric': metric
        }
        super().__init__('DBSCAN', hyperparameters)

        self._model = DBSCAN(**hyperparameters)
        self._cluster_labels: Optional[np.ndarray] = None
        self._core_samples: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DBSCANModel':
        """Train the DBSCAN model."""
        start_time = time.time()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self._cluster_labels = self._model.fit_predict(X_scaled)
        self._core_samples = X_scaled[self._cluster_labels != -1]
        self._X_train = X_scaled

        self.training_time = time.time() - start_time
        self.is_fitted = True
        self._training_timestamp = datetime.now()

        n_anomalies = (self._cluster_labels == -1).sum()
        self.logger.info(
            f"DBSCAN trained in {self.training_time:.2f}s, "
            f"found {n_anomalies} noise points"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1 = noise/anomaly, 0 = in cluster)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        # For new data, calculate distances to core samples
        predictions = []
        for x in X_scaled:
            if len(self._core_samples) == 0:
                predictions.append(1)
            else:
                distances = np.linalg.norm(self._core_samples - x, axis=1)
                min_dist = distances.min()
                is_anomaly = min_dist > self.hyperparameters['eps']
                predictions.append(int(is_anomaly))

        return np.array(predictions)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores based on distance to nearest core sample."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        scores = []
        for x in X_scaled:
            if len(self._core_samples) == 0:
                scores.append(1.0)
            else:
                distances = np.linalg.norm(self._core_samples - x, axis=1)
                min_dist = distances.min()
                scores.append(min_dist)

        return normalize_scores(np.array(scores), method='minmax')


class LocalOutlierFactorModel(BaseAnomalyModel):
    """
    Local Outlier Factor (LOF) anomaly detection model.

    LOF measures local deviation of density compared to neighbors.
    Points with substantially lower density are considered outliers.
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        novelty: bool = True
    ):
        """
        Initialize LOF model.

        Args:
            n_neighbors: Number of neighbors for density estimation
            contamination: Expected proportion of outliers
            novelty: If True, can predict on new data
        """
        hyperparameters = {
            'n_neighbors': n_neighbors,
            'contamination': contamination,
            'novelty': novelty
        }
        super().__init__('LocalOutlierFactor', hyperparameters)

        self._model = LocalOutlierFactor(**hyperparameters)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'LocalOutlierFactorModel':
        """Train the LOF model."""
        start_time = time.time()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self._model.fit(X_scaled)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        self._training_timestamp = datetime.now()

        self.logger.info(f"LOF trained in {self.training_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1 = anomaly, 0 = normal)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        predictions = self._model.predict(X_scaled)
        # Convert from sklearn format (-1 = anomaly, 1 = normal)
        return (predictions == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (higher = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        # score_samples returns negative LOF scores
        raw_scores = -self._model.score_samples(X_scaled)
        return normalize_scores(raw_scores, method='minmax')


class NNAutoencoderModel(BaseAnomalyModel):
    """
    Neural Network Autoencoder anomaly detection model.

    Uses sklearn's MLPRegressor as an autoencoder. Trains to reconstruct
    input data; anomalies are detected based on high reconstruction error.

    This implementation does NOT require TensorFlow and works with Python 3.13+.
    """

    def __init__(
        self,
        encoding_dim: int = 8,
        hidden_layers: Tuple[int, ...] = (32, 16, 8, 16, 32),
        max_iter: int = 500,
        learning_rate_init: float = 0.001,
        random_state: int = 42
    ):
        """
        Initialize Neural Network Autoencoder model.

        Args:
            encoding_dim: Dimension of the bottleneck layer (for reference)
            hidden_layers: Tuple of hidden layer sizes (encoder + decoder)
            max_iter: Maximum training iterations
            learning_rate_init: Initial learning rate
            random_state: Random seed for reproducibility
        """
        hyperparameters = {
            'encoding_dim': encoding_dim,
            'hidden_layers': hidden_layers,
            'max_iter': max_iter,
            'learning_rate_init': learning_rate_init,
            'random_state': random_state
        }
        super().__init__('NNAutoencoder', hyperparameters)

        self._threshold: Optional[float] = None
        self._train_errors: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'NNAutoencoderModel':
        """Train the Neural Network Autoencoder."""
        start_time = time.time()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Build autoencoder architecture
        # Input -> hidden_layers -> Output (same as input)
        self._model = MLPRegressor(
            hidden_layer_sizes=self.hyperparameters['hidden_layers'],
            activation='relu',
            solver='adam',
            learning_rate_init=self.hyperparameters['learning_rate_init'],
            max_iter=self.hyperparameters['max_iter'],
            random_state=self.hyperparameters['random_state'],
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )

        # Train autoencoder: input = output (reconstruction)
        self._model.fit(X_scaled, X_scaled)

        # Calculate reconstruction errors on training data
        reconstructed = self._model.predict(X_scaled)
        self._train_errors = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

        # Set threshold at 95th percentile of training errors
        self._threshold = np.percentile(self._train_errors, 95)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        self._training_timestamp = datetime.now()

        self.logger.info(
            f"NNAutoencoder trained in {self.training_time:.2f}s"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels based on reconstruction error."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        scores = self.score_samples(X)
        # Use normalized threshold at 0.5
        return (scores > 0.5).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores based on reconstruction error."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        reconstructed = self._model.predict(X_scaled)

        # Calculate mean squared error for each sample
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

        return normalize_scores(mse, method='minmax')


class KMeansModel(BaseAnomalyModel):
    """
    K-Means clustering-based anomaly detection model.

    Points far from cluster centroids are considered anomalies.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        init: str = 'k-means++',
        random_state: int = 42
    ):
        """
        Initialize K-Means model.

        Args:
            n_clusters: Number of clusters
            init: Initialization method
            random_state: Random seed
        """
        hyperparameters = {
            'n_clusters': n_clusters,
            'init': init,
            'random_state': random_state
        }
        super().__init__('KMeans', hyperparameters)

        self._model = KMeans(**hyperparameters)
        self._threshold: Optional[float] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'KMeansModel':
        """Train the K-Means model."""
        start_time = time.time()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self._model.fit(X_scaled)

        # Calculate distances and set threshold
        distances = self._get_distances(X_scaled)
        self._threshold = np.percentile(distances, 95)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        self._training_timestamp = datetime.now()

        self.logger.info(f"KMeans trained in {self.training_time:.2f}s")
        return self

    def _get_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate distance from each point to nearest centroid."""
        # Get cluster assignments
        labels = self._model.predict(X)
        centroids = self._model.cluster_centers_

        # Calculate distance to assigned centroid
        distances = np.array([
            np.linalg.norm(x - centroids[label])
            for x, label in zip(X, labels)
        ])

        return distances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1 if far from centroid)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        distances = self._get_distances(X_scaled)

        return (distances > self._threshold).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores based on distance to centroid."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        distances = self._get_distances(X_scaled)

        return normalize_scores(distances, method='minmax')


class OneClassSVMModel(BaseAnomalyModel):
    """
    One-Class SVM anomaly detection model.

    Learns a decision boundary around normal data. Points outside
    the boundary are considered anomalies.
    """

    def __init__(
        self,
        kernel: str = 'rbf',
        nu: float = 0.05,
        gamma: str = 'scale'
    ):
        """
        Initialize One-Class SVM model.

        Args:
            kernel: Kernel type
            nu: Upper bound on fraction of training errors
            gamma: Kernel coefficient
        """
        hyperparameters = {
            'kernel': kernel,
            'nu': nu,
            'gamma': gamma
        }
        super().__init__('OneClassSVM', hyperparameters)

        self._model = OneClassSVM(**hyperparameters)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OneClassSVMModel':
        """Train the One-Class SVM model."""
        start_time = time.time()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self._model.fit(X_scaled)

        self.training_time = time.time() - start_time
        self.is_fitted = True
        self._training_timestamp = datetime.now()

        self.logger.info(f"OneClassSVM trained in {self.training_time:.2f}s")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (1 = anomaly, 0 = normal)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        predictions = self._model.predict(X_scaled)
        # Convert from sklearn format (-1 = anomaly, 1 = normal)
        return (predictions == -1).astype(int)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (higher = more anomalous)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        # decision_function returns signed distance to boundary
        raw_scores = -self._model.decision_function(X_scaled)
        return normalize_scores(raw_scores, method='minmax')


# Backwards compatibility alias
LSTMAutoencoderModel = NNAutoencoderModel


class ModelFactory:
    """
    Factory class for creating and managing anomaly detection models.

    Provides methods for:
    - Creating models by type
    - Training multiple models
    - Ensemble predictions
    - Model persistence

    Attributes:
        models: Dictionary of trained models
        feature_names: Names of features used for training
    """

    MODEL_CLASSES = {
        'IsolationForest': IsolationForestModel,
        'DBSCAN': DBSCANModel,
        'LocalOutlierFactor': LocalOutlierFactorModel,
        'NNAutoencoder': NNAutoencoderModel,
        'LSTMAutoencoder': NNAutoencoderModel,  # Alias for backwards compatibility
        'KMeans': KMeansModel,
        'OneClassSVM': OneClassSVMModel
    }

    def __init__(self):
        """Initialize the ModelFactory."""
        self.logger = setup_logging(__name__)
        self.models: Dict[str, BaseAnomalyModel] = {}
        self.feature_names: List[str] = []
        self.scaler: Optional[StandardScaler] = None

    def create_model(
        self,
        model_type: str,
        hyperparameters: Optional[Dict] = None,
        model_name: Optional[str] = None
    ) -> BaseAnomalyModel:
        """
        Create a model instance.

        Args:
            model_type: Type of model to create
            hyperparameters: Custom hyperparameters (uses defaults if None)
            model_name: Custom name for the model

        Returns:
            Model instance
        """
        if model_type not in self.MODEL_CLASSES:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self.MODEL_CLASSES.keys())}"
            )

        # Get default hyperparameters and update with custom ones
        # Map LSTMAutoencoder to NNAutoencoder defaults
        param_key = 'NNAutoencoder' if model_type == 'LSTMAutoencoder' else model_type
        default_params = constants.DEFAULT_HYPERPARAMETERS.get(param_key, {})
        params = {**default_params, **(hyperparameters or {})}

        # Filter out params not applicable to NNAutoencoder
        if model_type in ['LSTMAutoencoder', 'NNAutoencoder']:
            valid_params = ['encoding_dim', 'hidden_layers', 'max_iter',
                          'learning_rate_init', 'random_state']
            params = {k: v for k, v in params.items() if k in valid_params}

        # Create model
        model_class = self.MODEL_CLASSES[model_type]
        model = model_class(**params)

        # Store with name
        name = model_name or f"{model_type}_{get_timestamp()}"
        self.models[name] = model

        self.logger.info(f"Created model: {name}")
        return model

    def train_model(
        self,
        model_name: str,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[BaseAnomalyModel, Dict]:
        """
        Train a specific model.

        Args:
            model_name: Name of the model to train
            X: Training data
            feature_names: Feature names

        Returns:
            Tuple of (trained model, training metrics)
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        model = self.models[model_name]
        model.feature_names = feature_names

        self.logger.info(f"Training model: {model_name}")
        model.fit(X)

        # Get predictions for metrics
        predictions = model.predict(X)
        scores = model.score_samples(X)

        metrics = {
            'model_name': model_name,
            'model_type': model.model_type,
            'hyperparameters': model.hyperparameters,
            'training_time': model.training_time,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'anomalies_detected': int(predictions.sum()),
            'anomaly_rate': float(predictions.mean()),
            'mean_score': float(scores.mean()),
            'max_score': float(scores.max()),
            'timestamp': model._training_timestamp
        }

        return model, metrics

    def train_all_models(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Dict]:
        """
        Train all registered models.

        Args:
            X: Training data
            feature_names: Feature names

        Returns:
            Dictionary of training metrics for each model
        """
        self.feature_names = feature_names
        all_metrics = {}

        for model_name in self.models:
            try:
                _, metrics = self.train_model(model_name, X, feature_names)
                all_metrics[model_name] = metrics
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                all_metrics[model_name] = {'error': str(e)}

        return all_metrics

    def predict(
        self,
        model_name: str,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Get predictions from a specific model.

        Args:
            model_name: Name of the model
            X: Data to predict on

        Returns:
            Array of predictions (1 = anomaly, 0 = normal)
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        return self.models[model_name].predict(X)

    def score(
        self,
        model_name: str,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Get anomaly scores from a specific model.

        Args:
            model_name: Name of the model
            X: Data to score

        Returns:
            Array of anomaly scores [0, 1]
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        return self.models[model_name].score_samples(X)

    def ensemble_predict(
        self,
        X: np.ndarray,
        voting_method: str = 'majority',
        model_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Combine predictions from multiple models.

        Args:
            X: Data to predict on
            voting_method: 'majority', 'consensus', or 'average_score'
            model_names: Models to include (all if None)

        Returns:
            Dictionary with ensemble predictions and individual results
        """
        model_names = model_names or list(self.models.keys())

        individual_predictions = {}
        individual_scores = {}

        for name in model_names:
            if name in self.models and self.models[name].is_fitted:
                individual_predictions[name] = self.models[name].predict(X)
                individual_scores[name] = self.models[name].score_samples(X)

        if not individual_predictions:
            raise ValueError("No fitted models available for prediction")

        # Stack predictions
        pred_array = np.vstack(list(individual_predictions.values()))
        score_array = np.vstack(list(individual_scores.values()))

        # Calculate ensemble results
        if voting_method == 'majority':
            ensemble_pred = (pred_array.mean(axis=0) >= 0.5).astype(int)
        elif voting_method == 'consensus':
            ensemble_pred = (pred_array.all(axis=0)).astype(int)
        elif voting_method == 'average_score':
            ensemble_score = score_array.mean(axis=0)
            ensemble_pred = (ensemble_score >= 0.5).astype(int)
        else:
            raise ValueError(f"Unknown voting method: {voting_method}")

        # Calculate agreement
        model_agreement = pred_array.mean(axis=0)

        return {
            'individual_predictions': individual_predictions,
            'individual_scores': individual_scores,
            'ensemble_label': ensemble_pred,
            'ensemble_score': score_array.mean(axis=0),
            'model_agreement': model_agreement,
            'n_models': len(individual_predictions)
        }

    def detect_historical_anomalies(
        self,
        model_name: str,
        X: np.ndarray,
        window_dates: pd.DatetimeIndex,
        percentile: float = 95
    ) -> pd.DataFrame:
        """
        Detect historical anomalies and return ranked results.

        Args:
            model_name: Model to use
            X: Feature matrix
            window_dates: Date index for the windows
            percentile: Percentile threshold for anomalies

        Returns:
            DataFrame with anomaly rankings
        """
        scores = self.score(model_name, X)
        threshold = np.percentile(scores, percentile)
        predictions = (scores >= threshold).astype(int)

        results = pd.DataFrame({
            'window_date': window_dates,
            'anomaly_score': scores,
            'is_anomaly': predictions,
            'percentile_rank': [
                (scores < s).sum() / len(scores) * 100
                for s in scores
            ]
        })

        # Sort by score descending
        results = results.sort_values('anomaly_score', ascending=False)
        results = results.reset_index(drop=True)

        return results

    def save_model(
        self,
        model_name: str,
        filepath: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save a model to disk.

        Args:
            model_name: Name of the model to save
            filepath: Save path (auto-generated if None)

        Returns:
            Path to saved file
        """
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        if filepath is None:
            filepath = (
                constants.SAVED_MODELS_DIR /
                f"{get_timestamp()}_{model_name}.pkl"
            )
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'model': self.models[model_name],
            'metadata': self.models[model_name].get_metadata(),
            'feature_names': self.feature_names
        }

        save_pickle(save_data, filepath)
        self.logger.info(f"Saved model to {filepath}")

        return filepath

    def load_model(
        self,
        filepath: Union[str, Path],
        model_name: Optional[str] = None
    ) -> BaseAnomalyModel:
        """
        Load a model from disk.

        Args:
            filepath: Path to model file
            model_name: Name to register model under

        Returns:
            Loaded model
        """
        save_data = load_pickle(filepath)
        model = save_data['model']

        name = model_name or Path(filepath).stem
        self.models[name] = model
        self.feature_names = save_data.get('feature_names', [])

        self.logger.info(f"Loaded model from {filepath}")
        return model

    def list_models(self) -> List[Dict]:
        """List all registered models with their metadata."""
        return [
            {
                'name': name,
                **model.get_metadata()
            }
            for name, model in self.models.items()
        ]

    def get_model(self, model_name: str) -> BaseAnomalyModel:
        """Get a model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")
        return self.models[model_name]

    def remove_model(self, model_name: str) -> None:
        """Remove a model from the factory."""
        if model_name in self.models:
            del self.models[model_name]
            self.logger.info(f"Removed model: {model_name}")
