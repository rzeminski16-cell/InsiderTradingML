"""
Anomaly Scoring and Thresholding for the Insider Trading Anomaly Detection System.

This module provides classes for:
- Normalizing anomaly scores across different models
- Adaptive threshold calculation
- Confidence scoring
- Risk level classification
- Model diagnostics and interpretability

Classes:
    AnomalyScorer: Main class for anomaly scoring operations
    ModelDiagnostics: Class for model validation and diagnostics
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from . import constants
from .utils import (
    setup_logging,
    normalize_scores,
    get_risk_level,
    calculate_zscore,
    get_percentile
)


class AnomalyScorer:
    """
    Anomaly scoring and thresholding system.

    Provides methods for normalizing scores, computing thresholds,
    calculating confidence, and determining risk levels.

    Attributes:
        historical_scores: Historical anomaly scores for baseline
        threshold: Current threshold value
        threshold_method: Method used for threshold calculation
    """

    def __init__(self):
        """Initialize the AnomalyScorer."""
        self.logger = setup_logging(__name__)
        self.historical_scores: Optional[np.ndarray] = None
        self.threshold: Optional[float] = None
        self.threshold_method: str = 'percentile'
        self.threshold_param: float = 95.0

    def set_historical_scores(self, scores: np.ndarray) -> None:
        """
        Set historical scores for baseline comparisons.

        Args:
            scores: Array of historical anomaly scores
        """
        self.historical_scores = np.asarray(scores)
        self.logger.info(f"Set {len(scores)} historical scores")

    def normalize_scores(
        self,
        raw_scores: np.ndarray,
        model_type: str = 'generic'
    ) -> np.ndarray:
        """
        Normalize model-specific scores to [0, 1] range.

        Different models produce scores in different ranges and with
        different interpretations. This method normalizes them for
        comparison.

        Args:
            raw_scores: Raw anomaly scores from model
            model_type: Type of model ('IsolationForest', 'DBSCAN', etc.)

        Returns:
            Normalized scores in [0, 1], where 1 = most anomalous
        """
        raw_scores = np.asarray(raw_scores).astype(float)

        if len(raw_scores) == 0:
            return raw_scores

        if model_type == 'IsolationForest':
            # IsolationForest: offset path length, more negative = more anomalous
            # Already normalized in model, but ensure range
            return normalize_scores(raw_scores, method='minmax')

        elif model_type == 'DBSCAN':
            # DBSCAN: distance-based score
            return normalize_scores(raw_scores, method='minmax')

        elif model_type == 'LocalOutlierFactor':
            # LOF: negative LOF scores
            return normalize_scores(raw_scores, method='minmax')

        elif model_type == 'LSTMAutoencoder':
            # LSTM: reconstruction error, use sigmoid normalization
            return normalize_scores(raw_scores, method='sigmoid')

        elif model_type == 'KMeans':
            # KMeans: distance to centroid
            return normalize_scores(raw_scores, method='minmax')

        elif model_type == 'OneClassSVM':
            # OneClassSVM: decision function, use sigmoid
            return normalize_scores(raw_scores, method='sigmoid')

        else:
            # Generic normalization
            return normalize_scores(raw_scores, method='minmax')

    def adaptive_threshold(
        self,
        scores: np.ndarray,
        method: str = 'percentile',
        param: float = 95.0
    ) -> float:
        """
        Compute anomaly threshold based on score distribution.

        Args:
            scores: Historical anomaly scores
            method: Threshold method
                   - 'percentile': Use nth percentile
                   - 'std': Use mean + n*std
                   - 'elbow': Identify knee in sorted scores
                   - 'iqr': Use IQR-based outlier detection
            param: Method-specific parameter

        Returns:
            Threshold value
        """
        scores = np.asarray(scores)

        if len(scores) == 0:
            return 0.5

        self.threshold_method = method
        self.threshold_param = param

        if method == 'percentile':
            threshold = np.percentile(scores, param)

        elif method == 'std':
            mean = np.mean(scores)
            std = np.std(scores)
            threshold = mean + param * std

        elif method == 'elbow':
            # Find elbow/knee point in sorted scores
            sorted_scores = np.sort(scores)[::-1]
            diffs = np.diff(sorted_scores)
            if len(diffs) > 0:
                # Find largest change
                elbow_idx = np.argmin(diffs)
                threshold = sorted_scores[elbow_idx]
            else:
                threshold = sorted_scores[0] * 0.9

        elif method == 'iqr':
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            threshold = q3 + param * iqr

        else:
            raise ValueError(f"Unknown threshold method: {method}")

        self.threshold = float(threshold)
        self.logger.info(
            f"Calculated threshold: {self.threshold:.4f} "
            f"(method={method}, param={param})"
        )

        return self.threshold

    def get_confidence_score(
        self,
        anomaly_score: Union[float, np.ndarray],
        threshold: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate confidence in the anomaly classification.

        Confidence measures how far the score is from the threshold.
        Higher confidence means more certainty in the classification.

        Args:
            anomaly_score: Anomaly score(s) in [0, 1]
            threshold: Decision threshold (uses stored threshold if None)

        Returns:
            Confidence score(s) in [0, 1]
        """
        if threshold is None:
            threshold = self.threshold or 0.5

        anomaly_score = np.asarray(anomaly_score)

        # Calculate distance from threshold, normalized
        is_anomaly = anomaly_score >= threshold

        confidence = np.where(
            is_anomaly,
            # For anomalies: how far above threshold (normalized to [0,1])
            (anomaly_score - threshold) / (1 - threshold + 1e-10),
            # For normal: how far below threshold (normalized to [0,1])
            (threshold - anomaly_score) / (threshold + 1e-10)
        )

        # Clip to [0, 1]
        confidence = np.clip(confidence, 0, 1)

        return confidence if len(confidence.shape) > 0 else float(confidence)

    def get_risk_level(
        self,
        anomaly_score: float,
        confidence: Optional[float] = None,
        model_agreement: Optional[float] = None
    ) -> str:
        """
        Determine risk level from score and confidence.

        Args:
            anomaly_score: Anomaly score in [0, 1]
            confidence: Confidence score in [0, 1]
            model_agreement: Fraction of models agreeing

        Returns:
            Risk level: 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', or 'NORMAL'
        """
        if confidence is None:
            confidence = self.get_confidence_score(anomaly_score)

        return get_risk_level(anomaly_score, confidence, model_agreement)

    def get_percentile_rank(
        self,
        score: float,
        distribution: Optional[np.ndarray] = None
    ) -> float:
        """
        Get percentile rank of a score in the historical distribution.

        Args:
            score: Anomaly score to rank
            distribution: Distribution to compare against

        Returns:
            Percentile rank (0-100)
        """
        if distribution is None:
            distribution = self.historical_scores

        if distribution is None or len(distribution) == 0:
            return 50.0

        return get_percentile(score, distribution)

    def classify_batch(
        self,
        scores: np.ndarray,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Classify a batch of scores with full details.

        Args:
            scores: Array of anomaly scores
            threshold: Decision threshold

        Returns:
            DataFrame with classification details
        """
        if threshold is None:
            threshold = self.threshold or 0.5

        scores = np.asarray(scores)
        predictions = (scores >= threshold).astype(int)
        confidences = self.get_confidence_score(scores, threshold)

        risk_levels = [
            self.get_risk_level(s, c)
            for s, c in zip(scores, confidences)
        ]

        percentile_ranks = [
            self.get_percentile_rank(s)
            for s in scores
        ]

        return pd.DataFrame({
            'anomaly_score': scores,
            'is_anomaly': predictions,
            'confidence': confidences,
            'risk_level': risk_levels,
            'percentile_rank': percentile_ranks
        })

    def compare_to_baseline(
        self,
        new_scores: np.ndarray,
        baseline_scores: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compare new scores to baseline distribution.

        Args:
            new_scores: New anomaly scores
            baseline_scores: Baseline scores for comparison

        Returns:
            Dictionary with comparison statistics
        """
        if baseline_scores is None:
            baseline_scores = self.historical_scores

        if baseline_scores is None or len(baseline_scores) == 0:
            return {'error': 'No baseline scores available'}

        new_scores = np.asarray(new_scores)
        baseline_scores = np.asarray(baseline_scores)

        # Basic statistics
        new_mean = np.mean(new_scores)
        new_std = np.std(new_scores)
        baseline_mean = np.mean(baseline_scores)
        baseline_std = np.std(baseline_scores)

        # Statistical tests
        t_stat, t_pvalue = stats.ttest_ind(new_scores, baseline_scores)
        ks_stat, ks_pvalue = stats.ks_2samp(new_scores, baseline_scores)

        # Percentile analysis
        above_95 = (new_scores > np.percentile(baseline_scores, 95)).sum()
        above_99 = (new_scores > np.percentile(baseline_scores, 99)).sum()

        return {
            'new_mean': float(new_mean),
            'new_std': float(new_std),
            'baseline_mean': float(baseline_mean),
            'baseline_std': float(baseline_std),
            'mean_difference': float(new_mean - baseline_mean),
            'mean_zscore': float(calculate_zscore(new_mean, baseline_mean, baseline_std)),
            't_statistic': float(t_stat),
            't_pvalue': float(t_pvalue),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'above_95_pctl': int(above_95),
            'above_99_pctl': int(above_99),
            'n_new': len(new_scores),
            'n_baseline': len(baseline_scores)
        }


class ModelDiagnostics:
    """
    Model diagnostics and interpretability analysis.

    Provides methods for model validation, feature importance analysis,
    and generating interpretable explanations for anomalies.
    """

    def __init__(self):
        """Initialize ModelDiagnostics."""
        self.logger = setup_logging(__name__)

    def generate_diagnostics(
        self,
        model,
        X_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Generate comprehensive model diagnostics.

        Args:
            model: Trained anomaly detection model
            X_train: Training data
            X_test: Test data
            feature_names: Feature names

        Returns:
            Dictionary with diagnostic information
        """
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        train_scores = model.score_samples(X_train)
        test_scores = model.score_samples(X_test)

        diagnostics = {
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_features': X_train.shape[1],
            'feature_names': feature_names,
            'n_anomalies_train': int(train_preds.sum()),
            'n_anomalies_test': int(test_preds.sum()),
            'anomaly_rate_train': float(train_preds.mean()),
            'anomaly_rate_test': float(test_preds.mean()),
            'train_score_mean': float(train_scores.mean()),
            'train_score_std': float(train_scores.std()),
            'test_score_mean': float(test_scores.mean()),
            'test_score_std': float(test_scores.std()),
            'model_type': model.model_type,
            'hyperparameters': model.hyperparameters
        }

        # Add warnings
        warnings = []
        if diagnostics['anomaly_rate_train'] > 0.2:
            warnings.append(
                "High anomaly rate in training data (>20%). "
                "Consider adjusting threshold or contamination parameter."
            )
        if abs(diagnostics['anomaly_rate_train'] - diagnostics['anomaly_rate_test']) > 0.1:
            warnings.append(
                "Large difference between train and test anomaly rates. "
                "Model may not generalize well."
            )
        if diagnostics['train_score_std'] < 0.01:
            warnings.append(
                "Very low score variance. Model may not be discriminating well."
            )

        diagnostics['warnings'] = warnings

        # Add recommendations
        recommendations = []
        if diagnostics['anomaly_rate_train'] > 0.15:
            recommendations.append(
                "Try increasing threshold or decreasing contamination parameter."
            )
        if diagnostics['n_samples_train'] < 50:
            recommendations.append(
                "Training data is small. Consider using more historical data."
            )

        diagnostics['recommendations'] = recommendations

        return diagnostics

    def feature_importance_analysis(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        n_iterations: int = 10
    ) -> Dict[str, float]:
        """
        Analyze feature importance using permutation importance.

        Args:
            model: Trained model
            X: Data to analyze
            feature_names: Feature names
            n_iterations: Number of permutation iterations

        Returns:
            Dictionary mapping feature names to importance scores
        """
        base_scores = model.score_samples(X)
        base_mean = np.mean(base_scores)

        importance_scores = {}

        for i, feature_name in enumerate(feature_names):
            score_diffs = []

            for _ in range(n_iterations):
                # Create permuted copy
                X_permuted = X.copy()
                X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

                # Get scores with permuted feature
                permuted_scores = model.score_samples(X_permuted)
                permuted_mean = np.mean(permuted_scores)

                # Calculate difference
                score_diffs.append(abs(permuted_mean - base_mean))

            importance_scores[feature_name] = float(np.mean(score_diffs))

        # Normalize to [0, 1]
        max_importance = max(importance_scores.values()) if importance_scores else 1
        if max_importance > 0:
            importance_scores = {
                k: v / max_importance
                for k, v in importance_scores.items()
            }

        return importance_scores

    def explain_anomaly(
        self,
        anomaly_idx: int,
        X: np.ndarray,
        feature_names: List[str],
        feature_matrix: pd.DataFrame,
        model
    ) -> Dict:
        """
        Generate explanation for why a specific point is flagged as anomaly.

        Args:
            anomaly_idx: Index of the anomaly
            X: Feature matrix
            feature_names: Feature names
            feature_matrix: Original feature DataFrame
            model: Trained model

        Returns:
            Dictionary with explanation details
        """
        if anomaly_idx >= len(X):
            raise ValueError(f"Invalid anomaly index: {anomaly_idx}")

        anomaly_point = X[anomaly_idx]
        anomaly_score = model.score_samples(X[anomaly_idx:anomaly_idx+1])[0]

        # Calculate feature statistics
        feature_means = np.mean(X, axis=0)
        feature_stds = np.std(X, axis=0)

        # Calculate z-scores for the anomaly
        z_scores = {}
        deviations = {}

        for i, name in enumerate(feature_names):
            if feature_stds[i] > 0:
                z = (anomaly_point[i] - feature_means[i]) / feature_stds[i]
            else:
                z = 0
            z_scores[name] = float(z)
            deviations[name] = {
                'value': float(anomaly_point[i]),
                'mean': float(feature_means[i]),
                'std': float(feature_stds[i]),
                'z_score': float(z),
                'percentile': float(get_percentile(anomaly_point[i], X[:, i]))
            }

        # Find top reasons (features with highest absolute z-scores)
        sorted_features = sorted(
            z_scores.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_reasons = []
        for feature_name, z in sorted_features[:5]:
            if abs(z) > 1.5:  # Only include significant deviations
                deviation = deviations[feature_name]
                direction = "above" if z > 0 else "below"
                top_reasons.append(
                    f"{feature_name} ({deviation['value']:.2f}) is "
                    f"{abs(z):.1f}σ {direction} mean ({deviation['mean']:.2f})"
                )

        return {
            'anomaly_idx': anomaly_idx,
            'anomaly_score': float(anomaly_score),
            'z_scores': z_scores,
            'feature_deviations': deviations,
            'top_reasons': top_reasons,
            'n_significant_deviations': sum(1 for z in z_scores.values() if abs(z) > 2)
        }

    def model_comparison_report(
        self,
        models: Dict,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """
        Generate comparison report for multiple models.

        Args:
            models: Dictionary of model_name -> model
            X: Data to evaluate
            feature_names: Feature names

        Returns:
            Dictionary with comparison details
        """
        comparison = {
            'n_models': len(models),
            'n_samples': len(X),
            'models': {}
        }

        all_predictions = {}
        all_scores = {}

        for name, model in models.items():
            if model.is_fitted:
                preds = model.predict(X)
                scores = model.score_samples(X)

                all_predictions[name] = preds
                all_scores[name] = scores

                comparison['models'][name] = {
                    'model_type': model.model_type,
                    'n_anomalies': int(preds.sum()),
                    'anomaly_rate': float(preds.mean()),
                    'mean_score': float(scores.mean()),
                    'max_score': float(scores.max()),
                    'min_score': float(scores.min())
                }

        # Calculate agreement between models
        if len(all_predictions) >= 2:
            pred_array = np.array(list(all_predictions.values()))
            agreement = np.mean(pred_array, axis=0)

            comparison['agreement'] = {
                'full_agreement': float((agreement == 1).sum() + (agreement == 0).sum()) / len(agreement),
                'majority_agreement': float(((agreement >= 0.5).astype(int) == (pred_array[0])).mean()),
                'mean_agreement': float(np.mean(agreement[agreement > 0])) if (agreement > 0).any() else 0
            }

            # Find unanimous anomalies
            unanimous_anomalies = np.where(agreement == 1)[0]
            comparison['unanimous_anomalies'] = unanimous_anomalies.tolist()

            # Find contested points
            contested = np.where((agreement > 0) & (agreement < 1))[0]
            comparison['contested_points'] = contested.tolist()

        return comparison

    def get_interpretability_summary(
        self,
        model,
        X: np.ndarray,
        feature_names: List[str],
        anomaly_indices: List[int]
    ) -> List[Dict]:
        """
        Generate interpretability summaries for multiple anomalies.

        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Feature names
            anomaly_indices: Indices of anomalies to explain

        Returns:
            List of explanation dictionaries
        """
        explanations = []

        for idx in anomaly_indices:
            try:
                explanation = self.explain_anomaly(
                    idx, X, feature_names, None, model
                )
                explanations.append(explanation)
            except Exception as e:
                self.logger.error(f"Error explaining anomaly {idx}: {e}")
                explanations.append({
                    'anomaly_idx': idx,
                    'error': str(e)
                })

        return explanations
