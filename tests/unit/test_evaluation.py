"""
Unit tests for evaluation module

Tests cover Power, FDR computation and metrics calculation.
"""

import unittest
import numpy as np
import pytest

from bakc_plus.evaluation import (
    compute_metrics,
    compute_power,
    compute_fdr,
    MetricsCalculator
)


class TestMetricsComputation(unittest.TestCase):
    """Test Power and FDR computation"""

    def test_compute_metrics_perfect_prediction(self):
        """Test metrics with perfect prediction"""
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])

        metrics = compute_metrics(y_true, y_pred)

        # Perfect prediction: TP=3, FP=0, TN=2, FN=0
        assert metrics['tp'] == 3
        assert metrics['fp'] == 0
        assert metrics['tn'] == 2
        assert metrics['fn'] == 0
        assert metrics['power'] == 1.0  # TP/(TP+FN) = 3/3
        assert metrics['fdr'] == 0.0  # FP/(FP+TP) = 0/3

    def test_compute_metrics_all_wrong(self):
        """Test metrics with all wrong predictions"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])

        metrics = compute_metrics(y_true, y_pred)

        # All wrong: TP=0, FP=2, TN=0, FN=2
        assert metrics['tp'] == 0
        assert metrics['fp'] == 2
        assert metrics['tn'] == 0
        assert metrics['fn'] == 2
        assert metrics['power'] == 0.0  # TP/(TP+FN) = 0/2
        assert metrics['fdr'] == 1.0  # FP/(FP+TP) = 2/2

    def test_compute_metrics_mixed(self):
        """Test metrics with mixed predictions"""
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])

        metrics = compute_metrics(y_true, y_pred)

        # TP=2 (positions 0,4), FP=1 (position 3), TN=1 (position 2), FN=1 (position 1)
        assert metrics['tp'] == 2
        assert metrics['fp'] == 1
        assert metrics['tn'] == 1
        assert metrics['fn'] == 1
        assert np.isclose(metrics['power'], 2.0/3.0)  # TP/(TP+FN) = 2/3
        assert np.isclose(metrics['fdr'], 1.0/3.0)  # FP/(FP+TP) = 1/3

    def test_compute_power_standalone(self):
        """Test standalone power computation"""
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0])

        power = compute_power(y_true, y_pred)

        # TP=2, FN=1 → Power = 2/3
        assert np.isclose(power, 2.0/3.0)

    def test_compute_fdr_standalone(self):
        """Test standalone FDR computation"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0])

        fdr = compute_fdr(y_true, y_pred)

        # TP=2, FP=1 → FDR = 1/3
        assert np.isclose(fdr, 1.0/3.0)

    def test_compute_metrics_no_anomalies(self):
        """Test with no anomalies in ground truth"""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0])

        metrics = compute_metrics(y_true, y_pred)

        # TP=0, FP=2, TN=2, FN=0
        assert metrics['power'] == 0.0  # No anomalies in ground truth
        assert metrics['fdr'] == 1.0  # All predictions are false

    def test_compute_metrics_no_predictions(self):
        """Test with no anomaly predictions"""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        metrics = compute_metrics(y_true, y_pred)

        # TP=0, FP=0, TN=2, FN=2
        assert metrics['power'] == 0.0  # Missed all anomalies
        assert metrics['fdr'] == 0.0  # No predictions made

    def test_compute_metrics_empty_input(self):
        """Test error handling for empty inputs"""
        with pytest.raises(ValueError):
            compute_metrics(np.array([]), np.array([1, 0]))

        with pytest.raises(ValueError):
            compute_metrics(np.array([1, 0]), np.array([]))

    def test_compute_metrics_length_mismatch(self):
        """Test error handling for length mismatch"""
        with pytest.raises(ValueError):
            compute_metrics(np.array([1, 0]), np.array([1, 0, 1]))


class TestMetricsCalculator(unittest.TestCase):
    """Test MetricsCalculator class"""

    def test_calculator_compute(self):
        """Test calculator compute method"""
        calculator = MetricsCalculator()
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])

        metrics = calculator.compute(y_true, y_pred)

        assert 'power' in metrics
        assert 'fdr' in metrics
        assert isinstance(metrics['power'], float)
        assert isinstance(metrics['fdr'], float)

    def test_calculator_compute_batch(self):
        """Test batch metrics computation"""
        calculator = MetricsCalculator()

        y_true_list = [
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0]),
            np.array([0, 0, 1, 1]),
        ]
        y_pred_list = [
            np.array([1, 1, 0, 0]),  # Perfect
            np.array([1, 1, 0, 0]),  # Mixed
            np.array([0, 1, 1, 1]),  # Mixed
        ]

        results = calculator.compute_batch(y_true_list, y_pred_list)

        assert 'power' in results
        assert 'fdr' in results
        assert 'mean_power' in results
        assert 'mean_fdr' in results
        assert len(results['power']) == 3
        assert len(results['fdr']) == 3

    def test_calculator_compute_batch_length_mismatch(self):
        """Test batch computation with length mismatch"""
        calculator = MetricsCalculator()

        y_true_list = [np.array([1, 0])]
        y_pred_list = [np.array([1, 0]), np.array([0, 1])]

        with pytest.raises(ValueError):
            calculator.compute_batch(y_true_list, y_pred_list)


class TestIntegration(unittest.TestCase):
    """Integration tests for metrics module"""

    def test_typical_anomaly_detection_scenario(self):
        """Test with typical anomaly detection scenario"""
        # Simulate: 10% anomalies, 90% detection rate, 5% FDR
        np.random.seed(42)
        n_samples = 1000
        n_anomalies = 100

        y_true = np.zeros(n_samples)
        y_true[:n_anomalies] = 1
        np.random.shuffle(y_true)

        # Simulate predictions: 90% power, 5% FDR
        y_pred = y_true.copy()
        # Miss 10% of anomalies
        anomaly_indices = np.where(y_true == 1)[0]
        miss_indices = np.random.choice(anomaly_indices, size=10, replace=False)
        y_pred[miss_indices] = 0

        # Add false positives (5% FDR)
        normal_indices = np.where(y_true == 0)[0]
        fp_indices = np.random.choice(normal_indices, size=5, replace=False)
        y_pred[fp_indices] = 1

        metrics = compute_metrics(y_true, y_pred)

        # Verify expected ranges
        assert 0.85 <= metrics['power'] <= 0.95  # ~90% power
        assert 0.0 <= metrics['fdr'] <= 0.1  # ~5% FDR


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
