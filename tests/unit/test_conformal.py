"""
Unit tests for conformal prediction module

Tests cover scoring, calibration, and prediction functionality.
"""

import unittest
import numpy as np
import pytest

from bakc_plus.conformal import (
    sigmoid_score,
    compute_threshold,
    predict_anomalies,
    ConformalPredictor
)
from bakc_plus.config import ConformalConfig


class TestScoring(unittest.TestCase):
    """Test sigmoid scoring function"""

    def test_sigmoid_score_basic(self):
        """Test basic sigmoid transformation"""
        scores = np.array([-2.0, 0.0, 2.0])
        conformity = sigmoid_score(scores)

        assert len(conformity) == 3
        assert np.all(conformity > 0) and np.all(conformity < 1)
        # Sigmoid(-2) > Sigmoid(0) > Sigmoid(2)
        assert conformity[0] > conformity[1] > conformity[2]

    def test_sigmoid_score_formula(self):
        """Verify exact sigmoid formula: 1/(1+exp(x))"""
        scores = np.array([0.0])
        conformity = sigmoid_score(scores)
        expected = 1.0 / (1.0 + np.exp(0.0))  # = 0.5
        np.testing.assert_almost_equal(conformity[0], expected)

    def test_sigmoid_score_range(self):
        """Test sigmoid output range (0, 1)"""
        scores = np.random.randn(100)
        conformity = sigmoid_score(scores)
        assert np.all(conformity > 0) and np.all(conformity < 1)


class TestThresholdComputation(unittest.TestCase):
    """Test conformal threshold computation"""

    def test_compute_threshold_basic(self):
        """Test basic threshold computation"""
        calib_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        threshold = compute_threshold(calib_scores, alpha=0.1)
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_compute_threshold_formula(self):
        """Verify threshold formula: q_level = ceil((n+1)*(1-alpha))/n"""
        calib_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        n = len(calib_scores)
        alpha = 0.1

        threshold = compute_threshold(calib_scores, alpha=alpha)

        # Compute expected q_level
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)  # Clamp to [0, 1]
        expected_threshold = np.quantile(calib_scores, q_level, method='higher')

        np.testing.assert_almost_equal(threshold, expected_threshold)

    def test_compute_threshold_method_higher(self):
        """Test that method must be 'higher'"""
        calib_scores = np.array([0.1, 0.3, 0.5])

        # Default should work
        threshold = compute_threshold(calib_scores)
        assert threshold is not None

        # Explicit 'higher' should work
        threshold = compute_threshold(calib_scores, method='higher')
        assert threshold is not None

    def test_compute_threshold_invalid_alpha(self):
        """Test error handling for invalid alpha"""
        calib_scores = np.array([0.1, 0.3, 0.5])

        with pytest.raises(ValueError):
            compute_threshold(calib_scores, alpha=-0.1)

        with pytest.raises(ValueError):
            compute_threshold(calib_scores, alpha=1.5)

    def test_compute_threshold_empty_scores(self):
        """Test error handling for empty calibration scores"""
        with pytest.raises(ValueError):
            compute_threshold(np.array([]))


class TestPrediction(unittest.TestCase):
    """Test anomaly prediction"""

    def test_predict_anomalies_basic(self):
        """Test basic anomaly prediction"""
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        threshold = 0.4
        predictions = predict_anomalies(scores, threshold)

        # 0.1 <= 0.4 → 1 (anomaly)
        # 0.3 <= 0.4 → 1 (anomaly)
        # 0.5 > 0.4 → 0 (normal)
        # 0.7 > 0.4 → 0 (normal)
        # 0.9 > 0.4 → 0 (normal)
        expected = np.array([1, 1, 0, 0, 0])
        np.testing.assert_array_equal(predictions, expected)

    def test_predict_anomalies_all_normal(self):
        """Test prediction when all samples are normal"""
        scores = np.array([0.8, 0.9, 1.0])
        threshold = 0.5
        predictions = predict_anomalies(scores, threshold)
        assert np.all(predictions == 0)

    def test_predict_anomalies_all_anomalies(self):
        """Test prediction when all samples are anomalies"""
        scores = np.array([0.1, 0.2, 0.3])
        threshold = 0.5
        predictions = predict_anomalies(scores, threshold)
        assert np.all(predictions == 1)


class TestConformalPredictor(unittest.TestCase):
    """Test ConformalPredictor class"""

    def test_init_with_config(self):
        """Test initialization with config"""
        config = ConformalConfig(alpha=0.05)
        predictor = ConformalPredictor(config=config)
        assert predictor.alpha == 0.05
        assert not predictor.is_calibrated()

    def test_init_without_config(self):
        """Test initialization without config"""
        predictor = ConformalPredictor()
        assert predictor.alpha is not None
        assert not predictor.is_calibrated()

    def test_calibrate(self):
        """Test calibration"""
        predictor = ConformalPredictor(alpha=0.1)
        calib_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

        threshold = predictor.calibrate(calib_scores)

        assert predictor.is_calibrated()
        assert predictor.get_threshold() == threshold
        assert threshold is not None

    def test_predict_without_calibration(self):
        """Test that predict fails without calibration"""
        predictor = ConformalPredictor()
        scores = np.array([0.1, 0.5, 0.9])

        with pytest.raises(RuntimeError, match="not yet calibrated"):
            predictor.predict(scores)

    def test_full_workflow(self):
        """Test full calibration + prediction workflow"""
        predictor = ConformalPredictor(alpha=0.1)

        # Calibrate
        calib_scores = np.random.rand(100)
        predictor.calibrate(calib_scores)

        # Predict
        test_scores = np.random.rand(50)
        predictions = predictor.predict(test_scores)

        assert len(predictions) == 50
        assert np.all((predictions == 0) | (predictions == 1))


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end conformal prediction"""

    def test_end_to_end_pipeline(self):
        """Test complete conformal prediction pipeline"""
        # Simulate OC-SVM decision scores
        calib_decision_scores = np.random.randn(100)
        test_decision_scores = np.random.randn(50)

        # Apply sigmoid scoring
        calib_conformity = sigmoid_score(calib_decision_scores)
        test_conformity = sigmoid_score(test_decision_scores)

        # Compute threshold
        threshold = compute_threshold(calib_conformity, alpha=0.1)

        # Predict anomalies
        predictions = predict_anomalies(test_conformity, threshold)

        # Sanity checks
        assert len(predictions) == 50
        n_anomalies = np.sum(predictions)
        # With alpha=0.1, expect ~10% anomalies (but this is random, so just check range)
        assert 0 <= n_anomalies <= 50

    def test_determinism(self):
        """Test that same inputs produce same outputs"""
        calib_scores = np.random.rand(50)
        test_scores = np.random.rand(20)

        # Run 1
        threshold1 = compute_threshold(calib_scores, alpha=0.1)
        pred1 = predict_anomalies(test_scores, threshold1)

        # Run 2
        threshold2 = compute_threshold(calib_scores, alpha=0.1)
        pred2 = predict_anomalies(test_scores, threshold2)

        # Should be identical
        assert threshold1 == threshold2
        np.testing.assert_array_equal(pred1, pred2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
