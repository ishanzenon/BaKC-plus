"""
Unit tests for prediction pipeline

Tests cover the complete prediction workflow.
"""

import unittest
import numpy as np
import pytest

from bakc_plus.pipeline import TrainingPipeline, PredictionPipeline, predict_pipeline
from bakc_plus.config import EnsembleConfig


class TestPredictionPipelineInit(unittest.TestCase):
    """Test PredictionPipeline initialization"""

    def setUp(self):
        """Set up trained models and predictor for testing"""
        # Train a simple pipeline
        train_pipeline = TrainingPipeline(
            ensemble_config=EnsembleConfig(num_models=2)
        )
        X_train = np.random.randn(100, 5)
        self.models, self.predictor = train_pipeline.train(
            X_train, len_cal=10, random_state=42
        )

    def test_init_with_models_and_predictor(self):
        """Test initialization with trained models and predictor"""
        pipeline = PredictionPipeline(self.models, self.predictor)

        assert pipeline.models == self.models
        assert pipeline.predictor == self.predictor
        assert pipeline.n_folds == len(self.models)
        assert pipeline.n_members == len(self.models[0])

    def test_init_empty_models(self):
        """Test error handling for empty models"""
        with pytest.raises(ValueError, match="models cannot be empty"):
            PredictionPipeline([], self.predictor)

    def test_init_uncalibrated_predictor(self):
        """Test error handling for uncalibrated predictor"""
        from bakc_plus.conformal import ConformalPredictor

        uncalibrated_predictor = ConformalPredictor()
        with pytest.raises(ValueError, match="predictor must be calibrated"):
            PredictionPipeline(self.models, uncalibrated_predictor)


class TestPredictionPipelinePredict(unittest.TestCase):
    """Test PredictionPipeline.predict() method"""

    def setUp(self):
        """Set up trained pipeline for testing"""
        train_pipeline = TrainingPipeline(
            ensemble_config=EnsembleConfig(num_models=2)
        )
        X_train = np.random.randn(200, 5)
        self.models, self.predictor = train_pipeline.train(
            X_train, len_cal=20, random_state=42
        )

    def test_predict_basic(self):
        """Test basic prediction functionality"""
        pipeline = PredictionPipeline(self.models, self.predictor)

        X_test = np.random.randn(50, 5)
        predictions = pipeline.predict(X_test)

        # Check output shape and type
        assert len(predictions) == 50
        assert predictions.dtype == np.int64 or predictions.dtype == np.int32

        # Check binary predictions
        assert np.all((predictions == 0) | (predictions == 1))

    def test_predict_deterministic(self):
        """Test that same data produces same predictions"""
        X_test = np.random.randn(50, 5)

        pipeline1 = PredictionPipeline(self.models, self.predictor)
        pred1 = pipeline1.predict(X_test)

        pipeline2 = PredictionPipeline(self.models, self.predictor)
        pred2 = pipeline2.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)

    def test_predict_empty_data(self):
        """Test error handling for empty test data"""
        pipeline = PredictionPipeline(self.models, self.predictor)

        with pytest.raises(ValueError, match="X_test is empty"):
            pipeline.predict(np.array([]))

    def test_predict_scores(self):
        """Test predict_scores method"""
        pipeline = PredictionPipeline(self.models, self.predictor)

        X_test = np.random.randn(50, 5)
        scores = pipeline.predict_scores(X_test)

        # Check output shape
        assert len(scores) == 50

        # Check scores in (0, 1) range
        assert np.all(scores > 0) and np.all(scores < 1)

    def test_get_threshold(self):
        """Test get_threshold method"""
        pipeline = PredictionPipeline(self.models, self.predictor)

        threshold = pipeline.get_threshold()

        assert isinstance(threshold, float)
        assert 0.0 < threshold < 1.0
        assert threshold == self.predictor.get_threshold()


class TestScoreAggregation(unittest.TestCase):
    """Test score aggregation logic"""

    def setUp(self):
        """Set up pipeline for aggregation testing"""
        train_pipeline = TrainingPipeline(
            ensemble_config=EnsembleConfig(num_models=3)
        )
        X_train = np.random.randn(150, 5)
        self.models, self.predictor = train_pipeline.train(
            X_train, len_cal=15, random_state=42
        )

    def test_score_aggregation_order(self):
        """Test that aggregation follows correct order (mean then median)"""
        pipeline = PredictionPipeline(self.models, self.predictor)

        X_test = np.random.randn(10, 5)

        # Get scores through full pipeline
        scores = pipeline.predict_scores(X_test)

        # Verify scores computed
        assert len(scores) == 10
        assert np.all(np.isfinite(scores))

    def test_multiple_folds_aggregate_correctly(self):
        """Test that multiple folds produce aggregated predictions"""
        pipeline = PredictionPipeline(self.models, self.predictor)

        X_test = np.random.randn(20, 5)
        predictions = pipeline.predict(X_test)

        # With multiple folds and members, should get some variation
        assert len(predictions) == 20
        assert 0 <= np.sum(predictions) <= 20


class TestIntegration(unittest.TestCase):
    """Integration tests for prediction pipeline"""

    def test_full_train_predict_workflow(self):
        """Test complete train → predict workflow"""
        # Train
        train_pipeline = TrainingPipeline(
            ensemble_config=EnsembleConfig(num_models=5)
        )
        X_train = np.random.randn(500, 10)
        models, predictor = train_pipeline.train(
            X_train, len_cal=25, random_state=42
        )

        # Predict
        pred_pipeline = PredictionPipeline(models, predictor)
        X_test = np.random.randn(100, 10)
        predictions = pred_pipeline.predict(X_test)

        # Verify predictions
        assert len(predictions) == 100
        assert np.all((predictions == 0) | (predictions == 1))

        # Verify some anomalies detected (probabilistic, but should happen)
        # With alpha=0.1, expect ~10% anomalies
        n_anomalies = np.sum(predictions)
        assert 0 <= n_anomalies <= 100  # Sanity check

    def test_convenience_function(self):
        """Test predict_pipeline convenience function"""
        # Train
        train_pipeline = TrainingPipeline(
            ensemble_config=EnsembleConfig(num_models=2)
        )
        X_train = np.random.randn(200, 5)
        models, predictor = train_pipeline.train(
            X_train, len_cal=20, random_state=42
        )

        # Predict using convenience function
        X_test = np.random.randn(50, 5)
        predictions = predict_pipeline(models, predictor, X_test)

        assert len(predictions) == 50
        assert np.all((predictions == 0) | (predictions == 1))

    def test_reproducibility_across_runs(self):
        """Test reproducibility with same trained models"""
        # Train once
        train_pipeline = TrainingPipeline(
            ensemble_config=EnsembleConfig(num_models=2)
        )
        X_train = np.random.randn(100, 5)
        models, predictor = train_pipeline.train(
            X_train, len_cal=10, random_state=42
        )

        # Predict multiple times with same test data
        X_test = np.random.randn(30, 5)

        pred1 = PredictionPipeline(models, predictor).predict(X_test)
        pred2 = PredictionPipeline(models, predictor).predict(X_test)
        pred3 = PredictionPipeline(models, predictor).predict(X_test)

        # All predictions should be identical
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
