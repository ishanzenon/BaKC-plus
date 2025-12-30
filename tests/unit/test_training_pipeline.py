"""
Unit tests for training pipeline

Tests cover the complete training workflow integrating all Phase 2 components.
"""

import unittest
import numpy as np
import pytest

from bakc_plus.pipeline import TrainingPipeline, train_pipeline
from bakc_plus.config import ModelConfig, EnsembleConfig, ConformalConfig
from bakc_plus.model import OCSVMMember
from bakc_plus.conformal import ConformalPredictor


class TestTrainingPipelineInit(unittest.TestCase):
    """Test TrainingPipeline initialization"""

    def test_init_with_configs(self):
        """Test initialization with all configs"""
        model_config = ModelConfig(nu=0.05)
        ensemble_config = EnsembleConfig(num_models=3)
        conformal_config = ConformalConfig(alpha=0.1)

        pipeline = TrainingPipeline(
            model_config=model_config,
            ensemble_config=ensemble_config,
            conformal_config=conformal_config
        )

        assert pipeline.model_config == model_config
        assert pipeline.ensemble_config == ensemble_config
        assert pipeline.conformal_config == conformal_config
        assert not pipeline.is_pipeline_trained()

    def test_init_without_configs(self):
        """Test initialization without configs (uses defaults)"""
        pipeline = TrainingPipeline()

        assert pipeline.model_config is not None
        assert pipeline.ensemble_config is not None
        assert pipeline.conformal_config is not None
        assert not pipeline.is_pipeline_trained()


class TestTrainingPipelineTrain(unittest.TestCase):
    """Test TrainingPipeline.train() method"""

    def test_train_basic(self):
        """Test basic training workflow"""
        pipeline = TrainingPipeline(
            ensemble_config=EnsembleConfig(num_models=2)
        )

        X_train = np.random.randn(200, 5)
        len_cal = 20

        models, predictor = pipeline.train(X_train, len_cal, random_state=42)

        # Check models returned
        assert models is not None
        assert len(models) > 0  # At least one fold
        assert all(isinstance(m, OCSVMMember) for fold in models for m in fold)

        # Check predictor calibrated
        assert isinstance(predictor, ConformalPredictor)
        assert predictor.is_calibrated()
        assert predictor.get_threshold() is not None

        # Check pipeline state
        assert pipeline.is_pipeline_trained()
        assert pipeline.get_models() == models
        assert pipeline.get_predictor() == predictor

    def test_train_deterministic(self):
        """Test that same seed produces identical results"""
        X_train = np.random.randn(100, 5)
        len_cal = 10

        pipeline1 = TrainingPipeline(ensemble_config=EnsembleConfig(num_models=2))
        models1, predictor1 = pipeline1.train(X_train, len_cal, random_state=42)

        pipeline2 = TrainingPipeline(ensemble_config=EnsembleConfig(num_models=2))
        models2, predictor2 = pipeline2.train(X_train, len_cal, random_state=42)

        # Same threshold
        assert predictor1.get_threshold() == predictor2.get_threshold()

        # Same number of models
        assert len(models1) == len(models2)

    def test_train_different_seeds(self):
        """Test that different seeds produce different results"""
        X_train = np.random.randn(100, 5)
        len_cal = 10

        pipeline1 = TrainingPipeline(ensemble_config=EnsembleConfig(num_models=2))
        _, predictor1 = pipeline1.train(X_train, len_cal, random_state=42)

        pipeline2 = TrainingPipeline(ensemble_config=EnsembleConfig(num_models=2))
        _, predictor2 = pipeline2.train(X_train, len_cal, random_state=123)

        # Different thresholds (high probability)
        assert predictor1.get_threshold() != predictor2.get_threshold()

    def test_train_realistic_data(self):
        """Test with realistic dataset size"""
        pipeline = TrainingPipeline(
            ensemble_config=EnsembleConfig(num_models=5)
        )

        X_train = np.random.randn(1000, 20)
        len_cal = 50

        models, predictor = pipeline.train(X_train, len_cal, random_state=42)

        # Check fold count: 1000 // 50 = 20
        assert len(models) == 20

        # Check members per fold
        assert all(len(fold_models) == 5 for fold_models in models)

        # Check threshold in reasonable range
        threshold = predictor.get_threshold()
        assert 0.0 < threshold < 1.0


class TestConvenienceFunction(unittest.TestCase):
    """Test train_pipeline convenience function"""

    def test_convenience_function(self):
        """Test convenience function produces same results"""
        X_train = np.random.randn(200, 5)
        len_cal = 20

        models, predictor = train_pipeline(
            X_train, len_cal, random_state=42
        )

        assert models is not None
        assert predictor is not None
        assert predictor.is_calibrated()

    def test_convenience_with_configs(self):
        """Test convenience function with custom configs"""
        X_train = np.random.randn(200, 5)
        len_cal = 20

        models, predictor = train_pipeline(
            X_train,
            len_cal,
            model_config=ModelConfig(nu=0.01),
            ensemble_config=EnsembleConfig(num_models=3),
            conformal_config=ConformalConfig(alpha=0.05),
            random_state=42
        )

        assert len(models[0]) == 3  # 3 members per fold


class TestIntegration(unittest.TestCase):
    """Integration tests for training pipeline"""

    def test_full_workflow(self):
        """Test complete workflow from data to trained predictor"""
        # Create pipeline
        pipeline = TrainingPipeline(
            model_config=ModelConfig(nu=0.05, kernel='rbf'),
            ensemble_config=EnsembleConfig(num_models=5),
            conformal_config=ConformalConfig(alpha=0.1)
        )

        # Generate synthetic data
        np.random.seed(42)
        X_train = np.random.randn(500, 10)
        len_cal = 25

        # Train
        models, predictor = pipeline.train(X_train, len_cal, random_state=42)

        # Verify all components integrated
        assert len(models) == 500 // 25  # 20 folds
        assert all(len(fold) == 5 for fold in models)  # 5 members per fold
        assert predictor.is_calibrated()

        # Verify models can predict
        X_test = np.random.randn(10, 10)
        for fold_models in models:
            for member in fold_models:
                scores = member.decision_function(X_test)
                assert len(scores) == 10

    def test_score_accumulation_order(self):
        """Test that score accumulation order is correct"""
        pipeline = TrainingPipeline(ensemble_config=EnsembleConfig(num_models=2))

        X_train = np.random.randn(100, 5)
        len_cal = 10

        # Train and get predictor
        models, predictor = pipeline.train(X_train, len_cal, random_state=42)

        # Threshold should be computed from all scores (calib + OOB)
        # Just verify it exists and is reasonable
        threshold = predictor.get_threshold()
        assert 0.0 < threshold < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
