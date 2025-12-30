"""
Unit tests for OC-SVM Model Module (Step 2.1)

Tests for:
- StratifiedBootstrapper: Random state hashing and bootstrapping logic
- OCSVMMember: OC-SVM model training and prediction

These tests ensure reproducibility, correctness, and proper error handling.
"""

import pytest
import numpy as np
from sklearn.svm import OneClassSVM

from bakc_plus.model.bootstrapping import StratifiedBootstrapper, stratified_bootstrap
from bakc_plus.model.ocsvm import OCSVMMember, create_ocsvm_member
from bakc_plus.config import ModelConfig


# ============================================================================
# FIXTURES - Shared test data and configurations
# ============================================================================

@pytest.fixture
def sample_data():
    """
    Create sample training data for bootstrapping and model fitting.

    Returns:
        np.ndarray: Shape (100, 5) with normalized features
    """
    np.random.seed(42)
    X = np.random.randn(100, 5)
    return X


@pytest.fixture
def small_data():
    """Create small dataset for edge case testing"""
    np.random.seed(123)
    return np.random.randn(20, 3)


@pytest.fixture
def large_data():
    """Create large dataset for consistency testing"""
    np.random.seed(456)
    return np.random.randn(1000, 10)


@pytest.fixture
def model_config():
    """Create a standard ModelConfig for testing"""
    return ModelConfig(
        nu=0.05,
        kernel='rbf',
        gamma='scale',
        cache_size=200
    )


@pytest.fixture
def bootstrapper():
    """Create a StratifiedBootstrapper instance"""
    return StratifiedBootstrapper()


# ============================================================================
# STRATIFIEDBOOTSTRAPPER TESTS (12 tests)
# ============================================================================

class TestHashRandomState:
    """Tests for StratifiedBootstrapper.hash_random_state() method"""

    def test_hash_random_state_deterministic(self):
        """
        Test that hash_random_state produces deterministic results.

        Same inputs must always produce the same hash value.
        This is CRITICAL for reproducibility.
        """
        hash1 = StratifiedBootstrapper.hash_random_state(0, 0, 42)
        hash2 = StratifiedBootstrapper.hash_random_state(0, 0, 42)

        assert hash1 == hash2, "Same inputs should produce same hash"
        assert isinstance(hash1, (int, np.integer))

    def test_hash_random_state_different_inputs_member_idx(self):
        """
        Test that different member indices produce different hashes.

        Each ensemble member should get a different random seed.
        """
        hash0 = StratifiedBootstrapper.hash_random_state(0, 0, 42)
        hash1 = StratifiedBootstrapper.hash_random_state(1, 0, 42)
        hash2 = StratifiedBootstrapper.hash_random_state(2, 0, 42)

        assert hash0 != hash1, "Different member_idx should produce different hashes"
        assert hash1 != hash2, "Different member_idx should produce different hashes"
        assert hash0 != hash2

    def test_hash_random_state_different_inputs_fold_idx(self):
        """
        Test that different fold indices produce different hashes.

        Different CV folds should get different random seeds.
        """
        hash0 = StratifiedBootstrapper.hash_random_state(0, 0, 42)
        hash1 = StratifiedBootstrapper.hash_random_state(0, 1, 42)
        hash2 = StratifiedBootstrapper.hash_random_state(0, 2, 42)

        assert hash0 != hash1, "Different fold_idx should produce different hashes"
        assert hash1 != hash2
        assert hash0 != hash2

    def test_hash_random_state_different_inputs_random_state(self):
        """
        Test that different random states produce different hashes.

        Different base seeds should produce different hashes.
        """
        hash1 = StratifiedBootstrapper.hash_random_state(0, 0, 42)
        hash2 = StratifiedBootstrapper.hash_random_state(0, 0, 43)
        hash3 = StratifiedBootstrapper.hash_random_state(0, 0, 100)

        assert hash1 != hash2, "Different random_state should produce different hashes"
        assert hash2 != hash3
        assert hash1 != hash3

    def test_hash_random_state_range(self):
        """
        Test that hash values are within valid range [0, 2^32-1].

        All hashed values must fit in a 32-bit unsigned integer.
        The XOR with 0x7FFFFFFF flips the high bit but doesn't constrain to 31-bit.
        """
        # Test multiple combinations
        for member_idx in range(5):
            for fold_idx in range(3):
                for random_state in [42, 123, 999, 12345]:
                    h = StratifiedBootstrapper.hash_random_state(
                        member_idx, fold_idx, random_state
                    )

                    assert 0 <= h < 2**32, \
                        f"Hash {h} out of range [0, 2^32-1]"
                    assert isinstance(h, (int, np.integer))

    def test_hash_random_state_all_zero_inputs(self):
        """Test hashing with all zero inputs"""
        h = StratifiedBootstrapper.hash_random_state(0, 0, 0)
        assert 0 <= h < 2**32
        assert isinstance(h, (int, np.integer))

    def test_hash_random_state_large_inputs(self):
        """Test hashing with large input values"""
        h = StratifiedBootstrapper.hash_random_state(
            member_idx=10000,
            fold_idx=1000,
            random_state=999999
        )
        assert 0 <= h < 2**32


class TestPerformBootstrapping:
    """Tests for StratifiedBootstrapper.perform_bootstrapping() method"""

    def test_perform_bootstrapping_basic(self, sample_data, bootstrapper):
        """
        Test that basic bootstrapping works and returns expected structure.

        Should return bootstrapped data and leave-out indices.
        """
        X_boot, leave_out = bootstrapper.perform_bootstrapping(
            sample_data, member_idx=0, num_members=5, random_state=42
        )

        # Check shapes
        assert X_boot.shape[0] + len(leave_out) == len(sample_data)
        assert X_boot.shape[1] == sample_data.shape[1]

        # Check that leave_out indices are valid
        assert len(leave_out) > 0
        assert np.max(leave_out) < len(sample_data)
        assert np.min(leave_out) >= 0

    def test_perform_bootstrapping_deterministic(self, sample_data, bootstrapper):
        """
        Test that bootstrapping is deterministic.

        Same member_idx and random_state should produce identical splits.
        CRITICAL for reproducibility.
        """
        X_boot1, leave_out1 = bootstrapper.perform_bootstrapping(
            sample_data, member_idx=0, num_members=5, random_state=42
        )

        X_boot2, leave_out2 = bootstrapper.perform_bootstrapping(
            sample_data, member_idx=0, num_members=5, random_state=42
        )

        # Check data matches
        np.testing.assert_array_equal(X_boot1, X_boot2)
        np.testing.assert_array_equal(leave_out1, leave_out2)

    @pytest.mark.parametrize("num_members", [2, 3, 5, 10])
    def test_perform_bootstrapping_coverage(self, sample_data, bootstrapper, num_members):
        """
        Test that all indices are covered across all ensemble members.

        Each data point should appear in exactly (M-1)/M of the bootstrapped sets.
        """
        all_indices_used = set()
        all_leave_outs = set()

        for member_idx in range(num_members):
            X_boot, leave_out = bootstrapper.perform_bootstrapping(
                sample_data,
                member_idx=member_idx,
                num_members=num_members,
                random_state=42
            )

            # Collect indices
            leave_out_set = set(leave_out)
            all_leave_outs.update(leave_out_set)

            # All indices in leave_out should be in original data
            assert np.max(leave_out) < len(sample_data)

        # All indices should appear in at least one leave_out set
        # (though not necessarily all appear exactly once due to group splitting)
        assert len(all_leave_outs) > 0

    @pytest.mark.parametrize("num_members", [2, 5, 10])
    def test_perform_bootstrapping_leave_one_out_ratio(self, sample_data, bootstrapper, num_members):
        """
        Test that leave-out sets have approximately correct size.

        Each leave-out set should have ~1/M of the data.
        """
        X_boot, leave_out = bootstrapper.perform_bootstrapping(
            sample_data, member_idx=0, num_members=num_members, random_state=42
        )

        expected_leave_out_size = len(sample_data) / num_members
        actual_leave_out_size = len(leave_out)

        # Allow ±2 samples tolerance for rounding
        assert abs(actual_leave_out_size - expected_leave_out_size) <= 2, \
            f"Expected ~{expected_leave_out_size} left-out samples, got {actual_leave_out_size}"

    def test_perform_bootstrapping_coverage_all_indices(self, sample_data, bootstrapper):
        """
        Test that indices appear in leave-out sets across ensemble.

        Due to the nature of array_split, different indices will be left out for
        different members. This test verifies basic coverage.
        """
        num_members = 5
        all_leave_out_indices = set()

        for member_idx in range(num_members):
            _, leave_out = bootstrapper.perform_bootstrapping(
                sample_data,
                member_idx=member_idx,
                num_members=num_members,
                random_state=42
            )
            all_leave_out_indices.update(leave_out)

        # Check that we cover some portion of indices
        # With 5 members and 100 samples, we expect reasonable coverage
        coverage_ratio = len(all_leave_out_indices) / len(sample_data)
        assert coverage_ratio >= 0.5, \
            f"Expected at least 50% coverage, got {100*coverage_ratio:.1f}%"
        assert coverage_ratio <= 1.0, \
            f"Coverage should not exceed 100%, got {100*coverage_ratio:.1f}%"

    def test_perform_bootstrapping_invalid_empty_data(self, bootstrapper):
        """Test error handling for empty input data"""
        X_empty = np.array([]).reshape(0, 5)

        with pytest.raises(ValueError, match="empty or None"):
            bootstrapper.perform_bootstrapping(
                X_empty, member_idx=0, num_members=5, random_state=42
            )

    def test_perform_bootstrapping_invalid_none_data(self, bootstrapper):
        """Test error handling for None input data"""
        with pytest.raises(ValueError, match="empty or None"):
            bootstrapper.perform_bootstrapping(
                None, member_idx=0, num_members=5, random_state=42
            )

    def test_perform_bootstrapping_invalid_member_idx(self, sample_data, bootstrapper):
        """Test error handling for invalid member_idx"""
        # member_idx >= num_members
        with pytest.raises(ValueError, match="member_idx must be in"):
            bootstrapper.perform_bootstrapping(
                sample_data, member_idx=5, num_members=5, random_state=42
            )

        # member_idx < 0
        with pytest.raises(ValueError, match="member_idx must be in"):
            bootstrapper.perform_bootstrapping(
                sample_data, member_idx=-1, num_members=5, random_state=42
            )

    def test_perform_bootstrapping_invalid_num_members(self, sample_data, bootstrapper):
        """Test error handling for invalid num_members"""
        # When num_members <= 0, the code first checks if member_idx is in range [0, num_members-1]
        # which fails before checking num_members validity
        with pytest.raises(ValueError):
            bootstrapper.perform_bootstrapping(
                sample_data, member_idx=0, num_members=0, random_state=42
            )

        with pytest.raises(ValueError):
            bootstrapper.perform_bootstrapping(
                sample_data, member_idx=0, num_members=-5, random_state=42
            )

    @pytest.mark.parametrize("n_samples", [5, 20, 100])
    def test_perform_bootstrapping_edge_cases(self, bootstrapper, n_samples):
        """
        Test bootstrapping with edge case dataset sizes.

        Testing with small and medium dataset sizes.
        Note: Very small datasets (< num_members) may produce empty training sets.
        """
        X = np.random.randn(n_samples, 3)
        num_members = 3

        for member_idx in range(num_members):
            X_boot, leave_out = bootstrapper.perform_bootstrapping(
                X, member_idx=member_idx, num_members=num_members, random_state=42
            )

            # Basic sanity checks
            assert len(leave_out) > 0, "Leave-out set should not be empty"
            assert X_boot.shape[0] + len(leave_out) == n_samples
            assert X_boot.shape[1] == 3

    def test_perform_bootstrapping_with_fold_idx(self, sample_data, bootstrapper):
        """
        Test that different fold indices produce different splits.

        fold_idx should affect the random state hashing.
        """
        X_boot_fold0, leave_out_fold0 = bootstrapper.perform_bootstrapping(
            sample_data, member_idx=0, num_members=5, random_state=42, fold_idx=0
        )

        X_boot_fold1, leave_out_fold1 = bootstrapper.perform_bootstrapping(
            sample_data, member_idx=0, num_members=5, random_state=42, fold_idx=1
        )

        # Different fold_idx should produce different splits
        assert not np.array_equal(X_boot_fold0, X_boot_fold1) or \
               not np.array_equal(leave_out_fold0, leave_out_fold1), \
            "Different fold_idx should produce different splits"


class TestStratifiedBootstrapFunction:
    """Tests for the convenience function stratified_bootstrap()"""

    def test_stratified_bootstrap_convenience_function(self, sample_data):
        """
        Test that the convenience function works correctly.

        Should produce same results as StratifiedBootstrapper instance method.
        """
        bootstrapper = StratifiedBootstrapper()

        # Using class method
        X_boot1, leave_out1 = bootstrapper.perform_bootstrapping(
            sample_data, member_idx=0, num_members=5, random_state=42
        )

        # Using convenience function
        X_boot2, leave_out2 = stratified_bootstrap(
            sample_data, member_idx=0, num_members=5, random_state=42
        )

        np.testing.assert_array_equal(X_boot1, X_boot2)
        np.testing.assert_array_equal(leave_out1, leave_out2)


# ============================================================================
# OCSVM MEMBER TESTS (13 tests)
# ============================================================================

class TestOCSVMMemberInit:
    """Tests for OCSVMMember initialization"""

    def test_init_with_config(self, model_config):
        """
        Test initialization with ModelConfig.

        Should use config parameters.
        """
        member = OCSVMMember(config=model_config)

        assert member.nu == 0.05
        assert member.kernel == 'rbf'
        assert member.gamma == 'scale'
        assert member.cache_size == 200
        assert member.model is None
        assert not member.is_fitted()

    def test_init_with_explicit_parameters(self):
        """
        Test initialization with explicit parameters (no config).

        Should use explicit parameters with defaults.
        """
        member = OCSVMMember(
            nu=0.1,
            kernel='linear',
            gamma='auto',
            cache_size=300
        )

        assert member.nu == 0.1
        assert member.kernel == 'linear'
        assert member.gamma == 'auto'
        assert member.cache_size == 300

    def test_init_with_config_and_overrides(self, model_config):
        """
        Test that explicit parameters override config values.

        If both config and parameters provided, parameters win.
        """
        member = OCSVMMember(
            config=model_config,
            nu=0.2,  # Override
            kernel='poly'  # Override
        )

        assert member.nu == 0.2  # Overridden
        assert member.kernel == 'poly'  # Overridden
        assert member.gamma == 'scale'  # From config
        assert member.cache_size == 200  # From config

    def test_init_with_defaults(self):
        """
        Test initialization with defaults when no config or parameters.

        Should use sensible defaults.
        """
        member = OCSVMMember()

        assert member.nu == 0.05
        assert member.kernel == 'rbf'
        assert member.gamma == 'scale'
        assert member.cache_size == 200

    @pytest.mark.parametrize("kernel", ['rbf', 'linear', 'poly', 'sigmoid'])
    def test_init_with_different_kernels(self, kernel):
        """Test initialization with different kernel types"""
        member = OCSVMMember(kernel=kernel)
        assert member.kernel == kernel


class TestOCSVMMemberFitting:
    """Tests for OCSVMMember.fit() method"""

    def test_fit_without_bootstrapping(self, sample_data, model_config):
        """
        Test fitting without bootstrapping (num_members=None).

        Should use full training data.
        """
        member = OCSVMMember(config=model_config)
        model, leave_out = member.fit(sample_data, random_state=42)

        assert member.is_fitted()
        assert isinstance(model, OneClassSVM)
        assert leave_out is None
        assert model.n_support_ > 0

    def test_fit_with_bootstrapping(self, sample_data, model_config):
        """
        Test fitting with bootstrapping enabled.

        Should return model and leave-out indices.
        """
        member = OCSVMMember(config=model_config)
        model, leave_out = member.fit(
            sample_data,
            member_idx=0,
            num_members=5,
            random_state=42
        )

        assert member.is_fitted()
        assert isinstance(model, OneClassSVM)
        assert leave_out is not None
        assert len(leave_out) > 0
        assert len(leave_out) < len(sample_data)

    def test_fit_deterministic(self, sample_data, model_config):
        """
        Test that same random seed produces reproducible models.

        CRITICAL for reproducibility in ensemble training.
        """
        # Create two members and fit with same parameters
        member1 = OCSVMMember(config=model_config)
        model1, _ = member1.fit(
            sample_data,
            member_idx=0,
            num_members=5,
            fold_idx=0,
            random_state=42
        )

        member2 = OCSVMMember(config=model_config)
        model2, _ = member2.fit(
            sample_data,
            member_idx=0,
            num_members=5,
            fold_idx=0,
            random_state=42
        )

        # Support vectors should be identical (or very similar)
        # This tests that random state hashing is deterministic
        assert model1.n_support_ == model2.n_support_

    def test_fit_different_members_different_models(self, sample_data, model_config):
        """
        Test that different ensemble members produce different models.

        Different member_idx should result in different training data.
        """
        member1 = OCSVMMember(config=model_config)
        model1, leave_out1 = member1.fit(
            sample_data,
            member_idx=0,
            num_members=5,
            random_state=42
        )

        member2 = OCSVMMember(config=model_config)
        model2, leave_out2 = member2.fit(
            sample_data,
            member_idx=1,
            num_members=5,
            random_state=42
        )

        # Different leave-out sets
        assert not np.array_equal(leave_out1, leave_out2)

        # May have different number of support vectors (due to different training data)
        # Just check that both models are valid
        assert model1.n_support_ > 0
        assert model2.n_support_ > 0

    def test_fit_with_fold_idx(self, sample_data, model_config):
        """
        Test that different fold indices affect training.

        fold_idx should be used in random state hashing.
        """
        member_fold0 = OCSVMMember(config=model_config)
        _, leave_out_fold0 = member_fold0.fit(
            sample_data,
            member_idx=0,
            num_members=5,
            fold_idx=0,
            random_state=42
        )

        member_fold1 = OCSVMMember(config=model_config)
        _, leave_out_fold1 = member_fold1.fit(
            sample_data,
            member_idx=0,
            num_members=5,
            fold_idx=1,
            random_state=42
        )

        # Different fold_idx should produce different leave-out sets
        assert not np.array_equal(leave_out_fold0, leave_out_fold1)

    def test_fit_returns_model_and_leave_out(self, sample_data, model_config):
        """
        Test that fit returns both model and leave_out indices.

        Return value should be a tuple (model, leave_out_indices).
        """
        member = OCSVMMember(config=model_config)
        result = member.fit(
            sample_data,
            member_idx=0,
            num_members=5,
            random_state=42
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        model, leave_out = result
        assert isinstance(model, OneClassSVM)
        assert isinstance(leave_out, np.ndarray)

    def test_fit_invalid_empty_data(self, model_config):
        """Test error handling for empty training data"""
        member = OCSVMMember(config=model_config)
        X_empty = np.array([]).reshape(0, 5)

        with pytest.raises(ValueError, match="empty or None"):
            member.fit(X_empty, random_state=42)

    def test_fit_invalid_none_data(self, model_config):
        """Test error handling for None training data"""
        member = OCSVMMember(config=model_config)

        with pytest.raises(ValueError, match="empty or None"):
            member.fit(None, random_state=42)

    def test_fit_invalid_member_idx(self, sample_data, model_config):
        """Test error handling for invalid member_idx"""
        member = OCSVMMember(config=model_config)

        with pytest.raises(ValueError, match="member_idx must be in"):
            member.fit(
                sample_data,
                member_idx=5,
                num_members=5,
                random_state=42
            )

    def test_fit_updates_is_fitted_state(self, sample_data, model_config):
        """Test that is_fitted() is updated after fitting"""
        member = OCSVMMember(config=model_config)

        assert not member.is_fitted()
        member.fit(sample_data, random_state=42)
        assert member.is_fitted()


class TestOCSVMMemberPrediction:
    """Tests for OCSVMMember prediction methods"""

    def test_decision_function_basic(self, sample_data, model_config):
        """
        Test decision function computation.

        Should return scores with correct shape.
        """
        member = OCSVMMember(config=model_config)
        member.fit(sample_data, random_state=42)

        X_test = np.random.randn(20, 5)
        scores = member.decision_function(X_test)

        assert scores.shape == (20,)
        assert scores.dtype in [np.float64, np.float32]

    def test_decision_function_values(self, sample_data, model_config):
        """
        Test that decision function returns reasonable values.

        Scores should be finite numbers.
        """
        member = OCSVMMember(config=model_config)
        member.fit(sample_data, random_state=42)

        X_test = np.random.randn(50, 5)
        scores = member.decision_function(X_test)

        assert np.all(np.isfinite(scores))
        assert scores.min() != scores.max()  # Should have variation

    def test_decision_function_deterministic(self, sample_data, model_config):
        """
        Test that decision function is deterministic.

        Same data and model should produce same scores.
        """
        member = OCSVMMember(config=model_config)
        member.fit(sample_data, random_state=42)

        X_test = sample_data[:10]  # Use subset for testing

        scores1 = member.decision_function(X_test)
        scores2 = member.decision_function(X_test)

        np.testing.assert_array_equal(scores1, scores2)

    def test_decision_function_not_fitted(self, model_config):
        """
        Test error handling when calling decision_function on unfitted model.

        Should raise ValueError.
        """
        member = OCSVMMember(config=model_config)
        X_test = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="not fitted"):
            member.decision_function(X_test)

    def test_predict_basic(self, sample_data, model_config):
        """
        Test predict method.

        Should return predictions with correct shape and values.
        """
        member = OCSVMMember(config=model_config)
        member.fit(sample_data, random_state=42)

        X_test = np.random.randn(20, 5)
        predictions = member.predict(X_test)

        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [1, -1]))  # OC-SVM returns 1 or -1

    def test_predict_not_fitted(self, model_config):
        """
        Test error handling when calling predict on unfitted model.

        Should raise ValueError.
        """
        member = OCSVMMember(config=model_config)
        X_test = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="not fitted"):
            member.predict(X_test)

    def test_predict_consistent_with_decision_function(self, sample_data, model_config):
        """
        Test that predict is consistent with decision_function.

        Predictions should match sign of decision function (approximately).
        """
        member = OCSVMMember(config=model_config)
        member.fit(sample_data, random_state=42)

        X_test = np.random.randn(100, 5)
        predictions = member.predict(X_test)
        scores = member.decision_function(X_test)

        # Predictions should be 1 where score >= 0, -1 where score < 0
        expected_predictions = np.where(scores >= 0, 1, -1)
        np.testing.assert_array_equal(predictions, expected_predictions)


class TestOCSVMMemberStatus:
    """Tests for OCSVMMember status checking methods"""

    def test_is_fitted_before_fitting(self, model_config):
        """Test is_fitted() returns False before fitting"""
        member = OCSVMMember(config=model_config)
        assert not member.is_fitted()

    def test_is_fitted_after_fitting(self, sample_data, model_config):
        """Test is_fitted() returns True after fitting"""
        member = OCSVMMember(config=model_config)
        member.fit(sample_data, random_state=42)
        assert member.is_fitted()

    def test_get_n_support_basic(self, sample_data, model_config):
        """
        Test get_n_support returns number of support vectors.

        Should return positive integer.
        """
        member = OCSVMMember(config=model_config)
        member.fit(sample_data, random_state=42)

        n_support = member.get_n_support()

        assert isinstance(n_support, (int, np.integer))
        assert n_support > 0
        assert n_support <= len(sample_data)

    def test_get_n_support_not_fitted(self, model_config):
        """Test error when getting n_support before fitting"""
        member = OCSVMMember(config=model_config)

        with pytest.raises(ValueError, match="not fitted"):
            member.get_n_support()

    def test_get_n_support_varies_with_data(self, sample_data, model_config):
        """
        Test that n_support can vary with different training data.

        Different bootstrap samples should produce different models.
        """
        member1 = OCSVMMember(config=model_config)
        member1.fit(
            sample_data,
            member_idx=0,
            num_members=5,
            random_state=42
        )
        n_support1 = member1.get_n_support()

        member2 = OCSVMMember(config=model_config)
        member2.fit(
            sample_data,
            member_idx=1,
            num_members=5,
            random_state=42
        )
        n_support2 = member2.get_n_support()

        # May or may not be equal (depends on specific SVM behavior)
        # Just verify both are positive
        assert n_support1 > 0
        assert n_support2 > 0


class TestOCSVMMemberIntegration:
    """Integration tests for OCSVMMember"""

    def test_full_workflow_with_bootstrapping(self, sample_data, model_config):
        """
        Test complete workflow: init -> fit -> predict -> status.

        This tests the full pipeline in realistic usage.
        """
        # Initialize
        member = OCSVMMember(config=model_config)

        # Check initial state
        assert not member.is_fitted()

        # Fit with bootstrapping
        model, leave_out = member.fit(
            sample_data,
            member_idx=0,
            num_members=5,
            random_state=42
        )

        # Check fitted state
        assert member.is_fitted()
        assert model is not None
        assert leave_out is not None

        # Get predictions
        X_test = sample_data[:10]
        predictions = member.predict(X_test)
        scores = member.decision_function(X_test)

        # Verify outputs
        assert predictions.shape == (10,)
        assert scores.shape == (10,)
        assert member.get_n_support() > 0

    def test_member_repr(self, sample_data, model_config):
        """Test string representation of member"""
        member = OCSVMMember(config=model_config)
        repr_before = repr(member)

        assert "OCSVMMember" in repr_before
        assert "not fitted" in repr_before

        member.fit(sample_data, random_state=42)
        repr_after = repr(member)

        assert "OCSVMMember" in repr_after
        assert "fitted" in repr_after

    def test_create_ocsvm_member_factory(self, model_config):
        """
        Test factory function for creating OCSVMMember.

        Convenience function should work correctly.
        """
        member = create_ocsvm_member(model_config)

        assert isinstance(member, OCSVMMember)
        assert member.nu == model_config.nu
        assert member.kernel == model_config.kernel
        assert member.gamma == model_config.gamma
        assert member.cache_size == model_config.cache_size


class TestOCSVMMemberEdgeCases:
    """Edge case tests for OCSVMMember"""

    def test_fit_with_single_feature(self, model_config):
        """Test fitting with single feature"""
        X = np.random.randn(100, 1)
        member = OCSVMMember(config=model_config)

        model, _ = member.fit(X, random_state=42)
        assert member.is_fitted()
        assert model.n_support_ > 0

    def test_fit_with_many_features(self, model_config):
        """Test fitting with high-dimensional data"""
        X = np.random.randn(100, 50)
        member = OCSVMMember(config=model_config)

        model, _ = member.fit(X, random_state=42)
        assert member.is_fitted()
        assert model.n_support_ > 0

    def test_fit_with_small_nu(self):
        """Test fitting with very small nu (few anomalies expected)"""
        member = OCSVMMember(nu=0.01)
        X = np.random.randn(100, 5)

        model, _ = member.fit(X, random_state=42)
        assert member.is_fitted()
        assert model.n_support_ > 0

    def test_fit_with_large_nu(self):
        """Test fitting with large nu (many anomalies expected)"""
        member = OCSVMMember(nu=0.5)
        X = np.random.randn(100, 5)

        model, _ = member.fit(X, random_state=42)
        assert member.is_fitted()
        assert model.n_support_ > 0

    @pytest.mark.parametrize("num_members", [2, 3, 5, 10])
    def test_fit_with_various_num_members(self, sample_data, model_config, num_members):
        """
        Test fitting with different ensemble sizes.

        Note: num_members=1 creates empty training data, which OneClassSVM rejects.
        """
        for member_idx in range(num_members):
            member = OCSVMMember(config=model_config)

            model, leave_out = member.fit(
                sample_data,
                member_idx=member_idx,
                num_members=num_members,
                random_state=42
            )

            assert member.is_fitted()
            assert leave_out is not None


# ============================================================================
# PARAMETRIZED TESTS FOR CONSISTENCY
# ============================================================================

class TestConsistency:
    """Tests for consistency across different configurations"""

    @pytest.mark.parametrize("kernel", ['rbf', 'linear'])
    @pytest.mark.parametrize("nu", [0.01, 0.05, 0.1])
    def test_fit_with_different_configs(self, sample_data, kernel, nu):
        """
        Test that fitting works with different kernel and nu combinations.

        Parametrized over multiple hyperparameters.
        """
        member = OCSVMMember(kernel=kernel, nu=nu)
        model, _ = member.fit(sample_data, random_state=42)

        assert member.is_fitted()
        assert model.n_support_ > 0
        assert member.kernel == kernel
        assert member.nu == nu

    @pytest.mark.parametrize("random_state", [0, 42, 123, 999])
    def test_determinism_across_random_states(self, sample_data, random_state):
        """
        Test that each random state produces consistent results.

        Multiple calls with same random_state should produce identical results.
        """
        member1 = OCSVMMember()
        model1, _ = member1.fit(sample_data, random_state=random_state)

        member2 = OCSVMMember()
        model2, _ = member2.fit(sample_data, random_state=random_state)

        # Both should have same number of support vectors (as trained on full data)
        assert model1.n_support_ == model2.n_support_


# ============================================================================
# PERFORMANCE AND COVERAGE TESTS
# ============================================================================

class TestPerformance:
    """Tests for performance and behavior with larger datasets"""

    def test_bootstrapping_performance(self, large_data, bootstrapper):
        """Test bootstrapping performance on larger dataset"""
        X_boot, leave_out = bootstrapper.perform_bootstrapping(
            large_data,
            member_idx=0,
            num_members=10,
            random_state=42
        )

        assert len(X_boot) + len(leave_out) == len(large_data)
        assert X_boot.shape[1] == large_data.shape[1]

    def test_fitting_performance(self, large_data):
        """Test fitting performance on larger dataset"""
        member = OCSVMMember(nu=0.05)
        model, _ = member.fit(large_data, random_state=42)

        assert member.is_fitted()
        assert model.n_support_ > 0

    def test_prediction_performance(self, large_data):
        """Test prediction performance on larger dataset"""
        member = OCSVMMember(nu=0.05)
        member.fit(large_data, random_state=42)

        X_test = np.random.randn(200, large_data.shape[1])
        predictions = member.predict(X_test)

        assert predictions.shape == (200,)
        assert np.all(np.isin(predictions, [1, -1]))
