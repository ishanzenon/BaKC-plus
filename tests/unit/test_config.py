"""
Unit tests for configuration module

Tests configuration loading, validation, and behavior.
"""

import pytest
import tempfile
import os
from pathlib import Path

from bakc_plus.config import (
    DataConfig,
    ModelConfig,
    EnsembleConfig,
    ConformalConfig,
    BaKCConfig,
)


class TestDataConfig:
    """Tests for DataConfig dataclass"""

    def test_data_config_defaults(self):
        """Test DataConfig has correct default values"""
        config = DataConfig(dataset_name="test")

        assert config.dataset_name == "test"
        assert config.data_dir == Path("./data/input")
        assert config.output_dir == Path("./output")
        assert config.train_fraction == 0.5
        assert config.len_cal is None
        assert config.len_test is None

    def test_data_config_path_conversion(self):
        """Test that string paths are converted to Path objects"""
        config = DataConfig(
            dataset_name="test",
            data_dir="./my/data",
            output_dir="./my/output"
        )

        assert isinstance(config.data_dir, Path)
        assert isinstance(config.output_dir, Path)
        assert config.data_dir == Path("./my/data")
        assert config.output_dir == Path("./my/output")

    def test_data_config_custom_values(self):
        """Test DataConfig with custom values"""
        config = DataConfig(
            dataset_name="cardio",
            data_dir=Path("./data/input"),
            output_dir=Path("./output/cardio"),
            train_fraction=0.6,
            len_cal=1000,
            len_test=2000,
        )

        assert config.dataset_name == "cardio"
        assert config.train_fraction == 0.6
        assert config.len_cal == 1000
        assert config.len_test == 2000


class TestModelConfig:
    """Tests for ModelConfig dataclass"""

    def test_model_config_defaults(self):
        """Test ModelConfig has correct default values"""
        config = ModelConfig()

        assert config.nu == 0.05
        assert config.kernel == "rbf"
        assert config.gamma == "scale"
        assert config.cache_size == 200
        assert config.verbose is False

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values"""
        config = ModelConfig(
            nu=0.1,
            kernel="linear",
            gamma="auto",
            cache_size=500,
            verbose=True,
        )

        assert config.nu == 0.1
        assert config.kernel == "linear"
        assert config.gamma == "auto"
        assert config.cache_size == 500
        assert config.verbose is True


class TestEnsembleConfig:
    """Tests for EnsembleConfig dataclass"""

    def test_ensemble_config_defaults(self):
        """Test EnsembleConfig has correct default values"""
        config = EnsembleConfig()

        assert config.num_models == 5
        assert config.num_folds is None
        assert config.num_test_splits == 20
        assert config.num_repetitions == 5
        assert config.random_state == 42
        assert config.use_multiprocessing is True
        assert config.num_workers is None

    def test_ensemble_config_custom_values(self):
        """Test EnsembleConfig with custom values"""
        config = EnsembleConfig(
            num_models=10,
            num_folds=5,
            num_test_splits=30,
            num_repetitions=10,
            random_state=123,
            use_multiprocessing=False,
            num_workers=4,
        )

        assert config.num_models == 10
        assert config.num_folds == 5
        assert config.num_test_splits == 30
        assert config.num_repetitions == 10
        assert config.random_state == 123
        assert config.use_multiprocessing is False
        assert config.num_workers == 4


class TestConformalConfig:
    """Tests for ConformalConfig dataclass"""

    def test_conformal_config_defaults(self):
        """Test ConformalConfig has correct default values"""
        config = ConformalConfig()

        assert config.alpha == 0.05
        assert config.scoring_method == "sigmoid"
        assert config.quantile_method == "higher"
        assert config.fold_aggregation == "mean"
        assert config.cross_fold_aggregation == "median"

    def test_conformal_config_custom_values(self):
        """Test ConformalConfig with custom values"""
        config = ConformalConfig(
            alpha=0.1,
            scoring_method="normalize",
            quantile_method="lower",
            fold_aggregation="median",
            cross_fold_aggregation="mean",
        )

        assert config.alpha == 0.1
        assert config.scoring_method == "normalize"
        assert config.quantile_method == "lower"
        assert config.fold_aggregation == "median"
        assert config.cross_fold_aggregation == "mean"


class TestBaKCConfig:
    """Tests for BaKCConfig main configuration class"""

    def test_bakc_config_creation(self):
        """Test BaKCConfig can be created with all sub-configs"""
        config = BaKCConfig(
            data=DataConfig(dataset_name="test"),
            model=ModelConfig(),
            ensemble=EnsembleConfig(),
            conformal=ConformalConfig(),
        )

        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.ensemble, EnsembleConfig)
        assert isinstance(config.conformal, ConformalConfig)
        assert config.save_models is True
        assert config.save_calibration is True
        assert config.save_predictions is True

    def test_load_default_yaml(self):
        """Test loading default.yaml configuration"""
        config = BaKCConfig.from_yaml('configs/default.yaml')

        # Check data config
        assert config.data.dataset_name == "cardio"
        assert config.data.train_fraction == 0.5

        # Check model config
        assert config.model.nu == 0.05
        assert config.model.kernel == "rbf"

        # Check ensemble config
        assert config.ensemble.num_models == 5
        assert config.ensemble.num_test_splits == 20
        assert config.ensemble.num_repetitions == 5
        assert config.ensemble.random_state == 42

        # Check conformal config
        assert config.conformal.alpha == 0.05
        assert config.conformal.scoring_method == "sigmoid"

    def test_load_cardio_yaml(self):
        """Test loading cardio.yaml configuration"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        # Check data config
        assert config.data.dataset_name == "cardio"
        assert str(config.data.output_dir) == "output/cardio"

        # Check critical baseline parameters are preserved
        assert config.model.nu == 0.05  # CRITICAL
        assert config.ensemble.num_models == 5  # CRITICAL
        assert config.ensemble.num_test_splits == 20  # CRITICAL
        assert config.ensemble.num_repetitions == 5  # CRITICAL
        assert config.ensemble.random_state == 42  # CRITICAL
        assert config.conformal.alpha == 0.05  # CRITICAL
        assert config.conformal.scoring_method == "sigmoid"  # CRITICAL
        assert config.conformal.quantile_method == "higher"  # CRITICAL
        assert config.conformal.fold_aggregation == "mean"  # CRITICAL
        assert config.conformal.cross_fold_aggregation == "median"  # CRITICAL

    def test_missing_yaml_file(self):
        """Test error when YAML file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            BaKCConfig.from_yaml('configs/nonexistent.yaml')

    def test_invalid_yaml_format(self):
        """Test error on malformed YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: {{{\n")
            temp_file = f.name

        try:
            with pytest.raises(Exception):  # yaml.YAMLError or similar
                BaKCConfig.from_yaml(temp_file)
        finally:
            os.unlink(temp_file)

    def test_empty_yaml_file(self):
        """Test error on empty YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")  # Empty file
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Empty or invalid YAML"):
                BaKCConfig.from_yaml(temp_file)
        finally:
            os.unlink(temp_file)


class TestConfigValidation:
    """Tests for configuration validation"""

    def test_validation_valid_config(self):
        """Test validation passes for valid configuration"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')
        config.validate()  # Should not raise

    def test_validation_invalid_alpha(self):
        """Test validation catches invalid alpha values"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        # Test alpha > 1
        config.conformal.alpha = 1.5
        with pytest.raises(ValueError, match="alpha must be in"):
            config.validate()

        # Test alpha <= 0
        config.conformal.alpha = 0.0
        with pytest.raises(ValueError, match="alpha must be in"):
            config.validate()

        # Test alpha < 0
        config.conformal.alpha = -0.1
        with pytest.raises(ValueError, match="alpha must be in"):
            config.validate()

    def test_validation_invalid_nu(self):
        """Test validation catches invalid nu values"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        # Test nu > 1
        config.model.nu = 1.5
        with pytest.raises(ValueError, match="nu must be in"):
            config.validate()

        # Test nu <= 0
        config.model.nu = 0.0
        with pytest.raises(ValueError, match="nu must be in"):
            config.validate()

    def test_validation_invalid_num_models(self):
        """Test validation catches invalid num_models"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        config.ensemble.num_models = 0
        with pytest.raises(ValueError, match="num_models must be positive"):
            config.validate()

        config.ensemble.num_models = -5
        with pytest.raises(ValueError, match="num_models must be positive"):
            config.validate()

    def test_validation_invalid_num_test_splits(self):
        """Test validation catches invalid num_test_splits"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        config.ensemble.num_test_splits = 0
        with pytest.raises(ValueError, match="num_test_splits must be positive"):
            config.validate()

    def test_validation_invalid_num_repetitions(self):
        """Test validation catches invalid num_repetitions"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        config.ensemble.num_repetitions = -1
        with pytest.raises(ValueError, match="num_repetitions must be positive"):
            config.validate()

    def test_validation_invalid_train_fraction(self):
        """Test validation catches invalid train_fraction"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        # Test train_fraction > 1
        config.data.train_fraction = 1.5
        with pytest.raises(ValueError, match="train_fraction must be in"):
            config.validate()

        # Test train_fraction <= 0
        config.data.train_fraction = 0.0
        with pytest.raises(ValueError, match="train_fraction must be in"):
            config.validate()

    def test_validation_invalid_scoring_method(self):
        """Test validation catches invalid scoring_method"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        config.conformal.scoring_method = "invalid_method"
        with pytest.raises(ValueError, match="scoring_method must be one of"):
            config.validate()

    def test_validation_invalid_kernel(self):
        """Test validation catches invalid kernel"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        config.model.kernel = "invalid_kernel"
        with pytest.raises(ValueError, match="kernel must be one of"):
            config.validate()

    def test_validation_invalid_fold_aggregation(self):
        """Test validation catches invalid fold_aggregation"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        config.conformal.fold_aggregation = "invalid"
        with pytest.raises(ValueError, match="fold_aggregation must be one of"):
            config.validate()

    def test_validation_invalid_cross_fold_aggregation(self):
        """Test validation catches invalid cross_fold_aggregation"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')

        config.conformal.cross_fold_aggregation = "invalid"
        with pytest.raises(ValueError, match="cross_fold_aggregation must be one of"):
            config.validate()

    def test_validation_creates_output_dir(self):
        """Test validation creates output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output"

            config = BaKCConfig.from_yaml('configs/cardio.yaml')
            config.data.output_dir = output_path

            assert not output_path.exists()
            config.validate()
            assert output_path.exists()
            assert output_path.is_dir()


class TestConfigSerialization:
    """Tests for configuration serialization (to_yaml)"""

    def test_config_round_trip(self):
        """Test loading, modifying, and saving configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name

        try:
            # Load config
            config = BaKCConfig.from_yaml('configs/cardio.yaml')

            # Modify some values
            config.model.nu = 0.1
            config.ensemble.num_models = 10

            # Save to temp file
            config.to_yaml(temp_file)

            # Load from temp file
            config2 = BaKCConfig.from_yaml(temp_file)

            # Check modified values persisted
            assert config2.model.nu == 0.1
            assert config2.ensemble.num_models == 10

            # Check other values remain the same
            assert config2.conformal.alpha == 0.05
            assert config2.ensemble.random_state == 42

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_config_repr(self):
        """Test configuration string representation"""
        config = BaKCConfig.from_yaml('configs/cardio.yaml')
        repr_str = repr(config)

        assert "BaKCConfig" in repr_str
        assert "cardio" in repr_str
        assert "0.05" in repr_str
        assert "5" in repr_str
        assert "42" in repr_str
