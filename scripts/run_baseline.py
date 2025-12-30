#!/usr/bin/env python
"""
Baseline verification script for BaKC-plus

This script runs the baseline experiment on the CARDIO dataset with exact
notebook configuration and compares results to baseline targets.

Expected Results (from notebook):
- Power: 90.29% (±2% tolerance)
- FDR: 8.47% (±2% tolerance)

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --dataset cardio --reps 5 --splits 20
    python scripts/run_baseline.py --output results/cardio_baseline.json
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from bakc_plus.config import ModelConfig, EnsembleConfig, ConformalConfig, DataConfig
from bakc_plus.data import DataLoader
from bakc_plus.pipeline import BaKCWorkflow
from bakc_plus.logger import get_logger

logger = get_logger(__name__)


def load_cardio_dataset(data_dir: str = "data/input") -> tuple:
    """
    Load CARDIO dataset

    Args:
        data_dir: Path to parent data directory containing cardio folder

    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    logger.info(f"Loading CARDIO dataset from {data_dir}")

    # Load dataset - DataLoader expects parent directory
    # It will construct path as data_dir / cardio / cardio.csv
    from pathlib import Path
    cardio_dir = Path(data_dir) / "cardio"

    config = DataConfig(dataset_name="cardio", data_dir=str(cardio_dir))
    loader = DataLoader(config)
    # Specify header=0 to read first row as column names
    df = loader.load_dataset("cardio", header=0)

    logger.info(f"Dataset loaded: shape={df.shape}")

    # Extract features and labels
    # Last column is the target (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Dataset info
    n_samples = len(y)
    n_features = X.shape[1]
    n_anomalies = np.sum(y == 1)
    anomaly_rate = 100.0 * n_anomalies / n_samples

    logger.info(
        f"Dataset: {n_samples} samples, {n_features} features, "
        f"{n_anomalies} anomalies ({anomaly_rate:.2f}%)"
    )

    return X, y


def run_baseline_experiment(
    dataset_name: str = "cardio",
    data_dir: str = "data/input",
    num_repetitions: int = 5,
    num_test_splits: int = 20,
    len_cal: int = 50,
    num_models: int = 5,
    alpha: float = 0.1,
    random_state: int = 42,
    output_file: str = None
) -> dict:
    """
    Run baseline experiment with exact notebook configuration

    Args:
        dataset_name: Dataset identifier
        data_dir: Path to data directory
        num_repetitions: Number of repetitions (J)
        num_test_splits: Number of test splits per repetition (L)
        len_cal: Calibration set size
        num_models: Number of ensemble members (M)
        alpha: Conformal prediction alpha (1 - coverage)
        random_state: Base random seed
        output_file: Optional path to save results

    Returns:
        Dictionary with experiment results and baseline comparison
    """
    logger.info("="*60)
    logger.info("BASELINE VERIFICATION EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Configuration:")
    logger.info(f"  num_repetitions (J) = {num_repetitions}")
    logger.info(f"  num_test_splits (L) = {num_test_splits}")
    logger.info(f"  len_cal = {len_cal}")
    logger.info(f"  num_models (M) = {num_models}")
    logger.info(f"  alpha = {alpha}")
    logger.info(f"  random_state = {random_state}")
    logger.info("="*60)

    # Load dataset
    X, y = load_cardio_dataset(data_dir)

    # Create configs
    model_config = ModelConfig(nu=0.05, kernel='rbf')
    ensemble_config = EnsembleConfig(num_models=num_models)
    conformal_config = ConformalConfig(alpha=alpha)

    # Create workflow
    workflow = BaKCWorkflow(
        model_config=model_config,
        ensemble_config=ensemble_config,
        conformal_config=conformal_config,
        num_repetitions=num_repetitions,
        num_test_splits=num_test_splits,
        len_cal=len_cal,
        test_size=0.2,
        random_state=random_state
    )

    # Run experiment
    logger.info("\nRunning experiment...")
    results = workflow.run_experiment(X, y)

    # Extract metrics
    power_mean = results['power_mean']
    power_std = results['power_std']
    power_p90 = results['power_p90']
    fdr_mean = results['fdr_mean']
    fdr_std = results['fdr_std']
    fdr_p90 = results['fdr_p90']

    # Baseline targets (from notebook)
    baseline_power = 0.9029  # 90.29%
    baseline_fdr = 0.0847    # 8.47%
    tolerance = 0.02         # ±2%

    # Check if within tolerance
    power_in_range = abs(power_mean - baseline_power) <= tolerance
    fdr_in_range = abs(fdr_mean - baseline_fdr) <= tolerance
    baseline_passed = power_in_range and fdr_in_range

    # Report results
    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Power: {power_mean:.4f} ± {power_std:.4f} (p90={power_p90:.4f})")
    logger.info(f"FDR:   {fdr_mean:.4f} ± {fdr_std:.4f} (p90={fdr_p90:.4f})")
    logger.info("")
    logger.info("Baseline Comparison:")
    logger.info(f"  Target Power: {baseline_power:.4f} ± {tolerance:.4f}")
    logger.info(f"  Actual Power: {power_mean:.4f} {'✓' if power_in_range else '✗'}")
    logger.info(f"  Target FDR:   {baseline_fdr:.4f} ± {tolerance:.4f}")
    logger.info(f"  Actual FDR:   {fdr_mean:.4f} {'✓' if fdr_in_range else '✗'}")
    logger.info("")
    logger.info(f"Baseline Status: {'PASSED ✓' if baseline_passed else 'FAILED ✗'}")
    logger.info("="*60)

    # Prepare output
    output = {
        'dataset': dataset_name,
        'configuration': {
            'num_repetitions': num_repetitions,
            'num_test_splits': num_test_splits,
            'len_cal': len_cal,
            'num_models': num_models,
            'alpha': alpha,
            'random_state': random_state,
        },
        'results': {
            'power_mean': float(power_mean),
            'power_std': float(power_std),
            'power_p90': float(power_p90),
            'power_per_rep': results['power_per_rep'].tolist(),
            'fdr_mean': float(fdr_mean),
            'fdr_std': float(fdr_std),
            'fdr_p90': float(fdr_p90),
            'fdr_per_rep': results['fdr_per_rep'].tolist(),
        },
        'baseline': {
            'target_power': baseline_power,
            'target_fdr': baseline_fdr,
            'tolerance': tolerance,
            'power_in_range': power_in_range,
            'fdr_in_range': fdr_in_range,
            'passed': baseline_passed,
        }
    }

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to: {output_path}")

    return output


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run baseline verification experiment on CARDIO dataset"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cardio',
        help='Dataset name (default: cardio)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/input',
        help='Data directory path (default: data/input)'
    )
    parser.add_argument(
        '--reps',
        type=int,
        default=5,
        help='Number of repetitions (default: 5)'
    )
    parser.add_argument(
        '--splits',
        type=int,
        default=20,
        help='Number of test splits per repetition (default: 20)'
    )
    parser.add_argument(
        '--len-cal',
        type=int,
        default=50,
        help='Calibration set size (default: 50)'
    )
    parser.add_argument(
        '--num-models',
        type=int,
        default=5,
        help='Number of ensemble members (default: 5)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Conformal prediction alpha (default: 0.1)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/cardio_baseline.json',
        help='Output file path (default: results/cardio_baseline.json)'
    )

    args = parser.parse_args()

    # Run experiment
    output = run_baseline_experiment(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        num_repetitions=args.reps,
        num_test_splits=args.splits,
        len_cal=args.len_cal,
        num_models=args.num_models,
        alpha=args.alpha,
        random_state=args.seed,
        output_file=args.output
    )

    # Exit code based on baseline status
    if output['baseline']['passed']:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
