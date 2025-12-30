#!/usr/bin/env python3
"""
BaKC-plus: Bagging with Conformal Prediction for Anomaly Detection
===================================================================
This script implements One-Class SVM with bagging and conformal prediction
for anomaly detection. It serves as a baseline reference implementation.

Original source: oc-svm-x-cv-x-bagging (1).ipynb
"""

import os
import sys
import logging
import shutil
import pickle
import pickle as pkl
from functools import partial
from statistics import stdev
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import OneClassSVM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
np.set_printoptions(threshold=sys.maxsize)

# Paths - adapted for local execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'input')
WORKDIR_PATH = os.path.join(BASE_DIR, 'output')
CALIB_PATH_PREFIX = 'calib'

# Dataset configuration
DATASET_FOLDER = 'gamma'
DATASET_FILE = 'gamma.csv'

# Model parameters
NUM_MODELS = 5
J, L = 5, 20  # J: number of random states, L: number of splits


def setup_directories():
    """Create necessary directories for model storage."""
    os.makedirs(WORKDIR_PATH, exist_ok=True)
    os.makedirs(os.path.join(WORKDIR_PATH, 'models'), exist_ok=True)
    os.makedirs(os.path.join(WORKDIR_PATH, 'calib', 'models'), exist_ok=True)
    os.makedirs(os.path.join(WORKDIR_PATH, 'train_artifacts'), exist_ok=True)
    logger.info(f"Directories set up at {WORKDIR_PATH}")


def delete_model_folder(folder_name='models'):
    """Delete model folder if it exists."""
    folder_path = os.path.join(WORKDIR_PATH, folder_name)
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        logger.info(f"Deleted folder: {folder_path}")


def create_model_folder(folder_name='models'):
    """Create model folder."""
    folder_path = os.path.join(WORKDIR_PATH, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    logger.info(f"Created folder: {folder_path}")


def load_dataset():
    """Load and preprocess the dataset."""
    dataset_path = os.path.join(DATA_DIR, DATASET_FOLDER, DATASET_FILE)
    logger.info(f"Loading dataset from: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df.rename(columns={'Class': 'y'}, inplace=True)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Dataset columns: {list(df.columns)}")
    logger.info(f"Dataset head:\n{df.head()}")
    logger.info(f"Total records: {len(df)}")

    return df


def split_inliers_outliers(df):
    """Split dataset into inliers and outliers."""
    inliers_df = df.loc[df['y'] == 0]
    outliers_df = df.loc[df['y'] == 1]

    logger.info(f"Inliers count: {len(inliers_df)}")
    logger.info(f"Outliers count: {len(outliers_df)}")

    return inliers_df, outliers_df


def compute_split_sizes(inliers_df):
    """Compute train/calibration/test split sizes."""
    len_train = len(inliers_df) // 2
    len_cal = min(2000, len_train // 2)
    len_test = min(2000, len_train // 3)
    outlier_pcent = 0.1
    len_test_outliers = round(outlier_pcent * len_test)
    len_test_inliers = round((1 - outlier_pcent) * len_test)

    logger.info(f"Split sizes - Train: {len_train}, Calibration: {len_cal}, "
                f"Test: {len_test}, Test outliers: {len_test_outliers}, "
                f"Test inliers: {len_test_inliers}")

    return len_train, len_cal, len_test, len_test_outliers, len_test_inliers


# Scoring functions
def scoring_function(x, m, n):
    """Custom scoring function for calibration."""
    return m**2 * (1 / (1 + x**2)**2) + n**2 * (x + n**2 - 5)**2


def normalize_scored_sample(calib):
    """Normalize calibration scores to [0, 1]."""
    minval, maxval = calib.min(), calib.max()
    return (calib - minval) / (maxval - minval)


def normalize_scored_sample_2(calib, ground_truth):
    """Normalize calibration scores with ground truth adjustment."""
    sgn_calib = np.where(calib < 0, -1, 1)
    ground_truth = np.where(ground_truth == 1, -1, 1)
    m = (sgn_calib + ground_truth)
    n = (sgn_calib - ground_truth)
    return 1 - normalize_scored_sample(scoring_function(calib, m, n))


def sigmoid_scored_sample(calib):
    """Apply sigmoid to calibration scores."""
    return 1 / (1 + np.exp(calib))


def signed_OHE_scored_sample(calib):
    """Convert calibration scores to binary predictions."""
    return np.where(calib < 0, 1, 0)


def score_samples(model, calib):
    """Score samples using model decision function (normalized)."""
    proba = model.decision_function(calib)
    return normalize_scored_sample(proba)


def score_samples_2(model, calib, ground_truth):
    """Score samples with ground truth adjustment."""
    proba = model.decision_function(calib)
    return normalize_scored_sample_2(proba, ground_truth)


def score_samples_3(model, calib, ground_truth):
    """Score samples using sigmoid transformation."""
    proba = model.decision_function(calib)
    return sigmoid_scored_sample(proba)


def score_samples_baseline(model, test, plot=False):
    """Score samples for baseline evaluation."""
    proba = model.decision_function(test)
    return signed_OHE_scored_sample(proba)


def perform_bootstrapping(X_train, member_idx, num_members, rnd):
    """Perform leave-out bootstrapping for ensemble member."""
    rnd_state = np.random.RandomState(rnd)
    indices = np.arange(len(X_train))
    rnd_state.shuffle(indices)
    index_sets = np.array_split(indices, num_members)
    leave_out_indices = index_sets[member_idx]
    mask = np.ones_like(indices, dtype=bool)
    mask[leave_out_indices] = False
    X_train_bootstrap = X_train[mask]
    return X_train_bootstrap, leave_out_indices


def fit_OCSVM_member(member_idx=0, num_members=None, fold_idx=0, X_train=None,
                     random_state=42, save=True):
    """Fit a One-Class SVM ensemble member."""
    if X_train is None:
        return None

    model = OneClassSVM(nu=0.05)
    rnd = hash((member_idx, fold_idx, random_state)) % 4294967296
    rnd = rnd ^ 0x7FFFFFFF

    if num_members is not None:
        X_train_bootstrap, leave_out_indices = perform_bootstrapping(
            X_train, member_idx, num_members, rnd)
    else:
        X_train_bootstrap = X_train
        leave_out_indices = None

    model.fit(X_train_bootstrap)

    if save:
        model_path = os.path.join(WORKDIR_PATH, 'calib', 'models',
                                   f'model_{fold_idx}_{member_idx}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    if leave_out_indices is not None:
        return model, leave_out_indices
    return model


def create_calibration_sets(train, random_state, len_cal, num_models):
    """Create calibration sets using K-Fold cross-validation."""
    calibration_scores = np.array([], dtype=np.float16)
    calibration_scores_std = np.array([], dtype=np.float16)
    models = []

    len_splits = len(train) // len_cal if len(train) < 20000 else 20
    kf = KFold(n_splits=len_splits, shuffle=True, random_state=random_state)
    X_train = train.drop('y', axis=1).to_numpy()
    ground_truth = train['y'].to_numpy()

    # Initialize tracking dictionaries
    visited = {}
    visited_i = {}

    logger.info(f"Creating calibration sets with {len_splits} splits")

    for i, (train_index, calib_index) in tqdm(enumerate(kf.split(train)),
                                               total=len_splits, desc="K-Fold"):
        calibration_scores_i = np.array([], dtype=np.float16)
        calibration_scores_i_leave_out = np.array([], dtype=np.float16)

        for j in range(num_models):
            model, leave_out_indices = fit_OCSVM_member(
                j, num_models, i, X_train[train_index], random_state)

            for idx in calib_index:
                if idx not in visited:
                    visited[idx] = 1
                else:
                    visited[idx] = visited[idx] + 1

            for idx in leave_out_indices:
                if idx not in visited_i:
                    visited_i[idx] = 1
                else:
                    visited_i[idx] = visited_i[idx] + 1

            calibration_scores_i_j = score_samples_3(
                model, X_train[calib_index], ground_truth[calib_index])
            calibration_scores_i = (np.vstack([calibration_scores_i, calibration_scores_i_j])
                                    if calibration_scores_i.shape[0] > 0
                                    else calibration_scores_i_j)

            calibration_scores_leave_out_j = score_samples_3(
                model, X_train[leave_out_indices], ground_truth[leave_out_indices])
            calibration_scores_i_leave_out = np.append(
                calibration_scores_i_leave_out, calibration_scores_leave_out_j)

            models.append(model)

        calibration_scores_i_std = np.std(calibration_scores_i, axis=1)
        calibration_scores_i = np.mean(calibration_scores_i, axis=1)

        calibration_scores = np.append(calibration_scores, calibration_scores_i)
        calibration_scores = np.append(calibration_scores, calibration_scores_i_leave_out)
        calibration_scores_std = np.append(calibration_scores_std, calibration_scores_i_std)

    logger.info(f"Unique visit counts: {set(visited_i.values())}")

    return (models, calibration_scores, calibration_scores_std)


def add_noise_to_dataframe(df, noise_level=0.1, clip=True):
    """Add uniform noise to dataframe values."""
    df = df.copy()
    for col in df.columns:
        noise = np.random.uniform(low=-1*noise_level, high=noise_level, size=len(df))
        df[col] += noise
        if clip:
            df[col] = np.clip(df[col], 0, 1)
    return df


def count_less_equal(C, T):
    """Count values in C less than or equal to each element in T."""
    sorted_C = np.sort(C)
    counts = np.searchsorted(sorted_C, T)
    return counts


def count_more_equal(C, T):
    """Count values in C greater than or equal to each element in T."""
    sorted_C = np.sort(C)
    counts = len(C) - np.searchsorted(sorted_C, T, side='right')
    return counts


def control_false_positives(p, alpha=0.2):
    """Control false positives using Benjamini-Hochberg procedure."""
    mask = stats.false_discovery_control(p, method='bh') < alpha
    mask = mask.astype(int)
    return mask


def get_p_values_for_X(models, X_test, ground_truth, qhat,
                       scoring_function=score_samples_3, plot_scores=False):
    """Get p-values for test samples."""
    if X_test.shape[0] <= 0:
        return None

    scores = np.stack([scoring_function(model, X_test, ground_truth)
                       for model in models], axis=1)
    scores = np.median(scores, axis=1)
    p_values = (scores > qhat).astype(int)
    return p_values


def get_power(p_values, ground_truth):
    """Calculate statistical power (true positive rate)."""
    if p_values is None:
        return None
    true_positives = np.sum(p_values == ground_truth)
    false_negatives = np.sum((1 - p_values) == ground_truth)
    power = true_positives / (true_positives + false_negatives)
    return power


def get_fdr(p_values, ground_truth):
    """Calculate false discovery rate."""
    if p_values is None:
        return None
    true_positives = np.sum(p_values == ground_truth)
    false_positives = np.sum(p_values == (1 - ground_truth))
    fdr = false_positives / (true_positives + false_positives)
    return fdr


def test_impl(models, calibration_scores, calibration_scores_std,
              inliers, outliers, random_state, alpha=0.05, plot=None):
    """Test implementation for a single fold."""
    inliers = inliers.copy()
    outliers = outliers.copy()

    inliers_in_test_df = inliers.sample(n=round(0.9*len(inliers)), random_state=random_state)

    test_df = pd.concat([inliers, outliers], ignore_index=True)

    n = len(calibration_scores)
    q_level = np.ceil((n+1)*(1-alpha))/n
    qhat = np.quantile(calibration_scores, q_level, method='higher')
    sigmahat = np.quantile(calibration_scores_std, q_level, method='higher')

    ground_truth = test_df['y'].values
    X_test = test_df.drop(['y'], axis=1).to_numpy()

    inliers_ground_truth = inliers['y'].values
    X_inliers = inliers.drop(['y'], axis=1).to_numpy()

    outliers_ground_truth = outliers['y'].values
    X_outliers = outliers.drop(['y'], axis=1).to_numpy()

    inlier_p_values = get_p_values_for_X(models, X_inliers, inliers_ground_truth, qhat)
    outlier_p_values = get_p_values_for_X(models, X_outliers, outliers_ground_truth, qhat)

    inliers['y_pred'] = inlier_p_values
    outliers['y_pred'] = outlier_p_values

    power_new = get_power(outlier_p_values, outliers_ground_truth)
    fdr_new = get_fdr(inlier_p_values, inliers_ground_truth)

    return power_new, fdr_new, inliers, outliers


def train_impl(random_state_offset, inliers_df, outliers_df, len_train, len_cal,
               num_models, L, save=False, load_saved=False):
    """Training implementation for a single random state."""
    random_state = 42 + random_state_offset

    train_df, test_df = train_test_split(inliers_df, test_size=len_train,
                                          random_state=random_state)

    inliers_test_df_shuffled = test_df.sample(n=len(test_df), random_state=random_state)
    outliers_df_shuffled = outliers_df.sample(n=len(outliers_df), random_state=random_state)

    inliers_indices = np.array_split(inliers_test_df_shuffled.index, L)
    outliers_indices = np.array_split(outliers_df_shuffled.index, L)

    if load_saved:
        dataset_name = os.path.splitext(DATASET_FILE)[0]
        folder_name = os.path.join(WORKDIR_PATH, 'train_artifacts', dataset_name)

        with open(os.path.join(folder_name, 'models.pkl'), 'rb') as f_models:
            models = pkl.load(f_models)
        with open(os.path.join(folder_name, 'calibration_scores.pkl'), 'rb') as f_cal:
            calibration_scores = pkl.load(f_cal)
        with open(os.path.join(folder_name, 'calibration_scores_std.pkl'), 'rb') as f_std:
            calibration_scores_std = pkl.load(f_std)
    else:
        logger.info(f"Training with {len(train_df)} samples")
        models, calibration_scores, calibration_scores_std = create_calibration_sets(
            train_df, random_state, len_cal, num_models)

    if save and not load_saved:
        dataset_name = os.path.splitext(DATASET_FILE)[0]
        folder_name = os.path.join(WORKDIR_PATH, 'train_artifacts', dataset_name)
        os.makedirs(folder_name, exist_ok=True)

        with open(os.path.join(folder_name, 'models.pkl'), 'wb') as f_models:
            pkl.dump(models, f_models)
        with open(os.path.join(folder_name, 'calibration_scores.pkl'), 'wb') as f_cal:
            pkl.dump(calibration_scores, f_cal)
        with open(os.path.join(folder_name, 'calibration_scores_std.pkl'), 'wb') as f_std:
            pkl.dump(calibration_scores_std, f_std)
        return None

    res = []
    for l in tqdm(range(L), desc=f"Testing splits (rs={random_state_offset})"):
        inliers_df_i = test_df.loc[inliers_indices[l]].reset_index(drop=True)
        outliers_df_i = outliers_df.loc[outliers_indices[l]].reset_index(drop=True)

        plot = True if l == 0 else None
        res.append(test_impl(models, calibration_scores, calibration_scores_std,
                             inliers=inliers_df_i, outliers=outliers_df_i,
                             random_state=l, plot=plot))

    return res


def evaluate_baseline(inliers_df, outliers_df, len_train, len_test_outliers, len_test_inliers):
    """Evaluate baseline One-Class SVM (no bagging, no conformal)."""
    logger.info("Evaluating baseline One-Class SVM...")

    random_state = 42
    train_df, test_df = train_test_split(inliers_df, test_size=len_train,
                                          random_state=random_state)

    outliers_in_train_df = outliers_df.sample(n=len_test_outliers, random_state=random_state)
    outliers_in_test_df = outliers_df.drop(outliers_in_train_df.index)

    inliers_in_test_df = test_df.sample(n=len_test_inliers, random_state=101)

    test_df = pd.concat([test_df, outliers_df], ignore_index=True)

    train_data = train_df.drop(['y'], axis=1).to_numpy()
    model = fit_OCSVM_member(X_train=train_data, random_state=random_state, save=False)

    ground_truth = test_df['y'].values

    X_inliers = test_df.drop(['y'], axis=1).to_numpy()
    inliers_ground_truth = test_df['y'].values

    X_outliers = outliers_df.drop(['y'], axis=1).to_numpy()
    outliers_ground_truth = outliers_df['y'].values

    inlier_p_values = score_samples_baseline(model, X_inliers)
    outlier_p_values = score_samples_baseline(model, X_outliers)

    power_new = get_power(outlier_p_values, outliers_ground_truth)
    fdr_new = get_fdr(inlier_p_values, inliers_ground_truth)

    logger.info(f"Baseline Results - Power: {power_new:.4f}, FDR: {fdr_new:.4f}")

    return power_new, fdr_new


def run_parallel_training(inliers_df, outliers_df, len_train, len_cal, num_models, J, L):
    """Run parallel training across multiple random states."""
    logger.info(f"Starting parallel training with J={J} random states, L={L} splits")

    # Create partial function with fixed arguments
    train_func = partial(train_impl,
                         inliers_df=inliers_df.copy(),
                         outliers_df=outliers_df.copy(),
                         len_train=len_train,
                         len_cal=len_cal,
                         num_models=num_models,
                         L=L)

    # Run in parallel
    j = range(J)
    with Pool() as pool:
        res = list(tqdm(pool.imap_unordered(train_func, j), total=J, desc="Training"))

    return res


def aggregate_results(res):
    """Aggregate results from parallel training runs."""
    power_list = []
    fdr_list = []
    inlier_df_list = []
    outlier_df_list = []

    for j in range(len(res)):
        if res[j] is None:
            continue
        for l in range(len(res[j])):
            if res[j][l] is None:
                continue
            if res[j][l][0] is not None:
                power_list.append(res[j][l][0])
            if res[j][l][1] is not None:
                fdr_list.append(res[j][l][1])
            if res[j][l][2] is not None:
                inlier_df_list.append(res[j][l][2])
            if res[j][l][3] is not None:
                outlier_df_list.append(res[j][l][3])

    logger.info(f"Collected {len(power_list)} power values and {len(fdr_list)} FDR values")

    return power_list, fdr_list, inlier_df_list, outlier_df_list


def print_summary_statistics(power_list, fdr_list):
    """Print summary statistics for power and FDR."""
    if len(power_list) > 0 and len(fdr_list) > 0:
        q90_power = np.quantile(power_list, 0.9)
        q90_fdr = np.quantile(fdr_list, 0.9)
        avg_power = sum(power_list) / len(power_list)
        avg_fdr = sum(fdr_list) / len(fdr_list)
        std_power = stdev(power_list)
        std_fdr = stdev(fdr_list)

        logger.info("=" * 60)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 60)
        logger.info("------------------- POWER -------------------")
        logger.info(f"Average Power:       {avg_power:.6f}")
        logger.info(f"90th Quantile Power: {q90_power:.6f}")
        logger.info(f"Std Power:           {std_power:.6f}")
        logger.info("--------------- FALSE DISCOVERY RATE -----------------")
        logger.info(f"Average FDR:         {avg_fdr:.6f}")
        logger.info(f"90th Quantile FDR:   {q90_fdr:.6f}")
        logger.info(f"Std FDR:             {std_fdr:.6f}")
        logger.info("=" * 60)

        return {
            'avg_power': avg_power,
            'q90_power': q90_power,
            'std_power': std_power,
            'avg_fdr': avg_fdr,
            'q90_fdr': q90_fdr,
            'std_fdr': std_fdr
        }

    return None


def save_predictions(inlier_df_list, outlier_df_list):
    """Save prediction results to CSV."""
    if inlier_df_list and outlier_df_list:
        inlier_df_pred = pd.concat(inlier_df_list, axis=0, ignore_index=True)
        outlier_df_pred = pd.concat(outlier_df_list, axis=0, ignore_index=True)
        test_df_pred = pd.concat([inlier_df_pred, outlier_df_pred], axis=0, ignore_index=True)

        predictions_path = os.path.join(WORKDIR_PATH, 'predictions.csv')
        test_df_pred.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to: {predictions_path}")

        logger.info(f"Inlier predictions distribution:\n{inlier_df_pred['y_pred'].value_counts()}")
        logger.info(f"Outlier predictions distribution:\n{outlier_df_pred['y_pred'].value_counts()}")

        return test_df_pred

    return None


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("BaKC-plus: Bagging with Conformal Prediction")
    logger.info("=" * 60)

    # Setup
    setup_directories()
    delete_model_folder('calib/models')
    create_model_folder('calib/models')

    # Load data
    df = load_dataset()
    inliers_df, outliers_df = split_inliers_outliers(df)

    # Compute split sizes
    len_train, len_cal, len_test, len_test_outliers, len_test_inliers = compute_split_sizes(inliers_df)

    # K-Fold demonstration
    logger.info("\nK-Fold split demonstration:")
    kf = KFold(n_splits=20, shuffle=True, random_state=42)
    for i, (train_index, calib_index) in enumerate(kf.split(df)):
        if i == 0:
            logger.info(f"Fold {i}: Train size = {len(train_index)}, Calib size = {len(calib_index)}")

    # Prepare arrays
    inliers_df_X = inliers_df.drop('y', axis=1)
    inliers_df_y = inliers_df['y']
    outliers_df_X = outliers_df.drop('y', axis=1)
    outliers_df_y = outliers_df['y']

    logger.info(f"\nInliers X shape: {inliers_df_X.shape}")
    logger.info(f"Outliers X shape: {outliers_df_X.shape}")
    logger.info(f"Outliers y head:\n{outliers_df_y.head()}")

    # Run parallel training
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)

    res = run_parallel_training(inliers_df, outliers_df, len_train, len_cal,
                                NUM_MODELS, J, L)

    # Aggregate results
    power_list, fdr_list, inlier_df_list, outlier_df_list = aggregate_results(res)

    # Print summary
    stats = print_summary_statistics(power_list, fdr_list)

    # Save predictions
    test_df_pred = save_predictions(inlier_df_list, outlier_df_list)

    # Save metrics
    if stats:
        metrics_path = os.path.join(WORKDIR_PATH, 'metrics.csv')
        metrics_df = pd.DataFrame([stats])
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to: {metrics_path}")

    # Evaluate baseline
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE EVALUATION")
    logger.info("=" * 60)
    baseline_power, baseline_fdr = evaluate_baseline(
        inliers_df, outliers_df, len_train, len_test_outliers, len_test_inliers)

    logger.info("\n" + "=" * 60)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 60)

    return stats, (baseline_power, baseline_fdr)


if __name__ == "__main__":
    main()
