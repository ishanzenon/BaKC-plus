# -*- coding: utf-8 -*-
"""BaKC+ Methodology Implementation for Anomaly Detection

This script implements the BaKC+ methodology for one-class anomaly detection,
as described in the NeurIPS 2025 paper: "On the Integration of
Cross-Conformal Prediction, Ensembles, and Bootstrapping for Uncertainty
Quantification in One-Class Anomaly Detection."

The method combines K-fold cross-validation based cross-conformal prediction
with ensembles and bootstrapping for anomaly detection. It provides statistical
guarantees for Type-I error control while maintaining high detection power.

Implementation Details:
- Base estimator: OneClassSVM
- Datasets: ADBench (Shuttle, Mammography, Cardio, Gamma, Musk, and Fraud)
- Evaluation metrics: Statistical power and false discovery rate (FDR)

Baseline approaches for comparison:
- Vanilla OneClassSVM
- Autoencoders (Note: Requires 'build_autoencoder' function and TensorFlow/Keras)
"""

## Environment Setup Instructions
# ---------------------------------
# To run this script, ensure you have Python 3.x installed.
# Create a virtual environment and install the required packages:
#
# python -m venv bakc_env
# source bakc_env/bin/activate  # On Windows: bakc_env\Scripts\activate
# pip install numpy pandas scikit-learn matplotlib scipy tqdm
#
# (Optional, for Autoencoder baseline):
# pip install tensorflow # If you intend to implement and use the autoencoder baseline
#
# Data Preparation:
# 1. Download the ADBench datasets.
# 2. Create a 'data' directory in the root of this project.
# 3. For each dataset (e.g., 'gamma'), create a subdirectory inside 'data'
#    (e.g., 'data/gamma/') and place the corresponding CSV file
#    (e.g., 'data/gamma/gamma.csv') there.
#
# Output Directories:
# The script will create an 'output' directory in the root for storing
# trained models ('output/models') and other artifacts ('output/artifacts').
# ---------------------------------

# Import necessary libraries
import os
import shutil
import sys
import pickle
from functools import partial
from statistics import stdev, mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import OneClassSVM
# from sklearn.neural_network import MLPClassifier # Not used in the final script, can be removed if confirmed
from sklearn.metrics import accuracy_score # Not used, can be removed
# from sklearn.metrics import log_loss # Not used, can be removed
from scipy import stats
from tqdm import tqdm
from multiprocessing import Pool

# TensorFlow/Keras imports for Autoencoder (uncomment if using)
# import tensorflow as tf
# from keras.models import Model, Sequential
# from keras.layers import Input, Dense
# from keras.optimizers import Adam

# Configuration Options
np.set_printoptions(threshold=sys.maxsize) # May cause performance issues with large arrays, consider if necessary

# Define base paths for data and outputs
BASE_DATA_PATH = 'data'
BASE_OUTPUT_PATH = 'output'
MODELS_DIR = os.path.join(BASE_OUTPUT_PATH, 'models', 'calib_models') # More specific path for calibration models
ARTIFACTS_DIR = os.path.join(BASE_OUTPUT_PATH, 'artifacts')

## Data Loading and Preprocessing
def load_and_preprocess_data(dataset_name):
    """
    Loads and preprocesses the specified ADBench dataset.
    Assumes data is in 'BASE_DATA_PATH/dataset_name/dataset_name.csv'.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'gamma', 'fraud', etc.).

    Returns:
        pandas.DataFrame: Preprocessed DataFrame containing features and labels.
                          Returns None if the file is not found.
    """
    # Corrected path construction for dataset name like "Shuttle" vs "shuttle.csv"
    # Assuming dataset_name is like "Shuttle", "Mammography"
    actual_dataset_file_name = dataset_name.lower() + '.csv' # Or derive from a mapping if names differ significantly
    data_path = os.path.join(BASE_DATA_PATH, dataset_name, actual_dataset_file_name)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the ADBench datasets are downloaded and placed in the 'data' directory as per instructions.")
        return None

    df = pd.read_csv(data_path)
    # Standardize column name for target variable, common issue with ADBench
    if 'class' in df.columns and 'y' not in df.columns:
        df.rename(columns={'class': 'y'}, inplace=True)
    elif 'Class' in df.columns and 'y' not in df.columns:
        df.rename(columns={'Class': 'y'}, inplace=True)
    
    # Ensure 'y' column exists
    if 'y' not in df.columns:
        print(f"Error: Target column 'y' (or 'Class'/'class') not found in {dataset_name}.")
        return None
        
    return df

## Helper Functions for Visualization and Data Manipulation
def add_jitter(arr, jitter_strength=0.02):
    """Adds jitter to the input array."""
    return arr + np.random.normal(0, jitter_strength, arr.shape)

def plot_df(df, x_idx, y_idx, y_col='y'):
    """Plots anomalies given a dataframe"""
    if y_col not in df.columns:
        print(f"Error: y_col '{y_col}' not found in DataFrame for plotting.")
        return
    X_plot = df.drop(y_col, axis=1)
    y_plot = df[y_col]

    normal_indices = np.where(y_plot == 0)[0]
    anomaly_indices = np.where(y_plot == 1)[0]

    X_jittered = X_plot.copy()
    # Ensure x_idx and y_idx are valid column indices
    if x_idx >= X_plot.shape[1] or y_idx >= X_plot.shape[1]:
        print("Error: x_idx or y_idx out of bounds for plotting.")
        return

    X_jittered.iloc[:, x_idx] = add_jitter(X_plot.iloc[:, x_idx]) # Use specific column for jitter
    X_jittered.iloc[:, y_idx] = add_jitter(X_plot.iloc[:, y_idx]) # Use specific column for jitter

    plt.figure(figsize=(8, 6))
    plt.scatter(X_plot.iloc[normal_indices, x_idx], X_jittered.iloc[normal_indices, y_idx], c='b', label='Normal', alpha=0.5, edgecolor='k')
    plt.scatter(X_plot.iloc[anomaly_indices, x_idx], X_jittered.iloc[anomaly_indices, y_idx], c='r', label='Anomaly', alpha=0.5, edgecolor='k')

    plt.title(f'Global Anomalies for feature V{X_plot.columns[x_idx]} and V{X_plot.columns[y_idx]}')
    plt.xlabel(f'V{X_plot.columns[x_idx]}')
    plt.ylabel(f'V{X_plot.columns[y_idx]}')
    plt.legend()
    plt.show()

def add_noise_to_dataframe(df, noise_level=0.1, clip=True):
    """Adds uniform random noise to each column of the DataFrame."""
    df_noisy = df.copy()
    for col in df_noisy.columns:
        if pd.api.types.is_numeric_dtype(df_noisy[col]): # Add noise only to numeric columns
            noise = np.random.uniform(low=-1 * noise_level, high=noise_level, size=len(df_noisy))
            df_noisy[col] += noise
            if clip:
                # Assuming data is scaled [0,1]. If not, clipping range might need adjustment.
                df_noisy[col] = np.clip(df_noisy[col], 0, 1)
    return df_noisy

## BaKC+ Implementation
def perform_bootstrapping(X_train_fold, member_idx, num_members_ensemble, rnd_seed_bootstrap):
    """
    Performs bootstrapping to create a subset of the training data for an ensemble member.

    Args:
        X_train_fold (numpy.ndarray): Training data for the current fold.
        member_idx (int): Index of the ensemble member.
        num_members_ensemble (int): Total number of ensemble members.
        rnd_seed_bootstrap (int): Random seed for reproducibility of this bootstrap sample.

    Returns:
        tuple: Bootstrapped training data (X_train_bootstrap) and indices of the
               left-out samples relative to X_train_fold (leave_out_indices).
    """
    rnd_state = np.random.RandomState(rnd_seed_bootstrap)
    n_samples = len(X_train_fold)
    
    # Standard bootstrapping: sample with replacement
    bootstrap_indices = rnd_state.choice(np.arange(n_samples), size=n_samples, replace=True)
    X_train_bootstrap = X_train_fold[bootstrap_indices]
    
    # Identify out-of-bag samples (left-out)
    all_indices = np.arange(n_samples)
    leave_out_indices = np.setdiff1d(all_indices, np.unique(bootstrap_indices))
    
    return X_train_bootstrap, leave_out_indices


def fit_OCSVM_member(member_idx=0, num_members_ensemble=None, fold_idx=0, X_train_fold=None, random_state_fold=42, save_model=True):
    """
    Fits a OneClassSVM model to a bootstrapped subset of the training data.

    Args:
        member_idx (int): Index of the ensemble member.
        num_members_ensemble (int): Total number of ensemble members. If None, no bootstrapping is done.
        fold_idx (int): Index of the K-fold cross-validation fold.
        X_train_fold (numpy.ndarray): Training data for the current fold.
        random_state_fold (int): Base random seed for this fold.
        save_model (bool): Whether to save the trained model.

    Returns:
        tuple: (Fitted OneClassSVM model, indices of the left-out samples (if bootstrapped else None))
    """
    if X_train_fold is None or len(X_train_fold) == 0:
        print("Warning: X_train_fold is None or empty in fit_OCSVM_member.")
        return None, None

    model = OneClassSVM(nu=0.05, kernel="rbf", gamma='scale') # Common defaults

    # Consistent random seed for bootstrapping based on member, fold, and base random state
    # Ensures that if num_members_ensemble is used, the bootstrap sample is reproducible
    rnd_seed_bootstrap = hash((member_idx, fold_idx, random_state_fold)) % (2**32 -1)

    if num_members_ensemble is not None and num_members_ensemble > 0:
        X_train_bootstrap, leave_out_indices = perform_bootstrapping(X_train_fold, member_idx, num_members_ensemble, rnd_seed_bootstrap)
    else:
        X_train_bootstrap = X_train_fold
        leave_out_indices = None # No leave-out set if not bootstrapping

    if len(X_train_bootstrap) == 0:
        print(f"Warning: Bootstrapped training data is empty for member {member_idx}, fold {fold_idx}.")
        return None, leave_out_indices # Cannot fit model on empty data

    model.fit(X_train_bootstrap)

    if save_model:
        # Ensure MODELS_DIR exists
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f'model_fold{fold_idx}_member{member_idx}.pkl')
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Error saving model {model_path}: {e}")


    return model, leave_out_indices


def manage_output_folders():
    """Creates or clears necessary output folders."""
    # Clear and create models directory
    if os.path.isdir(MODELS_DIR):
        shutil.rmtree(MODELS_DIR)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Create artifacts directory if it doesn't exist
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    # Create dataset-specific artifact subdirectories later, as needed

def normalize_scores(scores):
    """Normalizes scores to the range [0, 1]. Higher original scores become closer to 1."""
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val == min_val: # Avoid division by zero if all scores are the same
        return np.zeros_like(scores) if max_val == 0 else np.ones_like(scores) * 0.5 # Or handle as appropriate
    return (scores - min_val) / (max_val - min_val)

def score_to_conformity(scores):
    """
    Converts raw scores (e.g., OCSVM decision_function) to conformity scores.
    Higher decision_function scores (more normal) should map to higher conformity.
    This implementation uses a sigmoid-like transformation if scores can be negative,
    or simple normalization if scores are already in a non-decreasing range for normality.
    The paper should clarify the exact transformation from raw scores to conformity scores.
    Assuming decision_function where higher values are "more normal".
    We want high conformity for normal, low for anomalous.
    This function directly returns the decision_function values,
    as they are often used as non-conformity scores where lower means more anomalous.
    The BaKC+ paper might define a specific transformation for conformity scores.
    For OCSVM, decision_function values are such that positive values are inliers.
    If using these as *non-conformity* scores, they should be negated.
    Let's assume `model.decision_function(X)` gives scores where higher = more normal.
    A common non-conformity score is `-model.decision_function(X)`.
    The paper describes using scores from `decision_function`.
    The current `score_samples_3` uses `1 / (1 + np.exp(calib))` which means
    original `calib` (decision_function output) values that are higher (more normal)
    will result in smaller values from the sigmoid (closer to 0).
    This seems to be treating raw scores as "normality" scores, and then transforming
    them into something where higher values might indicate anomaly if qhat is a lower bound.
    Let's clarify: For CP, non-conformity scores are needed. If `decision_function` gives
    normality scores (higher = more normal), then non-conformity = `-decision_function` or `1 - normalized_decision_function`.
    The `score_samples_3` function seems to create p-value like scores.
    """
    # This function is a placeholder if a more direct conformity score is needed.
    # The current script uses `score_samples_3` which applies a sigmoid.
    return scores # Or apply a transformation as per BaKC+ definition.

def score_samples_ocsvm(model, X_samples):
    """Scores samples using the decision function of the OCSVM model."""
    if X_samples.shape[0] == 0:
        return np.array([])
    return model.decision_function(X_samples)

def transform_to_p_like_values(scores):
    """
    Transforms scores (e.g., from OCSVM decision_function) to [0,1] range
    where higher values suggest higher likelihood of being an anomaly,
    if the original scores had higher values for inliers.
    This is consistent with `score_samples_3`'s sigmoid approach.
    """
    # Sigmoid transformation: maps (-inf, +inf) to (0, 1)
    # If scores are high (normal), exp(scores) is large, 1/(1+exp) is small.
    # If scores are low (anomaly), exp(scores) is small, 1/(1+exp) is large (closer to 1).
    return 1 / (1 + np.exp(scores))


def create_calibration_scores_set(train_df_fold, base_random_state_fold, num_ensemble_members):
    """
    Creates calibration scores for a given training fold using K-fold internal splitting
    and bootstrapping for ensemble members.

    Args:
        train_df_fold (pandas.DataFrame): Training data for this main fold.
        base_random_state_fold (int): Random seed for reproducibility within this fold.
        num_ensemble_members (int): Number of OCSVM models in the ensemble.

    Returns:
        tuple: (list of fitted models, numpy.ndarray of calibration scores)
    """
    fold_models = []
    all_calibration_scores_for_fold = np.array([], dtype=np.float64)

    X_train_fold_np = train_df_fold.drop('y', axis=1).to_numpy()
    # y_train_fold_np = train_df_fold['y'].to_numpy() # Not directly used by OCSVM fit, but good for context

    # K-Fold for Cross-Conformal Prediction: Split train_df_fold into K_cal sub-folds
    # Each sub-fold i will use K_cal-1 parts for training member models, 1 part for calibration
    # For BaKC+, we generate calibration scores for each point in train_df_fold
    # by ensuring it's in the "calibration" part of an internal split.

    # Using K-fold to generate calibration scores for all points in train_df_fold
    # For each point x_i in train_df_fold, its calibration score is derived when x_i is in a hold-out set.
    # The ensemble is trained on the remaining data.
    n_samples_in_fold = len(X_train_fold_np)
    if n_samples_in_fold == 0:
        return [], np.array([])

    # Simplified: For each point, its calibration score is the median score from ensemble members
    # where each member was trained on a bootstrap sample of X_train_fold_np *excluding* that point (or an OOB scheme).
    # The current `fit_OCSVM_member` with `perform_bootstrapping` gives `leave_out_indices`.
    # We need to collect scores for these `leave_out_indices`.

    # Let's follow the paper's K-fold cross-conformal idea:
    # Split train_df_fold into K_ccp folds (e.g., 5 or 10)
    k_ccp_splits = min(5, n_samples_in_fold) if n_samples_in_fold > 1 else 1
    if k_ccp_splits == 1:
        # If not, the concept of cross-conformal prediction is harder to apply directly.
        # Alternative for small folds: use OOB scores from bootstrapped ensembles.
        print(f"Warning: Fold size {n_samples_in_fold} is too small for K-fold cross-conformal. Adjusting strategy.")
        # For now, let's assume we proceed by training ensemble on the whole fold and
        # use OOB scores if num_ensemble_members > 0.
        # This part needs to align perfectly with the paper's description of BaKC+.

        # If the paper implies that each member `j` of the ensemble is trained on a bootstrap of the *entire* `train_df_fold`,
        # then the calibration scores are the OOB scores.

        temp_calibration_scores_sum = np.zeros(n_samples_in_fold)
        temp_calibration_scores_count = np.zeros(n_samples_in_fold)

        for member_idx in range(num_ensemble_members):
            model, leave_out_indices = fit_OCSVM_member(
                member_idx=member_idx,
                num_members_ensemble=num_ensemble_members, # Pass this to enable bootstrapping
                fold_idx=base_random_state_fold, # Using base_random_state_fold as a unique id for this "meta-fold"
                X_train_fold=X_train_fold_np,
                random_state_fold=base_random_state_fold + member_idx, # Vary seed per member
                save_model=True # Save all members
            )
            if model is None:
                continue
            fold_models.append(model)

            if leave_out_indices is not None and len(leave_out_indices) > 0:
                # Scores for OOB samples
                raw_scores_oob = score_samples_ocsvm(model, X_train_fold_np[leave_out_indices])
                calib_scores_oob = transform_to_p_like_values(raw_scores_oob)

                temp_calibration_scores_sum[leave_out_indices] += calib_scores_oob
                temp_calibration_scores_count[leave_out_indices] += 1
        
        # Final calibration scores are the average OOB scores for points that were OOB at least once
        valid_indices = temp_calibration_scores_count > 0
        all_calibration_scores_for_fold = np.full(n_samples_in_fold, np.nan) # Initialize with NaN
        all_calibration_scores_for_fold[valid_indices] = temp_calibration_scores_sum[valid_indices] / temp_calibration_scores_count[valid_indices]
        all_calibration_scores_for_fold = all_calibration_scores_for_fold[~np.isnan(all_calibration_scores_for_fold)] # Remove NaNs for points never OOB

    else: # K-fold cross-conformal part
        kf_ccp = KFold(n_splits=k_ccp_splits, shuffle=True, random_state=base_random_state_fold)
        # Store all models trained across all CCP folds for later use or use only the last set?
        # Paper implies an ensemble of B models is available for test time.
        # Let's assume we build ONE main ensemble on the entire train_df_fold for test time predictions,
        # and the K-fold CCP is *only* for generating calibration scores.

        # Models for test-time prediction (ensemble trained on full train_df_fold):
        test_time_ensemble = []
        for member_idx in range(num_ensemble_members):
            # Seed for bootstrap should be consistent for this member across CCP folds if that's the design
            # Or, each member is just trained once on a bootstrap of the full train_df_fold
            model, _ = fit_OCSVM_member( # _ are leave_out_indices, not used for this main model training
                member_idx=member_idx,
                num_members_ensemble=num_ensemble_members,
                fold_idx=base_random_state_fold, # Unique ID for this training session
                X_train_fold=X_train_fold_np,
                random_state_fold=base_random_state_fold + member_idx + 1000, # Different seeds from CCP
                save_model=True # Save these main models
            )
            if model:
                test_time_ensemble.append(model)
        fold_models = test_time_ensemble # These are the models representing this fold

        # K-Fold CCP for calibration scores
        ccp_calibration_scores = np.zeros(n_samples_in_fold)
        for ccp_train_idx, ccp_calib_idx in kf_ccp.split(X_train_fold_np):
            X_ccp_train = X_train_fold_np[ccp_train_idx]
            X_ccp_calib = X_train_fold_np[ccp_calib_idx]

            if len(X_ccp_train) == 0 or len(X_ccp_calib) == 0:
                continue

            # For each ccp_calib point, get scores from an ensemble trained on X_ccp_train
            member_scores_for_calib_points = []
            for member_idx in range(num_ensemble_members):
                # Model trained only on X_ccp_train (and bootstrapped from it)
                # These models are temporary for calibration score generation, not saved long-term unless specified
                temp_model, _ = fit_OCSVM_member(
                    member_idx=member_idx,
                    num_members_ensemble=num_ensemble_members,
                    fold_idx=base_random_state_fold, # Could add ccp_fold_id here if needed
                    X_train_fold=X_ccp_train,
                    random_state_fold=base_random_state_fold + member_idx + 2000, # Different seeds
                    save_model=False # Don't save these temporary CCP models
                )
                if temp_model:
                    raw_scores_for_calib = score_samples_ocsvm(temp_model, X_ccp_calib)
                    p_like_scores_for_calib = transform_to_p_like_values(raw_scores_for_calib)
                    member_scores_for_calib_points.append(p_like_scores_for_calib)
            
            if member_scores_for_calib_points:
                # Aggregate scores from ensemble members (e.g., median or mean)
                aggregated_scores = np.median(np.array(member_scores_for_calib_points), axis=0)
                ccp_calibration_scores[ccp_calib_idx] = aggregated_scores
        
        all_calibration_scores_for_fold = ccp_calibration_scores

    return fold_models, all_calibration_scores_for_fold


## Statistical Evaluation Metrics
def get_q_hat(calibration_scores, alpha_sig):
    """Calculates the (1-alpha) quantile of calibration scores."""
    if len(calibration_scores) == 0:
        print("Warning: Calibration scores set is empty.")
        return np.inf # Or some other indicator of failure
    # q_level = 1 - alpha_sig # For p-values, threshold is alpha
    # For conformity scores (higher = more normal), q_hat is alpha-quantile
    # For non-conformity scores (higher = more anomalous), q_hat is (1-alpha)-quantile
    # Current `transform_to_p_like_values` creates scores where higher means more anomalous (p-value like)
    # So, a high score suggests anomaly. We need a threshold q_hat.
    # Predictions are p_test > q_hat means anomaly.
    # To control Type I error at alpha, q_hat should be (1-alpha) quantile of calibration scores.
    # (n+1)(1-alpha)/n correction for finite samples.
    n_cal = len(calibration_scores)
    q_level_corrected = np.ceil((n_cal + 1) * (1 - alpha_sig)) / n_cal
    q_level_corrected = min(1, max(0, q_level_corrected)) # Ensure it's within [0,1]
    
    q_hat = np.quantile(calibration_scores, q_level_corrected, method='higher')
    return q_hat

def get_predictions(test_scores_aggregated, q_hat_threshold):
    """Determines anomaly predictions based on test scores and q_hat."""
    # test_scores_aggregated are p-like values (higher = more anomalous)
    # Prediction is anomaly if score > q_hat
    return (test_scores_aggregated > q_hat_threshold).astype(int)


def calculate_power(predictions, ground_truth_anomalies):
    """Calculates statistical power (True Positive Rate for anomalies)."""
    if len(predictions) == 0: return 0.0
    true_positives = np.sum((predictions == 1) & (ground_truth_anomalies == 1))
    actual_positives = np.sum(ground_truth_anomalies == 1)
    return true_positives / actual_positives if actual_positives > 0 else 0.0


def calculate_fdr(predictions, ground_truth_inliers):
    """Calculates False Discovery Rate (for inliers predicted as anomalies)."""
    if len(predictions) == 0: return 0.0
    # ground_truth_inliers should be 0 for inliers, 1 for anomalies
    # We need FDR on inliers incorrectly flagged.
    # ground_truth typically: 0 for normal, 1 for anomaly.
    # So, false positives are where prediction is 1 (anomaly) but ground_truth is 0 (normal).
    false_positives = np.sum((predictions == 1) & (ground_truth_inliers == 0))
    total_predicted_positives = np.sum(predictions == 1)
    return false_positives / total_predicted_positives if total_predicted_positives > 0 else 0.0


## Training and Testing Implementation for one main run (e.g., one J loop iteration)
def train_and_test_single_run(data_df, dataset_name_str, run_idx, num_total_runs, params):
    """
    Performs one full train/test cycle for BaKC+.
    Splits data, trains model (getting calibration scores), and evaluates on test set.

    Args:
        data_df (pd.DataFrame): The full dataset for this run.
        dataset_name_str (str): Name of the dataset.
        run_idx (int): Index of this run (e.g., for averaging results over J runs).
        num_total_runs (int): Total number of runs (J).
        params (dict): Dictionary of parameters.

    Returns:
        dict: Results including power, FDR, and optionally predicted labels.
    """
    alpha = params['alpha']
    n_ensemble_members = params['num_models']
    main_random_state = params['base_random_state'] + run_idx # Vary seed per run

    # Split data: inliers for training/calibration, hold-out test set with inliers and outliers
    inliers_full_df = data_df[data_df['y'] == 0]
    outliers_full_df = data_df[data_df['y'] == 1]

    if len(inliers_full_df) == 0:
        print(f"Error: No inliers found in dataset {dataset_name_str} for run {run_idx}.")
        return {'power': 0, 'fdr': 0, 'error': "No inliers for training"}

    # Stratified split for train/test to maintain class proportions if desired,
    # or just split inliers and then add outliers to test set.
    # For OCSVM, train only on inliers.
    # Splitting inliers into a training set (for model fitting & calibration score generation)
    # and a test set (of inliers).
    # if len(inliers_full_df) < threshold.astype(int): # 1 for anomaly
    #
    # power_ae = calculate_power(predictions_ae, y_test_ae)
    # fdr_ae = calculate_fdr(predictions_ae, y_test_ae)
    #
    # return {'power': power_ae, 'fdr': fdr_ae, 'method': 'Autoencoder'}
    return {'power': 'N/A', 'fdr': 'N/A', 'method': 'Autoencoder (Commented Out)'}


## Visualization
def plot_score_distribution(scores, title='Score Distribution'):
    """Plots the distribution of scores."""
    plt.figure(figsize=(8,6))
    plt.hist(scores, bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

## Main Execution
if __name__ == "__main__":
    # Ensure output directories are ready
    manage_output_folders()

    # Select dataset
    # Available: "Shuttle", "Mammography", "Cardio", "Gamma", "Musk", "Fraud"
    # Note: "Fraud" dataset might be very large or require specific handling.
    # DATASET_TO_RUN = "Gamma" # Example: Choose one dataset
    DATASET_LIST = ["Shuttle", "Mammography", "Cardio", "Gamma", "Musk"] # List of datasets for evaluation

    # Common parameters for BaKC+
    bakc_params = {
        'alpha': 0.05,  # Target Type-I error rate (significance level)
        'num_models': 10,  # Number of ensemble members (B)
        'base_random_state': 42, # Base seed for reproducibility
        'test_inlier_ratio': 0.3 # Proportion of inliers to be used for testing
    }

    # Parameters for baseline OCSVM
    ocsvm_baseline_params = {
        'ocsvm_nu': 0.05, # Nu for vanilla OCSVM (often similar to alpha but distinct concept)
        'base_random_state': 42,
        'test_inlier_ratio': 0.3
    }

    # Parameters for Autoencoder baseline (if implemented and used)
    # ae_baseline_params = {
    #     'base_random_state': 42,
    #     'test_inlier_ratio': 0.3,
    #     'ae_epochs': 50,
    #     'ae_batch_size': 32,
    #     'ae_threshold_percentile': 95
    # }

    # Number of independent runs (J) for averaging results
    NUM_TOTAL_RUNS = 5 # For robust results, paper might use more (e.g., 20-30)

    results_summary = []

    for dataset_name in DATASET_LIST:
        print(f"\n--- Processing Dataset: {dataset_name} ---")
        df_full = load_and_preprocess_data(dataset_name)
        if df_full is None:
            print(f"Skipping dataset {dataset_name} due to loading error.")
            continue
        
        if 'y' not in df_full.columns:
            print(f"Skipping dataset {dataset_name} as target column 'y' is missing.")
            continue

        print(f"Dataset {dataset_name}: {len(df_full)} samples, {df_full['y'].value_counts().get(1,0)} anomalies.")

        # --- BaKC+ Evaluation ---
        bakc_run_results = []
        # Use multiprocessing Pool for parallel runs if J is large
        # For simplicity in this script, running serially. Can be adapted with Pool.
        # Example with Pool (ensure train_and_test_single_run is picklable and params are passed correctly):
        # pool_args = [(df_full.copy(), dataset_name, r_idx, NUM_TOTAL_RUNS, bakc_params) for r_idx in range(NUM_TOTAL_RUNS)]
        # with Pool(processes=min(NUM_TOTAL_RUNS, os.cpu_count() -1 if os.cpu_count() else 1)) as pool:
        #     bakc_run_results = list(tqdm(pool.starmap(train_and_test_single_run, pool_args), total=NUM_TOTAL_RUNS, desc=f"BaKC+ {dataset_name}"))
        
        for r_idx in tqdm(range(NUM_TOTAL_RUNS), desc=f"BaKC+ runs for {dataset_name}"):
            run_res = train_and_test_single_run(df_full.copy(), dataset_name, r_idx, NUM_TOTAL_RUNS, bakc_params)
            if 'error' not in run_res:
                bakc_run_results.append(run_res)

        # Aggregate BaKC+ results for this dataset
        if bakc_run_results:
            avg_power_bakc = mean([res['power'] for res in bakc_run_results if 'power' in res])
            avg_fdr_bakc = mean([res['fdr'] for res in bakc_run_results if 'fdr' in res])
            std_power_bakc = stdev([res['power'] for res in bakc_run_results if 'power' in res]) if len(bakc_run_results) > 1 else 0
            std_fdr_bakc = stdev([res['fdr'] for res in bakc_run_results if 'fdr' in res]) if len(bakc_run_results) > 1 else 0
            results_summary.append({
                'Dataset': dataset_name, 'Method': 'BaKC+',
                'Avg Power': avg_power_bakc, 'Std Power': std_power_bakc,
                'Avg FDR': avg_fdr_bakc, 'Std FDR': std_fdr_bakc
            })
            print(f"BaKC+ Results for {dataset_name} (Avg over {len(bakc_run_results)} runs):")
            print(f"  Avg Power: {avg_power_bakc:.4f} (Std: {std_power_bakc:.4f})")
            print(f"  Avg FDR:   {avg_fdr_bakc:.4f} (Std: {std_fdr_bakc:.4f})")
        else:
            print(f"BaKC+ evaluation failed or produced no results for {dataset_name}.")
            results_summary.append({'Dataset': dataset_name, 'Method': 'BaKC+', 'Avg Power': 'N/A', 'Avg FDR': 'N/A'})


        # --- Vanilla OCSVM Baseline Evaluation ---
        # For baselines, typically run once or average over fewer runs if computationally expensive,
        # but for fair comparison, similar #runs is better if feasible.
        # Here, running once for simplicity for the baseline.
        ocsvm_res = evaluate_baseline_ocsvm(df_full.copy(), ocsvm_baseline_params)
        if 'error' not in ocsvm_res:
            results_summary.append({
                'Dataset': dataset_name, 'Method': 'Vanilla OCSVM',
                'Avg Power': ocsvm_res['power'], 'Std Power': 0, # Single run
                'Avg FDR': ocsvm_res['fdr'], 'Std FDR': 0  # Single run
            })
            print(f"Vanilla OCSVM Results for {dataset_name}:")
            print(f"  Power: {ocsvm_res['power']:.4f}")
            print(f"  FDR:   {ocsvm_res['fdr']:.4f}")
        else:
            print(f"Vanilla OCSVM evaluation failed for {dataset_name}: {ocsvm_res.get('error')}")
            results_summary.append({'Dataset': dataset_name, 'Method': 'Vanilla OCSVM', 'Avg Power': 'N/A', 'Avg FDR': 'N/A'})


        # --- Autoencoder Baseline Evaluation (if implemented) ---
        # ae_res = evaluate_autoencoder_baseline(df_full.copy(), ae_baseline_params)
        # if 'error' not in ae_res and ae_res['power'] != 'N/A':
        #     results_summary.append({
        #         'Dataset': dataset_name, 'Method': 'Autoencoder',
        #         'Avg Power': ae_res['power'], 'Std Power': 0,
        #         'Avg FDR': ae_res['fdr'], 'Std FDR': 0
        #     })
        #     print(f"Autoencoder Results for {dataset_name}:")
        #     print(f"  Power: {ae_res['power']:.4f}")
        #     print(f"  FDR:   {ae_res['fdr']:.4f}")
        # else:
        #     print(f"Autoencoder evaluation skipped or failed for {dataset_name}.")
        #     results_summary.append({'Dataset': dataset_name, 'Method': 'Autoencoder', 'Avg Power': 'N/A', 'Avg FDR': 'N/A'})
        print(f"Autoencoder evaluation skipped for {dataset_name} (requires user implementation of 'build_autoencoder' and TF/Keras).")
        results_summary.append({'Dataset': dataset_name, 'Method': 'Autoencoder', 'Avg Power': 'N/A', 'Std Power':0, 'Avg FDR': 'N/A', 'Std FDR':0})


    # Print summary table (simulating Tables 1 & 2 from the paper)
    print("\n\n--- Overall Results Summary ---")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string())

    # Save summary to CSV
    summary_csv_path = os.path.join(BASE_OUTPUT_PATH, 'results_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nResults summary saved to: {summary_csv_path}")

    ## Reproducibility Notes
    # -------------------------
    # The results depend on the random seeds, dataset splits, and specific versions
    # of the libraries used.
    # Key parameters (`alpha`, `num_models`, `base_random_state`, `NUM_TOTAL_RUNS`)
    # are defined in the `__main__` block.
    # The `create_calibration_scores_set` function's K-fold CCP and bootstrapping logic
    # is central to BaKC+. Ensure its implementation matches the paper's description precisely.
    # The OCSVM `nu` parameter and kernel choices also affect performance.
    # For exact reproduction of paper results, library versions should be fixed (e.g., via requirements.txt).
    # -------------------------

    print("\nScript finished.")