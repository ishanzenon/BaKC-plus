# BaKC+ Anomaly Detection

Implementation of the BaKC+ methodology for one-class anomaly detection combining cross-conformal prediction, ensembles, and bootstrapping for uncertainty quantification.

## Features

- Combines K-fold cross-validation, cross-conformal prediction, ensembles, and bootstrapping
- Base estimator: OneClassSVM
- Provides statistical guarantees for Type-I error control with high detection power
- Benchmarked on ADBench datasets: Shuttle, Mammography, Cardio, Gamma, Musk, and Fraud
- Compared against Vanilla OneClassSVM and Autoencoder baselines

## Installation

```bash
# Create and activate virtual environment
python -m venv bakc_env
source bakc_env/bin/activate  # On Windows: bakc_env\Scripts\activate

# Install required packages
pip install numpy pandas scikit-learn matplotlib scipy tqdm

# Optional: For Autoencoder baseline
# pip install tensorflow
```

## Data Preparation

1. Download the ADBench datasets.
2. Create a `data` directory in the project root.
3. For each dataset (e.g., 'gamma'), create a subdirectory in `data` (e.g., `data/gamma/`) and place the corresponding CSV file (e.g., `data/gamma/gamma.csv`).

## Usage

Simply run the main script:

```bash
python bakc_plus.py
```

The script will:
1. Process each dataset in the configured list (Shuttle, Mammography, Cardio, Gamma, Musk)
2. Run the BaKC+ method and baseline comparisons 
3. Output performance metrics (statistical power and FDR)

## Output

- Trained models are saved in `output/models/calib_models/`
- Results summary is saved to `output/results_summary.csv`
- Artifacts are stored in `output/artifacts/`

## Citation

If you use this code in your research, please cite:

```
[Citation removed for anonymization]
```