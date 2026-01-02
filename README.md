# Crypto Market Prediction

A machine learning project for predicting cryptocurrency market trends using historical trading data from Kaggle's DRW Crypto Market Prediction competition.

Here's the kaggle competition link: https://www.kaggle.com/competitions/drw-crypto-market-prediction

## Project Overview

This project predicts cryptocurrency market movements from minute-level market features derived from time-series data. While the training set includes timestamps, the test set masks and shuffles them to prevent temporal leakage, so the model is evaluated in a per-row (tabular) inference setting.

## Project Structure

```
.
├── src/                      # Source code package
│   └── myproj/               # Main package
│       ├── __init__.py       # Package initialization
│       ├── data.py           # Data loading and validation
│       ├── features.py       # Feature engineering
│       ├── split.py          # Data splitting utilities
│       ├── models.py         # Model training and evaluation
│       ├── metrics.py        # Performance metrics
│       └── utils.py          # Save/load utilities
├── scripts/                  # Executable scripts
│   ├── make_features.py      # Generate engineered features
│   ├── train_ridge.py        # Train Ridge regression model
│   ├── predict.py            # Generate predictions
│   └── permutation_importance.py  # Feature importance analysis
├── configs/                  # Configuration files
│   └── ridge.yaml            # Ridge model configuration
├── artifacts/                # Saved models and scalers
│   ├── models/               # Trained models
│   └── scalers/              # Feature scalers
├── data/                     # Dataset directory
│   ├── train.parquet         # Original training data
│   ├── test.parquet          # Original test data
│   ├── train_fe.parquet      # Training with engineered features
│   ├── test_fe.parquet       # Test with engineered features
│   └── sample_submission.csv # Sample submission format
├── main.py                   # Legacy entry point
├── read_parquet.py           # Data exploration script
├── README.md                 # Project documentation
├── requirement.txt           # Python dependencies
└── setup.sh                  # Automated setup script
```

## Setup Environment

### Prerequisites

- Python 3.8 or higher
- Miniconda or Anaconda (recommended)
- Kaggle API credentials

### Setup

Miniconda is recommended. After creating environment, install the requirements:

```bash
# Create and activate conda environment
conda create -n crypto python=3.8
conda activate crypto

# Install dependencies
pip install -r requirement.txt

# Download dataset (requires Kaggle API setup)
kaggle competitions download -c drw-crypto-market-prediction
unzip drw-crypto-market-prediction.zip -d ./data/
```

Or simply use the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

The script will:
1. Create a conda environment named 'crypto'
2. Install all required dependencies
3. Download the competition dataset from Kaggle (requires API credentials)
4. Extract data to the `data/` directory

### Kaggle API Setup

To download the dataset, you need to set up Kaggle API credentials.

1. Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Click "Create New API Token" to download `kaggle.json`
3. Open the file and copy your API token
4. Set environment variable:

```bash
export KAGGLE_API_TOKEN="YOUR_API_TOKEN"
```

## Data Analysis

The competition dataset includes:
- `train.parquet`: Historical training data with market features
- `test.parquet`: Test data for predictions
- `sample_submission.csv`: Template for submission format

Use the script to explore the dataset (both `train.parquet` and `test.parquet`):

```bash
python scripts/read_parquet.py
```

This script will display:
- Basic dataset information (rows, columns)
- Column names and data types
- First 10 rows of data
- Statistical summary
- Missing value analysis

### Outcome
There are 786 columns: bid quantity, ask quantity, buy quantity, sell quantity, volume, 780 anonymized features and label (market price movement).

## Feature Engineering

The project adds 9 engineered features to the original 780 anonymized features:

1. **imbalance_best**: Bid-ask imbalance at best levels
2. **trade_imbalance**: Trade flow imbalance (buy vs sell)
3. **vol_log1p**: Log-transformed volume
4. **bid_qty_log1p**: Log-transformed bid quantity
5. **ask_qty_log1p**: Log-transformed ask quantity
6. **total_best_qty_log1p**: Log-transformed total best quantity
7. **book_to_trade_ratio_log1p**: Log-transformed order book to trade ratio
8. **bid_to_buy**: Bid quantity to buy quantity ratio
9. **ask_to_sell**: Ask quantity to sell quantity ratio

Generate engineered features:

```bash
python scripts/make_features.py
```

This creates `train_fe.parquet` and `test_fe.parquet` in the `data/` directory.

## Model Training

### Ridge Regression

Train a Ridge regression model with StandardScaler:

```bash
python scripts/train_ridge.py
```

The training pipeline:
1. Loads feature-engineered data
2. Performs train/validation split (80/20 random split)
3. Standardizes features (mean=0, std=1)
4. Trains Ridge model (alpha=1.0)
5. Evaluates on train and validation sets
6. Saves model and scaler to `artifacts/`

Configuration is managed via `configs/ridge.yaml`.

### Generate Predictions

Create submission file:

```bash
python scripts/predict.py
```

### Feature Importance Analysis

Analyze feature importance using permutation:

```bash
python scripts/permutation_importance.py
```

## Using the myproj Package

The project is structured as a Python package for reusability:

```python
from myproj import load_data, add_depth_features, split_data
from myproj import train_ridge_model, evaluate_model

# Load and engineer features
df = load_data('data/train.parquet')
df_fe = add_depth_features(df)

# Split data
X_train, X_val, y_train, y_val = split_data(df_fe, test_size=0.2, method='random')

# Train and evaluate
model, scaler = train_ridge_model(X_train, y_train, alpha=1.0)
metrics = evaluate_model(model, scaler, X_val, y_val)
```

## Data Notes

The only difference between training set and testing set is: the timestamp of testing set is masked and shuffled and the label is set as 0 (this is what we are going to predict).

The training set has 525,886 rows and the testing set has 538,150 rows.

## Model Performance

### Ridge Regression + StandardScaler

**Baseline Results** (alpha=1.0, 780 original + 9 engineered features):

- Train Set: R² = 0.148, RMSE = 0.932, MAE = 0.619
- Validation Set: R² = 0.141, RMSE = 0.937, MAE = 0.623

### Ablation Study

To evaluate feature contribution, run experiments:

- **Experiment A**: Only 780 original features (baseline)
- **Experiment B**: Only 9 engineered features (signal strength test)
- **Experiment C**: All 789 features (combined performance)
- **Experiment D**: Permutation importance (feature ranking on validation set)

In financial modeling, small but stable improvements (e.g., R² +0.002) are meaningful as gains accumulate.

## License

This project is for educational and competition purposes.
