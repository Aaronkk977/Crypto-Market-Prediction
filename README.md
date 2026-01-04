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

**Outcome**:
There are 786 columns: bid quantity, ask quantity, buy quantity, sell quantity, volume, 780 anonymized features and label (market price movement).

### Time-Based Grouping

We found that data points with similar timestamps exhibit similar structures. Therefore, we grouped all data within the same hour together, resulting in 8,784 groups (24 hours × 366 days).

### Correlated Features

We identified 96 pairs of highly correlated features (|r| > 0.995). Due to overlapping relationships among these pairs, we removed 73 redundant features (9.4% reduction), reducing the feature set from 780 to 707 original features.

### Correlation between Features and Label



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

I have tried ridge regression, elastic net and lightGBM.

### Ridge Regression

Train a Ridge regression model with StandardScaler:

```bash
python scripts/train_ridge.py
```

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

**Training Configuration:**
- Model: RidgeCV with 3-fold internal CV for alpha selection
- Data Split: 80% train / 20% validation (random split)
- Feature Scaling: StandardScaler (mean=0, std=1)
- Evaluation Metric: Pearson correlation (aligned with Kaggle)

#### Feature Importance (After Correlation Filtering)

```bash
python scripts/permutation_importatnce.py
```

**Top 20 Features by Permutation Importance (Pearson Correlation):**

| Feature | Importance | Std |
|---------|------------|-----|
| X263 | 0.1204 | ±0.0020 |
| X762 | 0.1180 | ±0.0015 |
| X386 | 0.1163 | ±0.0016 |
| X380 | 0.1139 | ±0.0013 |
| X591 | 0.1090 | ±0.0011 |
| X16 | 0.1088 | ±0.0017 |
| X779 | 0.1037 | ±0.0015 |
| X277 | 0.1012 | ±0.0013 |
| X416 | 0.0939 | ±0.0013 |
| X127 | 0.0938 | ±0.0014 |
| X780 | 0.0938 | ±0.0013 |
| X52 | 0.0921 | ±0.0013 |
| X214 | 0.0889 | ±0.0010 |
| X219 | 0.0886 | ±0.0016 |
| X51 | 0.0870 | ±0.0013 |
| X21 | 0.0827 | ±0.0016 |
| X332 | 0.0827 | ±0.0014 |
| X254 | 0.0823 | ±0.0016 |
| X268 | 0.0798 | ±0.0012 |
| X72 | 0.0791 | ±0.0013 |

**Engineered Features Importance:**

| Feature | Importance | Std |
|---------|------------|-----|
| vol_log1p | 0.001461 | ±0.000285 |
| book_to_trade_ratio_log1p | 0.001232 | ±0.000140 |
| imbalance_best | 0.000518 | ±0.000159 |
| bid_qty_log1p | 0.000440 | ±0.000117 |
| trade_imbalance | 0.000266 | ±0.000083 |
| total_best_qty_log1p | 0.000110 | ±0.000063 |
| ask_to_sell | 0.000087 | ±0.000064 |
| bid_to_buy | 0.000002 | ±0.000007 |
| ask_qty_log1p | -0.000001 | ±0.000002 |

**Analysis:**
- **Model trained on 707 filtered features** (removed 73 highly correlated features, 9.4% reduction)
- Feature importance ranking changed significantly after removing redundant features
- Top features (X263, X762, X386) show importance ~100x higher than engineered features
- Average engineered feature importance: ~4.5×10⁻⁴
- Engineered features continue to provide marginal but consistent signal
- Results saved to: `results/importance/permutation_importance.csv`

### LightGBM

5-fold, tracking pearson score while training
Private score: 0.05462
Public score: 0.04803

adjusting training rounds from 2000 to 2500 did not improve the score on test set while validation score did keep going up.
Private Score: 0.03637
Public score: 0.04847

ablation
============================================================
ABLATION STUDY SUMMARY
============================================================
N Features   Val Pearson          Val RMSE             Best Iter
------------------------------------------------------------
30           0.685097 ± 0.006545  0.746899 ± 0.009092  2500
50           0.721540 ± 0.008342  0.712206 ± 0.010727  2500
100          0.749898 ± 0.004479  0.683133 ± 0.011124  2500
200          0.765003 ± 0.004981  0.664640 ± 0.012870  2500
500          0.773249 ± 0.004759  0.654769 ± 0.012312  2500


2000 rounds with 200 features
Private score: 0.05044
Score: 0.04909


## License

This project is for educational and competition purposes.
