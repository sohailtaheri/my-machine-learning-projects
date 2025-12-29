# Bulldozer Price Prediction 🚜

An end-to-end machine learning project for predicting the sale price of bulldozers at auction using Random Forest Regression and time-series features.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Feature Engineering](#feature-engineering)
- [Kaggle Submission](#kaggle-submission)
- [Improvements](#improvements)
- [References](#references)

## 🎯 Overview

This project implements a machine learning model that predicts the auction sale price of bulldozers based on their usage, equipment type, and configuration. It uses Random Forest Regression with extensive feature engineering to handle time-series data and missing values.

**Problem**: How well can we predict the future sale price of a bulldozer given its characteristics and previous examples of how much similar bulldozers have been sold for?

**Solution**: Random Forest Regression with time-series feature engineering, handling missing values through median imputation and categorical encoding.

**Evaluation**: RMSLE (Root Mean Squared Log Error) - Kaggle competition metric

> **Note**: This project is part of the [Complete Machine Learning & Data Science Bootcamp 2025](https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/) on Udemy - a comprehensive zero-to-mastery course covering machine learning, deep learning, and data science.

## ✨ Features

- **Clean OOP Architecture**: Modular `BulldozerPricePredictor` class for entire ML pipeline
- **Time-Series Features**: Extract year, month, day, day of week, and day of year from sale dates
- **Missing Value Handling**:
  - Numerical: Median imputation with binary indicators
  - Categorical: Encoding with missing value flags
- **Hyperparameter Tuning**: RandomizedSearchCV for automated parameter optimization
- **Model Persistence**: Save and load trained models
- **Feature Importance**: Visualize which features matter most
- **Kaggle Ready**: Generate submission files in Kaggle format
- **Evaluation Metrics**: MAE, RMSLE, and R² score
- **Fast Training**: Optimized with `n_jobs=-1` for parallel processing

## 📊 Dataset

The project uses the **Blue Book for Bulldozers** dataset from Kaggle's competition.

- **Training Records**: 412,698 auction records (1989-2011)
- **Validation Records**: Data from 2012 (January-April)
- **Test Records**: Data from 2012 (May-November)
- **Features**: 53 columns including:
  - Machine specifications (model, year, hours used)
  - Equipment details (enclosure, drive system, hydraulics)
  - Auction information (date, location, auctioneer)
  - Target variable: `SalePrice`

### Download Dataset

1. Visit [Kaggle's Bluebook for Bulldozers Competition](https://www.kaggle.com/competitions/bluebook-for-bulldozers)
2. Download the dataset files:
   - `Train.zip` - Training data through 2011
   - `Valid.zip` - Validation data (Jan-Apr 2012)
   - `Test.zip` - Test data (May-Nov 2012)
   - `Data Dictionary.xlsx` - Feature descriptions

3. Extract the files:
   ```bash
   unzip Train.zip
   unzip Valid.zip
   unzip Test.zip
   ```

4. Combine training and validation (as done in the notebook):
   ```python
   import pandas as pd
   train = pd.read_csv("Train.csv")
   valid = pd.read_csv("Valid.csv")
   df = pd.concat([train, valid], ignore_index=True)
   df.to_csv("data/TrainAndValid.csv", index=False)
   ```

Your directory structure should look like:
```
bulldozer-price-predict/
├── data/
│   ├── TrainAndValid.csv   # Combined training data
│   ├── Test.csv            # Test data
│   └── submission.csv      # Generated predictions
```

## 🔧 Requirements

### System Requirements
- **Python**: 3.8+ (tested with Python 3.11)
- **OS**: macOS, Linux, or Windows
- **RAM**: 8GB+ (16GB recommended for full dataset)
- **Disk Space**: 2GB+
- **CPU**: Multi-core processor recommended (uses parallel processing)

### Python Dependencies

Core libraries:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `matplotlib` - Visualization

## 📥 Installation

### Step 1: Clone or Navigate to Project Directory

```bash
cd /path/to/bulldozer-price-predict
```

### Step 2: Create Virtual Environment

#### On macOS/Linux:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify activation (should show path to .venv)
which python
```

#### On Windows:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Verify activation
where python
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install required packages
pip install numpy pandas scikit-learn matplotlib jupyter
```

### Step 4: Verify Installation

```bash
python -c "import pandas as pd; import sklearn; print('Pandas:', pd.__version__); print('Scikit-learn:', sklearn.__version__)"
```

Expected output:
```
Pandas: 2.x.x
Scikit-learn: 1.x.x
```

### Deactivating Virtual Environment

When you're done working:

```bash
deactivate
```

## 📁 Project Structure

```
bulldozer-price-predict/
│
├── data/
│   ├── TrainAndValid.csv              # Combined training/validation data
│   ├── Test.csv                       # Test data for predictions
│   └── submission.csv                 # Generated Kaggle submission
│
├── bulldoz.ipynb                      # Jupyter notebook (original exploration)
├── bulldoz.py                         # Main Python module (OOP version)
├── bulldoz_structure_mindmap.md       # Code architecture visualization
├── README.md                          # This file
└── feature_importance.png             # Feature importance visualization
```

## 🚀 Usage

### Quick Start

```python
from bulldoz import BulldozerPricePredictor

# Initialize predictor
predictor = BulldozerPricePredictor(random_state=42)

# Load data
predictor.load_data("data/TrainAndValid.csv", parse_dates=True)

# Preprocess data
df_processed = predictor.preprocess_data(predictor.df)

# Split data (2012 for validation)
predictor.split_data(df_processed, split_year=2012)

# Train model with optimal hyperparameters
predictor.train(
    n_estimators=40,
    max_depth=10,
    max_features=0.5,
    min_samples_split=14,
    min_samples_leaf=3
)

# Evaluate model
scores = predictor.evaluate()

# Plot feature importance
fig = predictor.plot_feature_importance(n=20)
plt.show()
```

### Training with Hyperparameter Tuning

```python
from bulldoz import BulldozerPricePredictor

# Initialize and load data
predictor = BulldozerPricePredictor()
predictor.load_data("data/TrainAndValid.csv", parse_dates=True)

# Preprocess and split
df_processed = predictor.preprocess_data(predictor.df)
predictor.split_data(df_processed)

# Automatic hyperparameter tuning
rs_model = predictor.hyperparameter_tuning(
    n_iter=20,
    cv=5,
    max_samples=10000  # Faster training
)

# Evaluate tuned model
predictor.evaluate()
```

### Quick Training (Subset of Data)

```python
from bulldoz import BulldozerPricePredictor

# For faster experimentation
predictor = BulldozerPricePredictor()
predictor.load_data("data/TrainAndValid.csv", parse_dates=True)

df_processed = predictor.preprocess_data(predictor.df)
predictor.split_data(df_processed)

# Train on subset for speed
predictor.train(max_samples=10000, n_estimators=50)
predictor.evaluate()
```

### Making Predictions on Test Data

```python
from bulldoz import BulldozerPricePredictor

# After training your model
predictor.save_predictions(
    test_data_path="data/Test.csv",
    output_path="data/submission.csv"
)

# This creates a Kaggle-ready submission file
```

### Standalone Preprocessing Function

```python
from bulldoz import preprocess_data_standalone
import pandas as pd

# Load data
df = pd.read_csv("data/TrainAndValid.csv", parse_dates=["saledate"])

# Preprocess
df_processed = preprocess_data_standalone(df)

# Now ready for modeling
```

### Running the Jupyter Notebook

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Launch Jupyter
jupyter notebook

# Open bulldoz.ipynb in your browser
```

## 🏗️ Model Architecture

### Random Forest Regression

```
Input Features (102 columns after preprocessing)
      ↓
┌─────────────────────────────────────────┐
│   Random Forest Regressor               │
│                                         │
│   ├── n_estimators: 40 trees           │
│   ├── max_depth: 10                    │
│   ├── max_features: 0.5 (50%)          │
│   ├── min_samples_split: 14            │
│   ├── min_samples_leaf: 3              │
│   └── random_state: 42                 │
│                                         │
│   Each tree:                            │
│   ├── Bootstrap sampling                │
│   ├── Feature randomization             │
│   └── Recursive binary splitting        │
└─────────────────────────────────────────┘
      ↓
Ensemble Averaging (40 predictions)
      ↓
Output: Predicted SalePrice ($)
```

### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Algorithm | Random Forest Regressor | Ensemble of decision trees |
| Number of Trees | 40 | Found via hyperparameter tuning |
| Max Depth | 10 | Prevents overfitting |
| Max Features | 0.5 | 50% of features per split |
| Min Samples Split | 14 | Minimum samples to split node |
| Min Samples Leaf | 3 | Minimum samples in leaf |
| Loss Function | MSE | Mean Squared Error |
| Parallelization | -1 (all cores) | Fast training |

## 📈 Results

### Model Performance

**Training on 401,125 samples (pre-2012):**
**Validation on 11,573 samples (2012):**

| Metric | Training | Validation |
|--------|----------|------------|
| MAE | $6,579 | $7,601 |
| RMSLE | 0.294 | 0.307 |
| R² Score | 0.817 | 0.815 |

### Notes on Performance

- **Strong R² Score (0.815)**: Model explains 81.5% of price variance
- **Low RMSLE (0.307)**: Good performance on Kaggle's metric
- **Small Train/Val Gap**: Minimal overfitting
- **MAE ~$7,600**: Average prediction error on validation set

### Training Time

- **Quick Training** (max_samples=10000): ~1.5 seconds
- **Full Training** (401,125 samples): ~5 seconds
- **Hyperparameter Tuning** (20 iterations, 5-fold CV): ~21 seconds

All times on modern multi-core CPU with `n_jobs=-1`.

## 🔧 Feature Engineering

### Time-Series Features

From `saledate`, we extract:

```python
saleYear         # Year of sale (1989-2012)
saleMonth        # Month (1-12)
saleDay          # Day of month (1-31)
saleDayofweek    # Day of week (0-6)
saleDayofyear    # Day of year (1-365)
```

### Missing Value Strategy

**Numerical Features**:
- Impute with median (robust to outliers)
- Add binary `_is_missing` indicator

**Categorical Features**:
- Convert to integer codes
- Missing values encoded as 0
- Add binary `_is_missing` indicator

### Feature Preprocessing Pipeline

```
Raw Data (53 columns, many missing values)
      ↓
1. Parse saledate → datetime
      ↓
2. Extract time features (5 new columns)
      ↓
3. Drop saledate column
      ↓
4. Convert strings → ordered categories
      ↓
5. Fill numerical missing → median + indicator
      ↓
6. Encode categorical → codes + indicator
      ↓
Processed Data (103 columns, no missing values)
```

### Top Features by Importance

Based on trained model:

1. **YearMade** - Manufacturing year
2. **saleYear** - Year sold
3. **ModelID** - Specific model identifier
4. **MachineHoursCurrentMeter** - Hours of usage
5. **saleMonth** - Seasonality effects
6. **ProductSize** - Size category
7. **Enclosure** - Cab type
8. **YearMade_is_missing** - Missing indicator
9. **fiModelDesc** - Model description
10. **fiBaseModel** - Base model type

View full feature importance:
```python
predictor.plot_feature_importance(n=20)
```

## 🏆 Kaggle Submission

### Generate Submission File

```python
# After training your model
predictor.save_predictions(
    test_data_path="data/Test.csv",
    output_path="data/submission.csv"
)

# This creates: data/submission.csv
```

### Submission File Format

The CSV file contains:
- `SalesID` column: Unique identifier for each test record
- `SalePrice` column: Predicted price in USD

### Submit to Kaggle

1. Go to [Bluebook for Bulldozers Competition](https://www.kaggle.com/competitions/bluebook-for-bulldozers)
2. Click "Submit Predictions"
3. Upload `data/submission.csv`
4. View your RMSLE score

## 🔄 Improvements

### Ways to Improve Model Performance

1. **Advanced Feature Engineering**
   ```python
   # Age of machine at sale
   df['MachineAge'] = df['saleYear'] - df['YearMade']

   # Usage rate
   df['UsageRate'] = df['MachineHoursCurrentMeter'] / df['MachineAge']

   # Price trends over time
   df['AvgPriceByYear'] = df.groupby('saleYear')['SalePrice'].transform('mean')
   ```

2. **Better Handling of Missing Data**
   ```python
   # KNN imputation for numerical features
   from sklearn.impute import KNNImputer

   # Iterative imputation
   from sklearn.experimental import enable_iterative_imputer
   from sklearn.impute import IterativeImputer
   ```

3. **Try Different Algorithms**
   - Gradient Boosting (XGBoost, LightGBM, CatBoost)
   - Linear models with regularization (Ridge, Lasso)
   - Neural networks for tabular data
   - Ensemble of multiple models

4. **Hyperparameter Optimization**
   ```python
   # More extensive search
   predictor.hyperparameter_tuning(n_iter=100, cv=10)

   # Grid search for precision
   from sklearn.model_selection import GridSearchCV

   # Bayesian optimization
   from skopt import BayesSearchCV
   ```

5. **Cross-Validation Strategy**
   ```python
   # Time-series cross-validation
   from sklearn.model_selection import TimeSeriesSplit

   # Multiple validation years
   # Train on 1989-2010, validate on 2011, test on 2012
   ```

6. **Feature Selection**
   ```python
   # Remove low-importance features
   # Reduce correlation between features
   # Use recursive feature elimination
   from sklearn.feature_selection import RFE
   ```

7. **Outlier Detection**
   ```python
   # Identify and handle price outliers
   # Remove or cap extreme values
   # Use robust scaling
   ```

8. **Categorical Encoding**
   ```python
   # Try different encoding strategies
   - Target encoding (mean encoding)
   - One-hot encoding (for low-cardinality)
   - Entity embeddings
   ```

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Error**
```python
# Train on subset of data
predictor.train(max_samples=50000)

# Or use less trees
predictor.train(n_estimators=20)
```

**2. Slow Training**
```python
# Reduce training data
predictor.train(max_samples=10000)

# Reduce trees
predictor.train(n_estimators=20, max_depth=5)

# Verify parallel processing
import os
print(f"CPU cores: {os.cpu_count()}")
```

**3. Module Not Found Error**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install numpy pandas scikit-learn matplotlib
```

**4. Date Parsing Issues**
```python
# Ensure parse_dates is True
predictor.load_data("data/TrainAndValid.csv", parse_dates=True)

# Or manually parse
df['saledate'] = pd.to_datetime(df['saledate'])
```

**5. Column Mismatch in Test Data**
```python
# The save_predictions() method handles this automatically
# It adds missing columns and reorders them
# But if doing manually:

missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = False

X_test = X_test[X_train.columns]
```

## 📚 References

### Course
- [Complete Machine Learning & Data Science Bootcamp 2025](https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/) - Zero to Mastery course on Udemy

### Competition & Dataset
- [Kaggle: Bluebook for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers)
- [Competition Evaluation (RMSLE)](https://www.kaggle.com/competitions/bluebook-for-bulldozers/overview/evaluation)
- [Data Dictionary](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data)

### Documentation & Papers
- [Scikit-learn: Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Random Forests Paper (Breiman, 2001)](https://link.springer.com/article/10.1023/A:1010933404324)
- [Scikit-learn: Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [Handling Missing Data](https://scikit-learn.org/stable/modules/impute.html)

### Time-Series & Feature Engineering
- [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Time-Series Feature Extraction](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)

## 📝 License

This project is for educational purposes. Dataset is from Kaggle's Bluebook for Bulldozers competition.

## 👤 Author

**Sohail**

- GitHub: [@sohail](https://github.com/sohail)
- Project: Bulldozer Price Prediction using Machine Learning

## 🙏 Acknowledgments

- Kaggle for providing the dataset and competition platform
- Fast Iron (competition data provider)
- Scikit-learn team for the excellent machine learning library
- The machine learning community for tutorials and resources

---

## 🚦 Quick Command Reference

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib jupyter

# Train (Quick)
python -c "from bulldoz import BulldozerPricePredictor; p = BulldozerPricePredictor(); p.load_data('data/TrainAndValid.csv'); df = p.preprocess_data(p.df); p.split_data(df); p.train(max_samples=10000); p.evaluate()"

# Train (Full)
python bulldoz.py

# Jupyter
jupyter notebook bulldoz.ipynb

# Make Predictions
python -c "from bulldoz import BulldozerPricePredictor; p = BulldozerPricePredictor(); p.load_data('data/TrainAndValid.csv'); df = p.preprocess_data(p.df); p.split_data(df); p.train(); p.save_predictions('data/Test.csv', 'data/submission.csv')"

# Deactivate
deactivate
```

---

## 📊 Project Highlights

✅ **Complete ML Pipeline** - From raw data to Kaggle submission
✅ **Production-Ready Code** - OOP design, modular, reusable
✅ **Comprehensive Documentation** - Code comments, docstrings, README
✅ **Feature Engineering** - Time-series features, missing value handling
✅ **Model Optimization** - Hyperparameter tuning, parallel processing
✅ **Visualization** - Feature importance, performance metrics
✅ **Real-World Dataset** - 412K+ records, 53 features, industry data

---

**Happy Predicting! 🚜💰**

For detailed code architecture, see `bulldoz_structure_mindmap.md`
