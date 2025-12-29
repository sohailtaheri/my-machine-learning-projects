# Bulldozer Price Predictor - Code Structure Mindmap

```
bulldoz.py
│
├── 📦 IMPORTS
│   ├── numpy (np)
│   ├── pandas (pd)
│   ├── matplotlib.pyplot (plt)
│   └── sklearn
│       ├── RandomForestRegressor
│       ├── RandomizedSearchCV
│       └── metrics (mean_squared_log_error, mean_absolute_error, r2_score)
│
├── 🎯 MAIN CLASS: BulldozerPricePredictor
│   │
│   ├── 🔧 Constructor & Attributes
│   │   └── __init__(random_state=42)
│   │       ├── self.random_state
│   │       ├── self.model
│   │       ├── self.df
│   │       ├── self.df_preprocessed
│   │       ├── self.X_train
│   │       ├── self.y_train
│   │       ├── self.X_valid
│   │       └── self.y_valid
│   │
│   ├── 📥 DATA LOADING
│   │   └── load_data(filepath, parse_dates=True)
│   │       ├── Reads CSV file
│   │       ├── Parses saledate column
│   │       ├── Sorts by date
│   │       └── Returns: pd.DataFrame
│   │
│   ├── 🔄 DATA PREPROCESSING
│   │   │
│   │   ├── add_datetime_features(df)
│   │   │   ├── Extracts from saledate:
│   │   │   │   ├── saleYear
│   │   │   │   ├── saleMonth
│   │   │   │   ├── saleDay
│   │   │   │   ├── saleDayofweek
│   │   │   │   └── saleDayofyear
│   │   │   ├── Drops saledate column
│   │   │   └── Returns: pd.DataFrame
│   │   │
│   │   ├── handle_missing_values(df)
│   │   │   ├── For Numerical Columns:
│   │   │   │   ├── Add "_is_missing" indicator
│   │   │   │   └── Fill with median
│   │   │   ├── For Categorical Columns:
│   │   │   │   ├── Convert strings to categories
│   │   │   │   ├── Add "_is_missing" indicator
│   │   │   │   └── Convert to numeric codes (+1)
│   │   │   └── Returns: pd.DataFrame
│   │   │
│   │   └── preprocess_data(df)
│   │       ├── Calls add_datetime_features()
│   │       ├── Calls handle_missing_values()
│   │       └── Returns: pd.DataFrame
│   │
│   ├── ✂️ DATA SPLITTING
│   │   └── split_data(df, split_year=2012, target_col="SalePrice")
│   │       ├── Splits by year
│   │       ├── Training: year != 2012
│   │       ├── Validation: year == 2012
│   │       ├── Separates X and y
│   │       └── Returns: (X_train, y_train, X_valid, y_valid)
│   │
│   ├── 🤖 MODEL TRAINING
│   │   │
│   │   ├── train(n_estimators, max_depth, max_features, ...)
│   │   │   ├── Creates RandomForestRegressor
│   │   │   ├── Configurable hyperparameters:
│   │   │   │   ├── n_estimators (default: 40)
│   │   │   │   ├── max_depth (default: 10)
│   │   │   │   ├── max_features (default: 0.5)
│   │   │   │   ├── min_samples_split (default: 14)
│   │   │   │   ├── min_samples_leaf (default: 3)
│   │   │   │   └── max_samples (default: None)
│   │   │   ├── Fits model on training data
│   │   │   └── Returns: RandomForestRegressor
│   │   │
│   │   └── hyperparameter_tuning(param_grid, n_iter=20, cv=5, ...)
│   │       ├── Uses RandomizedSearchCV
│   │       ├── Default parameter grid:
│   │       │   ├── n_estimators: [10-100]
│   │       │   ├── max_depth: [None, 3, 5, 10]
│   │       │   ├── min_samples_split: [2-20]
│   │       │   ├── min_samples_leaf: [1-20]
│   │       │   └── max_features: [0.5, 1, "sqrt"]
│   │       ├── Performs cross-validation
│   │       ├── Updates self.model with best estimator
│   │       └── Returns: RandomizedSearchCV
│   │
│   ├── 📊 MODEL EVALUATION
│   │   └── evaluate(verbose=True)
│   │       ├── Makes predictions on train & validation
│   │       ├── Calculates metrics:
│   │       │   ├── Training MAE
│   │       │   ├── Validation MAE
│   │       │   ├── Training RMSLE
│   │       │   ├── Validation RMSLE
│   │       │   ├── Training R²
│   │       │   └── Validation R²
│   │       └── Returns: dict of scores
│   │
│   ├── 🔮 PREDICTIONS
│   │   ├── predict(X)
│   │   │   ├── Makes predictions on new data
│   │   │   └── Returns: np.array
│   │   │
│   │   └── save_predictions(test_data_path, output_path, id_col)
│   │       ├── Loads test data
│   │       ├── Preprocesses test data
│   │       ├── Ensures column alignment
│   │       ├── Makes predictions
│   │       ├── Creates submission DataFrame
│   │       ├── Saves to CSV
│   │       └── Returns: pd.DataFrame
│   │
│   └── 📈 VISUALIZATION
│       └── plot_feature_importance(n=20, figsize=(10, 8))
│           ├── Extracts feature importances
│           ├── Sorts by importance
│           ├── Creates horizontal bar plot
│           ├── Shows top n features
│           └── Returns: matplotlib.figure.Figure
│
├── 🔧 STANDALONE FUNCTIONS
│   │
│   ├── rmsle(y_true, y_pred)
│   │   ├── Calculates Root Mean Squared Log Error
│   │   ├── Uses sklearn's mean_squared_log_error
│   │   └── Returns: float
│   │
│   └── preprocess_data_standalone(df)
│       ├── Independent preprocessing function
│       ├── Adds datetime features
│       ├── Handles missing values
│       └── Returns: pd.DataFrame
│
└── 🚀 MAIN EXECUTION
    └── main()
        ├── 1. Initialize predictor
        ├── 2. Load data
        ├── 3. Preprocess data
        ├── 4. Split data (year 2012 for validation)
        ├── 5. Train model (with optimal hyperparameters)
        ├── 6. Evaluate model
        ├── 7. Plot feature importance
        ├── 8. Save predictions
        └── Returns: None


═══════════════════════════════════════════════════════════════

WORKFLOW: Complete ML Pipeline
═══════════════════════════════════════════════════════════════

    START
      │
      ├─→ [1] LOAD DATA
      │     └─→ load_data()
      │
      ├─→ [2] PREPROCESS
      │     ├─→ add_datetime_features()
      │     └─→ handle_missing_values()
      │
      ├─→ [3] SPLIT DATA
      │     └─→ split_data()
      │           ├─→ Train Set (pre-2012)
      │           └─→ Validation Set (2012)
      │
      ├─→ [4] TRAIN MODEL
      │     ├─→ Option A: train() [with known hyperparameters]
      │     └─→ Option B: hyperparameter_tuning() [auto-tune]
      │
      ├─→ [5] EVALUATE
      │     └─→ evaluate()
      │           ├─→ MAE
      │           ├─→ RMSLE (competition metric)
      │           └─→ R² Score
      │
      ├─→ [6] ANALYZE
      │     └─→ plot_feature_importance()
      │
      ├─→ [7] PREDICT
      │     └─→ save_predictions()
      │           ├─→ Load test data
      │           ├─→ Preprocess
      │           └─→ Generate submission.csv
      │
    END


═══════════════════════════════════════════════════════════════

KEY DESIGN PATTERNS
═══════════════════════════════════════════════════════════════

🎨 Object-Oriented Design
   └─→ Single class encapsulates entire ML pipeline

🔄 Pipeline Pattern
   └─→ Sequential steps: Load → Preprocess → Split → Train → Evaluate

🎯 Separation of Concerns
   ├─→ Data handling methods
   ├─→ Preprocessing methods
   ├─→ Model training methods
   └─→ Evaluation/visualization methods

📦 Encapsulation
   └─→ Internal state (df, model, X_train, etc.) protected in class

♻️ Reusability
   ├─→ Standalone functions for common operations
   └─→ Configurable parameters with sensible defaults

🔧 Flexibility
   ├─→ Custom hyperparameters
   ├─→ Optional hyperparameter tuning
   └─→ Verbose/quiet modes


═══════════════════════════════════════════════════════════════

USAGE EXAMPLES
═══════════════════════════════════════════════════════════════

Example 1: Basic Usage
─────────────────────
from bulldoz import BulldozerPricePredictor

predictor = BulldozerPricePredictor()
predictor.load_data("data/TrainAndValid.csv")
df = predictor.preprocess_data(predictor.df)
predictor.split_data(df)
predictor.train()
predictor.evaluate()


Example 2: With Hyperparameter Tuning
─────────────────────────────────────
predictor = BulldozerPricePredictor()
predictor.load_data("data/TrainAndValid.csv")
df = predictor.preprocess_data(predictor.df)
predictor.split_data(df)
predictor.hyperparameter_tuning(n_iter=20, cv=5)
predictor.evaluate()


Example 3: Quick Prediction
───────────────────────────
predictor = BulldozerPricePredictor()
# ... load and train model ...
predictor.save_predictions(
    test_data_path="data/Test.csv",
    output_path="submission.csv"
)


Example 4: Standalone Preprocessing
───────────────────────────────────
from bulldoz import preprocess_data_standalone

df = pd.read_csv("data.csv", parse_dates=["saledate"])
df_processed = preprocess_data_standalone(df)


═══════════════════════════════════════════════════════════════
```
