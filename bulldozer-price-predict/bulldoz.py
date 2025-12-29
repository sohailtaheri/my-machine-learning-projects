"""
Bulldozer Price Prediction using Machine Learning

This module contains a complete implementation for predicting bulldozer sale prices
using Random Forest Regression. The model predicts future sale prices based on
bulldozer characteristics and historical auction data.

Problem Definition:
    How well can we predict the future sale price of a bulldozer given its
    characteristics and previous examples of how much similar bulldozers have been sold for?

Evaluation Metric:
    RMSLE (Root Mean Squared Log Error) between actual and predicted auction prices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score


class BulldozerPricePredictor:
    """
    A class for predicting bulldozer sale prices using Random Forest Regression.

    This class handles the entire machine learning pipeline including:
    - Data loading and preprocessing
    - Feature engineering (datetime features)
    - Handling missing values
    - Model training and evaluation
    - Hyperparameter tuning
    - Making predictions
    """

    def __init__(self, random_state=42):
        """
        Initialize the BulldozerPricePredictor.

        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.df = None
        self.df_preprocessed = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None

    def load_data(self, filepath, parse_dates=True):
        """
        Load bulldozer data from CSV file.

        Args:
            filepath (str): Path to the CSV file
            parse_dates (bool): Whether to parse the saledate column as datetime

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        date_cols = ["saledate"] if parse_dates else None
        self.df = pd.read_csv(filepath, low_memory=False, parse_dates=date_cols)

        # Sort by date for time series consistency
        if parse_dates and "saledate" in self.df.columns:
            self.df.sort_values(by=["saledate"], inplace=True, ascending=True)

        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df

    def add_datetime_features(self, df):
        """
        Add datetime-based features from the saledate column.

        Args:
            df (pd.DataFrame): Dataframe with saledate column

        Returns:
            pd.DataFrame: Dataframe with additional datetime features
        """
        if "saledate" not in df.columns:
            print("Warning: 'saledate' column not found. Skipping datetime features.")
            return df

        df["saleYear"] = df.saledate.dt.year
        df["saleMonth"] = df.saledate.dt.month
        df["saleDay"] = df.saledate.dt.day
        df["saleDayofweek"] = df.saledate.dt.dayofweek
        df["saleDayofyear"] = df.saledate.dt.dayofyear

        # Drop the original saledate column
        df.drop("saledate", axis=1, inplace=True)

        return df

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataframe.

        For numerical columns: Fill with median and add a binary indicator column
        For categorical columns: Convert to category codes and add binary indicator

        Args:
            df (pd.DataFrame): Dataframe with missing values

        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        # First, convert string columns to categorical
        for label, content in df.items():
            if pd.api.types.is_string_dtype(content):
                df[label] = content.astype("category").cat.as_ordered()

        # Fill numerical missing values with median
        for label, content in df.items():
            if pd.api.types.is_numeric_dtype(content):
                if pd.isnull(content).sum() > 0:
                    # Add binary column indicating missing values
                    df[label + "_is_missing"] = pd.isnull(content)
                    # Fill with median
                    df[label] = content.fillna(content.median())

        # Convert categorical variables to numbers
        for label, content in df.items():
            if not pd.api.types.is_numeric_dtype(content):
                # Add binary column indicating missing values
                df[label + "_is_missing"] = pd.isnull(content)
                # Convert to category codes (+1 because pandas uses -1 for missing)
                df[label] = pd.Categorical(content).codes + 1

        return df

    def preprocess_data(self, df):
        """
        Complete preprocessing pipeline for the dataframe.

        Args:
            df (pd.DataFrame): Raw dataframe

        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df_processed = df.copy()
        df_processed = self.add_datetime_features(df_processed)
        df_processed = self.handle_missing_values(df_processed)

        print(f"Data preprocessed. Shape: {df_processed.shape}")
        return df_processed

    def split_data(self, df, split_year=2012, target_col="SalePrice"):
        """
        Split data into training and validation sets based on year.

        Args:
            df (pd.DataFrame): Preprocessed dataframe
            split_year (int): Year to use for validation set
            target_col (str): Name of the target column

        Returns:
            tuple: (X_train, y_train, X_valid, y_valid)
        """
        # Split based on year
        df_val = df[df.saleYear == split_year]
        df_train = df[df.saleYear != split_year]

        # Separate features and target
        self.X_train = df_train.drop(target_col, axis=1)
        self.y_train = df_train[target_col]
        self.X_valid = df_val.drop(target_col, axis=1)
        self.y_valid = df_val[target_col]

        print(f"Training set size: {len(self.X_train)}")
        print(f"Validation set size: {len(self.X_valid)}")

        return self.X_train, self.y_train, self.X_valid, self.y_valid

    def train(self, n_estimators=40, max_depth=10, max_features=0.5,
              min_samples_split=14, min_samples_leaf=3, max_samples=None,
              n_jobs=-1, verbose=True):
        """
        Train a Random Forest Regressor model.

        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees
            max_features (float): Max features to consider for splits
            min_samples_split (int): Minimum samples required to split
            min_samples_leaf (int): Minimum samples required in leaf
            max_samples (int): Maximum samples to use for each tree
            n_jobs (int): Number of parallel jobs
            verbose (bool): Whether to print training info

        Returns:
            RandomForestRegressor: Trained model
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Please run split_data() first.")

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            n_jobs=n_jobs,
            random_state=self.random_state
        )

        if verbose:
            print("Training model...")

        self.model.fit(self.X_train, self.y_train)

        if verbose:
            print("Model training completed!")

        return self.model

    def evaluate(self, verbose=True):
        """
        Evaluate the model on both training and validation sets.

        Args:
            verbose (bool): Whether to print scores

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Please run train() first.")

        train_preds = self.model.predict(self.X_train)
        val_preds = self.model.predict(self.X_valid)

        scores = {
            "Training MAE": mean_absolute_error(self.y_train, train_preds),
            "Valid MAE": mean_absolute_error(self.y_valid, val_preds),
            "Training RMSLE": rmsle(self.y_train, train_preds),
            "Valid RMSLE": rmsle(self.y_valid, val_preds),
            "Training R^2": r2_score(self.y_train, train_preds),
            "Valid R^2": r2_score(self.y_valid, val_preds)
        }

        if verbose:
            print("\n=== Model Evaluation Scores ===")
            for metric, value in scores.items():
                print(f"{metric}: {value:.4f}")

        return scores

    def hyperparameter_tuning(self, param_grid=None, n_iter=20, cv=5,
                            max_samples=10000, verbose=True):
        """
        Perform hyperparameter tuning using RandomizedSearchCV.

        Args:
            param_grid (dict): Parameter grid for random search
            n_iter (int): Number of parameter settings sampled
            cv (int): Number of cross-validation folds
            max_samples (int): Maximum samples per tree for faster training
            verbose (bool): Verbosity level

        Returns:
            RandomizedSearchCV: Fitted random search object
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Please run split_data() first.")

        # Default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                "n_estimators": np.arange(10, 100, 10),
                "max_depth": [None, 3, 5, 10],
                "min_samples_split": np.arange(2, 20, 2),
                "min_samples_leaf": np.arange(1, 20, 2),
                "max_features": [0.5, 1, "sqrt"],
                "max_samples": [max_samples]
            }

        base_model = RandomForestRegressor(
            n_jobs=-1,
            random_state=self.random_state,
            max_samples=max_samples
        )

        rs_model = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            verbose=verbose
        )

        print("Starting hyperparameter tuning...")
        rs_model.fit(self.X_train, self.y_train)

        print(f"\nBest parameters: {rs_model.best_params_}")

        self.model = rs_model.best_estimator_

        return rs_model

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X (pd.DataFrame): Features to predict on

        Returns:
            np.array: Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Please run train() first.")

        return self.model.predict(X)

    def plot_feature_importance(self, n=20, figsize=(10, 8)):
        """
        Plot feature importance of the trained model.

        Args:
            n (int): Number of top features to display
            figsize (tuple): Figure size

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        if self.model is None:
            raise ValueError("Model not trained. Please run train() first.")

        # Create dataframe of features and importances
        importance_df = pd.DataFrame({
            "features": self.X_train.columns,
            "feature_importances": self.model.feature_importances_
        }).sort_values("feature_importances", ascending=False).reset_index(drop=True)

        # Plot top n features
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(importance_df["features"][:n], importance_df["feature_importances"][:n])
        ax.set_ylabel("Features")
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {n} Feature Importances")
        ax.invert_yaxis()
        plt.tight_layout()

        return fig

    def save_predictions(self, test_data_path, output_path, id_col="SalesID"):
        """
        Load test data, make predictions, and save to CSV.

        Args:
            test_data_path (str): Path to test data CSV
            output_path (str): Path to save predictions
            id_col (str): Name of the ID column

        Returns:
            pd.DataFrame: Submission dataframe
        """
        # Load and preprocess test data
        df_test = pd.read_csv(test_data_path, low_memory=False, parse_dates=["saledate"])
        df_test_processed = self.preprocess_data(df_test)

        # Ensure columns match training data
        missing_cols = set(self.X_train.columns) - set(df_test_processed.columns)
        for col in missing_cols:
            df_test_processed[col] = False

        # Reorder columns to match training data
        df_test_processed = df_test_processed[self.X_train.columns]

        # Make predictions
        predictions = self.predict(df_test_processed)

        # Create submission dataframe
        submission = pd.DataFrame({
            id_col: df_test[id_col],
            "SalePrice": predictions
        })

        # Save to CSV
        submission.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

        return submission


def rmsle(y_true, y_pred):
    """
    Calculate Root Mean Squared Log Error.

    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values

    Returns:
        float: RMSLE score
    """
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def preprocess_data_standalone(df):
    """
    Standalone preprocessing function for data transformation.

    This function can be used independently of the BulldozerPricePredictor class.

    Args:
        df (pd.DataFrame): Raw dataframe with saledate column

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Add datetime parameters for saledate column
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayofweek"] = df.saledate.dt.dayofweek
    df["saleDayofyear"] = df.saledate.dt.dayofyear

    # Remove saledate column
    df.drop("saledate", axis=1, inplace=True)

    # Fill numeric rows with median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum() > 0:
                df[label + "_is_missing"] = pd.isnull(content)
                df[label] = content.fillna(content.median())
        else:
            df[label + "_is_missing"] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes + 1

    return df


def main():
    """
    Main function demonstrating usage of BulldozerPricePredictor.
    """
    # Initialize predictor
    predictor = BulldozerPricePredictor(random_state=42)

    # Load data
    predictor.load_data("data/TrainAndValid.csv", parse_dates=True)

    # Preprocess data
    df_processed = predictor.preprocess_data(predictor.df)

    # Split data
    predictor.split_data(df_processed, split_year=2012)

    # Train model with ideal hyperparameters
    predictor.train(
        n_estimators=40,
        max_depth=10,
        max_features=0.5,
        min_samples_split=14,
        min_samples_leaf=3,
        max_samples=None
    )

    # Evaluate model
    scores = predictor.evaluate()

    # Plot feature importance
    fig = predictor.plot_feature_importance(n=20)
    plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Make predictions on test data and save
    predictor.save_predictions(
        test_data_path="data/Test.csv",
        output_path="data/submission.csv"
    )

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
