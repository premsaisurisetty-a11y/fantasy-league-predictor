
import numpy as np
import pandas as pd


class FantasyEnsembleRegressor:
    """
    Hybrid Ensemble combining a tuned LightGBM Regressor and a regularized Ridge Regressor.

    LightGBM captures non-linear relationships and feature interactions.
    Ridge Regression captures linear relationships between features and fantasy points.

    Their predictions are combined using configurable weights to produce
    the final fantasy score prediction.
    """

    def __init__(
        self,
        lgb_model,
        ridge_model,
        lgb_features,
        cat_cols,
        encoded_feature_columns,
        lgb_weight=0.35,
        ridge_weight=0.65
    ):
        # Store the trained LightGBM and Ridge models.
        self.lgb_model = lgb_model
        self.ridge_model = ridge_model

        # Store the feature names and categorical columns used during training.
        self.lgb_features = lgb_features
        self.cat_cols = cat_cols

        # Store the exact feature-column order expected by the trained Ridge model.
        self.encoded_feature_columns = encoded_feature_columns

        # Store the weights used to combine the two models' predictions.
        self.lgb_weight = lgb_weight
        self.ridge_weight = ridge_weight

        # Expose LightGBM's feature importance values when available.
        # If unavailable, create a zero-filled array of the appropriate length.
        self.feature_importances_ = getattr(
            lgb_model,
            "feature_importances_",
            np.zeros(len(lgb_features))
        )

        # Store the best training iteration from LightGBM, if available.
        self.best_iteration_ = getattr(
            lgb_model,
            "best_iteration_",
            None
        )

    def predict(self, X):
        # Work with a copy if the input is already a DataFrame.
        # Otherwise, convert the input into a DataFrame using the expected feature names.
        X_df = (
            X.copy()
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(X, columns=self.lgb_features)
        )

        # ---------------------------------------------------------
        # STEP 1: Generate predictions using LightGBM
        # ---------------------------------------------------------

        # Select the required features in the same order used during training.
        # LightGBM predicts fantasy points using non-linear patterns in the data.
        lgb_preds = self.lgb_model.predict(X_df[self.lgb_features])

        # ---------------------------------------------------------
        # STEP 2: Prepare input and generate Ridge predictions
        # ---------------------------------------------------------

        # Convert categorical columns into numerical one-hot encoded columns.
        # drop_first=True removes the first category's indicator column
        # for each categorical feature.
        X_enc = pd.get_dummies(
            X_df[self.lgb_features],
            columns=self.cat_cols,
            drop_first=True
        )

        # Ensure every feature column expected by the trained Ridge model exists.
        # If an input category was missing, its corresponding encoded column
        # may not have been created, so add that column with zeros.
        for col in self.encoded_feature_columns:
            if col not in X_enc.columns:
                X_enc[col] = 0

        # Arrange the encoded columns in exactly the same order as during training.
        # This prevents the Ridge model from receiving features in the wrong order.
        X_enc = X_enc[self.encoded_feature_columns]

        # Generate fantasy point predictions using the trained Ridge model.
        ridge_preds = self.ridge_model.predict(X_enc)

        # ---------------------------------------------------------
        # STEP 3: Combine both models' predictions
        # ---------------------------------------------------------

        # Convert predictions to NumPy arrays and calculate the weighted average.
        # Default weights:
        #   LightGBM = 35%
        #   Ridge    = 65%
        # The combined result is the final predicted fantasy score.
        return (
            self.lgb_weight * np.array(lgb_preds)
            + self.ridge_weight * np.array(ridge_preds)
        )
