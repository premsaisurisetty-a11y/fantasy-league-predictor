import numpy as np
import pandas as pd

class FantasyEnsembleRegressor:
    """
    Hybrid Ensemble combining Tuned LightGBM Regressor and Regularized Ridge Regressor.
    Blends non-linear interaction splits from LightGBM with robust linear point-system weights from Ridge.
    """
    def __init__(self, lgb_model, ridge_model, lgb_features, cat_cols, encoded_feature_columns, lgb_weight=0.35, ridge_weight=0.65):
        self.lgb_model = lgb_model
        self.ridge_model = ridge_model
        self.lgb_features = lgb_features
        self.cat_cols = cat_cols
        self.encoded_feature_columns = encoded_feature_columns
        self.lgb_weight = lgb_weight
        self.ridge_weight = ridge_weight
        self.feature_importances_ = getattr(lgb_model, "feature_importances_", np.zeros(len(lgb_features)))
        self.best_iteration_ = getattr(lgb_model, "best_iteration_", None)

    def predict(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.lgb_features)
        
        # 1. LightGBM Prediction
        lgb_preds = self.lgb_model.predict(X_df[self.lgb_features])
        
        # 2. Ridge Prediction with One-Hot Alignment
        X_enc = pd.get_dummies(X_df[self.lgb_features], columns=self.cat_cols, drop_first=True)
        for col in self.encoded_feature_columns:
            if col not in X_enc.columns:
                X_enc[col] = 0
        X_enc = X_enc[self.encoded_feature_columns]
        ridge_preds = self.ridge_model.predict(X_enc)
        
        # 3. Weighted Blend
        return self.lgb_weight * np.array(lgb_preds) + self.ridge_weight * np.array(ridge_preds)
