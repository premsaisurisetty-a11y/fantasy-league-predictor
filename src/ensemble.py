import numpy as np
import pandas as pd

class FantasyEnsemblePredictor:
    """
    Predictor wrapper that blends LightGBM and Ridge models.
    Supports initialization from a dictionary bundle or direct arguments.
    """
    def __init__(self, bundle=None, lgb_model=None, ridge_model=None, lgb_features=None, cat_cols=None, encoded_feature_columns=None, lgb_weight=0.35, ridge_weight=0.65, feature_importances=None):
        if bundle is not None and isinstance(bundle, dict):
            self.lgb_model = bundle.get("lgb_model")
            self.ridge_model = bundle.get("ridge_model")
            self.lgb_weight = bundle.get("lgb_weight", 0.35)
            self.ridge_weight = bundle.get("ridge_weight", 0.65)
            self.features = bundle.get("features", [])
            self.cat_cols = bundle.get("categorical_columns", [])
            self.encoded_feature_columns = bundle.get("encoded_feature_columns", [])
            self.feature_importances_ = np.array(bundle.get("feature_importances", []))
            self.best_iteration_ = bundle.get("metrics", {}).get("best_iteration", 502)
        else:
            self.lgb_model = lgb_model
            self.ridge_model = ridge_model
            self.lgb_weight = lgb_weight
            self.ridge_weight = ridge_weight
            self.features = lgb_features or []
            self.cat_cols = cat_cols or []
            self.encoded_feature_columns = encoded_feature_columns or []
            if feature_importances is not None:
                self.feature_importances_ = np.array(feature_importances)
            elif lgb_model is not None and hasattr(lgb_model, "feature_importances_"):
                self.feature_importances_ = lgb_model.feature_importances_
            else:
                self.feature_importances_ = np.zeros(len(self.features))
            self.best_iteration_ = getattr(lgb_model, "best_iteration_", 502)

    def predict(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.features)
        
        # 1. LightGBM prediction
        lgb_preds = self.lgb_model.predict(X_df[self.features])
        
        # 2. Ridge prediction
        X_enc = pd.get_dummies(X_df[self.features], columns=self.cat_cols, drop_first=True)
        for col in self.encoded_feature_columns:
            if col not in X_enc.columns:
                X_enc[col] = 0
        X_enc = X_enc[self.encoded_feature_columns]
        ridge_preds = self.ridge_model.predict(X_enc)
        
        # 3. Weighted blend
        return self.lgb_weight * np.array(lgb_preds) + self.ridge_weight * np.array(ridge_preds)

# Backwards compatibility alias
FantasyEnsembleRegressor = FantasyEnsemblePredictor
