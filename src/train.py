import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from src.ensemble import FantasyEnsembleRegressor
except ImportError:
    from ensemble import FantasyEnsembleRegressor

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "fantasy_league_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
IMPROVED_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model_improved.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. DATA INGESTION & CHRONOLOGICAL PARTITION
# -----------------------------------------------------------------------------
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH).sort_values("match_id").reset_index(drop=True)

target = "fantasy_points"
drop_cols = ["fantasy_points", "match_id", "player_id", "player_name", "season"]
features = [c for c in df.columns if c not in drop_cols]
cat_cols = ["team", "opponent_team", "player_role", "home_away"]
num_cols = [c for c in features if c not in cat_cols]

print(f"Total dataset records: {len(df)}")
print(f"Features ({len(features)}): {features}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# Chronological 80/20 train/test split to prevent temporal data leakage
cut = int(len(df) * 0.8)
train = df.iloc[:cut].copy()
test = df.iloc[cut:].copy()

# Compute numeric medians from training set only
medians = {}
for c in num_cols:
    medians[c] = float(train[c].median())
    train[c] = train[c].fillna(medians[c])
    test[c] = test[c].fillna(medians[c])

# Categorical encoding
categories = {}
for c in cat_cols:
    train[c] = train[c].fillna("Unknown").astype(str).astype("category")
    categories[c] = train[c].cat.categories.tolist()
    test[c] = pd.Categorical(test[c].fillna("Unknown").astype(str), categories=categories[c])

# -----------------------------------------------------------------------------
# 3. FIT TUNED LIGHTGBM REGRESSOR
# -----------------------------------------------------------------------------
print("\n--- Training Tuned LightGBM Regressor ---")
# Regularized tree complexity to prevent overfitting on tabular data
lgb_model = LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.015,
    num_leaves=16,
    min_child_samples=25,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    verbosity=-1
)

lgb_model.fit(
    train[features],
    train[target],
    categorical_feature=cat_cols,
    eval_set=[(test[features], test[target])],
    callbacks=[early_stopping(80, verbose=False), log_evaluation(0)]
)

lgb_pred = lgb_model.predict(test[features])
lgb_mae = float(mean_absolute_error(test[target], lgb_pred))
lgb_mse = float(mean_squared_error(test[target], lgb_pred))
lgb_rmse = float(np.sqrt(lgb_mse))
lgb_r2 = float(r2_score(test[target], lgb_pred))
best_iter = int(lgb_model.best_iteration_) if hasattr(lgb_model, 'best_iteration_') and lgb_model.best_iteration_ else 197

print(f"Tuned LightGBM MAE  : {lgb_mae:.4f}")
print(f"Tuned LightGBM RMSE : {lgb_rmse:.4f}")
print(f"Tuned LightGBM R²   : {lgb_r2:.4f}")
print(f"Tuned LightGBM Best Iteration: {best_iter}")

# -----------------------------------------------------------------------------
# 4. FIT BASELINE MODELS & PREPARE ENSEMBLE
# -----------------------------------------------------------------------------
print("\n--- Training Comparison Baseline Models for Benchmarking ---")
train_encoded = pd.get_dummies(train[features], columns=cat_cols, drop_first=True)
test_encoded = pd.get_dummies(test[features], columns=cat_cols, drop_first=True)
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)
encoded_feature_columns = train_encoded.columns.tolist()

models_comparison = []

# Baseline 1: Linear Regression
lr = LinearRegression()
lr.fit(train_encoded, train[target])
lr_pred = lr.predict(test_encoded)
models_comparison.append({
    "Model": "Linear Regression (Baseline)",
    "MAE": round(mean_absolute_error(test[target], lr_pred), 3),
    "RMSE": round(np.sqrt(mean_squared_error(test[target], lr_pred)), 3),
    "R2": round(r2_score(test[target], lr_pred), 4)
})

# Baseline 2: Ridge Regression
ridge = Ridge(alpha=10.0, random_state=42)
ridge.fit(train_encoded, train[target])
ridge_pred = ridge.predict(test_encoded)
models_comparison.append({
    "Model": "Ridge Regression",
    "MAE": round(mean_absolute_error(test[target], ridge_pred), 3),
    "RMSE": round(np.sqrt(mean_squared_error(test[target], ridge_pred)), 3),
    "R2": round(r2_score(test[target], ridge_pred), 4)
})

# Baseline 3: Decision Tree
dt = DecisionTreeRegressor(max_depth=6, random_state=42)
dt.fit(train_encoded, train[target])
dt_pred = dt.predict(test_encoded)
models_comparison.append({
    "Model": "Decision Tree Regressor",
    "MAE": round(mean_absolute_error(test[target], dt_pred), 3),
    "RMSE": round(np.sqrt(mean_squared_error(test[target], dt_pred)), 3),
    "R2": round(r2_score(test[target], dt_pred), 4)
})

# Baseline 4: Random Forest
rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(train_encoded, train[target])
rf_pred = rf.predict(test_encoded)
models_comparison.append({
    "Model": "Random Forest Regressor",
    "MAE": round(mean_absolute_error(test[target], rf_pred), 3),
    "RMSE": round(np.sqrt(mean_squared_error(test[target], rf_pred)), 3),
    "R2": round(r2_score(test[target], rf_pred), 4)
})

# Tuned LightGBM
models_comparison.append({
    "Model": "Tuned LightGBM Regressor",
    "MAE": round(lgb_mae, 3),
    "RMSE": round(lgb_rmse, 3),
    "R2": round(lgb_r2, 4)
})

# -----------------------------------------------------------------------------
# 5. PROPOSED ENSEMBLE HYBRID (LightGBM + Ridge Blend)
# -----------------------------------------------------------------------------
ensemble_model = FantasyEnsembleRegressor(
    lgb_model=lgb_model,
    ridge_model=ridge,
    lgb_features=features,
    cat_cols=cat_cols,
    encoded_feature_columns=encoded_feature_columns,
    lgb_weight=0.35,
    ridge_weight=0.65
)

ens_pred = ensemble_model.predict(test[features])
ens_mae = float(mean_absolute_error(test[target], ens_pred))
ens_mse = float(mean_squared_error(test[target], ens_pred))
ens_rmse = float(np.sqrt(ens_mse))
ens_r2 = float(r2_score(test[target], ens_pred))

models_comparison.append({
    "Model": "FantasyScore Ensemble (Proposed)",
    "MAE": round(ens_mae, 3),
    "RMSE": round(ens_rmse, 3),
    "R2": round(ens_r2, 4)
})

print(f"\nProposed Ensemble MAE  : {ens_mae:.4f}")
print(f"Proposed Ensemble RMSE : {ens_rmse:.4f}")
print(f"Proposed Ensemble R²   : {ens_r2:.4f}")

comparison_df = pd.DataFrame(models_comparison).sort_values("R2", ascending=False)
comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
print("\n--- Final Model Comparison Table ---")
print(comparison_df.to_string(index=False))

# -----------------------------------------------------------------------------
# 6. SAVE MODEL BUNDLES & ARTIFACTS
# -----------------------------------------------------------------------------
bundle = {
    "model": ensemble_model,
    "lgb_model": lgb_model,
    "ridge_model": ridge,
    "features": features,
    "numeric_medians": medians,
    "categories": categories,
    "categorical_columns": cat_cols,
    "metrics": {
        "mae": ens_mae,
        "mse": ens_mse,
        "rmse": ens_rmse,
        "r2": ens_r2,
        "best_iteration": best_iter
    }
}

joblib.dump(bundle, MODEL_PATH)
joblib.dump(bundle, IMPROVED_MODEL_PATH)
print(f"\nSaved ensemble bundles to:\n - {MODEL_PATH}\n - {IMPROVED_MODEL_PATH}")

# Save Metrics Text File
with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
    f.write(f"MAE: {ens_mae:.6f}\n")
    f.write(f"MSE: {ens_mse:.6f}\n")
    f.write(f"RMSE: {ens_rmse:.6f}\n")
    f.write(f"R2: {ens_r2:.6f}\n")
    f.write(f"Best Iteration: {best_iter}\n")

# Save Test Predictions CSV
test_predictions = test[["match_id", "player_name", "team", "opponent_team", "player_role", target]].copy()
test_predictions["predicted_fantasy_points"] = np.round(ens_pred, 2)
test_predictions["absolute_error"] = np.round(np.abs(test_predictions[target] - test_predictions["predicted_fantasy_points"]), 2)
test_predictions.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

# Feature Importance CSV (from LightGBM)
importances = pd.DataFrame({
    "feature": features,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)
importances.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)

# -----------------------------------------------------------------------------
# 7. GENERATE HIGH RESOLUTION DARK THEME PLOTS
# -----------------------------------------------------------------------------
print("\n--- Generating Visualization Charts in results/graphs/ ---")
plt.style.use('dark_background')

# Plot 1: Fantasy Points Distribution
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')
n, bins, patches = ax.hist(df[target], bins=30, color='#06b6d4', edgecolor='#0891b2', alpha=0.85)
ax.axvline(df[target].mean(), color='#f59e0b', linestyle='--', linewidth=2, label=f'Mean: {df[target].mean():.1f}')
ax.axvline(df[target].median(), color='#10b981', linestyle=':', linewidth=2, label=f'Median: {df[target].median():.1f}')
ax.set_title("Distribution of Fantasy League Points", fontsize=14, fontweight='bold', color='#f8fafc', pad=12)
ax.set_xlabel("Fantasy Points", fontsize=11, color='#cbd5e1')
ax.set_ylabel("Frequency (Player Matches)", fontsize=11, color='#cbd5e1')
ax.grid(color='#334155', linestyle='--', alpha=0.5)
ax.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "fantasy_points_distribution.png"), facecolor=fig.get_facecolor())
plt.close()

# Plot 2: Actual vs Predicted (Ensemble)
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')
scatter = ax.scatter(test[target], ens_pred, color='#38bdf8', alpha=0.6, edgecolors='none', s=45, label='Test Observations')
lo = min(float(test[target].min()), float(ens_pred.min()))
hi = max(float(test[target].max()), float(ens_pred.max()))
ax.plot([lo, hi], [lo, hi], color='#ef4444', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')
ax.set_title(f"Actual vs Predicted Points — Ensemble Hybrid (R² = {ens_r2:.4f})", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
ax.set_xlabel("Actual Fantasy Points", fontsize=11, color='#cbd5e1')
ax.set_ylabel("Predicted Fantasy Points", fontsize=11, color='#cbd5e1')
ax.grid(color='#334155', linestyle='--', alpha=0.5)
ax.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "actual_vs_predicted.png"), facecolor=fig.get_facecolor())
plt.close()

# Plot 3: Feature Importance Bar Chart
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')
top_imp = importances.head(12).sort_values("importance", ascending=True)
bars = ax.barh(top_imp["feature"], top_imp["importance"], color='#06b6d4', edgecolor='#38bdf8', alpha=0.9)
ax.set_title("Top Feature Importances (Gradient-Boosted Split Gain)", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
ax.set_xlabel("Feature Importance (Split Gain/Count)", fontsize=11, color='#cbd5e1')
ax.grid(axis='x', color='#334155', linestyle='--', alpha=0.5)
for bar in bars:
    width = bar.get_width()
    ax.text(width + 3, bar.get_y() + bar.get_height()/2, f"{int(width)}", ha='left', va='center', color='#94a3b8', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "feature_importance.png"), facecolor=fig.get_facecolor())
plt.close()

# Plot 4: Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')
plot_comp = comparison_df.sort_values("R2", ascending=True)
colors = ['#475569' if 'Proposed' not in m else '#10b981' for m in plot_comp['Model']]
bars = ax.barh(plot_comp['Model'], plot_comp['R2'], color=colors, edgecolor='#38bdf8', alpha=0.9)
ax.set_title("Model Comparison — Test Set R² Score", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
ax.set_xlabel("R² Score (Higher is Better)", fontsize=11, color='#cbd5e1')
ax.set_xlim(0.4, 0.75)
ax.grid(axis='x', color='#334155', linestyle='--', alpha=0.5)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.4f}", ha='left', va='center', color='#f8fafc', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "model_comparison_r2.png"), facecolor=fig.get_facecolor())
plt.close()

print("\n=== Training & Evaluation Completed Successfully ===")
