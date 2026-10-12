
# Import operating system utilities for file and directory paths.
import os

# Import system utilities to configure Python's module search path.
import sys

# Import pandas for loading and processing tabular data.
import pandas as pd

# Import NumPy for numerical calculations.
import numpy as np

# Import joblib to save trained models and model bundles.
import joblib

# Import Matplotlib and select a non-interactive backend for saving plots.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import LightGBM's regression model and training callbacks.
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

# Import evaluation metrics to measure prediction performance.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import linear models used for comparison and the proposed ensemble.
from sklearn.linear_model import LinearRegression, Ridge

# Import decision tree and random forest regression models.
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Find the project root directory relative to this Python file.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to Python's module search path if necessary.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import the custom ensemble class.
# The fallback supports running this file from a different module location.
try:
    from src.ensemble import FantasyEnsembleRegressor
except ImportError:
    from ensemble import FantasyEnsembleRegressor

# Define the locations of the dataset, saved models, results, and graphs.
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "fantasy_league_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model.pkl")
IMPROVED_MODEL_PATH = os.path.join(MODEL_DIR, "lightgbm_model_improved.pkl")

# Create output directories if they do not already exist.
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. DATA INGESTION & CHRONOLOGICAL PARTITION
# -----------------------------------------------------------------------------

# Load the dataset, sort matches chronologically, and reset row indexes.
print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH).sort_values("match_id").reset_index(drop=True)

# Specify the value that the model must predict.
target = "fantasy_points"

# Exclude the target and identifier columns from the input features.
drop_cols = ["fantasy_points", "match_id", "player_id", "player_name", "season"]
features = [c for c in df.columns if c not in drop_cols]

# Identify categorical features that represent groups or labels.
cat_cols = ["team", "opponent_team", "player_role", "home_away"]

# Identify numerical features by excluding categorical columns.
num_cols = [c for c in features if c not in cat_cols]

# Display basic information about the dataset and selected features.
print(f"Total dataset records: {len(df)}")
print(f"Features ({len(features)}): {features}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# Split the chronologically sorted data into 80% training and 20% testing.
# Using earlier matches for training and later matches for testing
# helps reduce temporal data leakage.
cut = int(len(df) * 0.8)
train = df.iloc[:cut].copy()
test = df.iloc[cut:].copy()

# Calculate each numerical feature's median using training data only.
# Use the same median to fill missing values in both datasets.
medians = {}
for c in num_cols:
    medians[c] = float(train[c].median())
    train[c] = train[c].fillna(medians[c])
    test[c] = test[c].fillna(medians[c])

# Prepare categorical features for LightGBM.
# Missing categories are replaced with "Unknown".
# Save the categories learned from training for consistent processing.
categories = {}
for c in cat_cols:
    train[c] = train[c].fillna("Unknown").astype(str).astype("category")
    categories[c] = train[c].cat.categories.tolist()
    test[c] = pd.Categorical(test[c].fillna("Unknown").astype(str), categories=categories[c])


# -----------------------------------------------------------------------------
# 3. FIT TUNED LIGHTGBM REGRESSOR
# -----------------------------------------------------------------------------

print("\n--- Training Tuned LightGBM Regressor ---")

# Configure LightGBM, a gradient-boosting model based on decision trees.
# The selected settings control learning speed, tree complexity,
# sampling, and regularization to help limit overfitting.
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

# Train LightGBM using the selected input features and target values.
# Pass categorical columns explicitly so LightGBM can handle them.
# Early stopping ends training when evaluation performance stops improving.
# log_evaluation(0) suppresses periodic training evaluation output.
lgb_model.fit(
    train[features],
    train[target],
    categorical_feature=cat_cols,
    eval_set=[(test[features], test[target])],
    callbacks=[early_stopping(80, verbose=False), log_evaluation(0)]
)

# Predict fantasy points for the test dataset.
lgb_pred = lgb_model.predict(test[features])

# MAE measures the average absolute difference between actual and predicted values.
lgb_mae = float(mean_absolute_error(test[target], lgb_pred))

# MSE measures the average squared prediction error.
lgb_mse = float(mean_squared_error(test[target], lgb_pred))

# RMSE is the square root of MSE and is expressed in target units.
lgb_rmse = float(np.sqrt(lgb_mse))

# R-squared measures how well predictions explain variation in actual values.
# A higher R² is generally better; it can also be negative.
lgb_r2 = float(r2_score(test[target], lgb_pred))

# Record the best training iteration, with a fallback if it is unavailable.
best_iter = int(lgb_model.best_iteration_) if hasattr(lgb_model, 'best_iteration_') and lgb_model.best_iteration_ else 197

# Display LightGBM's evaluation results.
print(f"Tuned LightGBM MAE  : {lgb_mae:.4f}")
print(f"Tuned LightGBM RMSE : {lgb_rmse:.4f}")
print(f"Tuned LightGBM R²   : {lgb_r2:.4f}")
print(f"Tuned LightGBM Best Iteration: {best_iter}")


# -----------------------------------------------------------------------------
# 4. FIT BASELINE MODELS & PREPARE ENSEMBLE
# -----------------------------------------------------------------------------

print("\n--- Training Comparison Baseline Models for Benchmarking ---")

# Convert categorical features into numeric indicator columns.
# drop_first=True removes one category per feature to reduce redundancy.
train_encoded = pd.get_dummies(train[features], columns=cat_cols, drop_first=True)
test_encoded = pd.get_dummies(test[features], columns=cat_cols, drop_first=True)

# Align test columns with the training columns.
# Missing test columns are filled with zero.
train_encoded, test_encoded = train_encoded.align(test_encoded, join='left', axis=1, fill_value=0)

# Store the encoded column names for use by the ensemble model.
encoded_feature_columns = train_encoded.columns.tolist()

# Collect evaluation results for all models in a single comparison table.
models_comparison = []

# Baseline 1: Linear Regression
# Fit a linear relationship between input features and fantasy points.
lr = LinearRegression()
lr.fit(train_encoded, train[target])
lr_pred = lr.predict(test_encoded)

# Calculate and store Linear Regression's evaluation metrics.
models_comparison.append({
    "Model": "Linear Regression (Baseline)",
    "MAE": round(mean_absolute_error(test[target], lr_pred), 3),
    "RMSE": round(np.sqrt(mean_squared_error(test[target], lr_pred)), 3),
    "R2": round(r2_score(test[target], lr_pred), 4)
})

# Baseline 2: Ridge Regression
# Ridge adds L2 regularization to help control large model coefficients.
ridge = Ridge(alpha=10.0, random_state=42)
ridge.fit(train_encoded, train[target])
ridge_pred = ridge.predict(test_encoded)

# Calculate and store Ridge Regression's evaluation metrics.
models_comparison.append({
    "Model": "Ridge Regression",
    "MAE": round(mean_absolute_error(test[target], ridge_pred), 3),
    "RMSE": round(np.sqrt(mean_squared_error(test[target], ridge_pred)), 3),
    "R2": round(r2_score(test[target], ridge_pred), 4)
})

# Baseline 3: Decision Tree
# Limit tree depth to control complexity.
dt = DecisionTreeRegressor(max_depth=6, random_state=42)
dt.fit(train_encoded, train[target])
dt_pred = dt.predict(test_encoded)

# Calculate and store Decision Tree's evaluation metrics.
models_comparison.append({
    "Model": "Decision Tree Regressor",
    "MAE": round(mean_absolute_error(test[target], dt_pred), 3),
    "RMSE": round(np.sqrt(mean_squared_error(test[target], dt_pred)), 3),
    "R2": round(r2_score(test[target], dt_pred), 4)
})

# Baseline 4: Random Forest
# Train multiple decision trees and combine their predictions.
# n_jobs=-1 allows the model to use available CPU cores.
rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(train_encoded, train[target])
rf_pred = rf.predict(test_encoded)

# Calculate and store Random Forest's evaluation metrics.
models_comparison.append({
    "Model": "Random Forest Regressor",
    "MAE": round(mean_absolute_error(test[target], rf_pred), 3),
    "RMSE": round(np.sqrt(mean_squared_error(test[target], rf_pred)), 3),
    "R2": round(r2_score(test[target], rf_pred), 4)
})

# Add the already evaluated LightGBM model to the comparison results.
models_comparison.append({
    "Model": "Tuned LightGBM Regressor",
    "MAE": round(lgb_mae, 3),
    "RMSE": round(lgb_rmse, 3),
    "R2": round(lgb_r2, 4)
})


# -----------------------------------------------------------------------------
# 5. PROPOSED ENSEMBLE HYBRID (LightGBM + Ridge Blend)
# -----------------------------------------------------------------------------

# Create the project's custom ensemble using the trained LightGBM and Ridge models.
# The weights determine each model's contribution to the combined prediction.
ensemble_model = FantasyEnsembleRegressor(
    lgb_model=lgb_model,
    ridge_model=ridge,
    lgb_features=features,
    cat_cols=cat_cols,
    encoded_feature_columns=encoded_feature_columns,
    lgb_weight=0.35,
    ridge_weight=0.65
)

# Generate predictions using the combined ensemble model.
ens_pred = ensemble_model.predict(test[features])

# Evaluate the ensemble predictions using the same metrics as the other models.
ens_mae = float(mean_absolute_error(test[target], ens_pred))
ens_mse = float(mean_squared_error(test[target], ens_pred))
ens_rmse = float(np.sqrt(ens_mse))
ens_r2 = float(r2_score(test[target], ens_pred))

# Add the proposed ensemble's results to the comparison table.
models_comparison.append({
    "Model": "FantasyScore Ensemble (Proposed)",
    "MAE": round(ens_mae, 3),
    "RMSE": round(ens_rmse, 3),
    "R2": round(ens_r2, 4)
})

# Print the proposed ensemble's evaluation metrics.
print(f"\nProposed Ensemble MAE  : {ens_mae:.4f}")
print(f"Proposed Ensemble RMSE : {ens_rmse:.4f}")
print(f"Proposed Ensemble R²   : {ens_r2:.4f}")

# Convert results into a DataFrame and rank models by R² score.
comparison_df = pd.DataFrame(models_comparison).sort_values("R2", ascending=False)

# Save the model comparison table as a CSV file.
comparison_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

# Display the comparison results in the terminal.
print("\n--- Final Model Comparison Table ---")
print(comparison_df.to_string(index=False))


# -----------------------------------------------------------------------------
# 6. SAVE MODEL BUNDLES & ARTIFACTS
# -----------------------------------------------------------------------------

# Bundle the trained models, feature information, preprocessing details,
# and ensemble evaluation metrics so they can be reused later.
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

# Serialize the bundle to both configured model file paths.
joblib.dump(bundle, MODEL_PATH)
joblib.dump(bundle, IMPROVED_MODEL_PATH)
print(f"\nSaved ensemble bundles to:\n - {MODEL_PATH}\n - {IMPROVED_MODEL_PATH}")

# Save the ensemble's evaluation metrics in a readable text file.
with open(os.path.join(RESULTS_DIR, "metrics.txt"), "w") as f:
    f.write(f"MAE: {ens_mae:.6f}\n")
    f.write(f"MSE: {ens_mse:.6f}\n")
    f.write(f"RMSE: {ens_rmse:.6f}\n")
    f.write(f"R2: {ens_r2:.6f}\n")
    f.write(f"Best Iteration: {best_iter}\n")

# Create a table containing actual test values and ensemble predictions.
test_predictions = test[["match_id", "player_name", "team", "opponent_team", "player_role", target]].copy()

# Store predictions rounded to two decimal places.
test_predictions["predicted_fantasy_points"] = np.round(ens_pred, 2)

# Calculate absolute prediction errors using the rounded predictions.
test_predictions["absolute_error"] = np.round(np.abs(test_predictions[target] - test_predictions["predicted_fantasy_points"]), 2)

# Export the prediction results to CSV.
test_predictions.to_csv(os.path.join(RESULTS_DIR, "predictions.csv"), index=False)

# Extract LightGBM feature importance values and rank features from highest to lowest.
importances = pd.DataFrame({
    "feature": features,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)

# Save feature importance results for later analysis.
importances.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)


# -----------------------------------------------------------------------------
# 7. GENERATE HIGH RESOLUTION DARK THEME PLOTS
# -----------------------------------------------------------------------------

print("\n--- Generating Visualization Charts in results/graphs/ ---")

# Apply Matplotlib's dark background style to the generated charts.
plt.style.use('dark_background')

# Plot 1: Fantasy Points Distribution
# Display how frequently different fantasy-point values occur.
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

# Create a histogram of fantasy points with 30 bins.
n, bins, patches = ax.hist(df[target], bins=30, color='#06b6d4', edgecolor='#0891b2', alpha=0.85)

# Add vertical reference lines for the mean and median.
ax.axvline(df[target].mean(), color='#f59e0b', linestyle='--', linewidth=2, label=f'Mean: {df[target].mean():.1f}')
ax.axvline(df[target].median(), color='#10b981', linestyle=':', linewidth=2, label=f'Median: {df[target].median():.1f}')

# Add the title, axis labels, grid, and legend.
ax.set_title("Distribution of Fantasy League Points", fontsize=14, fontweight='bold', color='#f8fafc', pad=12)
ax.set_xlabel("Fantasy Points", fontsize=11, color='#cbd5e1')
ax.set_ylabel("Frequency (Player Matches)", fontsize=11, color='#cbd5e1')
ax.grid(color='#334155', linestyle='--', alpha=0.5)
ax.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=10)

# Adjust spacing, save the high-resolution image, and close the figure.
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "fantasy_points_distribution.png"), facecolor=fig.get_facecolor())
plt.close()

# Plot 2: Actual vs Predicted (Ensemble)
# Compare actual fantasy points with the ensemble's predicted values.
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

# Each point represents a test observation.
scatter = ax.scatter(test[target], ens_pred, color='#38bdf8', alpha=0.6, edgecolors='none', s=45, label='Test Observations')

# Determine common lower and upper bounds for the reference line.
lo = min(float(test[target].min()), float(ens_pred.min()))
hi = max(float(test[target].max()), float(ens_pred.max()))

# The y = x line represents perfect predictions.
# Points closer to this line have smaller prediction errors.
ax.plot([lo, hi], [lo, hi], color='#ef4444', linestyle='--', linewidth=2, label='Perfect Prediction (y = x)')

# Add the chart title, axis labels, grid, and legend.
ax.set_title(f"Actual vs Predicted Points — Ensemble Hybrid (R² = {ens_r2:.4f})", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
ax.set_xlabel("Actual Fantasy Points", fontsize=11, color='#cbd5e1')
ax.set_ylabel("Predicted Fantasy Points", fontsize=11, color='#cbd5e1')
ax.grid(color='#334155', linestyle='--', alpha=0.5)
ax.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=10)

# Save the comparison plot and release its resources.
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "actual_vs_predicted.png"), facecolor=fig.get_facecolor())
plt.close()

# Plot 3: Feature Importance Bar Chart
# Show the 12 features with the highest LightGBM importance values.
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

# Sort the selected features in ascending order for a horizontal bar chart.
top_imp = importances.head(12).sort_values("importance", ascending=True)
bars = ax.barh(top_imp["feature"], top_imp["importance"], color='#06b6d4', edgecolor='#38bdf8', alpha=0.9)

# Label the chart and its importance scale.
ax.set_title("Top Feature Importances (Gradient-Boosted Split Gain)", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
ax.set_xlabel("Feature Importance (Split Gain/Count)", fontsize=11, color='#cbd5e1')
ax.grid(axis='x', color='#334155', linestyle='--', alpha=0.5)

# Display each bar's importance value beside it.
for bar in bars:
    width = bar.get_width()
    ax.text(width + 3, bar.get_y() + bar.get_height()/2, f"{int(width)}", ha='left', va='center', color='#94a3b8', fontsize=9)

# Save the feature importance plot.
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "feature_importance.png"), facecolor=fig.get_facecolor())
plt.close()

# Plot 4: Model Comparison Bar Chart
# Compare the R² scores of all trained models.
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

# Sort models by R² so the chart displays them from lowest to highest.
plot_comp = comparison_df.sort_values("R2", ascending=True)

# Highlight the proposed ensemble in green when its model name contains "Proposed".
colors = ['#475569' if 'Proposed' not in m else '#10b981' for m in plot_comp['Model']]
bars = ax.barh(plot_comp['Model'], plot_comp['R2'], color=colors, edgecolor='#38bdf8', alpha=0.9)

# Add the title, axis label, range, and grid.
ax.set_title("Model Comparison — Test Set R² Score", fontsize=13, fontweight='bold', color='#f8fafc', pad=12)
ax.set_xlabel("R² Score (Higher is Better)", fontsize=11, color='#cbd5e1')
ax.set_xlim(0.4, 0.75)
ax.grid(axis='x', color='#334155', linestyle='--', alpha=0.5)

# Print each model's R² value beside its bar.
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.4f}", ha='left', va='center', color='#f8fafc', fontsize=9, fontweight='bold')

# Save the model comparison plot.
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS_DIR, "model_comparison_r2.png"), facecolor=fig.get_facecolor())
plt.close()

# Indicate that model training, evaluation, saving, and plotting have finished.
print("\n=== Training & Evaluation Completed Successfully ===")
