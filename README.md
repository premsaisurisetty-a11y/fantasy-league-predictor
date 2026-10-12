# Fantasy League Score Prediction Using LightGBM Gradient-Boosted Trees

An end-to-end Machine Learning web application and analytics platform for predicting player fantasy points using gradient-boosted decision trees.

---

## 1. Abstract

This project introduces **FantasyScore AI**, an advanced machine learning system utilizing a **Hybrid Ensemble (Tuned LightGBM Regressor + Ridge Regularized Regression)** to predict fantasy scores from multi-faceted match context, historical form, batting, and bowling metrics. Trained on a chronological dataset of 2,741 match instances across 163 players and 130 matches, the model prevents look-ahead data leakage by strictly partitioning 80% earlier matches for training and 20% subsequent matches for evaluation. Missing numerical features are imputed using training-set medians, while categorical parameters are handled natively by LightGBM's optimal split-finding algorithm. The model achieves a Mean Absolute Error (MAE) of **7.80**, a Root Mean Squared Error (RMSE) of **9.79**, and an $R^2$ score of **0.7035**, outperforming all standard baseline models (Linear Regression, Ridge, Random Forest, Decision Tree). The accompanying interactive Flask web application features smart auto-fill player profiling, live leaderboard rankings, an automated optimal squad builder, and in-depth model explainability.

---

## 2. Problem Statement

Selecting an optimal fantasy league lineup requires evaluating vast and volatile sports statistics—ranging from recent form and batting strike rates to bowling economy and opponent difficulty. Manual assessment is error-prone, biased, and incapable of capturing non-linear interactions across mixed categorical and numerical factors. The goal is to formulate a supervised regression model that reliably forecasts a continuous target variable (`fantasy_points`) and surfaces actionable predictions through an accessible web interface.

---

## 3. Project Objectives

1. **End-to-End Pipeline**: Develop a complete ML pipeline: Data Ingestion $\rightarrow$ Preprocessing $\rightarrow$ Chronological Train/Test Partition $\rightarrow$ LightGBM Training $\rightarrow$ Metric Evaluation $\rightarrow$ Web Deployment.
2. **Leakage-Free Feature Engineering**: Implement chronological splitting sorted by `match_id` and training-set-only median imputation.
3. **High-Accuracy Regression**: Train a tuned LightGBM model utilizing histogram-based tree learning with early stopping.
4. **Smart Prediction Dashboard**: Provide automatic player profile pre-population, customizable scenario testing, and feature contribution breakdowns.
5. **Dynamic Leaderboard & Analytics**: Rank players across the league and present comprehensive evaluation metrics, baseline comparisons, and diagnostic visualizations.

---

## 4. Dataset Overview

- **Source File**: `data/raw/fantasy_league_dataset.csv`
- **Total Records**: 2,741 match rows
- **Features**: 22 columns (1 target, 4 identifier/metadata, 4 categorical, 13 numerical)
- **Target Variable**: `fantasy_points` (Continuous numeric)
- **Unique Players**: 163
- **Unique Matches**: 130
- **Season**: 2026

### Feature Breakdown

| Category | Columns |
| :--- | :--- |
| **Identifiers** | `match_id`, `player_id`, `player_name`, `season` |
| **Match Context** | `team`, `opponent_team`, `player_role`, `home_away`, `opponent_strength` |
| **Batting Performance** | `batting_runs`, `batting_strike_rate`, `batting_average`, `boundaries` |
| **Bowling Performance** | `bowling_wickets`, `bowling_economy`, `bowling_average`, `overs_bowled` |
| **Form & Availability** | `recent_fantasy_points`, `recent_average_points`, `player_form`, `playing_time_percentage` |
| **Target Variable** | `fantasy_points` |

---

## 5. Technology Stack

- **Machine Learning**: LightGBM, Scikit-learn, NumPy, Pandas
- **Model Serialization**: Joblib
- **Visualization**: Matplotlib
- **Backend**: Python 3.11, Flask
- **Frontend**: HTML5, Vanilla CSS3 (Custom Dark Theme Design System), JavaScript (ES6+)

---

## 6. Methodology & Architecture

```text
       +---------------------------------------------+
       |   Historical Dataset (2,741 Records)        |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Chronological Sort (by match_id)          |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   80/20 Train/Test Split (No Data Leakage)  |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Preprocessing & Median Imputation         |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   LightGBM Gradient-Boosted Trees Fitting   |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Model Evaluation (MAE, MSE, RMSE, R²)     |
       +---------------------------------------------+
                              |
                              v
       +---------------------------------------------+
       |   Flask Web Application & Player Rankings   |
       +---------------------------------------------+
```

---

## 7. Evaluation Metrics & Results

Evaluation was performed on the unseen 20% test partition (549 match records).

### Model Performance Metrics

| Metric | Formula / Description | Saved Project Result |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$ | **7.80** |
| **Mean Squared Error (MSE)** | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | **95.91** |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\text{MSE}}$ | **9.79** |
| **$R^2$ Score (Variance Explained)** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | **0.7035** |
| **Best Tree Iteration** | Optimal early-stopping iteration | **502** |

### Baseline & Model Benchmarking Comparison

| Model | MAE | RMSE | $R^2$ |
| :--- | :---: | :---: | :---: |
| **FantasyScore Ensemble (Proposed - LightGBM + Ridge)** | **7.804** | **9.793** | **0.7035** |
| **Ridge Regression** | 7.829 | 9.857 | 0.6997 |
| **Linear Regression (Baseline)** | 7.859 | 9.876 | 0.6985 |
| **Tuned LightGBM Regressor** | 8.026 | 10.008 | 0.6904 |
| **Random Forest Regressor** | 8.678 | 10.808 | 0.6389 |
| **Decision Tree Regressor** | 9.805 | 12.360 | 0.5278 |

---

## 8. Feature Importance

Based on decision split gain from `results/feature_importance.csv`:
1. `batting_runs` (338 splits)
2. `recent_fantasy_points` (315 splits)
3. `playing_time_percentage` (263 splits)
4. `bowling_wickets` (256 splits)
5. `recent_average_points` (248 splits)
6. `batting_average` (244 splits)
7. `batting_strike_rate` (240 splits)
8. `opponent_strength` (232 splits)
9. `bowling_economy` (207 splits)
10. `bowling_average` (186 splits)

---

## 9. Project Directory Structure

```text
Fantasy_Score_Prediction/
│
├── data/
│   ├── raw/
│   │   └── fantasy_league_dataset.csv
│   └── processed/
│       └── fantasy_league_cleaned.csv
│
├── models/
│   ├── lightgbm_model.pkl
│   └── lightgbm_model_improved.pkl
│
├── results/
│   ├── eda_graphs/
│   │   ├── 01_target_distribution.png
│   │   ├── 02_correlation_heatmap.png
│   │   ├── 03_role_performance_distribution.png
│   │   ├── 04_batting_impact_analysis.png
│   │   ├── 05_bowling_impact_analysis.png
│   │   ├── 06_form_and_recent_trends.png
│   │   ├── 07_venue_and_opponent_strength.png
│   │   ├── 08_team_performance_comparison.png
│   │   ├── 09_missing_values_audit.png
│   │   └── 10_outlier_spread_boxplots.png
│   ├── graphs/
│   │   ├── actual_vs_predicted.png
│   │   ├── fantasy_points_distribution.png
│   │   └── feature_importance.png
│   ├── eda_summary_statistics.csv
│   ├── eda_correlation_matrix.csv
│   ├── feature_importance.csv
│   ├── metrics.txt
│   ├── model_comparison.csv
│   └── predictions.csv
│
├── app/
│   ├── app.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── rankings.html
│   │   └── model.html
│   └── static/
│       ├── style.css
│       └── script.js
│
├── src/
│   ├── eda.py
│   ├── train.py
│   └── predict.py
│
├── notebooks/
│   ├── EDA_Fantasy_League.ipynb
│   └── fantasy_prediction.ipynb
│
├── EDA_REPORT.md
├── requirements.txt
├── README.md
└── RUN_APP.txt
```

---

## 10. Installation & How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Exploratory Data Analysis (EDA)
```bash
python src/eda.py
```
*Or open and run the interactive Jupyter notebook:*
```bash
jupyter notebook notebooks/EDA_Fantasy_League.ipynb
```

### Step 3: (Optional) Re-Train the Model
```bash
python src/train.py
```

### Step 4: Launch the Web Application
```bash
python app/app.py
```

### Step 4: Access in Web Browser
Open your browser at:
```text
http://127.0.0.1:5000
```

---

## 11. Future Scope

1. **Live Sports API Integration**: Real-time ball-by-ball score and odds ingestion.
2. **Pitch & Weather Context**: Incorporating humidity, pitch wear, and ground dimensions.
3. **Player Injury & Matchup Matrix**: Head-to-head batter vs. bowler matchup features.
4. **Automated Lineup Optimization**: Linear programming solver (Knapsack optimization) for auto-generating optimal 11-player fantasy squads under salary cap constraints.
5. **SHAP TreeExplainer**: Deep local and global SHAP interpretability graphs.
