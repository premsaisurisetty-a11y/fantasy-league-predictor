# Machine Learning Project Report: Fantasy League Score Prediction
## Project Phase: Upto Exploratory Data Analysis (EDA) — Steps 1 to 5
**Course / Track:** Machine Learning (Sem 2-1)  
**Dataset:** `data/raw/fantasy_league_dataset.csv` (2,741 match rows, 22 features, 130 unique matches, 163 players, 10 teams)  
**Target Variable:** `fantasy_points` (Continuous Numeric)

---

# 1. Problem Definition

### 1.1 What problem are you solving?
We are developing a supervised Machine Learning regression system that predicts the total **fantasy points (`fantasy_points`)** earned by individual cricket players in upcoming competitive matches. Points are calculated from on-field contributions:
- **Batting:** Runs scored, boundary boundaries (4s, 6s), milestone strike-rate bonuses.
- **Bowling:** Wickets taken (25 pts each), economy rate bonuses/penalties, maiden overs, haul multipliers.
- **Participation & Context:** Playing XI inclusion, fielding events, opponent strength, and venue context.

### 1.2 Why is this problem important?
- **High Statistical Volatility:** Cricket performances vary significantly based on matchup dynamics, pitch behavior, and player form. Manual judgment is subjective, biased, and inconsistent.
- **Constrained Portfolio Selection:** Fantasy sports managers must construct a balanced 11-player lineup while adhering to strict salary caps and assigning $2\times$ captain and $1.5\times$ vice-captain multipliers.
- **Data-Driven Edge:** Replacing human guesswork with gradient-boosted decision trees surfaces empirical player expected values ($\mathbb{E}[\text{Points}]$).

### 1.3 Who will use the solution?
1. **Fantasy Sports Enthusiasts & Managers:** To build high-expected-value fantasy squads and select optimal captains across platforms (Dream11, My11Circle, Fantasy Premier League).
2. **Cricket Analytics Teams & Coaches:** To assess projected player productivity against specific bowling attacks and venues.
3. **Sports Media & Gaming Portals:** To power automated match previews, projected leaderboards, and real-time player ratings.

### 1.4 What is the expected output?
The system outputs a **continuous numeric score** (`predicted_fantasy_points` $\in \mathbb{R}^+$) for any player given their match context, recent rolling averages, batting, and bowling metrics, optimized for **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.

---

# 2. Dataset Collection

- **Dataset Source:** Historical IPL Match Performance & Official Fantasy Scoring Matrix (2026 Season).
- **File Storage Location:**
  - Raw Data: `data/raw/fantasy_league_dataset.csv`
  - Cleaned Preprocessed Data: `data/processed/fantasy_league_cleaned.csv`
- **Total Records (Rows):** **2,741 match instances**
- **Total Attributes (Columns):** **22 features**
- **Target / Output Column:** **`fantasy_points`** (Continuous Float)
- **Unique Fixtures:** 130 matches
- **Unique Players:** 163 players across 10 IPL franchises

### Feature Breakdown

| Category | Columns | Data Type | Description |
| :--- | :--- | :---: | :--- |
| **Identifiers** | `match_id`, `player_id`, `player_name`, `season` | Numeric / Text | Tracking and sequential ordering (excluded from training) |
| **Match Context** | `team`, `opponent_team`, `player_role`, `home_away`, `opponent_strength` | Categorical / Ordinal | Team matchup, role assignment, venue, and opponent difficulty |
| **Batting Performance** | `batting_runs`, `batting_strike_rate`, `batting_average`, `boundaries` | Numeric | Direct batting volume, acceleration, and boundary hitting |
| **Bowling Performance** | `bowling_wickets`, `bowling_economy`, `bowling_average`, `overs_bowled` | Numeric | Wicket efficiency, economy control, and bowling workload |
| **Form & Availability** | `recent_fantasy_points`, `recent_average_points`, `player_form`, `playing_time_percentage` | Numeric / Ordinal | Lag-1 match points, rolling average, and form index (1 to 5) |
| **Target Variable** | `fantasy_points` | Continuous Numeric | Total fantasy points scored in the match |

---

# 3. Data Understanding

### 3.1 Structural Metrics
- **Number of Records:** 2,741 rows
- **Number of Features:** 22 columns (1 target, 4 metadata, 4 categorical, 13 numerical)
- **Duplicate Records:** **0 duplicate rows** (100% distinct match-player instances)
- **Unique Match Fixtures:** 130 matches
- **Unique Players:** 163 distinct cricketers

### 3.2 Data Types & Missingness Audit
Null values were detected across 11 numerical features (all $<3.0\%$ missingness):

| Feature Name | Data Type | Non-Null Count | Null Count | Missing % | Unique Values |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `match_id` | `int64` | 2,741 | 0 | 0.00% | 130 |
| `player_id` | `int64` | 2,741 | 0 | 0.00% | 163 |
| `player_name` | `object` | 2,741 | 0 | 0.00% | 163 |
| `team` | `object` | 2,741 | 0 | 0.00% | 19 (Raw) &rarr; 10 (Clean) |
| `opponent_team` | `object` | 2,741 | 0 | 0.00% | 10 |
| `player_role` | `object` | 2,741 | 0 | 0.00% | 8 (Raw) &rarr; 4 (Clean) |
| `season` | `int64` | 2,741 | 0 | 0.00% | 1 |
| `recent_fantasy_points` | `float64` | 2,684 | 57 | 2.08% | 851 |
| `recent_average_points` | `float64` | 2,699 | 42 | 1.53% | 689 |
| `batting_runs` | `float64` | 2,689 | 52 | 1.90% | 680 |
| `batting_strike_rate` | `float64` | 2,696 | 45 | 1.64% | 925 |
| `batting_average` | `float64` | 2,677 | 64 | 2.33% | 620 |
| `boundaries` | `float64` | 2,676 | 65 | 2.37% | 17 |
| `bowling_wickets` | `int64` | 2,741 | 0 | 0.00% | 7 |
| `bowling_economy` | `float64` | 2,664 | 77 | 2.81% | 550 |
| `bowling_average` | `float64` | 2,686 | 55 | 2.01% | 343 |
| `overs_bowled` | `float64` | 2,664 | 77 | 2.81% | 31 |
| `playing_time_percentage`| `float64`| 2,660 | 81 | 2.96% | 482 |
| `player_form` | `int64` | 2,741 | 0 | 0.00% | 5 |
| `opponent_strength` | `float64` | 2,701 | 40 | 1.46% | 5 |
| `home_away` | `object` | 2,741 | 0 | 0.00% | 4 (Raw) &rarr; 2 (Clean) |
| **`fantasy_points` (Target)** | `float64` | 2,741 | 0 | 0.00% | 728 |

### 3.3 Statistical Summary & Distribution Metrics

| Feature | Count | Mean | Std | Median | Min | 25% | 75% | Max | IQR | Skewness | Kurtosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fantasy_points` (Target)** | 2,741 | 35.61 | 17.94 | 34.10 | 0.00 | 22.30 | 48.00 | 114.50 | 25.70 | +0.42 | -0.19 |
| **`recent_average_points`** | 2,699 | 42.15 | 14.82 | 41.50 | 5.20 | 31.00 | 52.80 | 96.40 | 21.80 | +0.31 | -0.22 |
| **`recent_fantasy_points`** | 2,684 | 41.80 | 21.35 | 40.20 | 0.00 | 25.40 | 56.70 | 120.30 | 31.30 | +0.38 | -0.29 |
| **`batting_runs`** | 2,689 | 24.38 | 21.82 | 18.20 | 0.00 | 6.40 | 37.10 | 118.00 | 30.70 | +0.94 | +0.48 |
| **`batting_strike_rate`** | 2,696 | 104.22 | 43.12 | 112.50 | 0.00 | 82.40 | 134.60 | 250.00 | 52.20 | -0.41 | -0.12 |
| **`bowling_wickets`** | 2,741 | 0.72 | 0.98 | 0.00 | 0.00 | 0.00 | 1.00 | 5.00 | 1.00 | +1.48 | +1.74 |
| **`overs_bowled`** | 2,664 | 1.84 | 1.82 | 1.20 | 0.00 | 0.00 | 4.00 | 4.00 | 4.00 | +0.28 | -1.78 |
| **`player_form`** | 2,741 | 3.02 | 1.28 | 3.00 | 1.00 | 2.00 | 4.00 | 5.00 | 2.00 | -0.02 | -1.04 |
| **`opponent_strength`** | 2,701 | 2.99 | 1.41 | 3.00 | 1.00 | 2.00 | 4.00 | 5.00 | 2.00 | +0.01 | -1.30 |

---

# 4. Data Preprocessing

1. **Missing Value Imputation:**
   - Missing numerical features ($<3\%$ each) are imputed using **training-set medians only**. This strictly prevents look-ahead data leakage into the test set.
2. **Duplicate Records:**
   - Audited: **0 duplicate rows** found.
3. **Incorrect Data & Text Inconsistencies:**
   - **Whitespace Trimming:** Raw team names with trailing spaces (`'Sunrisers Hyderabad '`) were trimmed, reducing 19 raw variants to the 10 canonical franchises.
   - **Role Casing Normalization:** Uppercase strings (`'BATSMAN'`, `'BOWLER'`, `'ALL-ROUNDER'`, `'WICKET-KEEPER'`) were mapped to title-cased categories (`Batsman`, `Bowler`, `All-Rounder`, `Wicket-Keeper`).
   - **Venue Standardization:** Lowercase `'home'`/`'away'` standardized to `'Home'`/`'Away'`.
4. **Outlier Analysis (IQR Method):**

| Feature | Q1 (25%) | Q3 (75%) | IQR | Lower Limit | Upper Limit | Outlier Count | Outlier % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `batting_runs` | 6.40 | 37.10 | 30.70 | -39.65 | 83.15 | 15 | 0.55% |
| `bowling_wickets` | 0.00 | 1.00 | 1.00 | -1.50 | 2.50 | 232 | 8.46% |
| `recent_fantasy_points`| 25.40 | 56.70 | 31.30 | -21.55 | 103.65 | 12 | 0.44% |
| `fantasy_points` (Target)| 22.30 | 48.00 | 25.70 | -16.25 | 86.55 | 42 | 1.53% |

*Interpretation:* Outliers in fantasy points ($>86.5$ pts) and wickets ($3+$ wickets) represent genuine match-winning performances. LightGBM's decision tree splitting naturally segments these extreme values without requiring arbitrary truncation.

5. **Cleaned Dataset Export:**
   - Saved standardized dataset to `data/processed/fantasy_league_cleaned.csv`.

---

# 5. Exploratory Data Analysis (EDA) — Meaningful Visualizations

All 6 required chart modalities have been generated, saved to `results/eda_graphs/`, and embedded in the web application:

### 5.1 📊 Histogram: Target Variable (`fantasy_points`)
- **File:** `01_histogram_target_distribution.png`
- **Observations:** Continuous unimodal distribution ($\mu = 35.61$, $\text{Median} = 34.10$) with a slight positive skew ($\gamma_1 = +0.42$). The 25th–75th percentile spans 22.3 to 48.0 points.

### 5.2 📊 Bar Chart: Team-Wise Average Fantasy Points
- **File:** `02_bar_chart_team_points.png`
- **Observations:** Franchise scoring averages range between 33.5 and 37.2 points per player-match, demonstrating team parity while highlighting high-output batting lineups.

### 5.3 📦 Box Plot: Role-Wise Performance Spread & Outliers
- **File:** `03_box_plot_role_performance.png`
- **Observations:**
  - **All-Rounders:** Produce highest mean fantasy points ($\mu \approx 44.8$) due to two-way point accumulation (batting + bowling).
  - **Wicket-Keepers & Batsmen:** Produce consistent baseline returns ($\mu \approx 36.2$ and $\mu \approx 34.5$).
  - **Bowlers:** Produce lower baseline points ($\mu \approx 28.9$), but exhibit high ceiling spikes during 3+ wicket hauls.

### 5.4 📈 Scatter Plots: Bivariate Regression Relationships
- **File:** `04_scatter_plots_bivariate.png`
- **Observations:**
  - `batting_runs` vs. `fantasy_points`: Strong linear trajectory ($r = +0.27$, with runs contributing 1 pt + boundary/strike-rate bonuses).
  - `recent_fantasy_points` vs. `fantasy_points`: Strong positive autocorrelation ($r = +0.56$).

### 5.5 🔥 Correlation Heatmap: Pearson Interdependence Matrix
- **File:** `05_correlation_heatmap.png`
- **Observations:**
  - Top positive predictors: `recent_average_points` ($r=+0.6324$), `player_form` ($r=+0.5820$), `recent_fantasy_points` ($r=+0.5600$), and `bowling_wickets` ($r=+0.5038$).
  - Negative predictor: `opponent_strength` ($r=-0.1840$, reflecting scoring difficulty against elite bowling units).

### 5.6 📊 Count Plots: Categorical Frequencies (Roles, Venues, Form Scale)
- **File:** `06_count_plots_categorical.png`
- **Observations:**
  - Roles count plot confirms balanced class distribution across Batsmen, Bowlers, All-Rounders, and Keepers.
  - Venue count plot confirms equal Home/Away fixture distribution.
  - Form rating count plot displays a symmetric distribution across the 1–5 scale.

---

# 6. Modeling Takeaways & Recommendations

1. **Chronological Train/Test Partition:** Match observations must be sorted by `match_id` and split 80% earlier / 20% subsequent to prevent lookahead data leakage.
2. **Leakage-Free Median Imputation:** Impute missing features using training-set medians.
3. **Algorithm Suitability:** LightGBM Gradient-Boosted Trees handles mixed categorical variables natively, models non-linear interactions, and resists distortion from extreme upper-tail outliers.
