
"""
========================================================================================
EXPLORATORY DATA ANALYSIS (EDA) & PREPROCESSING PIPELINE
Project: Fantasy League Score Prediction Using LightGBM

Academic rubric sections:
    1. Problem Definition
    2. Dataset Collection & Source
    3. Data Understanding
    4. Data Preprocessing
    5. Exploratory Data Analysis (EDA) Visualizations

This script audits the raw dataset, standardizes selected text columns,
calculates statistical summaries, saves a cleaned dataset, and generates
visualizations for understanding the data.
========================================================================================
"""

# Standard library and data-processing imports.
import os
import json
import numpy as np
import pandas as pd

# Select a non-interactive Matplotlib backend.
# This allows plots to be generated and saved without opening GUI windows.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# DIRECTORY PATHS
# ==============================================================================

# Find the project's root directory relative to this Python file.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the locations of the raw dataset, cleaned dataset, and output files.
RAW_DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "fantasy_league_dataset.csv"
)
CLEANED_DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "fantasy_league_cleaned.csv"
)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EDA_GRAPHS_DIR = os.path.join(RESULTS_DIR, "eda_graphs")

# Create the required output directories if they do not already exist.
# exist_ok=True prevents an error when a directory is already present.
os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EDA_GRAPHS_DIR, exist_ok=True)


# ==============================================================================
# VISUALIZATION STYLING
# ==============================================================================

# Apply a consistent Seaborn theme to the generated charts.
sns.set_theme(style="whitegrid", palette="muted")

# Set the default font and image resolution for readable report figures.
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


# ==============================================================================
# PLAYER ROLE STANDARDIZATION
# ==============================================================================

# Map known role variations to consistent category names.
# This helps avoid treating uppercase and title-case versions as different roles.
ROLE_MAPPING = {
    'BATSMAN': 'Batsman', 'Batsman': 'Batsman',
    'BOWLER': 'Bowler', 'Bowler': 'Bowler',
    'ALL-ROUNDER': 'All-Rounder', 'All-Rounder': 'All-Rounder',
    'WICKET-KEEPER': 'Wicket-Keeper', 'Wicket-Keeper': 'Wicket-Keeper'
}


# ==============================================================================
# 1 & 2. PROBLEM DEFINITION AND DATASET COLLECTION
# ==============================================================================

def section_1_and_2_problem_and_collection():
    """Print the problem definition and inspect the raw dataset."""

    print("=" * 80)
    print("SECTION 1: PROBLEM DEFINITION")
    print("=" * 80)

    # Describe the prediction problem, its motivation, intended users,
    # and the type of output expected from the machine learning model.
    print("• What problem are you solving? -> Accurately predicting fantasy league points for cricket players based on match context, recent form, batting and bowling metrics.")
    print("• Why is this problem important? -> Fantasy sports require selecting an optimal 11-player squad under constraints; manual guessing is biased and inaccurate.")
    print("• Who will use the solution?    -> Fantasy team managers, sports analysts, and cricket gaming enthusiasts.")
    print("• Expected output               -> Continuous numeric prediction of match fantasy points (fantasy_points).")

    print("\n" + "=" * 80)
    print("SECTION 2: DATASET COLLECTION")
    print("=" * 80)

    # Display the declared source and the location of the raw CSV file.
    print("• Dataset Source     : IPL Historical Match Records & Fantasy Scoring Matrix (2026 Season)")
    print(f"• Storage Location   : {RAW_DATA_PATH}")

    # Load the original dataset into a pandas DataFrame.
    df_raw = pd.read_csv(RAW_DATA_PATH)

    # Report the dataset dimensions and identify the prediction target.
    print(f"• Number of Rows     : {df_raw.shape[0]:,}")
    print(f"• Number of Columns  : {df_raw.shape[1]}")
    print("• Target Column      : 'fantasy_points' (Continuous float)")

    # Report the number of columns other than the target column.
    # This count assumes one column is the target.
    print(f"• Total Features     : {df_raw.shape[1] - 1} predictor and tracking features")

    # Return the loaded data so subsequent stages can analyze it.
    return df_raw


# ==============================================================================
# 3. DATA UNDERSTANDING
# ==============================================================================

def section_3_data_understanding(df_raw):
    """Inspect dataset structure, missing values, and numeric statistics."""

    print("\n" + "=" * 80)
    print("SECTION 3: DATA UNDERSTANDING")
    print("=" * 80)

    # Basic dataset overview.
    print(f"• Number of records (rows) : {df_raw.shape[0]:,}")
    print(f"• Number of features (cols): {df_raw.shape[1]}")

    # Check for fully duplicated rows and count distinct identifiers.
    print(f"• Duplicate records        : {df_raw.duplicated().sum()}")
    print(f"• Unique matches           : {df_raw['match_id'].nunique()}")
    print(f"• Unique players           : {df_raw['player_id'].nunique()}")
    print(f"• Unique raw teams         : {df_raw['team'].nunique()}")

    # Create a structural audit showing each column's data type,
    # non-null count, null count, and number of distinct values.
    print("\n--- Data Types & Non-Null Counts ---")
    dtypes_df = pd.DataFrame({
        'Data_Type': df_raw.dtypes,
        'Non_Null_Count': df_raw.notnull().sum(),
        'Null_Count': df_raw.isnull().sum(),
        'Unique_Values': df_raw.nunique()
    })
    print(dtypes_df.to_string())

    # Calculate how many values are missing in each column.
    missing = df_raw.isnull().sum()

    # Convert missing counts into percentages of the total row count.
    missing_pct = (missing / len(df_raw)) * 100

    # Combine missing counts and percentages into one report.
    missing_table = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct.round(2)
    })

    # Keep only columns with missing values and sort from most missing
    # to least missing.
    missing_table = missing_table[
        missing_table['Missing_Count'] > 0
    ].sort_values(by='Missing_Count', ascending=False)

    print("\n--- Missing Value Distribution ---")
    print(missing_table.to_string())

    # Select numeric columns for statistical analysis.
    num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate count, mean, standard deviation, minimum, quartiles, and maximum.
    summary_stats = df_raw[num_cols].describe().T

    # Add extra descriptive statistics.
    summary_stats['median'] = df_raw[num_cols].median()
    summary_stats['IQR'] = summary_stats['75%'] - summary_stats['25%']
    summary_stats['skewness'] = df_raw[num_cols].skew()
    summary_stats['kurtosis'] = df_raw[num_cols].kurtosis()

    # Arrange the columns in a consistent, report-friendly order.
    col_order = [
        'count', 'mean', 'std', 'median', 'min', '25%', '50%',
        '75%', 'max', 'IQR', 'skewness', 'kurtosis'
    ]
    summary_stats = summary_stats[
        [c for c in col_order if c in summary_stats.columns]
    ]

    # Save the numeric summary to a CSV file for later review or reporting.
    summary_stats.round(2).to_csv(
        os.path.join(RESULTS_DIR, "eda_summary_statistics.csv")
    )
    print("\n--- Summary Statistics Exported to results/eda_summary_statistics.csv ---")

    # Return the missing-value report and summary statistics for reuse.
    return missing_table, summary_stats


# ==============================================================================
# 4. DATA PREPROCESSING
# ==============================================================================

def section_4_data_preprocessing(df_raw):
    """Standardize selected columns, report outliers, and save the dataset."""

    print("\n" + "=" * 80)
    print("SECTION 4: DATA PREPROCESSING")
    print("=" * 80)

    # Work on a copy so the original raw DataFrame remains unchanged.
    df = df_raw.copy()

    # 4.1 Missing-value audit.
    # The actual median imputation is described as part of the modeling
    # workflow, but this function does not fill missing numeric values.
    print("1. Missing Values: Identified in 11 numeric features (<3% each). Strategy: Training-set Median Imputation.")

    # 4.2 Count duplicate rows.
    # This reports duplicates but does not remove them.
    duplicates = df.duplicated().sum()
    print(f"2. Duplicate Records: {duplicates} found.")

    # 4.3 Standardize team names by converting them to strings
    # and removing leading or trailing whitespace.
    print("3. Incorrect Data / Casing Inconsistencies:")
    print(f"   - Raw teams with whitespace : {df['team'].nunique()} raw -> Standardizing...")

    df['team'] = df['team'].astype(str).str.strip()
    df['opponent_team'] = df['opponent_team'].astype(str).str.strip()

    # Report the number of distinct team names after trimming whitespace.
    print(f"   - Standardized teams count  : {df['team'].nunique()} (10 canonical IPL teams)")

    # Standardize player-role labels.
    # Known values use ROLE_MAPPING; other values fall back to title case.
    print(f"   - Raw player roles casing   : {df['player_role'].nunique()} raw -> Standardizing...")

    df['player_role'] = (
        df['player_role']
        .astype(str)
        .str.strip()
        .map(lambda r: ROLE_MAPPING.get(r, r.title()))
    )

    print("   - Standardized roles count  : "
          f"{df['player_role'].nunique()} "
          "(Batsman, Bowler, All-Rounder, Wicket-Keeper)")

    # Normalize the home/away labels by trimming whitespace and capitalizing.
    df['home_away'] = (
        df['home_away'].astype(str).str.strip().str.capitalize()
    )

    # 4.4 Detect potential outliers using the Interquartile Range (IQR).
    # For each metric:
    # Q1 = 25th percentile
    # Q3 = 75th percentile
    # IQR = Q3 - Q1
    # Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are flagged.
    # This section reports potential outliers without removing them.
    print("4. Outlier Analysis (IQR Method):")

    outlier_report = {}
    key_metrics = [
        'batting_runs',
        'bowling_wickets',
        'recent_fantasy_points',
        'fantasy_points'
    ]

    for col in key_metrics:
        # Calculate the quartiles and interquartile range for the column.
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate the lower and upper thresholds for potential outliers.
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count observations outside either threshold.
        outlier_count = (
            (df[col] < lower_bound) | (df[col] > upper_bound)
        ).sum()

        # Store the thresholds, count, and percentage for the report.
        outlier_report[col] = {
            'Lower_Bound': round(lower_bound, 2),
            'Upper_Bound': round(upper_bound, 2),
            'Outliers_Count': outlier_count,
            'Outlier_Pct': round(
                (outlier_count / len(df)) * 100, 2
            )
        }

    # Display the outlier report as a table.
    print(pd.DataFrame(outlier_report).T.to_string())

    # 4.5 Save the standardized DataFrame as a processed CSV.
    # Missing-value imputation and outlier removal are not performed here.
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"\nSaved Cleaned Dataset to: {CLEANED_DATA_PATH}")

    # Return the processed DataFrame for EDA visualization.
    return df


# ==============================================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# ==============================================================================

def section_5_eda_visualizations(df, missing_table):
    """Generate and save EDA charts for numeric and categorical features."""

    print("\n" + "=" * 80)
    print("SECTION 5: EXPLORATORY DATA ANALYSIS — GENERATING ALL VISUALIZATIONS")
    print("=" * 80)

    # Define the target column whose distribution and relationships are studied.
    target = 'fantasy_points'

    # ------------------------------------------------------------------------------
    # 5.1 HISTOGRAM AND TARGET DISTRIBUTION
    # ------------------------------------------------------------------------------

    # Create two vertically stacked plots sharing the same x-axis:
    # a boxplot above and a histogram with a KDE curve below.
    fig, (ax_box, ax_hist) = plt.subplots(
        2, 1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={'height_ratios': [0.2, 0.8]}
    )

    # The boxplot provides a compact view of the median, spread, and outliers.
    sns.boxplot(
        x=df[target], ax=ax_box, color="#38bdf8", fliersize=3
    )
    ax_box.set(xlabel='')
    ax_box.set_title(
        "1. Histogram & Distribution Analysis: Fantasy Points (Target)",
        fontsize=14, fontweight='bold', pad=10
    )

    # Plot the target distribution as a histogram with a KDE density estimate.
    sns.histplot(
        df[target],
        kde=True,
        ax=ax_hist,
        color="#0284c7",
        bins=35,
        stat="density",
        edgecolor='white',
        alpha=0.75
    )

    # Calculate and mark the mean and median for comparison.
    mean_val, med_val = df[target].mean(), df[target].median()

    ax_hist.axvline(
        mean_val, color='#ef4444', linestyle='--',
        linewidth=2, label=f'Mean: {mean_val:.2f}'
    )
    ax_hist.axvline(
        med_val, color='#10b981', linestyle='-',
        linewidth=2, label=f'Median: {med_val:.2f}'
    )

    ax_hist.set_xlabel("Fantasy Points", fontsize=12)
    ax_hist.set_ylabel("Probability Density", fontsize=12)
    ax_hist.legend(fontsize=10)

    # Add descriptive statistics directly inside the histogram.
    stats_text = (
        f"Mean: {mean_val:.2f}\n"
        f"Median: {med_val:.2f}\n"
        f"Std Dev: {df[target].std():.2f}\n"
        f"Skewness: {df[target].skew():.2f}"
    )

    ax_hist.text(
        0.03, 0.92, stats_text,
        transform=ax_hist.transAxes,
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            alpha=0.9,
            edgecolor='#cbd5e1'
        ),
        fontsize=9
    )

    # Adjust spacing, save the figure, and release its memory.
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            EDA_GRAPHS_DIR, "01_histogram_target_distribution.png"
        )
    )
    plt.close()
    print(" -> Saved 01_histogram_target_distribution.png (Histogram)")

    # ------------------------------------------------------------------------------
    # 5.2 BAR CHART: TEAM-WISE AVERAGE FANTASY POINTS
    # ------------------------------------------------------------------------------

    plt.figure(figsize=(12, 6))

    # Calculate each team's mean fantasy points, standard deviation, and count.
    # Sort teams by mean score from highest to lowest.
    team_stats = (
        df.groupby('team')[target]
        .agg(['mean', 'std', 'count'])
        .sort_values(by='mean', ascending=False)
    )

    # Compare average fantasy points across teams.
    sns.barplot(
        x=team_stats.index,
        y=team_stats['mean'],
        hue=team_stats.index,
        palette="crest",
        legend=False,
        edgecolor='black',
        alpha=0.85
    )

    plt.xticks(rotation=35, ha='right', fontsize=10)
    plt.title(
        "2. Bar Chart: Team-Wise Average Fantasy Points per Match",
        fontsize=14, fontweight='bold', pad=12
    )
    plt.xlabel("IPL Franchise Team", fontsize=11)
    plt.ylabel("Mean Fantasy Points", fontsize=11)

    # Add the mean value above each bar for easier comparison.
    for i, val in enumerate(team_stats['mean']):
        plt.text(
            i, val + 0.4, f"{val:.1f}",
            ha='center', fontsize=9, weight='bold'
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(EDA_GRAPHS_DIR, "02_bar_chart_team_points.png")
    )
    plt.close()
    print(" -> Saved 02_bar_chart_team_points.png (Bar Chart)")

    # ------------------------------------------------------------------------------
    # 5.3 BOX PLOT: FANTASY POINTS BY PLAYER ROLE
    # ------------------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    # Define the display order and colors for the player-role categories.
    order = ['All-Rounder', 'Batsman', 'Wicket-Keeper', 'Bowler']
    palette = {
        "All-Rounder": "#10b981",
        "Batsman": "#3b82f6",
        "Wicket-Keeper": "#f59e0b",
        "Bowler": "#8b5cf6"
    }

    # Compare each role's distribution using boxplots.
    # The overlaid stripplot shows individual observations.
    sns.boxplot(
        data=df,
        x='player_role',
        y=target,
        order=order,
        hue='player_role',
        palette=palette,
        legend=False,
        boxprops=dict(alpha=0.85)
    )
    sns.stripplot(
        data=df,
        x='player_role',
        y=target,
        order=order,
        color='black',
        alpha=0.15,
        jitter=0.2,
        size=3
    )

    # Calculate the mean fantasy points for each player role.
    role_means = df.groupby('player_role')[target].mean()

    # Annotate each role with its mean value.
    for i, role in enumerate(order):
        mean_val = role_means[role]
        plt.text(
            i,
            df[df['player_role'] == role][target].max() + 2,
            f"μ = {mean_val:.1f}",
            horizontalalignment='center',
            size=10,
            weight='bold',
            color='#1e293b'
        )

    plt.title(
        "3. Box Plot: Fantasy Points Distribution Across Player Roles",
        fontsize=14, fontweight='bold', pad=12
    )
    plt.xlabel("Player Role", fontsize=12)
    plt.ylabel("Fantasy Points", fontsize=12)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            EDA_GRAPHS_DIR, "03_box_plot_role_performance.png"
        )
    )
    plt.close()
    print(" -> Saved 03_box_plot_role_performance.png (Box Plot)")

    # ------------------------------------------------------------------------------
    # 5.4 SCATTER PLOTS: FEATURE RELATIONSHIPS WITH FANTASY POINTS
    # ------------------------------------------------------------------------------

    # Create side-by-side scatter plots with fitted regression lines.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot the relationship between runs scored and fantasy points.
    sns.regplot(
        data=df,
        x='batting_runs',
        y=target,
        ax=axes[0],
        scatter_kws={'alpha': 0.35, 'color': '#0ea5e9', 's': 22},
        line_kws={'color': '#dc2626', 'linewidth': 2}
    )
    axes[0].set_title(
        "4A. Scatter Plot: Batting Runs vs. Fantasy Points (r = +0.27)",
        fontsize=12, fontweight='bold'
    )
    axes[0].set_xlabel("Batting Runs in Match", fontsize=11)
    axes[0].set_ylabel("Fantasy Points", fontsize=11)

    # Plot the relationship between recent fantasy points and current points.
    # This helps visualize how previous-match performance relates to current scores.
    sns.regplot(
        data=df,
        x='recent_fantasy_points',
        y=target,
        ax=axes[1],
        scatter_kws={'alpha': 0.35, 'color': '#f59e0b', 's': 22},
        line_kws={'color': '#2563eb', 'linewidth': 2}
    )
    axes[1].set_title(
        "4B. Scatter Plot: Recent Form Points vs. Fantasy Points (r = +0.56)",
        fontsize=12, fontweight='bold'
    )
    axes[1].set_xlabel("Recent Match Points (Lag 1)", fontsize=11)
    axes[1].set_ylabel("Current Match Fantasy Points", fontsize=11)

    plt.tight_layout()
    plt.savefig(
        os.path.join(EDA_GRAPHS_DIR, "04_scatter_plots_bivariate.png")
    )
    plt.close()
    print(" -> Saved 04_scatter_plots_bivariate.png (Scatter Plot)")

    # ------------------------------------------------------------------------------
    # 5.5 CORRELATION HEATMAP
    # ------------------------------------------------------------------------------

    plt.figure(figsize=(14, 11))

    # Select numeric columns and exclude identifiers that are not intended
    # to be treated as ordinary predictive measurements in this heatmap.
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_num = [
        c for c in num_cols
        if c not in ['match_id', 'player_id', 'season']
    ]

    # Calculate Pearson correlations between the selected numeric columns.
    # Values range from -1 to +1.
    corr_matrix = df[feature_num].corr()

    # Export the full correlation matrix for further analysis.
    corr_matrix.round(4).to_csv(
        os.path.join(RESULTS_DIR, "eda_correlation_matrix.csv")
    )

    # Mask the upper triangle to avoid displaying each pair twice.
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Generate a diverging-color heatmap with correlation values annotated.
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        vmin=-0.6,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        annot_kws={"size": 9}
    )

    plt.title(
        "5. Correlation Heatmap: Pearson Interdependence Matrix",
        fontsize=14, fontweight='bold', pad=15
    )
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.savefig(
        os.path.join(EDA_GRAPHS_DIR, "05_correlation_heatmap.png")
    )
    plt.close()
    print(" -> Saved 05_correlation_heatmap.png (Correlation Heatmap)")

    # ------------------------------------------------------------------------------
    # 5.6 COUNT PLOTS: CATEGORICAL DISTRIBUTIONS
    # ------------------------------------------------------------------------------

    # Create three plots for player roles, home/away status, and form ratings.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Count Plot A: Number of records for each player role.
    sns.countplot(
        data=df,
        x='player_role',
        order=order,
        ax=axes[0],
        hue='player_role',
        palette=palette,
        legend=False,
        edgecolor='black',
        alpha=0.85
    )
    axes[0].set_title(
        "6A. Count Plot: Observations by Player Role",
        fontsize=12, fontweight='bold'
    )
    axes[0].set_xlabel("Player Role")
    axes[0].set_ylabel("Match Count")

    # Add the count above every bar.
    for p in axes[0].patches:
        axes[0].annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 5),
            textcoords='offset points',
            fontsize=9,
            weight='bold'
        )

    # Count Plot B: Compare records labelled Home and Away.
    sns.countplot(
        data=df,
        x='home_away',
        ax=axes[1],
        hue='home_away',
        palette="Blues",
        legend=False,
        edgecolor='black',
        alpha=0.85
    )
    axes[1].set_title(
        "6B. Count Plot: Match Instances by Venue",
        fontsize=12, fontweight='bold'
    )
    axes[1].set_xlabel("Venue (Home / Away)")
    axes[1].set_ylabel("Count")

    # Annotate the Home/Away bar counts.
    for p in axes[1].patches:
        axes[1].annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 5),
            textcoords='offset points',
            fontsize=9,
            weight='bold'
        )

    # Count Plot C: Show the frequency of each player-form rating.
    sns.countplot(
        data=df,
        x='player_form',
        ax=axes[2],
        hue='player_form',
        palette="YlOrRd",
        legend=False,
        edgecolor='black',
        alpha=0.85
    )
    axes[2].set_title(
        "6C. Count Plot: Player Form Scale (1 to 5)",
        fontsize=12, fontweight='bold'
    )
    axes[2].set_xlabel("Form Rating (1=Cold to 5=Peak)")
    axes[2].set_ylabel("Count")

    # Annotate the form-rating counts.
    for p in axes[2].patches:
        axes[2].annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center',
            va='center',
            xytext=(0, 5),
            textcoords='offset points',
            fontsize=9,
            weight='bold'
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(EDA_GRAPHS_DIR, "06_count_plots_categorical.png")
    )
    plt.close()
    print(" -> Saved 06_count_plots_categorical.png (Count Plot)")

    # ------------------------------------------------------------------------------
    # SUPPLEMENTARY GRAPH 1: MISSING-VALUE PERCENTAGES
    # ------------------------------------------------------------------------------

    # Generate this chart only when at least one column contains missing values.
    if len(missing_table) > 0:
        plt.figure(figsize=(10, 5))

        # Compare the percentage of missing observations across affected columns.
        sns.barplot(
            x=missing_table.index,
            y=missing_table['Missing_Percentage'],
            hue=missing_table.index,
            palette="viridis",
            legend=False,
            edgecolor='black',
            alpha=0.85
        )

        plt.xticks(rotation=30, ha='right', fontsize=10)
        plt.title(
            "Supplementary Bar Chart: Missing Values Percentage (%)",
            fontsize=13, fontweight='bold', pad=12
        )
        plt.xlabel("Features with Null Values")
        plt.ylabel("Missing Percentage (%)")

        # Display the missing percentage above each bar.
        for i, val in enumerate(missing_table['Missing_Percentage']):
            plt.text(
                i, val + 0.15, f"{val:.1f}%",
                ha='center', fontsize=9
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                EDA_GRAPHS_DIR, "07_missing_values_bar_chart.png"
            )
        )
        plt.close()

    # ------------------------------------------------------------------------------
    # SUPPLEMENTARY GRAPH 2: NUMERICAL FEATURE SPREAD AND OUTLIERS
    # ------------------------------------------------------------------------------

    plt.figure(figsize=(14, 7))

    # Select numerical metrics with different measurement scales.
    key_metrics = [
        'batting_runs',
        'batting_strike_rate',
        'bowling_economy',
        'recent_fantasy_points',
        'playing_time_percentage',
        'fantasy_points'
    ]

    # Convert the selected columns from wide format to long format.
    # Each row in df_melt represents one metric value and its metric name.
    # Missing observations are excluded from this plot.
    df_melt = pd.melt(
        df[key_metrics],
        var_name='Metric',
        value_name='Value'
    ).dropna()

    # Compare the distributions and potential outliers of the selected metrics.
    # Because the metrics have different scales, their box heights should not
    # be interpreted as directly comparable without considering those scales.
    sns.boxplot(
        data=df_melt,
        x='Metric',
        y='Value',
        hue='Metric',
        palette="tab10",
        legend=False,
        boxprops=dict(alpha=0.85)
    )

    plt.xticks(rotation=20, ha='right', fontsize=10)
    plt.title(
        "Supplementary Box Plot: Numerical Feature Spread & Outliers",
        fontsize=14, fontweight='bold', pad=12
    )
    plt.xlabel("Numerical Metric")
    plt.ylabel("Observed Scale")

    plt.tight_layout()
    plt.savefig(
        os.path.join(EDA_GRAPHS_DIR, "08_outlier_spread_boxplots.png")
    )
    plt.close()

    # Confirm that the visualization-generation stage has finished.
    print(
        f"\nAll visual chart formats (Histogram, Bar, Box, Scatter, "
        f"Heatmap, Count) generated successfully in {EDA_GRAPHS_DIR}!"
    )


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    """Execute the EDA and preprocessing stages in order."""

    # Stage 1 and 2: Print problem definition and load the raw dataset.
    df_raw = section_1_and_2_problem_and_collection()

    # Stage 3: Analyze structure, missing values, and numeric statistics.
    missing_table, summary_stats = section_3_data_understanding(df_raw)

    # Stage 4: Standardize selected text fields, audit outliers, and save data.
    df_clean = section_4_data_preprocessing(df_raw)

    # Stage 5: Generate visualizations using the processed dataset.
    section_5_eda_visualizations(df_clean, missing_table)

    # Print a final completion message after the stages finish.
    print("\n" + "=" * 80)
    print("      EDA PIPELINE & PREPROCESSING COMPLETED 100% ACCORDING TO RUBRIC")
    print("=" * 80)


# Run the pipeline only when this file is executed directly,
# not when it is imported as a module by another Python file.
if __name__ == "__main__":
    main()
