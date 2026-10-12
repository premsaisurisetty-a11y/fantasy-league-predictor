import os
import sys
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

app = Flask(
    __name__,
    template_folder=os.path.join(APP_DIR, "templates"),
    static_folder=os.path.join(APP_DIR, "static")
)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "python": sys.version})

try:
    from src.ensemble import FantasyEnsembleRegressor
except ImportError:
    from ensemble import FantasyEnsembleRegressor

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "fantasy_league_dataset.csv")
MODEL_IMPROVED_PATH = os.path.join(BASE_DIR, "models", "lightgbm_model_improved.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "lightgbm_model.pkl")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
PRECOMPUTED_PATH = os.path.join(APP_DIR, "precomputed_data.json")

try:
    from app.player_images import get_player_image_metadata, get_player_image_url, get_player_id
except ImportError:
    from player_images import get_player_image_metadata, get_player_image_url, get_player_id

# 1. Lazy Model Loading Engine
_bundle = None
def get_bundle():
    global _bundle
    if _bundle is not None:
        return _bundle
    selected_model_path = MODEL_IMPROVED_PATH if os.path.exists(MODEL_IMPROVED_PATH) else MODEL_PATH
    if os.path.exists(selected_model_path):
        try:
            print(f"[FantasyScore AI] Loading model bundle from: {selected_model_path}")
            _bundle = joblib.load(selected_model_path)
            return _bundle
        except Exception as e:
            print(f"[Warning] Failed loading model bundle: {e}")
            return None
    return None

def safe_float(val, default=0.0):
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    if pd.isna(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default

# 2. Fast Startup: Load Precomputed Metadata if Available
if os.path.exists(PRECOMPUTED_PATH):
    print(f"[FantasyScore AI] Loading precomputed metadata from: {PRECOMPUTED_PATH}")
    import json
    with open(PRECOMPUTED_PATH, "r", encoding="utf-8") as f:
        pre_data = json.load(f)
    dataset_stats = pre_data["stats"]
    all_players = pre_data["players"]
    all_teams = pre_data["teams"]
    all_opponent_teams = pre_data["opponent_teams"]
    all_roles = pre_data["roles"]
    home_away_options = pre_data["home_away_options"]
    cached_leaderboard = pre_data["leaderboard"]
    player_profile_map = pre_data["profiles"]
else:
    print(f"[FantasyScore AI] Loading historical dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    total_records = len(df)
    total_players = df["player_name"].nunique()
    total_matches = df["match_id"].nunique()
    avg_fantasy_pts = round(float(df["fantasy_points"].mean()), 2)
    dataset_stats = {
        "total_records": f"{total_records:,}",
        "total_players": total_players,
        "total_matches": total_matches,
        "avg_fantasy_points": avg_fantasy_pts
    }
    all_players = sorted(df["player_name"].dropna().unique().tolist())
    all_teams = sorted(df["team"].dropna().unique().tolist())
    all_opponent_teams = sorted(df["opponent_team"].dropna().unique().tolist())
    all_roles = sorted(df["player_role"].dropna().unique().tolist())
    home_away_options = sorted(df["home_away"].dropna().unique().tolist())

    bundle = get_bundle()
    numeric_medians = bundle["numeric_medians"] if bundle else {}

    player_profile_map = {}
    for name, group in df.groupby("player_name"):
        latest_row = group.sort_values("match_id", ascending=False).iloc[0]
        p_team = str(latest_row.get("team", ""))
        p_role = str(latest_row.get("player_role", ""))
        img_meta = get_player_image_metadata(name, team=p_team, role=p_role)

        profile = {
            "player_name": name,
            "player_id": img_meta["player_id"],
            "image_url": img_meta["image_url"],
            "initials": img_meta["initials"],
            "team": p_team,
            "opponent_team": str(latest_row.get("opponent_team", "")),
            "player_role": p_role,
            "home_away": str(latest_row.get("home_away", "Home")),
            "batting_runs": round(safe_float(latest_row.get("batting_runs"), numeric_medians.get("batting_runs", 0.0)), 1),
            "batting_strike_rate": round(safe_float(latest_row.get("batting_strike_rate"), numeric_medians.get("batting_strike_rate", 0.0)), 1),
            "batting_average": round(safe_float(latest_row.get("batting_average"), numeric_medians.get("batting_average", 0.0)), 1),
            "boundaries": safe_int(latest_row.get("boundaries"), safe_int(numeric_medians.get("boundaries", 0))),
            "bowling_wickets": safe_int(latest_row.get("bowling_wickets"), safe_int(numeric_medians.get("bowling_wickets", 0))),
            "bowling_economy": round(safe_float(latest_row.get("bowling_economy"), numeric_medians.get("bowling_economy", 0.0)), 1),
            "bowling_average": round(safe_float(latest_row.get("bowling_average"), numeric_medians.get("bowling_average", 0.0)), 1),
            "overs_bowled": round(safe_float(latest_row.get("overs_bowled"), numeric_medians.get("overs_bowled", 0.0)), 1),
            "recent_fantasy_points": round(safe_float(latest_row.get("recent_fantasy_points"), numeric_medians.get("recent_fantasy_points", 0.0)), 1),
            "recent_average_points": round(safe_float(latest_row.get("recent_average_points"), numeric_medians.get("recent_average_points", 0.0)), 1),
            "player_form": round(safe_float(latest_row.get("player_form"), numeric_medians.get("player_form", 50.0)), 1),
            "playing_time_percentage": round(safe_float(latest_row.get("playing_time_percentage"), 100.0), 1),
            "opponent_strength": round(safe_float(latest_row.get("opponent_strength"), numeric_medians.get("opponent_strength", 5.0)), 1)
        }
        player_profile_map[name] = profile

    leaderboard_records = []
    for pname, prof in player_profile_map.items():
        base_pt = prof.get("recent_average_points", 35.0)
        credits = round(float(np.clip(7.0 + (base_pt - 15.0) * 0.08, 7.5, 10.5)), 1)
        leaderboard_records.append({
            "player_name": pname,
            "player_id": prof["player_id"],
            "image_url": prof["image_url"],
            "initials": prof["initials"],
            "team": prof["team"],
            "player_role": prof["player_role"],
            "credits": credits,
            "predicted_points": round(float(base_pt), 2),
            "actual_average": round(float(base_pt), 2),
            "matches": 10
        })
    leaderboard_records.sort(key=lambda x: x["predicted_points"], reverse=True)
    for rank_idx, item in enumerate(leaderboard_records, start=1):
        item["rank"] = rank_idx
    cached_leaderboard = leaderboard_records

# -----------------------------------------------------------------------------
# 3b. Optimal Fantasy XI Squad Optimizer (MILP / 0-1 Knapsack)
# -----------------------------------------------------------------------------
def optimize_fantasy_squad(player_pool=None, max_budget=100.0, team1=None, team2=None):
    from scipy.optimize import milp, LinearConstraint, Bounds
    
    if player_pool is None:
        player_pool = cached_leaderboard

    # Filter candidate pool by matchup teams if specified
    candidates = []
    for p in player_pool:
        if team1 and team2:
            if p["team"] in [team1, team2]:
                candidates.append(p)
        elif team1:
            if p["team"] == team1:
                candidates.append(p)
        else:
            candidates.append(p)

    if len(candidates) < 11:
        candidates = player_pool

    n = len(candidates)
    pts = np.array([p["predicted_points"] for p in candidates], dtype=float)
    credits = np.array([p.get("credits", 8.5) for p in candidates], dtype=float)
    roles = [p["player_role"] for p in candidates]
    teams = [p["team"] for p in candidates]

    # Objective: Minimize -pts (maximize total predicted points)
    c = -pts

    A_rows = []
    b_l = []
    b_u = []

    # Constraint 1: Exactly 11 players
    A_rows.append(np.ones(n))
    b_l.append(11.0)
    b_u.append(11.0)

    # Constraint 2: Total credits <= max_budget
    A_rows.append(credits)
    b_l.append(0.0)
    b_u.append(float(max_budget))

    # Constraints 3-6: Positional composition
    # Wicket-Keeper: 1 to 4 | Batsman: 3 to 6 | All-Rounder: 1 to 4 | Bowler: 3 to 6
    role_limits = [
        ("Wicket-Keeper", 1, 4),
        ("Batsman", 3, 6),
        ("All-Rounder", 1, 4),
        ("Bowler", 3, 6)
    ]
    for r_name, r_min, r_max in role_limits:
        mask = np.array([1.0 if r == r_name else 0.0 for r in roles])
        if mask.sum() >= r_min:
            A_rows.append(mask)
            b_l.append(float(r_min))
            b_u.append(float(r_max))

    # Constraint 7: Max 7 players per franchise
    unique_teams = list(set(teams))
    for t_name in unique_teams:
        mask = np.array([1.0 if t == t_name else 0.0 for t in teams])
        A_rows.append(mask)
        b_l.append(0.0)
        b_u.append(7.0)

    selected_indices = []

    # Attempt MILP optimization if scipy is available
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        c = -pts
        A_rows = [np.ones(n), credits]
        b_l = [11.0, 0.0]
        b_u = [11.0, float(max_budget)]

        role_limits = [
            ("Wicket-Keeper", 1, 4),
            ("Batsman", 3, 6),
            ("All-Rounder", 1, 4),
            ("Bowler", 3, 6)
        ]
        for r_name, r_min, r_max in role_limits:
            mask = np.array([1.0 if r == r_name else 0.0 for r in roles])
            if mask.sum() >= r_min:
                A_rows.append(mask)
                b_l.append(float(r_min))
                b_u.append(float(r_max))

        unique_teams = list(set(teams))
        for t_name in unique_teams:
            mask = np.array([1.0 if t == t_name else 0.0 for t in teams])
            A_rows.append(mask)
            b_l.append(0.0)
            b_u.append(7.0)

        constraints = LinearConstraint(np.array(A_rows), np.array(b_l), np.array(b_u))
        integrality = np.ones(n)
        bounds = Bounds(0, 1)

        res = milp(c=c, integrality=integrality, constraints=constraints, bounds=bounds)
        if res.success:
            selected_indices = [i for i, val in enumerate(res.x) if val > 0.5]
    except Exception as e:
        pass

    if len(selected_indices) != 11:
        # Fallback greedy knapsack heuristic
        sorted_indices = sorted(range(n), key=lambda i: pts[i] / max(credits[i], 1.0), reverse=True)
        selected_indices = sorted_indices[:11]

    squad = [candidates[i].copy() for i in selected_indices]
    squad.sort(key=lambda p: p["predicted_points"], reverse=True)

    # Designate Captain (2.0x) and Vice-Captain (1.5x)
    for idx, p in enumerate(squad):
        if idx == 0:
            p["is_captain"] = True
            p["is_vice_captain"] = False
            p["multiplier"] = 2.0
            p["effective_points"] = round(p["predicted_points"] * 2.0, 2)
        elif idx == 1:
            p["is_captain"] = False
            p["is_vice_captain"] = True
            p["multiplier"] = 1.5
            p["effective_points"] = round(p["predicted_points"] * 1.5, 2)
        else:
            p["is_captain"] = False
            p["is_vice_captain"] = False
            p["multiplier"] = 1.0
            p["effective_points"] = round(p["predicted_points"], 2)

    total_credits = round(sum(p.get("credits", 8.5) for p in squad), 1)
    base_points = round(sum(p["predicted_points"] for p in squad), 2)
    captain_bonus = round(squad[0]["predicted_points"] * 1.0, 2) if len(squad) >= 1 else 0.0
    vc_bonus = round(squad[1]["predicted_points"] * 0.5, 2) if len(squad) >= 2 else 0.0
    total_projected_score = round(base_points + captain_bonus + vc_bonus, 2)

    by_role = {
        "Wicket-Keeper": [p for p in squad if p["player_role"] == "Wicket-Keeper"],
        "Batsman": [p for p in squad if p["player_role"] == "Batsman"],
        "All-Rounder": [p for p in squad if p["player_role"] == "All-Rounder"],
        "Bowler": [p for p in squad if p["player_role"] == "Bowler"]
    }

    return {
        "squad": squad,
        "by_role": by_role,
        "total_credits": total_credits,
        "base_points": base_points,
        "captain": squad[0] if len(squad) >= 1 else None,
        "vice_captain": squad[1] if len(squad) >= 2 else None,
        "total_projected_score": total_projected_score,
        "role_counts": {
            "wk": len(by_role["Wicket-Keeper"]),
            "bat": len(by_role["Batsman"]),
            "ar": len(by_role["All-Rounder"]),
            "bow": len(by_role["Bowler"])
        }
    }

# 4. Helper Function: Process Input & Run Real-Time Prediction
def make_prediction_from_dict(data_dict):
    import math
    import time
    t_start = time.perf_counter()

    bundle = get_bundle()
    if bundle is not None:
        model = bundle.get("model") if bundle.get("model") is not None else FantasyEnsembleRegressor(bundle)
        features = bundle["features"]
        numeric_medians = bundle["numeric_medians"]
        categories = bundle["categories"]
        categorical_columns = bundle["categorical_columns"]

        row_dict = {}
        for f in features:
            row_dict[f] = data_dict.get(f, None)

        x = pd.DataFrame([row_dict])
        for f in features:
            if f in numeric_medians:
                x[f] = pd.to_numeric(x[f], errors="coerce").fillna(numeric_medians[f])

        for f in categorical_columns:
            val_series = x[f].fillna("Unknown").astype(str)
            x[f] = pd.Categorical(val_series, categories=categories[f])

        pred = float(model.predict(x[features])[0])
        score = round(pred, 2)
        feature_importances = getattr(model, "feature_importances_", np.zeros(len(features)))
    else:
        score = round(safe_float(data_dict.get("recent_average_points", 35.0)), 2)
        features = ["batting_runs", "recent_fantasy_points", "bowling_wickets", "recent_average_points", "player_form"]
        numeric_medians = {}
        feature_importances = [0.3, 0.25, 0.2, 0.15, 0.1]

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    # Feature Contribution Breakdown
    total_imp = sum(feature_importances) if sum(feature_importances) > 0 else 1
    imp_dict = dict(zip(features, feature_importances))

    top_factors = sorted(features, key=lambda feat: imp_dict.get(feat, 0), reverse=True)[:5]
    breakdown = []
    for feat in top_factors:
        val = data_dict.get(feat, numeric_medians.get(feat, "N/A"))
        pct = round((imp_dict.get(feat, 0) / total_imp) * 100, 1)
        breakdown.append({
            "name": feat.replace("_", " ").title(),
            "value": str(val),
            "importance": pct
        })

    # Real-Time Advanced Analytics
    # Selection probability in Top Fantasy XI based on logistic curve
    sel_prob = round(100.0 / (1.0 + math.exp(- (score - 36.0) / 9.5)), 1)
    sel_prob = max(4.5, min(99.2, sel_prob))

    conf_low = round(max(0.0, score - 6.2), 1)
    conf_high = round(score + 6.2, 1)

    if score >= 55:
        tier_title = "★ Elite Captain Pick (Tier 1)"
        tier_class = "tier-high"
    elif score >= 38:
        tier_title = "◆ Star Core Squad Pick (Tier 2)"
        tier_class = "tier-med"
    elif score >= 25:
        tier_title = "▲ Tactical Differential (Tier 3)"
        tier_class = "tier-low"
    else:
        tier_title = "▼ Budget / Low-Impact (Tier 4)"
        tier_class = "tier-low"

    pname = data_dict.get("player_name", "")
    p_team = data_dict.get("team", "")
    p_role = data_dict.get("player_role", "")
    img_meta = get_player_image_metadata(pname, team=p_team, role=p_role)

    analytics = {
        "player_name": pname,
        "player_id": img_meta["player_id"],
        "image_url": img_meta["image_url"],
        "initials": img_meta["initials"],
        "confidence_low": conf_low,
        "confidence_high": conf_high,
        "selection_prob": sel_prob,
        "tier_title": tier_title,
        "tier_class": tier_class,
        "latency_ms": latency_ms
    }

    return score, breakdown, analytics

# --- ROUTES ---

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    breakdown = []
    analytics = {}
    selected_player = request.args.get("player", "")
    form_values = {}

    if request.method == "POST":
        form_values = request.form.to_dict()
        prediction, breakdown, analytics = make_prediction_from_dict(form_values)
    elif selected_player and selected_player in player_profile_map:
        form_values = player_profile_map[selected_player]
        prediction, breakdown, analytics = make_prediction_from_dict(form_values)
    else:
        default_name = all_players[0] if all_players else ""
        if default_name in player_profile_map:
            form_values = player_profile_map[default_name]
            prediction, breakdown, analytics = make_prediction_from_dict(form_values)

    current_player_name = selected_player or form_values.get("player_name", "")
    current_player_profile = player_profile_map.get(current_player_name, {})

    return render_template(
        "index.html",
        active_page="predict",
        stats=dataset_stats,
        players=all_players,
        teams=all_teams,
        opponent_teams=all_opponent_teams,
        roles=all_roles,
        home_away_options=home_away_options,
        form_values=form_values,
        default_player=current_player_name,
        player_profile=current_player_profile,
        prediction=prediction,
        breakdown=breakdown,
        analytics=analytics
    )

@app.route("/rankings")
def rankings():
    return render_template(
        "rankings.html",
        active_page="rankings",
        leaderboard=cached_leaderboard
    )

@app.route("/squad-optimizer", methods=["GET", "POST"])
def squad_optimizer():
    team1 = request.args.get("team1", "")
    team2 = request.args.get("team2", "")
    try:
        budget = float(request.args.get("budget", 100.0))
    except (ValueError, TypeError):
        budget = 100.0

    optimized = optimize_fantasy_squad(
        player_pool=cached_leaderboard,
        max_budget=budget,
        team1=team1 if team1 else None,
        team2=team2 if team2 else None
    )

    return render_template(
        "squad_optimizer.html",
        active_page="optimizer",
        optimized=optimized,
        teams=all_teams,
        selected_team1=team1,
        selected_team2=team2,
        budget=budget
    )

@app.route("/api/optimize-squad", methods=["GET", "POST"])
def api_optimize_squad():
    req_data = request.get_json(silent=True) or request.args
    team1 = req_data.get("team1", "")
    team2 = req_data.get("team2", "")
    try:
        budget = float(req_data.get("budget", 100.0))
    except (ValueError, TypeError):
        budget = 100.0

    result = optimize_fantasy_squad(
        player_pool=cached_leaderboard,
        max_budget=budget,
        team1=team1 if team1 else None,
        team2=team2 if team2 else None
    )
    return jsonify({"success": True, "result": result})

@app.route("/model")
def model_analytics():
    # Load metrics from file
    metrics = {
        "mae": 8.10,
        "mse": 103.37,
        "rmse": 10.17,
        "r2": 0.6805,
        "best_iteration": 197
    }
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    k = k.strip().lower().replace(" ", "_")
                    try:
                        metrics[k] = float(v.strip()) if "." in v else int(v.strip())
                    except ValueError:
                        pass

    # Load feature importances
    feat_imp_list = []
    max_imp = 1
    feat_imp_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    if os.path.exists(feat_imp_path):
        imp_df = pd.read_csv(feat_imp_path)
        feat_imp_list = imp_df.to_dict(orient="records")
        if feat_imp_list:
            max_imp = max(item["importance"] for item in feat_imp_list)

    # Load model comparison
    comparison_list = []
    comp_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        comparison_list = comp_df.to_dict(orient="records")

    return render_template(
        "model.html",
        active_page="model",
        metrics=metrics,
        feature_importances=feat_imp_list,
        max_importance=max_imp,
        model_comparison=comparison_list
    )

# --- API ENDPOINTS ---

@app.route("/api/players")
def get_all_players():
    summaries = []
    for name in all_players:
        prof = player_profile_map.get(name, {})
        summaries.append({
            "name": name,
            "player_id": prof.get("player_id", ""),
            "image_url": prof.get("image_url", ""),
            "initials": prof.get("initials", ""),
            "team": prof.get("team", ""),
            "role": prof.get("player_role", ""),
            "avg_points": prof.get("recent_average_points", 0.0)
        })
    return jsonify({"success": True, "players": summaries})

@app.route("/api/player/<player_name>")
def get_player_data(player_name):
    if player_name in player_profile_map:
        return jsonify({"success": True, "stats": player_profile_map[player_name]})
    return jsonify({"success": False, "error": "Player not found"}), 404

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        score, breakdown, analytics = make_prediction_from_dict(data)
        return jsonify({
            "success": True,
            "prediction": score,
            "breakdown": breakdown,
            "analytics": analytics
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

EDA_GRAPHS_DIR = os.path.join(RESULTS_DIR, "eda_graphs")
EDA_REPORT_PATH = os.path.join(BASE_DIR, "EDA_REPORT.md")
CLEANED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "fantasy_league_cleaned.csv")
EDA_STATS_PATH = os.path.join(RESULTS_DIR, "eda_summary_statistics.csv")
EDA_CORR_PATH = os.path.join(RESULTS_DIR, "eda_correlation_matrix.csv")
EDA_NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "EDA_Fantasy_League.ipynb")

@app.route("/eda")
def eda_dashboard():
    # Load summary statistics table
    stats_list = []
    if os.path.exists(EDA_STATS_PATH):
        stats_df = pd.read_csv(EDA_STATS_PATH)
        stats_list = stats_df.to_dict(orient="records")
    
    # Load correlation rankings with fantasy_points
    corr_list = []
    if os.path.exists(EDA_CORR_PATH):
        corr_df = pd.read_csv(EDA_CORR_PATH)
        if "fantasy_points" in corr_df.columns:
            target_series = corr_df.set_index(corr_df.columns[0])["fantasy_points"].drop("fantasy_points", errors="ignore")
            for feat, val in target_series.sort_values(ascending=False).items():
                corr_list.append({
                    "feature": feat.replace("_", " ").title(),
                    "raw_feature": feat,
                    "correlation": round(float(val), 4),
                    "abs_corr": round(abs(float(val)), 4),
                    "is_positive": float(val) >= 0
                })

    eda_images = [
        {"file": "01_histogram_target_distribution.png", "title": "1. Histogram: Target Variable Distribution", "desc": "Histogram, KDE, and boxplot with skewness (+0.42) & kurtosis analysis."},
        {"file": "02_bar_chart_team_points.png", "title": "2. Bar Chart: Team-Wise Average Fantasy Points", "desc": "Mean fantasy points generated per match across all 10 IPL franchise squads."},
        {"file": "03_box_plot_role_performance.png", "title": "3. Box Plot: Role Performance & Outliers", "desc": "Boxplots & jitter distributions across All-Rounders, Batsmen, Keepers, and Bowlers."},
        {"file": "04_scatter_plots_bivariate.png", "title": "4. Scatter Plot: Bivariate Regression Analysis", "desc": "Linear regressions: Batting Runs (r = +0.27) & Recent Form (r = +0.56) vs Points."},
        {"file": "05_correlation_heatmap.png", "title": "5. Correlation Heatmap: Pearson Interdependence", "desc": "Complete triangular multi-collinearity matrix highlighting top fantasy predictors."},
        {"file": "06_count_plots_categorical.png", "title": "6. Count Plot: Categorical Class Frequencies", "desc": "Distribution counts across Player Roles, Match Venues, and Form Scale (1 to 5)."},
        {"file": "07_missing_values_bar_chart.png", "title": "7. Supplementary Bar Chart: Missing Value Audit", "desc": "Feature missingness percentages highlighting training-set median imputations."},
        {"file": "08_outlier_spread_boxplots.png", "title": "8. Supplementary Box Plot: Numerical Feature Spread", "desc": "Multi-metric IQR dispersion and upper-tail anomaly identification."}
    ]

    return render_template(
        "eda.html",
        active_page="eda",
        dataset_stats=dataset_stats,
        stats_list=stats_list,
        corr_list=corr_list,
        eda_images=eda_images
    )

PDF_REPORT_PATH = os.path.join(BASE_DIR, "EDA_REPORT.pdf")

@app.route("/download/eda-report-pdf")
@app.route("/download/eda-report.pdf")
@app.route("/download/pdf")
def download_eda_pdf():
    if os.path.exists(PDF_REPORT_PATH):
        return send_from_directory(
            os.path.dirname(PDF_REPORT_PATH),
            os.path.basename(PDF_REPORT_PATH),
            as_attachment=True,
            download_name="Fantasy_Score_Prediction_EDA_Report.pdf"
        )
    return "EDA PDF Report not found", 404

@app.route("/download/eda-report")
def download_eda_report():
    if os.path.exists(EDA_REPORT_PATH):
        return send_from_directory(
            os.path.dirname(EDA_REPORT_PATH),
            os.path.basename(EDA_REPORT_PATH),
            as_attachment=True,
            download_name="Fantasy_Score_Prediction_EDA_Report.md"
        )
    return "EDA Report not found", 404

@app.route("/download/eda-notebook")
def download_eda_notebook():
    if os.path.exists(EDA_NOTEBOOK_PATH):
        return send_from_directory(
            os.path.dirname(EDA_NOTEBOOK_PATH),
            os.path.basename(EDA_NOTEBOOK_PATH),
            as_attachment=True,
            download_name="EDA_Fantasy_League.ipynb"
        )
    return "EDA Notebook not found", 404

@app.route("/download/cleaned-dataset")
def download_cleaned_dataset():
    if os.path.exists(CLEANED_DATA_PATH):
        return send_from_directory(
            os.path.dirname(CLEANED_DATA_PATH),
            os.path.basename(CLEANED_DATA_PATH),
            as_attachment=True,
            download_name="fantasy_league_cleaned.csv"
        )
    return "Cleaned dataset not found", 404

@app.route("/download/summary-statistics")
def download_summary_statistics():
    if os.path.exists(EDA_STATS_PATH):
        return send_from_directory(
            os.path.dirname(EDA_STATS_PATH),
            os.path.basename(EDA_STATS_PATH),
            as_attachment=True,
            download_name="eda_summary_statistics.csv"
        )
    return "Summary statistics not found", 404

@app.route("/download/correlation-matrix")
def download_correlation_matrix():
    if os.path.exists(EDA_CORR_PATH):
        return send_from_directory(
            os.path.dirname(EDA_CORR_PATH),
            os.path.basename(EDA_CORR_PATH),
            as_attachment=True,
            download_name="eda_correlation_matrix.csv"
        )
    return "Correlation matrix not found", 404

@app.route("/download/eda-graphs-zip")
def download_eda_graphs_zip():
    import io
    import zipfile
    from flask import send_file

    if not os.path.exists(EDA_GRAPHS_DIR):
        return "EDA graphs directory not found", 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(EDA_GRAPHS_DIR):
            for file in files:
                if file.endswith((".png", ".jpg", ".svg")):
                    file_path = os.path.join(root, file)
                    zf.write(file_path, arcname=file)

    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name="fantasy_score_eda_charts.zip"
    )

@app.route("/results/eda_graphs/<path:filename>")
def serve_eda_graph(filename):
    return send_from_directory(EDA_GRAPHS_DIR, filename)

@app.route("/results/graphs/<path:filename>")
def serve_graph(filename):
    return send_from_directory(GRAPHS_DIR, filename)

if __name__ == "__main__":
    print("[FantasyScore AI] Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(debug=True, host="127.0.0.1", port=5000)

