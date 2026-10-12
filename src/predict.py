import sys
import os
import argparse
import joblib
import pandas as pd
import numpy as np

# Resolve project base directory to ensure portability across working directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from src.ensemble import FantasyEnsembleRegressor
except ImportError:
    from ensemble import FantasyEnsembleRegressor

# Support both improved ensemble and standard model bundles
model_candidates = [
    os.path.join(BASE_DIR, "models", "lightgbm_model_improved.pkl"),
    os.path.join(BASE_DIR, "models", "lightgbm_model.pkl")
]
model_path = next((p for p in model_candidates if os.path.exists(p)), None)

if not model_path:
    print(f"Error: No model bundle found in {os.path.join(BASE_DIR, 'models')}. Please run src/train.py first.")
    sys.exit(1)

bundle = joblib.load(model_path)
model = bundle["model"]
features = bundle["features"]
medians = bundle["numeric_medians"]
categories = bundle["categories"]
cat_cols = bundle["categorical_columns"]

def parse_args():
    parser = argparse.ArgumentParser(description="Run Fantasy Score Prediction on input CSV data.")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file containing player/match records")
    parser.add_argument("-o", "--output", type=str, default=None, help="Optional path to save output predictions CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    input_path = args.input_csv

    if not os.path.exists(input_path):
        # Also check relative to BASE_DIR if not found directly
        alt_path = os.path.join(BASE_DIR, input_path)
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            print(f"Error: Input file '{args.input_csv}' does not exist.")
            sys.exit(1)

    df_in = pd.read_csv(input_path)
    if len(df_in) == 0:
        print("Warning: Input CSV file is empty.")
        sys.exit(0)

    x = df_in.copy()

    # Gracefully handle missing feature columns with training set medians
    for c in features:
        if c in medians:
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce").fillna(medians[c])
            else:
                x[c] = medians[c]

    # Gracefully handle categorical features
    for c in cat_cols:
        if c in x.columns:
            val_series = x[c].fillna("Unknown").astype(str)
            x[c] = pd.Categorical(val_series, categories=categories[c])
        else:
            x[c] = pd.Categorical(["Unknown"] * len(x), categories=categories[c])

    # Run batch prediction
    preds = model.predict(x[features])
    df_in["predicted_fantasy_points"] = np.round(preds, 2)

    # Determine summary display columns
    preferred_cols = ["match_id", "player_name", "team", "opponent_team", "player_role", "predicted_fantasy_points"]
    display_cols = [c for c in preferred_cols if c in df_in.columns]
    if not display_cols:
        display_cols = ["predicted_fantasy_points"]

    print(f"\nLoaded model: {os.path.basename(model_path)}")
    print(f"Input records: {len(df_in)}")
    print("\n--- Predicted Fantasy Scores (Preview) ---")
    print(df_in[display_cols].head(30).to_string(index=False))
    if len(df_in) > 30:
        print(f"\n... and {len(df_in) - 30} more rows.")

    # Save output if requested
    if args.output:
        out_path = args.output
        if not os.path.isabs(out_path):
            out_path = os.path.join(BASE_DIR, out_path)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        df_in.to_csv(out_path, index=False)
        print(f"\n[Success] Full predictions saved to: {out_path}")

if __name__ == "__main__":
    main()
