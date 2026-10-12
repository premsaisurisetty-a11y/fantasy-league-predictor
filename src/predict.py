
# Import modules for system paths, command-line arguments, and file handling.
import sys
import os
import argparse

# Import joblib to load the previously trained model bundle.
import joblib

# Import pandas for reading and processing CSV files.
import pandas as pd

# Import NumPy for numerical operations and rounding predictions.
import numpy as np

# Resolve the project base directory using this script's location.
# This helps the script work even when launched from another directory.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to Python's module search path if it is not present.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import the custom ensemble model class.
# Use the alternative import path if the first import is unavailable.
try:
    from src.ensemble import FantasyEnsembleRegressor
except ImportError:
    from ensemble import FantasyEnsembleRegressor

# Define the possible model bundle paths.
# Prefer the improved model bundle, then fall back to the standard bundle.
model_candidates = [
    os.path.join(BASE_DIR, "models", "lightgbm_model_improved.pkl"),
    os.path.join(BASE_DIR, "models", "lightgbm_model.pkl")
]

# Select the first model file that exists.
model_path = next((p for p in model_candidates if os.path.exists(p)), None)

# Stop execution if neither trained model bundle is available.
if not model_path:
    print(f"Error: No model bundle found in {os.path.join(BASE_DIR, 'models')}. Please run src/train.py first.")
    sys.exit(1)

# Load the saved bundle containing the model and preprocessing information.
bundle = joblib.load(model_path)

# Extract the trained prediction model and required preprocessing details.
model = bundle.get("model") if bundle.get("model") is not None else FantasyEnsembleRegressor(bundle)
features = bundle["features"]
medians = bundle["numeric_medians"]
categories = bundle["categories"]
cat_cols = bundle["categorical_columns"]

# Define command-line arguments accepted by the prediction script.
def parse_args():
    # Create a parser with a description of the script's purpose.
    parser = argparse.ArgumentParser(description="Run Fantasy Score Prediction on input CSV data.")

    # Require the path to the CSV file containing player and match records.
    parser.add_argument("input_csv", type=str, help="Path to input CSV file containing player/match records")

    # Allow an optional path for saving the prediction results.
    parser.add_argument("-o", "--output", type=str, default=None, help="Optional path to save output predictions CSV")

    # Read and return the arguments supplied by the user.
    return parser.parse_args()

# Main function that loads input data, predicts scores, and optionally saves results.
def main():
    # Parse the command-line arguments.
    args = parse_args()
    input_path = args.input_csv

    # Check whether the supplied input CSV path exists.
    if not os.path.exists(input_path):
        # If it does not, also check for the file relative to the project root.
        alt_path = os.path.join(BASE_DIR, input_path)
        if os.path.exists(alt_path):
            input_path = alt_path
        else:
            # Stop with an error if the input file cannot be found.
            print(f"Error: Input file '{args.input_csv}' does not exist.")
            sys.exit(1)

    # Read the input CSV into a pandas DataFrame.
    df_in = pd.read_csv(input_path)

    # Stop if the input file contains no data rows.
    if len(df_in) == 0:
        print("Warning: Input CSV file is empty.")
        sys.exit(0)

    # Create a copy so preprocessing does not directly modify the original DataFrame.
    x = df_in.copy()

    # Handle missing numerical feature columns and missing numerical values.
    # Use the medians calculated from the training dataset for consistency.
    for c in features:
        if c in medians:
            if c in x.columns:
                # Convert values to numbers; invalid values become NaN.
                # Fill missing or invalid values with the stored training median.
                x[c] = pd.to_numeric(x[c], errors="coerce").fillna(medians[c])
            else:
                # Add a missing numerical feature using its training median.
                x[c] = medians[c]

    # Prepare categorical features using the categories learned during training.
    for c in cat_cols:
        if c in x.columns:
            # Replace missing categories with "Unknown" and convert values to strings.
            val_series = x[c].fillna("Unknown").astype(str)

            # Restrict category values to the categories stored in the model bundle.
            x[c] = pd.Categorical(val_series, categories=categories[c])
        else:
            # If a categorical feature is missing, create it with "Unknown" values.
            x[c] = pd.Categorical(["Unknown"] * len(x), categories=categories[c])

    # Predict fantasy points for all input records in one batch.
    preds = model.predict(x[features])

    # Add the predicted scores to the original input data, rounded to two decimals.
    df_in["predicted_fantasy_points"] = np.round(preds, 2)

    # Choose useful columns to display in the prediction summary.
    # Include only columns that actually exist in the input data.
    preferred_cols = ["match_id", "player_name", "team", "opponent_team", "player_role", "predicted_fantasy_points"]
    display_cols = [c for c in preferred_cols if c in df_in.columns]

    # If none of the preferred columns exist, display only the predicted scores.
    if not display_cols:
        display_cols = ["predicted_fantasy_points"]

    # Display the selected model, number of records, and prediction preview.
    print(f"\nLoaded model: {os.path.basename(model_path)}")
    print(f"Input records: {len(df_in)}")
    print("\n--- Predicted Fantasy Scores (Preview) ---")

    # Print at most the first 30 rows in a readable table format.
    print(df_in[display_cols].head(30).to_string(index=False))

    # Inform the user if additional records were not shown in the preview.
    if len(df_in) > 30:
        print(f"\n... and {len(df_in) - 30} more rows.")

    # Save the complete prediction results only when an output path is provided.
    if args.output:
        out_path = args.output

        # Resolve relative output paths from the project root directory.
        if not os.path.isabs(out_path):
            out_path = os.path.join(BASE_DIR, out_path)

        # Create the destination directory if it does not already exist.
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        # Save all original input columns along with the predicted scores.
        df_in.to_csv(out_path, index=False)

        # Display the location of the saved prediction file.
        print(f"\n[Success] Full predictions saved to: {out_path}")

# Run the main function only when this file is executed directly.
# Importing this file from another module will not automatically run main().
if __name__ == "__main__":
    main()
