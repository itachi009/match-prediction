import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from paths import ensure_data_layout, get_paths, resolve_repo_path

PATHS = get_paths()
REPO_ROOT = PATHS["repo_root"]
DATA_DIR = PATHS["data_dir"]

ACTIVE_MODEL_FILE = REPO_ROOT / "active_model.json"
MODEL_PATH_DEFAULT = REPO_ROOT / "model.pkl"
FEATURES_PATH_DEFAULT = REPO_ROOT / "model_features.pkl"
LATEST_STATS_FILE = DATA_DIR / "latest_stats.csv"
CLEAN_MATCHES_FILE = DATA_DIR / "clean_matches.csv"


_migration = ensure_data_layout()


def resolve_active_model():
    payload = {
        "model_path": MODEL_PATH_DEFAULT,
        "features_path": FEATURES_PATH_DEFAULT,
    }
    if not ACTIVE_MODEL_FILE.exists():
        return payload
    try:
        with open(ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
            current = json.load(f)
        payload["model_path"] = resolve_repo_path(current.get("model_path", str(MODEL_PATH_DEFAULT)), REPO_ROOT)
        payload["features_path"] = resolve_repo_path(current.get("features_path", str(FEATURES_PATH_DEFAULT)), REPO_ROOT)
    except Exception:
        pass
    return payload


def load_artifacts():
    try:
        active = resolve_active_model()
        model = joblib.load(active["model_path"])
        features = joblib.load(active["features_path"])
        stats = pd.read_csv(LATEST_STATS_FILE)
        return model, features, stats
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}")
        sys.exit(1)


def get_h2h(p1, p2):
    try:
        # Load minimal cols for speed
        df = pd.read_csv(CLEAN_MATCHES_FILE, usecols=["p1_name", "p2_name", "target"])

        # Filter for P1 vs P2 (Dataset is symmetrical, so we just look for P1 as p1 and P2 as p2)
        mask = (df["p1_name"] == p1) & (df["p2_name"] == p2)
        p1_vs_p2 = df[mask]

        p1_wins = p1_vs_p2["target"].sum()

        # We also need to check if there are matches where they played but roles swapped?
        # No, symmetrical dataset guarantees that IF they played, there is a row with P1 as p1 and P2 as p2.
        # So we don't need to check p1_name==P2.

        count = len(p1_vs_p2)
        p2_wins = count - p1_wins

        return p1_wins, p2_wins
    except Exception:
        # If file missing or error
        return 0, 0


def predict(p1_name, p2_name, surface):
    print(f"Analyzing: {p1_name} vs {p2_name} on {surface}...")

    model, feature_names, stats_df = load_artifacts()

    # Get Player Stats
    p1_row = stats_df[stats_df["player"] == p1_name]
    p2_row = stats_df[stats_df["player"] == p2_name]

    if p1_row.empty:
        print(f"Error: Player '{p1_name}' not found in database.")
        return
    if p2_row.empty:
        print(f"Error: Player '{p2_name}' not found in database.")
        return

    p1_row = p1_row.iloc[0]
    p2_row = p2_row.iloc[0]

    # Build Feature Vector
    data = {}

    # 1. Ranks
    r1 = p1_row["rank"]
    r2 = p2_row["rank"]

    data["rank_diff"] = r1 - r2
    data["log_rank_p1"] = np.log(r1) if r1 > 0 else 0
    data["log_rank_p2"] = np.log(r2) if r2 > 0 else 0

    # 2. H2H
    h1, h2 = get_h2h(p1_name, p2_name)
    data["p1_h2h"] = h1
    data["p2_h2h"] = h2

    # 3. Rolling Stats
    # Keys in latest_stats start with 'general_' or '{surface}_'
    # Features in model start with 'p1_' or 'p2_'
    # We need to map:
    #   p1_{stat} -> p1_row['general_{stat}']
    #   p2_{stat} -> p2_row['general_{stat}']

    # List of metrics used in features.py
    metrics = [
        "ace_pct",
        "df_pct",
        "1st_in_pct",
        "1st_won_pct",
        "2nd_won_pct",
        "return_pts_won_pct",
        "bp_convert_pct",
    ]

    for m in metrics:
        data[f"p1_p_{m}"] = p1_row.get(f"general_p_{m}", 0)
        data[f"p2_p_{m}"] = p2_row.get(f"general_p_{m}", 0)

    # Surface Win Pct
    # Model feature: p1_p_surface_win_pct
    # latest_stats: {surface}_win_pct

    s_key = f"{surface}_win_pct"
    data["p1_p_surface_win_pct"] = p1_row.get(s_key, 0.5)
    data["p2_p_surface_win_pct"] = p2_row.get(s_key, 0.5)

    # Create DataFrame with correct column order
    input_df = pd.DataFrame([data])

    # Align with model features
    # Ensure all features exist in data dict, default to 0
    ordered_data = {c: [data.get(c, 0)] for c in feature_names}
    final_input = pd.DataFrame(ordered_data)

    # Force numeric types to avoid XGBoost error
    final_input = final_input.astype(float)

    # Predict
    prob = model.predict_proba(final_input)[0][1]

    print(f"\nPrediction for {surface}:")
    print(f"{p1_name}: {prob*100:.1f}%")
    print(f"{p2_name}: {(1-prob)*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis Match Prediction")
    parser.add_argument("--p1", required=True, help="Player 1 Name")
    parser.add_argument("--p2", required=True, help="Player 2 Name")
    parser.add_argument("--surface", required=True, choices=["Hard", "Clay", "Grass"], help="Surface")

    args = parser.parse_args()
    predict(args.p1, args.p2, args.surface)
