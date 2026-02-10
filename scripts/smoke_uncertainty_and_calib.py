import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import backtest
from reporting.buckets import make_bucket_report, make_probability_bucket_edges


def main():
    backtest.ensure_data_layout()
    df = backtest.load_features_and_predictions()
    subset = df.head(2000).copy()
    if subset.empty:
        print("[SMOKE] subset vuoto, impossibile procedere.")
        return

    subset["level"] = backtest.infer_match_levels(subset).fillna("A")
    y = backtest.extract_binary_target(subset).fillna(0).astype(int)
    subset["y"] = y

    calib_input = subset[["tourney_date", "level", "prob_p1_win", "y"]].copy()
    calib_input = calib_input.rename(columns={"y": "p1_is_real_winner"})
    calib_bundle = backtest.fit_level_calibrators(calib_input, n_min_calib=backtest.MP_N_MIN_CALIB)

    p_calibrated = []
    for row in subset.itertuples(index=False):
        _, _, p_final = backtest.apply_probability_pipeline(
            row.prob_p1_win,
            2.0,
            backtest.BASELINE_CONFIG,
            calibrator=calib_bundle["global"],
            level=row.level,
            level_calibrators=calib_bundle["by_level"],
        )
        p_calibrated.append(p_final)
    subset["p_calibrated"] = np.asarray(p_calibrated, dtype=float)

    p_edges = make_probability_bucket_edges(step=backtest.MP_BUCKET_STEP, start=0.50, end=1.00)
    if len(p_edges) < 2:
        p_edges = np.array([0.50, 1.00], dtype=float)
    subset["p_bucket"] = pd.cut(subset["p_calibrated"], bins=p_edges, include_lowest=True, right=False)
    subset["uncertainty_bucket"] = pd.cut(
        pd.to_numeric(subset.get("uncertainty_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0),
        bins=np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.000001], dtype=float),
        include_lowest=True,
        right=False,
    )

    p_report = make_bucket_report(subset, "p_bucket", "p_calibrated", "y")
    u_report = make_bucket_report(subset, "uncertainty_bucket", "p_calibrated", "y")

    p_path = backtest.RUNS_DIR / "reliability_by_p_bucket.csv"
    u_path = backtest.RUNS_DIR / "reliability_by_uncertainty_bucket.csv"
    p_report.to_csv(p_path, index=False)
    u_report.to_csv(u_path, index=False)

    print(f"[SMOKE] calibration_dir: {calib_bundle.get('calibration_dir')}")
    print(f"[SMOKE] p_bucket csv: {p_path}")
    print(f"[SMOKE] uncertainty_bucket csv: {u_path}")
    print("[SMOKE] p_bucket head:")
    print(p_report.head(5).to_string(index=False))
    print("[SMOKE] uncertainty_bucket head:")
    print(u_report.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
