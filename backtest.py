import itertools
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

try:
    from utils import DataNormalizer, standardize_dates
except ImportError:
    print("[ERROR] utils.py not found. Please ensure it exists.")
    sys.exit(1)


# --- FILE CONFIG ---
FEATURES_FILE = "processed_features.csv"
METADATA_FILE = "clean_matches.csv"
ODDS_FILE_LOCAL = "2024_test.csv"
MODEL_FILE = "model_v9_balanced.pkl"
MODEL_FEATURES_FILE = "model_features.pkl"
PLOT_FILE = "real_backtest.png"
BASELINE_CONFIG_FILE = "backtest_baseline_config.json"
VALIDATION_REPORT_FILE = "backtest_validation_report.json"
WALKFORWARD_REPORT_FILE = "backtest_walkforward_report.json"
STRESS_REPORT_FILE = "backtest_stress_report.json"
RELIABILITY_PLOT_FILE = "reliability_curve.png"
RELIABILITY_TABLE_FILE = "reliability_table.csv"


# --- STRATEGY BASELINE (POINT 1: FROZEN BASELINE) ---
BASELINE_CONFIG = {
    "name": "v11_oriented_fair_kelly_oos_gate_conservative",
    "min_edge": 0.06,
    "min_ev": 0.06,
    "min_confidence": 0.64,
    "prob_shrink": 0.62,
    "kelly_fraction": 0.01,
    "max_stake_pct": 0.01,
    "max_overround": 1.12,
    "min_odds": 1.55,
    "max_odds": 3.00,
    "max_bet_share": 0.08,
    "min_kelly_f": 0.10,
    "min_signal_score": 0.005,
    "edge_slope_by_odds": 0.02,
    "payout_haircut_pct": 0.00,
    "commission_pct": 0.00,
    "slippage_pct": 0.00,
    "residual_shrink_odds_2_5": 0.96,
    "residual_shrink_odds_3_0": 0.95,
}


# --- TEMPORAL VALIDATION (POINT 3) ---
H1_H2_SPLIT_DATE = "2024-07-01"
MIN_BETS_FOR_TUNING = 15
MIN_TRAIN_MATCHES_WF = 300
MIN_TEST_MATCHES_WF = 120
BOOTSTRAP_SAMPLES = 3000
WF_TUNED_TRAIN_ADV_THRESHOLD = 0.25
MAX_TUNING_EVALS = 5000
TUNING_RANDOM_SEED = 42
TUNING_REFINEMENT_TOPK = 80
TUNING_GRID = {
    "min_edge": [0.05, 0.055, 0.06, 0.065],
    "min_ev": [0.04, 0.05, 0.06],
    "min_confidence": [0.60, 0.62, 0.64],
    "prob_shrink": [0.58, 0.62, 0.66],
    "kelly_fraction": [0.01, 0.015, 0.02],
    "max_stake_pct": [0.01, 0.015, 0.02],
    "max_bet_share": [0.08, 0.10, 0.12],
    "min_kelly_f": [0.05, 0.07, 0.10],
    "min_signal_score": [0.003, 0.004, 0.005],
    "edge_slope_by_odds": [0.015, 0.02, 0.03],
    "residual_shrink_odds_2_5": [0.96, 0.98],
    "residual_shrink_odds_3_0": [0.92, 0.95, 0.96],
}

STRESS_SCENARIOS = [
    {"name": "baseline", "payout_haircut_pct": 0.00, "commission_pct": 0.00, "slippage_pct": 0.00},
    {"name": "mild", "payout_haircut_pct": 0.01, "commission_pct": 0.002, "slippage_pct": 0.002},
    {"name": "medium", "payout_haircut_pct": 0.02, "commission_pct": 0.003, "slippage_pct": 0.003},
    {"name": "hard", "payout_haircut_pct": 0.03, "commission_pct": 0.005, "slippage_pct": 0.005},
]


def freeze_baseline_config(config, path=BASELINE_CONFIG_FILE):
    payload = {
        "frozen_at": pd.Timestamp.utcnow().isoformat(),
        "model_file": MODEL_FILE,
        "features_file": FEATURES_FILE,
        "odds_file": ODDS_FILE_LOCAL,
        "config": config,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"    Baseline config salvata in: {path}")


def get_model_features(df_feats):
    try:
        model_features = joblib.load(MODEL_FEATURES_FILE)
        print(f"    [INFO] Feature del modello caricate ({len(model_features)}).")
        return model_features
    except Exception:
        print("[WARN] model_features.pkl non trovato. Uso fallback.")
        fallback = [
            "surface_elo_p1",
            "surface_elo_p2",
            "surface_elo_diff",
            "fatigue_p1",
            "fatigue_p2",
            "fatigue_diff",
            "log_rank_diff",
            "h2h_diff",
            "p1_vs_hand_win_pct",
            "p1_decider_win_pct",
            "p2_decider_win_pct",
            "p1_ace_pct",
            "p2_ace_pct",
            "p1_1st_won_pct",
            "p2_1st_won_pct",
        ]
        return [c for c in fallback if c in df_feats.columns]


def normalize_surface(val):
    if pd.isna(val):
        return "UNK"
    s = str(val).strip().lower()
    if "clay" in s:
        return "CLAY"
    if "grass" in s:
        return "GRASS"
    if "carpet" in s:
        return "CARPET"
    if "hard" in s:
        return "HARD"
    return s.upper()


def get_directed_id(n1, n2):
    return f"{str(n1)}||{str(n2)}"


def load_features_and_predictions():
    df_feats = pd.read_csv(FEATURES_FILE, low_memory=False)
    model = joblib.load(MODEL_FILE)
    model_features = get_model_features(df_feats)

    if "date" in df_feats.columns and "tourney_date" not in df_feats.columns:
        df_feats = df_feats.rename(columns={"date": "tourney_date"})

    df_feats["tourney_date"] = pd.to_datetime(df_feats["tourney_date"], errors="coerce")
    df_feats = df_feats[df_feats["tourney_date"].dt.year == 2024].copy()

    missing = [c for c in model_features if c not in df_feats.columns]
    for c in missing:
        print(f"    [WARN] Feature mancante '{c}' - Riempimento con 0.")
        df_feats[c] = 0.0

    X = df_feats[model_features].fillna(0)
    df_feats["prob_p1_win"] = model.predict_proba(X.values)[:, 1]
    return df_feats


def load_and_normalize_odds(normalizer):
    df_odds = pd.read_csv(ODDS_FILE_LOCAL, sep=";", encoding="latin1")
    map_cols = {
        "Winner": "winner_raw",
        "Loser": "loser_raw",
        "Date": "tourney_date",
        "Surface": "surface_raw",
        "B365W": "odds_w",
        "B365L": "odds_l",
    }
    df_odds = df_odds.rename(columns=map_cols)

    df_odds["tourney_date"] = standardize_dates(df_odds["tourney_date"], "tennis_data")
    for c in ["odds_w", "odds_l"]:
        if c in df_odds.columns:
            df_odds[c] = df_odds[c].astype(str).str.replace(",", ".").astype(float)
    df_odds = df_odds.dropna(subset=["odds_w", "odds_l"]).copy()

    df_odds["winner_name"] = df_odds["winner_raw"].apply(normalizer.convert_name)
    df_odds["loser_name"] = df_odds["loser_raw"].apply(normalizer.convert_name)
    failed = df_odds["winner_name"].isna() | df_odds["loser_name"].isna()
    failed_count = int(failed.sum())
    if failed_count > 0:
        print(f"    [WARN] Fallita normalizzazione per {failed_count} match. Esempi:")
        print(df_odds[failed][["winner_raw", "loser_raw"]].head(3))

    df_odds = df_odds.dropna(subset=["winner_name", "loser_name"]).copy()
    total = len(df_odds) + failed_count
    print(f"    Quote pronte: {len(df_odds)} match (su {total} originali).")
    return df_odds


def build_oriented_merged_dataset(df_feats, df_odds):
    df_feats = df_feats.copy()
    df_feats["directed_id"] = df_feats.apply(lambda x: get_directed_id(x["p1_name"], x["p2_name"]), axis=1)
    if "surface" in df_feats.columns:
        df_feats["surface_key"] = df_feats["surface"].apply(normalize_surface)
        use_surface_merge = df_feats["surface_key"].nunique(dropna=True) > 1
    else:
        df_feats["surface_key"] = "UNK"
        use_surface_merge = False

    feats_merge = df_feats[["tourney_date", "directed_id", "surface_key", "prob_p1_win"]].copy()
    feats_merge = feats_merge.sort_values("tourney_date")

    df_odds = df_odds.copy()
    if "surface_raw" not in df_odds.columns:
        df_odds["surface_raw"] = "UNK"
    df_odds["surface_key"] = df_odds["surface_raw"].apply(normalize_surface)
    df_odds = df_odds.sort_values("tourney_date").reset_index(drop=True)
    df_odds["odds_row_id"] = df_odds.index

    odds_side_w = pd.DataFrame(
        {
            "odds_row_id": df_odds["odds_row_id"],
            "tourney_date": df_odds["tourney_date"],
            "surface_key": df_odds["surface_key"],
            "p1_name": df_odds["winner_name"],
            "p2_name": df_odds["loser_name"],
            "odd_p1": df_odds["odds_w"],
            "odd_p2": df_odds["odds_l"],
            "p1_is_real_winner": True,
        }
    )
    odds_side_l = pd.DataFrame(
        {
            "odds_row_id": df_odds["odds_row_id"],
            "tourney_date": df_odds["tourney_date"],
            "surface_key": df_odds["surface_key"],
            "p1_name": df_odds["loser_name"],
            "p2_name": df_odds["winner_name"],
            "odd_p1": df_odds["odds_l"],
            "odd_p2": df_odds["odds_w"],
            "p1_is_real_winner": False,
        }
    )
    df_odds_oriented = pd.concat([odds_side_w, odds_side_l], ignore_index=True)
    df_odds_oriented["directed_id"] = df_odds_oriented.apply(
        lambda x: get_directed_id(x["p1_name"], x["p2_name"]), axis=1
    )
    df_odds_oriented = df_odds_oriented.sort_values("tourney_date")

    merge_by = ["directed_id", "surface_key"] if use_surface_merge else ["directed_id"]

    merged_1 = pd.merge_asof(
        df_odds_oriented,
        feats_merge,
        on="tourney_date",
        by=merge_by,
        tolerance=pd.Timedelta("1d"),
        direction="nearest",
    )
    matched_1 = merged_1.dropna(subset=["prob_p1_win"]).copy()

    unmatched_mask = merged_1["prob_p1_win"].isna()
    df_odds_unmatched = df_odds_oriented.loc[unmatched_mask].copy()
    merged_2 = pd.merge_asof(
        df_odds_unmatched,
        feats_merge,
        on="tourney_date",
        by=merge_by,
        tolerance=pd.Timedelta("3d"),
        direction="nearest",
    )
    matched_2 = merged_2.dropna(subset=["prob_p1_win"]).copy()

    merged = pd.concat([matched_1, matched_2], ignore_index=True)
    merged = merged.sort_values(["tourney_date", "odds_row_id"]).drop_duplicates(
        subset=["odds_row_id", "directed_id"], keep="first"
    )

    stats = {
        "merge_by": merge_by,
        "oriented_rows": int(len(df_odds_oriented)),
        "matched_step1": int(len(matched_1)),
        "matched_step2": int(len(matched_2)),
        "matched_total_sides": int(len(merged)),
        "matched_unique_matches": int(merged["odds_row_id"].nunique()),
    }
    return merged.sort_values("tourney_date"), stats


def compute_max_drawdown_pct(equity):
    if len(equity) == 0:
        return 0.0
    eq = np.asarray(equity, dtype=float)
    running_max = np.maximum.accumulate(eq)
    drawdowns = (running_max - eq) / np.where(running_max <= 0, 1.0, running_max)
    return float(np.max(drawdowns) * 100.0)


def compute_bootstrap_roi_ci(bet_pnls, initial_bankroll, samples=BOOTSTRAP_SAMPLES, seed=42):
    if not bet_pnls:
        return np.nan, np.nan
    arr = np.asarray(bet_pnls, dtype=float) / float(initial_bankroll) * 100.0
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(samples, len(arr)), replace=True).sum(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def compute_monthly_roi(equity_df, initial_bankroll):
    if equity_df.empty:
        return pd.DataFrame(columns=["month", "start_bankroll", "end_bankroll", "roi_pct"])

    eq = equity_df.copy()
    eq = eq.sort_values("tourney_date")
    eq["month"] = eq["tourney_date"].dt.to_period("M").astype(str)

    monthly_rows = []
    prev_end = float(initial_bankroll)
    for month, g in eq.groupby("month", sort=True):
        end_bankroll = float(g["bankroll"].iloc[-1])
        roi_pct = ((end_bankroll - prev_end) / prev_end * 100.0) if prev_end > 0 else np.nan
        monthly_rows.append(
            {
                "month": month,
                "start_bankroll": round(prev_end, 2),
                "end_bankroll": round(end_bankroll, 2),
                "roi_pct": round(roi_pct, 2),
            }
        )
        prev_end = end_bankroll

    return pd.DataFrame(monthly_rows)


def compute_ece_mce(prob, y, bins=10):
    if len(prob) == 0:
        return np.nan, np.nan, np.array([]), np.array([]), np.array([])
    p = np.asarray(prob, dtype=float).clip(0.0, 1.0)
    t = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=False)

    ece = 0.0
    mce = 0.0
    centers = []
    obs = []
    weights = []
    n_total = len(p)

    for b in range(bins):
        mask = bin_ids == b
        n = int(mask.sum())
        if n == 0:
            continue
        avg_p = float(p[mask].mean())
        avg_t = float(t[mask].mean())
        gap = abs(avg_p - avg_t)
        ece += (n / n_total) * gap
        mce = max(mce, gap)
        centers.append(avg_p)
        obs.append(avg_t)
        weights.append(n / n_total)

    return float(ece), float(mce), np.array(centers), np.array(obs), np.array(weights)


def fit_probability_calibrator(df_side, min_train=200, min_val=120):
    def identity_predict(x):
        return np.asarray(x, dtype=float).clip(0.0, 1.0)

    if df_side.empty:
        return {
            "name": "identity",
            "predict": identity_predict,
            "selection_primary_metric": "ece",
            "selection_reason": "empty_dataset",
            "metrics": {
                "brier_raw_val": np.nan,
                "brier_sigmoid_val": np.nan,
                "brier_isotonic_val": np.nan,
                "ece_raw_val": np.nan,
                "ece_sigmoid_val": np.nan,
                "ece_isotonic_val": np.nan,
                "n_train": 0,
                "n_val": 0,
            },
        }

    df = df_side.sort_values("tourney_date").copy()
    df["raw"] = df["prob_p1_win"].astype(float).clip(0.0, 1.0)
    df["y"] = df["p1_is_real_winner"].astype(int)

    n = len(df)
    split_idx = max(min_train, int(n * 0.70))
    if split_idx >= n - min_val:
        split_idx = n - min_val

    if split_idx <= 0 or (n - split_idx) < min_val:
        brier_raw = float(np.mean((df["raw"] - df["y"]) ** 2)) if n > 0 else np.nan
        ece_raw, _, _, _, _ = compute_ece_mce(df["raw"].to_numpy(), df["y"].to_numpy(), bins=10) if n > 0 else (np.nan, np.nan, np.array([]), np.array([]), np.array([]))
        return {
            "name": "identity",
            "predict": identity_predict,
            "selection_primary_metric": "ece",
            "selection_reason": "insufficient_validation_window",
            "metrics": {
                "brier_raw_val": brier_raw,
                "brier_sigmoid_val": np.nan,
                "brier_isotonic_val": np.nan,
                "ece_raw_val": float(ece_raw),
                "ece_sigmoid_val": np.nan,
                "ece_isotonic_val": np.nan,
                "n_train": int(max(0, split_idx)),
                "n_val": int(max(0, n - split_idx)),
            },
        }

    tr = df.iloc[:split_idx]
    va = df.iloc[split_idx:]
    x_tr = tr["raw"].to_numpy()
    y_tr = tr["y"].to_numpy()
    x_va = va["raw"].to_numpy()
    y_va = va["y"].to_numpy()

    metrics = {
        "brier_raw_val": float(np.mean((x_va - y_va) ** 2)),
        "brier_sigmoid_val": np.nan,
        "brier_isotonic_val": np.nan,
        "ece_raw_val": np.nan,
        "ece_sigmoid_val": np.nan,
        "ece_isotonic_val": np.nan,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
    }
    metrics["ece_raw_val"] = float(compute_ece_mce(x_va, y_va, bins=10)[0])

    sigmoid = LogisticRegression(max_iter=1000, solver="lbfgs")
    sigmoid.fit(x_tr.reshape(-1, 1), y_tr)
    p_sig = sigmoid.predict_proba(x_va.reshape(-1, 1))[:, 1]
    metrics["brier_sigmoid_val"] = float(np.mean((p_sig - y_va) ** 2))
    metrics["ece_sigmoid_val"] = float(compute_ece_mce(p_sig, y_va, bins=10)[0])

    isotonic_model = None
    if len(np.unique(x_tr)) >= 25 and len(tr) >= 250:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x_tr, y_tr)
        p_iso = np.asarray(iso.predict(x_va))
        metrics["brier_isotonic_val"] = float(np.mean((p_iso - y_va) ** 2))
        metrics["ece_isotonic_val"] = float(compute_ece_mce(p_iso, y_va, bins=10)[0])
        isotonic_model = iso

    candidates = [("identity", metrics["ece_raw_val"], metrics["brier_raw_val"])]
    candidates.append(("sigmoid", metrics["ece_sigmoid_val"], metrics["brier_sigmoid_val"]))
    if isotonic_model is not None and np.isfinite(metrics["ece_isotonic_val"]) and np.isfinite(metrics["brier_isotonic_val"]):
        candidates.append(("isotonic", metrics["ece_isotonic_val"], metrics["brier_isotonic_val"]))

    candidates = [c for c in candidates if np.isfinite(c[1]) and np.isfinite(c[2])]
    if not candidates:
        return {
            "name": "identity",
            "predict": identity_predict,
            "selection_primary_metric": "ece",
            "selection_reason": "no_valid_candidates",
            "metrics": metrics,
        }

    best_name, best_ece, best_brier = sorted(candidates, key=lambda x: (x[1], x[2]))[0]
    guardrail_margin = 0.003
    selection_reason = f"min_ece_then_brier:{best_name}"
    if best_brier > metrics["brier_raw_val"] + guardrail_margin:
        best_name = "identity"
        selection_reason = "guardrail_brier_degradation"

    if best_name == "identity":
        predict_fn = identity_predict
    elif best_name == "sigmoid":
        predict_fn = lambda x: sigmoid.predict_proba(np.asarray(x, dtype=float).reshape(-1, 1))[:, 1]
    else:
        predict_fn = lambda x: np.asarray(isotonic_model.predict(np.asarray(x, dtype=float)))

    return {
        "name": best_name,
        "predict": predict_fn,
        "selection_primary_metric": "ece",
        "selection_reason": selection_reason,
        "metrics": metrics,
    }


def apply_probability_pipeline(raw_prob, odd_raw, cfg, calibrator=None):
    raw = float(np.clip(raw_prob, 0.0, 1.0))
    if calibrator is not None:
        calibrated = float(np.clip(calibrator["predict"]([raw])[0], 0.0, 1.0))
    else:
        calibrated = raw

    # Residual shrink only on extreme odds.
    final_p = calibrated
    if odd_raw >= 2.5:
        s = float(cfg.get("residual_shrink_odds_2_5", 1.0))
        final_p = 0.5 + (final_p - 0.5) * s
    if odd_raw >= 3.0:
        s = float(cfg.get("residual_shrink_odds_3_0", 1.0))
        final_p = 0.5 + (final_p - 0.5) * s

    return raw, calibrated, float(np.clip(final_p, 0.0, 1.0))


def simulate_strategy(
    df_sim,
    config,
    initial_bankroll=1000.0,
    compute_bootstrap=True,
    calibrator=None,
    fixed_selection=None,
    fixed_sizing_no_cost=False,
):
    cfg = dict(config)
    cfg.setdefault("payout_haircut_pct", 0.0)
    cfg.setdefault("commission_pct", 0.0)
    cfg.setdefault("slippage_pct", 0.0)
    cfg.setdefault("residual_shrink_odds_2_5", 1.0)
    cfg.setdefault("residual_shrink_odds_3_0", 1.0)

    if df_sim.empty:
        return {
            "bankroll": initial_bankroll,
            "roi": 0.0,
            "bets": 0,
            "wins": 0,
            "win_rate": 0.0,
            "candidate_bets": 0,
            "max_allowed_bets": 0,
            "max_drawdown_pct": 0.0,
            "profit_factor": np.nan,
            "bootstrap_roi_ci": (np.nan, np.nan),
            "monthly_roi": pd.DataFrame(columns=["month", "start_bankroll", "end_bankroll", "roi_pct"]),
            "history": [initial_bankroll],
            "skip": {},
            "selected_decisions": {},
        }

    skip = {
        "overround": 0,
        "low_conf": 0,
        "low_edge_ev": 0,
        "odds_range": 0,
        "dyn_edge": 0,
        "low_kelly": 0,
        "low_signal": 0,
        "rank_cap": 0,
    }

    if fixed_selection is None:
        decisions = {}
        for row in df_sim.itertuples(index=False):
            row_id = int(row.odds_row_id)
            odd_p1_raw = float(row.odd_p1)
            odd_p2 = float(row.odd_p2)
            odd_p1_eff = 1.0 + (odd_p1_raw - 1.0) * (1.0 - float(cfg["payout_haircut_pct"]))
            fixed_cost_rate = float(cfg["commission_pct"]) + float(cfg["slippage_pct"])

            implied_p1 = 1 / odd_p1_raw if odd_p1_raw > 0 else np.nan
            implied_p2 = 1 / odd_p2 if odd_p2 > 0 else np.nan
            if np.isnan(implied_p1) or np.isnan(implied_p2):
                skip["low_edge_ev"] += 1
                continue

            overround = implied_p1 + implied_p2
            if overround <= 0 or overround > float(cfg["max_overround"]):
                skip["overround"] += 1
                continue

            fair_p1 = implied_p1 / overround
            _, p_cal, p_model = apply_probability_pipeline(row.prob_p1_win, odd_p1_raw, cfg, calibrator=calibrator)
            if p_model < float(cfg["min_confidence"]):
                skip["low_conf"] += 1
                continue

            edge = p_model - fair_p1
            ev = p_model * odd_p1_eff - 1.0 - fixed_cost_rate
            req_edge = float(cfg["min_edge"]) + float(cfg["edge_slope_by_odds"]) * max(0.0, odd_p1_raw - 2.20)
            odds_in_range = float(cfg["min_odds"]) <= odd_p1_raw <= float(cfg["max_odds"])

            if not (edge >= req_edge and ev >= float(cfg["min_ev"]) and odds_in_range):
                if edge >= float(cfg["min_edge"]) and edge < req_edge and ev >= float(cfg["min_ev"]) and odds_in_range:
                    skip["dyn_edge"] += 1
                elif edge >= req_edge and ev >= float(cfg["min_ev"]) and not odds_in_range:
                    skip["odds_range"] += 1
                else:
                    skip["low_edge_ev"] += 1
                continue

            b = odd_p1_eff - 1.0
            kelly_f = (p_model * odd_p1_eff - 1.0 - fixed_cost_rate) / b if b > 0 else 0.0
            if kelly_f < float(cfg["min_kelly_f"]):
                skip["low_kelly"] += 1
                continue

            signal = edge * max(ev, 0) * max(kelly_f, 0)
            if signal < float(cfg["min_signal_score"]):
                skip["low_signal"] += 1
                continue

            decision = {
                "p_raw": float(row.prob_p1_win),
                "p_calibrated": p_cal,
                "p_sel": p_model,
                "odd_sel_raw": odd_p1_raw,
                "odd_sel_eff": odd_p1_eff,
                "signal": signal,
                "p1_is_real_winner": bool(row.p1_is_real_winner),
            }
            prev = decisions.get(row_id)
            if prev is None or decision["signal"] > prev["signal"]:
                decisions[row_id] = decision

        unique_matches = int(df_sim["odds_row_id"].nunique())
        max_allowed_bets = max(1, int(unique_matches * float(cfg["max_bet_share"])))
        ranked = sorted(decisions.items(), key=lambda kv: kv[1]["signal"], reverse=True)
        if len(ranked) > max_allowed_bets:
            skip["rank_cap"] = len(ranked) - max_allowed_bets
            ranked = ranked[:max_allowed_bets]
        selected = dict(ranked)
        candidate_bets = len(decisions)
    else:
        selected = {int(k): dict(v) for k, v in fixed_selection.items()}
        candidate_bets = len(selected)
        max_allowed_bets = len(selected)

    bankroll = float(initial_bankroll)
    bets_placed = 0
    wins = 0
    history = [bankroll]
    bet_pnls = []
    equity_rows = []

    timeline = df_sim.groupby("odds_row_id", as_index=False)["tourney_date"].min().sort_values("tourney_date")
    for row in timeline.itertuples(index=False):
        row_id = int(row.odds_row_id)
        date = pd.to_datetime(row.tourney_date)
        decision = selected.get(row_id)

        if decision is None:
            history.append(bankroll)
            equity_rows.append({"tourney_date": date, "bankroll": bankroll})
            continue

        p_sel = float(decision["p_sel"])
        odd_sel_raw = float(decision["odd_sel_raw"])
        odd_sel_eff = 1.0 + (odd_sel_raw - 1.0) * (1.0 - float(cfg["payout_haircut_pct"]))
        b_eff = odd_sel_eff - 1.0
        fixed_cost_rate = float(cfg["commission_pct"]) + float(cfg["slippage_pct"])

        if fixed_sizing_no_cost:
            b_raw = odd_sel_raw - 1.0
            f = (p_sel * odd_sel_raw - 1.0) / b_raw if b_raw > 0 else 0.0
        else:
            f = (p_sel * odd_sel_eff - 1.0 - fixed_cost_rate) / b_eff if b_eff > 0 else 0.0

        stake = bankroll * float(cfg["kelly_fraction"]) * f
        stake = max(0.0, min(stake, bankroll * float(cfg["max_stake_pct"])))

        if stake > 1.0:
            bets_placed += 1
            bankroll -= stake
            cost = stake * fixed_cost_rate
            bankroll -= cost
            pnl = -stake - cost
            if bool(decision["p1_is_real_winner"]):
                bankroll += stake * odd_sel_eff
                pnl = stake * (odd_sel_eff - 1.0) - cost
                wins += 1
            bet_pnls.append(pnl)

        history.append(bankroll)
        equity_rows.append({"tourney_date": date, "bankroll": bankroll})

    roi = ((bankroll - initial_bankroll) / initial_bankroll) * 100.0
    win_rate = (wins / bets_placed * 100.0) if bets_placed > 0 else 0.0
    max_drawdown_pct = compute_max_drawdown_pct(history)

    gross_profit = sum(p for p in bet_pnls if p > 0)
    gross_loss = -sum(p for p in bet_pnls if p < 0)
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = np.inf
    else:
        profit_factor = np.nan

    if compute_bootstrap:
        ci_low, ci_high = compute_bootstrap_roi_ci(bet_pnls, initial_bankroll, samples=BOOTSTRAP_SAMPLES)
    else:
        ci_low, ci_high = np.nan, np.nan

    equity_df = pd.DataFrame(equity_rows)
    monthly_df = compute_monthly_roi(equity_df, initial_bankroll)

    return {
        "bankroll": float(bankroll),
        "roi": float(roi),
        "bets": int(bets_placed),
        "wins": int(wins),
        "win_rate": float(win_rate),
        "candidate_bets": int(candidate_bets),
        "max_allowed_bets": int(max_allowed_bets),
        "max_drawdown_pct": float(max_drawdown_pct),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else np.inf,
        "bootstrap_roi_ci": (float(ci_low), float(ci_high)),
        "monthly_roi": monthly_df,
        "history": history,
        "skip": skip,
        "selected_decisions": selected,
    }


def print_strategy_report(result):
    print("\n============================================================")
    print("RISULTATI FINALI")
    print("============================================================")
    print(f"Bankroll: €{result['bankroll']:.2f}")
    print(f"ROI: {result['roi']:.2f}%")
    if result["bets"] > 0:
        print(f"Bets: {result['bets']} | Wins: {result['wins']} ({result['win_rate']:.1f}%)")
    else:
        print("Bets: 0")

    print(f"Candidate bets: {result['candidate_bets']} | Max allowed: {result['max_allowed_bets']}")
    s = result["skip"]
    print(
        f"Skip overround: {s.get('overround', 0)} | Skip low-conf: {s.get('low_conf', 0)} | "
        f"Skip low edge/EV: {s.get('low_edge_ev', 0)}"
    )
    print(
        f"Skip odds range: {s.get('odds_range', 0)} | Skip dyn-edge: {s.get('dyn_edge', 0)} | "
        f"Skip low Kelly: {s.get('low_kelly', 0)} | Skip low signal: {s.get('low_signal', 0)} | "
        f"Skip rank cap: {s.get('rank_cap', 0)}"
    )

    # Point 2: robustness block
    ci_low, ci_high = result["bootstrap_roi_ci"]
    ci_txt = f"[{ci_low:.2f}%, {ci_high:.2f}%]" if np.isfinite(ci_low) and np.isfinite(ci_high) else "n/a"
    pf = result["profit_factor"]
    pf_txt = f"{pf:.2f}" if np.isfinite(pf) else "inf"
    print(f"Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"Profit Factor: {pf_txt}")
    print(f"ROI CI 95% (bootstrap): {ci_txt}")

    monthly = result["monthly_roi"]
    if not monthly.empty:
        print("\nROI mensile:")
        print(monthly[["month", "roi_pct", "end_bankroll"]].to_string(index=False))


def config_conservativeness_score(cfg):
    return (
        float(cfg["min_edge"])
        + float(cfg["min_ev"])
        + 0.50 * float(cfg["min_confidence"])
        + 0.50 * (1.0 - float(cfg["prob_shrink"]))
        + 5.0 * float(cfg["min_signal_score"])
        + 0.20 * float(cfg["min_kelly_f"])
        - 0.50 * float(cfg["max_bet_share"])
    )


def tune_strategy_config(df_train, baseline_config, calibrator=None, progress_label="Tuning", print_progress=False):
    score_formula = "roi - 0.50*max_drawdown_pct + 0.05*min(profit_factor,2.0) - penalty_low_bets - gate_penalty"
    keys = list(TUNING_GRID.keys())
    full_combos = list(itertools.product(*[TUNING_GRID[k] for k in keys]))
    sampled_mode = len(full_combos) > MAX_TUNING_EVALS
    if sampled_mode:
        rng = np.random.default_rng(TUNING_RANDOM_SEED)
        sampled_idx = rng.choice(len(full_combos), size=MAX_TUNING_EVALS, replace=False)
        combos = [full_combos[int(i)] for i in sampled_idx]
    else:
        combos = list(full_combos)

    rows = []
    candidates = []

    def evaluate_combo(values):
        cfg = dict(baseline_config)
        for k, v in zip(keys, values):
            cfg[k] = float(v)

        res = simulate_strategy(df_train, cfg, initial_bankroll=1000.0, compute_bootstrap=False, calibrator=calibrator)
        pf = float(res["profit_factor"]) if np.isfinite(res["profit_factor"]) else 2.0
        pf = min(pf, 2.0)
        penalty_low_bets = max(0, MIN_BETS_FOR_TUNING - int(res["bets"])) * 0.10
        gate_penalty = 5.0 if float(res["max_drawdown_pct"]) > 5.0 else 0.0
        score = float(res["roi"]) - 0.50 * float(res["max_drawdown_pct"]) + 0.05 * pf - penalty_low_bets - gate_penalty
        cons = config_conservativeness_score(cfg)
        feasible_gate = float(res["max_drawdown_pct"]) <= 5.0 and int(res["bets"]) >= 20

        row = {
            "score": float(score),
            "roi": float(res["roi"]),
            "max_dd": float(res["max_drawdown_pct"]),
            "bets": int(res["bets"]),
            "profit_factor": pf,
            "penalty_low_bets": penalty_low_bets,
            "gate_penalty": gate_penalty,
            "feasible_gate": bool(feasible_gate),
            "conservativeness": cons,
            "min_edge": cfg["min_edge"],
            "min_ev": cfg["min_ev"],
            "min_confidence": cfg["min_confidence"],
            "prob_shrink": cfg["prob_shrink"],
            "kelly_fraction": cfg["kelly_fraction"],
            "max_stake_pct": cfg["max_stake_pct"],
            "max_bet_share": cfg["max_bet_share"],
            "min_kelly_f": cfg["min_kelly_f"],
            "min_signal_score": cfg["min_signal_score"],
            "edge_slope_by_odds": cfg["edge_slope_by_odds"],
            "residual_shrink_odds_2_5": cfg["residual_shrink_odds_2_5"],
            "residual_shrink_odds_3_0": cfg["residual_shrink_odds_3_0"],
        }
        cand = {
            "cfg": cfg,
            "res": res,
            "score": float(score),
            "cons": float(cons),
            "feasible_gate": bool(feasible_gate),
            "values": tuple(float(v) for v in values),
        }
        return row, cand

    for i, values in enumerate(combos, start=1):
        row, cand = evaluate_combo(values)
        rows.append(row)
        candidates.append(cand)

        if print_progress and i % 40 == 0:
            print(f"    {progress_label}: {i}/{len(combos)}")

    refinement_evals = 0
    if sampled_mode and candidates and TUNING_REFINEMENT_TOPK > 0:
        top = sorted(candidates, key=lambda x: (x["score"], x["cons"]), reverse=True)[: int(TUNING_REFINEMENT_TOPK)]
        key_to_vals = {k: [float(v) for v in TUNING_GRID[k]] for k in keys}
        key_to_idx = {k: {float(v): i for i, v in enumerate(vals)} for k, vals in key_to_vals.items()}
        seen = set(c["values"] for c in candidates)
        refine_values = []
        for c in top:
            base_vals = list(c["values"])
            for j, k in enumerate(keys):
                vals = key_to_vals[k]
                idx_map = key_to_idx[k]
                cur_idx = idx_map[float(base_vals[j])]
                for n_idx in [cur_idx - 1, cur_idx, cur_idx + 1]:
                    if n_idx < 0 or n_idx >= len(vals):
                        continue
                    nv = list(base_vals)
                    nv[j] = float(vals[n_idx])
                    t = tuple(nv)
                    if t in seen:
                        continue
                    seen.add(t)
                    refine_values.append(t)

        for i, values in enumerate(refine_values, start=1):
            row, cand = evaluate_combo(values)
            rows.append(row)
            candidates.append(cand)
            refinement_evals += 1
            if print_progress and i % 80 == 0:
                print(f"    {progress_label} refinement: {i}/{len(refine_values)}")

    if not candidates:
        best_cfg = dict(baseline_config)
        best_res = simulate_strategy(df_train, best_cfg, initial_bankroll=1000.0, compute_bootstrap=False, calibrator=calibrator)
        pf = float(best_res["profit_factor"]) if np.isfinite(best_res["profit_factor"]) else 2.0
        penalty_low_bets = max(0, MIN_BETS_FOR_TUNING - int(best_res["bets"])) * 0.10
        gate_penalty = 5.0 if float(best_res["max_drawdown_pct"]) > 5.0 else 0.0
        best_score = float(best_res["roi"] - 0.50 * best_res["max_drawdown_pct"] + 0.05 * min(pf, 2.0) - penalty_low_bets - gate_penalty)
        df_tuning = pd.DataFrame(rows)
        diagnostics = {
            "score_formula": score_formula,
            "num_configs_total": int(len(full_combos)),
            "sampled_mode": bool(sampled_mode),
            "num_configs_evaluated_stage1": int(len(combos)),
            "num_configs_evaluated_refinement": int(refinement_evals),
            "num_configs_evaluated_total": int(len(rows)),
            "num_configs_dd_le_5": 0,
            "num_configs_feasible": 0,
            "selected_from_feasible_pool": False,
        }
        return best_cfg, best_res, float(best_score), df_tuning, diagnostics

    feasible_pool = [c for c in candidates if c["feasible_gate"]]
    search_pool = feasible_pool if feasible_pool else candidates
    best = sorted(search_pool, key=lambda x: (x["score"], x["cons"]), reverse=True)[0]
    best_cfg = best["cfg"]
    best_res = best["res"]
    best_score = float(best["score"])

    df_tuning = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    diagnostics = {
        "score_formula": score_formula,
        "num_configs_total": int(len(full_combos)),
        "sampled_mode": bool(sampled_mode),
        "num_configs_evaluated_stage1": int(len(combos)),
        "num_configs_evaluated_refinement": int(refinement_evals),
        "num_configs_evaluated_total": int(len(rows)),
        "num_configs_dd_le_5": int(sum(1 for c in candidates if c["res"]["max_drawdown_pct"] <= 5.0)),
        "num_configs_feasible": int(len(feasible_pool)),
        "selected_from_feasible_pool": bool(len(feasible_pool) > 0),
    }
    return best_cfg, best_res, float(best_score), df_tuning, diagnostics


def build_calibration_report(df_sim, baseline_config, calibrator=None, bins=10):
    cols = [
        "bin",
        "n",
        "avg_pred_raw",
        "calibrated_pred",
        "avg_pred_shrunk",
        "win_rate",
        "gap_raw",
        "gap_calibrated",
        "gap_shrunk",
        "ece_component",
    ]
    if df_sim.empty:
        empty_table = pd.DataFrame(columns=cols)
        return (
            {
                "brier_raw": np.nan,
                "brier_calibrated": np.nan,
                "brier_calibrated_shrunk": np.nan,
                "ece_raw": np.nan,
                "ece_calibrated": np.nan,
                "mce_raw": np.nan,
                "mce_calibrated": np.nan,
                "plot_file": RELIABILITY_PLOT_FILE,
                "table_file": RELIABILITY_TABLE_FILE,
                "calibrator_name": "none",
            },
            empty_table,
        )

    cal = df_sim[["prob_p1_win", "odd_p1", "p1_is_real_winner"]].copy()
    cal["y"] = cal["p1_is_real_winner"].astype(int)
    cal["pred_raw"] = cal["prob_p1_win"].astype(float).clip(0.0, 1.0)

    calibrated_arr = []
    shrunk_arr = []
    for row in cal.itertuples(index=False):
        _, p_cal, p_final = apply_probability_pipeline(row.pred_raw, float(row.odd_p1), baseline_config, calibrator=calibrator)
        calibrated_arr.append(p_cal)
        shrunk_arr.append(p_final)
    cal["pred_calibrated"] = np.asarray(calibrated_arr, dtype=float).clip(0.0, 1.0)
    cal["pred_shrunk"] = np.asarray(shrunk_arr, dtype=float).clip(0.0, 1.0)

    y = cal["y"].to_numpy()
    p_raw = cal["pred_raw"].to_numpy()
    p_cal = cal["pred_calibrated"].to_numpy()
    p_shr = cal["pred_shrunk"].to_numpy()

    brier_raw = float(np.mean((p_raw - y) ** 2))
    brier_cal = float(np.mean((p_cal - y) ** 2))
    brier_shr = float(np.mean((p_shr - y) ** 2))

    ece_raw, mce_raw, _, _, _ = compute_ece_mce(p_raw, y, bins=bins)
    ece_cal, mce_cal, _, _, _ = compute_ece_mce(p_cal, y, bins=bins)

    edges = np.linspace(0.0, 1.0, bins + 1)
    cal["bin"] = pd.cut(cal["pred_calibrated"], bins=edges, include_lowest=True)
    grouped = (
        cal.groupby("bin", observed=False)
        .agg(
            n=("y", "size"),
            avg_pred_raw=("pred_raw", "mean"),
            calibrated_pred=("pred_calibrated", "mean"),
            avg_pred_shrunk=("pred_shrunk", "mean"),
            win_rate=("y", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n"] > 0].copy()
    total_n = grouped["n"].sum()
    grouped["gap_raw"] = grouped["avg_pred_raw"] - grouped["win_rate"]
    grouped["gap_calibrated"] = grouped["calibrated_pred"] - grouped["win_rate"]
    grouped["gap_shrunk"] = grouped["avg_pred_shrunk"] - grouped["win_rate"]
    grouped["ece_component"] = (grouped["n"] / total_n) * grouped["gap_calibrated"].abs()

    for c in ["avg_pred_raw", "calibrated_pred", "avg_pred_shrunk", "win_rate", "gap_raw", "gap_calibrated", "gap_shrunk", "ece_component"]:
        grouped[c] = grouped[c] * 100.0
    grouped["bin"] = grouped["bin"].astype(str)

    # Reliability plot
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.plot(grouped["avg_pred_raw"] / 100.0, grouped["win_rate"] / 100.0, marker="o", label="Raw")
    plt.plot(grouped["calibrated_pred"] / 100.0, grouped["win_rate"] / 100.0, marker="s", label="Calibrated")
    plt.plot(grouped["avg_pred_shrunk"] / 100.0, grouped["win_rate"] / 100.0, marker="^", label="Calibrated+Shrink")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed win rate")
    plt.title("Reliability Curve (2024 matched sides)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RELIABILITY_PLOT_FILE)
    grouped.to_csv(RELIABILITY_TABLE_FILE, index=False)

    report = {
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "brier_calibrated_shrunk": brier_shr,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "mce_raw": mce_raw,
        "mce_calibrated": mce_cal,
        "calibrator_name": calibrator["name"] if calibrator is not None else "identity",
        "plot_file": RELIABILITY_PLOT_FILE,
        "table_file": RELIABILITY_TABLE_FILE,
    }
    return report, grouped


def run_stress_tests(df_sim, baseline_config, calibrator=None, baseline_selection=None):
    print("\n[8] Stress Test (haircut + commission + slippage)...")
    fixed_rows = []
    adaptive_rows = []

    for scenario in STRESS_SCENARIOS:
        cfg = dict(baseline_config)
        cfg.update(scenario)
        res_fixed = simulate_strategy(
            df_sim,
            cfg,
            initial_bankroll=1000.0,
            compute_bootstrap=False,
            calibrator=calibrator,
            fixed_selection=baseline_selection,
            fixed_sizing_no_cost=True,
        )
        res_adaptive = simulate_strategy(
            df_sim,
            cfg,
            initial_bankroll=1000.0,
            compute_bootstrap=False,
            calibrator=calibrator,
            fixed_selection=None,
            fixed_sizing_no_cost=False,
        )

        fixed_rows.append(
            {
                "scenario": scenario["name"],
                "payout_haircut_pct": scenario["payout_haircut_pct"],
                "commission_pct": scenario["commission_pct"],
                "slippage_pct": scenario["slippage_pct"],
                "roi": res_fixed["roi"],
                "bets": res_fixed["bets"],
                "max_drawdown_pct": res_fixed["max_drawdown_pct"],
                "profit_factor": res_fixed["profit_factor"],
            }
        )
        adaptive_rows.append(
            {
                "scenario": scenario["name"],
                "payout_haircut_pct": scenario["payout_haircut_pct"],
                "commission_pct": scenario["commission_pct"],
                "slippage_pct": scenario["slippage_pct"],
                "roi": res_adaptive["roi"],
                "bets": res_adaptive["bets"],
                "max_drawdown_pct": res_adaptive["max_drawdown_pct"],
                "profit_factor": res_adaptive["profit_factor"],
            }
        )

    fixed_df = pd.DataFrame(fixed_rows)
    adaptive_df = pd.DataFrame(adaptive_rows)
    print("    fixed_bets_cost_impact")
    print(fixed_df.to_string(index=False))
    print("    adaptive_with_costs")
    print(adaptive_df.to_string(index=False))

    payload = {
        "fixed_bets_cost_impact": fixed_rows,
        "adaptive_with_costs": adaptive_rows,
    }
    fixed_rois = [r["roi"] for r in fixed_rows]
    monotonic_non_increasing = all(fixed_rois[i] >= fixed_rois[i + 1] for i in range(len(fixed_rois) - 1))
    payload["fixed_mode_monotonic_non_increasing_roi"] = monotonic_non_increasing
    if not monotonic_non_increasing:
        print("    [WARN] Fixed mode ROI non monotono rispetto ai costi.")
    with open(STRESS_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"    Report stress salvato in: {STRESS_REPORT_FILE}")
    return payload


def run_walk_forward_validation(df_sim, baseline_config):
    print("\n[9] Walk-Forward Multi-Split (quarterly)...")
    if df_sim.empty:
        print("    [WARN] Dataset vuoto. Salto walk-forward.")
        return None

    tmp = df_sim.copy()
    tmp["quarter"] = tmp["tourney_date"].dt.to_period("Q")
    quarters = sorted(tmp["quarter"].unique())
    if len(quarters) < 3:
        print("    [WARN] Non abbastanza quarter per walk-forward.")
        return None

    baseline_bank = 1000.0
    tuned_bank = 1000.0
    n_valid_folds = 0
    rows = []

    for i in range(1, len(quarters)):
        train_quarters = quarters[:i]
        test_quarter = quarters[i]

        df_train = tmp[tmp["quarter"].isin(train_quarters)].copy()
        df_test = tmp[tmp["quarter"] == test_quarter].copy()

        train_matches = int(df_train["odds_row_id"].nunique()) if not df_train.empty else 0
        test_matches = int(df_test["odds_row_id"].nunique()) if not df_test.empty else 0
        fold_valid = True
        skip_reason = ""
        if train_matches < MIN_TRAIN_MATCHES_WF:
            fold_valid = False
            skip_reason = f"train_matches<{MIN_TRAIN_MATCHES_WF}"
        elif test_matches < MIN_TEST_MATCHES_WF:
            fold_valid = False
            skip_reason = f"test_matches<{MIN_TEST_MATCHES_WF}"

        if not fold_valid:
            rows.append(
                {
                    "test_quarter": str(test_quarter),
                    "train_matches": train_matches,
                    "test_matches": test_matches,
                    "fold_valid": False,
                    "skip_reason": skip_reason,
                    "train_best_score": np.nan,
                    "train_best_roi": np.nan,
                    "baseline_test_roi": np.nan,
                    "tuned_test_roi": np.nan,
                    "baseline_test_bets": 0,
                    "tuned_test_bets": 0,
                    "baseline_bankroll_end": baseline_bank,
                    "tuned_bankroll_end": tuned_bank,
                    "best_min_edge": np.nan,
                    "best_min_ev": np.nan,
                    "best_min_confidence": np.nan,
                    "best_prob_shrink": np.nan,
                }
            )
            continue

        fold_calibrator = fit_probability_calibrator(df_train)
        best_cfg, best_h_train, best_score, _, _ = tune_strategy_config(
            df_train,
            baseline_config,
            calibrator=fold_calibrator,
            progress_label=f"WF {test_quarter}",
            print_progress=False,
        )
        base_train = simulate_strategy(
            df_train,
            baseline_config,
            initial_bankroll=1000.0,
            compute_bootstrap=False,
            calibrator=fold_calibrator,
        )
        tuned_train_adv = float(best_h_train["roi"]) - float(base_train["roi"])
        use_tuned_policy = tuned_train_adv >= WF_TUNED_TRAIN_ADV_THRESHOLD
        selected_cfg = best_cfg if use_tuned_policy else dict(baseline_config)

        base_test = simulate_strategy(
            df_test,
            baseline_config,
            initial_bankroll=baseline_bank,
            compute_bootstrap=False,
            calibrator=fold_calibrator,
        )
        tuned_test = simulate_strategy(
            df_test,
            selected_cfg,
            initial_bankroll=tuned_bank,
            compute_bootstrap=False,
            calibrator=fold_calibrator,
        )

        rows.append(
            {
                "test_quarter": str(test_quarter),
                "train_matches": train_matches,
                "test_matches": test_matches,
                "fold_valid": True,
                "skip_reason": "",
                "train_best_score": best_score,
                "train_best_roi": best_h_train["roi"],
                "train_baseline_roi": base_train["roi"],
                "train_tuned_adv_roi": tuned_train_adv,
                "use_tuned_policy": bool(use_tuned_policy),
                "baseline_test_roi": base_test["roi"],
                "tuned_test_roi": tuned_test["roi"],
                "baseline_test_bets": base_test["bets"],
                "tuned_test_bets": tuned_test["bets"],
                "baseline_bankroll_end": base_test["bankroll"],
                "tuned_bankroll_end": tuned_test["bankroll"],
                "best_min_edge": selected_cfg["min_edge"],
                "best_min_ev": selected_cfg["min_ev"],
                "best_min_confidence": selected_cfg["min_confidence"],
                "best_prob_shrink": selected_cfg["prob_shrink"],
            }
        )

        baseline_bank = base_test["bankroll"]
        tuned_bank = tuned_test["bankroll"]
        n_valid_folds += 1

    if not rows:
        print("    [WARN] Nessun fold walk-forward valido.")
        return None

    wf_df = pd.DataFrame(rows)
    print(wf_df.to_string(index=False))
    overall_baseline_roi = (baseline_bank - 1000.0) / 1000.0 * 100.0 if n_valid_folds > 0 else np.nan
    overall_tuned_roi = (tuned_bank - 1000.0) / 1000.0 * 100.0 if n_valid_folds > 0 else np.nan
    print(f"    Valid folds: {n_valid_folds}")
    print(f"    Walk-forward chained ROI baseline (valid folds): {overall_baseline_roi:.2f}%")
    print(f"    Walk-forward chained ROI tuned (valid folds): {overall_tuned_roi:.2f}%")

    wf_report = {
        "overall_baseline_roi": float(overall_baseline_roi),
        "overall_tuned_roi": float(overall_tuned_roi),
        "n_valid_folds": int(n_valid_folds),
        "chained_roi_valid_folds": {
            "baseline": float(overall_baseline_roi),
            "tuned": float(overall_tuned_roi),
        },
        "rows": wf_df.to_dict(orient="records"),
    }
    with open(WALKFORWARD_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(wf_report, f, indent=2)
    print(f"    Report walk-forward salvato in: {WALKFORWARD_REPORT_FILE}")
    return wf_report


def run_temporal_validation(df_sim, baseline_config):
    print("\n[7] Validazione Temporale (H1 tuning -> H2 test)...")
    split_date = pd.Timestamp(H1_H2_SPLIT_DATE)
    df_h1 = df_sim[df_sim["tourney_date"] < split_date].copy()
    df_h2 = df_sim[df_sim["tourney_date"] >= split_date].copy()

    h1_matches = int(df_h1["odds_row_id"].nunique()) if not df_h1.empty else 0
    h2_matches = int(df_h2["odds_row_id"].nunique()) if not df_h2.empty else 0
    print(f"    Split date: {split_date.date()} | H1 match: {h1_matches} | H2 match: {h2_matches}")

    if h1_matches == 0 or h2_matches == 0:
        print("    [WARN] Split H1/H2 non valido. Salto validazione temporale.")
        return None

    calibrator_h1 = fit_probability_calibrator(df_h1)

    baseline_h1 = simulate_strategy(
        df_h1,
        baseline_config,
        initial_bankroll=1000.0,
        compute_bootstrap=False,
        calibrator=calibrator_h1,
    )
    baseline_h2 = simulate_strategy(
        df_h2,
        baseline_config,
        initial_bankroll=1000.0,
        compute_bootstrap=True,
        calibrator=calibrator_h1,
    )
    print(f"    Baseline H1 ROI: {baseline_h1['roi']:.2f}% | Bets: {baseline_h1['bets']}")
    print(f"    Baseline H2 ROI: {baseline_h2['roi']:.2f}% | Bets: {baseline_h2['bets']}")
    print(f"    Calibrator selezionato su H1: {calibrator_h1['name']}")

    keys = list(TUNING_GRID.keys())
    combos = list(itertools.product(*[TUNING_GRID[k] for k in keys]))
    print(f"    Tuning grid size: {len(combos)} combinazioni")

    best_cfg, best_h1, best_score, df_tuning, tuning_diag = tune_strategy_config(
        df_h1,
        baseline_config,
        calibrator=calibrator_h1,
        progress_label="Tuning progress",
        print_progress=True,
    )

    tuned_h2 = simulate_strategy(
        df_h2,
        best_cfg,
        initial_bankroll=1000.0,
        compute_bootstrap=True,
        calibrator=calibrator_h1,
    )
    print(f"    Best H1 ROI: {best_h1['roi']:.2f}% | Bets: {best_h1['bets']} | Score: {best_score:.2f}")
    print(f"    Tuned H2 ROI: {tuned_h2['roi']:.2f}% | Bets: {tuned_h2['bets']}")

    print("\n    Top 10 combinazioni H1:")
    cols = [
        "score",
        "roi",
        "max_dd",
        "bets",
        "min_edge",
        "min_ev",
        "min_confidence",
        "prob_shrink",
        "max_bet_share",
        "min_signal_score",
    ]
    print(df_tuning[cols].head(10).to_string(index=False))

    report = {
        "split_date": H1_H2_SPLIT_DATE,
        "baseline_config": baseline_config,
        "baseline_h1": {
            "roi": baseline_h1["roi"],
            "bets": baseline_h1["bets"],
            "max_drawdown_pct": baseline_h1["max_drawdown_pct"],
        },
        "baseline_h2": {
            "roi": baseline_h2["roi"],
            "bets": baseline_h2["bets"],
            "max_drawdown_pct": baseline_h2["max_drawdown_pct"],
            "bootstrap_roi_ci": baseline_h2["bootstrap_roi_ci"],
        },
        "best_h1_config": best_cfg,
        "best_h1_result": {
            "roi": best_h1["roi"],
            "bets": best_h1["bets"],
            "max_drawdown_pct": best_h1["max_drawdown_pct"],
            "score": best_score,
        },
        "tuned_h2_result": {
            "roi": tuned_h2["roi"],
            "bets": tuned_h2["bets"],
            "max_drawdown_pct": tuned_h2["max_drawdown_pct"],
            "bootstrap_roi_ci": tuned_h2["bootstrap_roi_ci"],
        },
        "top10_h1": df_tuning.head(10).to_dict(orient="records"),
        "calibrator_h1": {
            "name": calibrator_h1["name"],
            "metrics": calibrator_h1["metrics"],
        },
        "tuning_diagnostics": tuning_diag,
    }
    return report


def evaluate_oos_gate(consolidated):
    reasons = []
    temporal = consolidated.get("temporal_validation") or {}
    walk = consolidated.get("walk_forward") or {}
    baseline_result = consolidated.get("baseline_result") or {}
    calib = consolidated.get("calibration_metrics") or {}

    wf_baseline = walk.get("overall_baseline_roi")
    wf_tuned = walk.get("overall_tuned_roi")
    mdd = baseline_result.get("max_drawdown_pct")
    ece_raw = calib.get("ece_raw")
    ece_cal = calib.get("ece_calibrated")

    c1 = wf_baseline is not None and np.isfinite(wf_baseline) and wf_baseline > 0
    c2 = (
        wf_baseline is not None
        and wf_tuned is not None
        and np.isfinite(wf_baseline)
        and np.isfinite(wf_tuned)
        and (wf_tuned >= wf_baseline - 0.25)
    )
    c3 = mdd is not None and np.isfinite(mdd) and mdd <= 5.0
    ece_eps = 1e-6
    c4 = (
        ece_raw is not None
        and ece_cal is not None
        and np.isfinite(ece_raw)
        and np.isfinite(ece_cal)
        and (ece_cal <= ece_raw + ece_eps)
    )

    if not c1:
        reasons.append("walkforward_baseline_roi<=0")
    if not c2:
        reasons.append("walkforward_tuned_underperform_baseline_threshold")
    if not c3:
        reasons.append("max_drawdown_over_5pct")
    if not c4:
        reasons.append("ece_calibrated_worse_than_raw_tolerance")

    passed = c1 and c2 and c3 and c4
    n_ok = sum([c1, c2, c3, c4])
    if passed:
        status = "GO"
    elif n_ok >= 3:
        status = "GO_WITH_CAUTION"
    else:
        status = "NO_GO"

    return {
        "status": status,
        "pass": bool(passed),
        "checks": {
            "walkforward_baseline_roi_gt_0": bool(c1),
            "walkforward_tuned_vs_baseline_threshold": bool(c2),
            "max_drawdown_le_5pct": bool(c3),
            "ece_calibrated_lt_ece_raw": bool(c4),
        },
        "reasons": reasons,
    }


def run_backtest_v7():
    print("============================================================")
    print("BACKTEST V10 - ORIENTED VALUE ENGINE")
    print("============================================================")

    # 1. INITIALIZE NORMALIZER
    print("[1] Inizializzazione Normalizzatore (Training Set)...")
    if not os.path.exists(METADATA_FILE):
        print(f"    [ERROR] {METADATA_FILE} mancante.")
        return

    df_train = pd.read_csv(METADATA_FILE, usecols=["p1_name", "p2_name"])
    valid_names = set(df_train["p1_name"].dropna()).union(set(df_train["p2_name"].dropna()))
    normalizer = DataNormalizer(valid_names)
    print(f"    Normalizzatore pronto con {len(valid_names)} giocatori unici.")

    # 2. LOAD FEATURES + MODEL PROBS
    print("\n[2] Caricamento Feature & Predizioni (Sackmann)...")
    try:
        df_feats = load_features_and_predictions()
    except Exception as e:
        print(f"    [ERROR] Errore features/modello: {e}")
        return

    # 3. LOAD ODDS
    print("\n[3] Caricamento e Normalizzazione Quote (Tennis-Data)...")
    try:
        df_odds = load_and_normalize_odds(normalizer)
    except Exception as e:
        print(f"    [ERROR] Errore quote: {e}")
        return

    # 4. MERGE
    print("\n[4] Unione Dati (Training vs Test) - Merge Asof...")
    df_sim, merge_stats = build_oriented_merged_dataset(df_feats, df_odds)
    print(f"    Merge by: {merge_stats['merge_by']}")
    print(f"    Righe quote orientate: {merge_stats['oriented_rows']}")
    print(f"    Lati uniti (step 1, ±1g): {merge_stats['matched_step1']}")
    print(f"    Lati recuperati fallback (step 2, ±3g): {merge_stats['matched_step2']}")
    print(f"    Lati totali uniti: {merge_stats['matched_total_sides']}")
    print(f"    Match unici coperti: {merge_stats['matched_unique_matches']}")

    if df_sim.empty:
        print("    [WARN] Nessun match unito: ROI non calcolabile.")
        return

    # 5. BASELINE SIMULATION
    print("\n[5] Simulazione Value Betting (Fair Odds + Kelly + Top Signals)...")
    freeze_baseline_config(BASELINE_CONFIG, BASELINE_CONFIG_FILE)
    split_date = pd.Timestamp(H1_H2_SPLIT_DATE)
    df_pre_h2 = df_sim[df_sim["tourney_date"] < split_date].copy()
    global_calibrator = fit_probability_calibrator(df_pre_h2)
    print(f"    Calibrator globale (pre-H2): {global_calibrator['name']}")
    baseline_result = simulate_strategy(
        df_sim,
        BASELINE_CONFIG,
        initial_bankroll=1000.0,
        compute_bootstrap=True,
        calibrator=global_calibrator,
    )
    print_strategy_report(baseline_result)

    plt.figure()
    plt.plot(baseline_result["history"])
    plt.title("Backtest V10 (Oriented + Robust)")
    plt.savefig(PLOT_FILE)
    print(f"Grafico salvato: {PLOT_FILE}")

    # 6. CALIBRAZIONE (Reliability Curve + bins)
    print("\n[6] Calibrazione Probabilistica (Reliability bins)...")
    calibration_report, calibration_table = build_calibration_report(df_sim, BASELINE_CONFIG, calibrator=global_calibrator, bins=10)
    print(
        f"    Brier raw: {calibration_report['brier_raw']:.4f} | "
        f"Brier calibrated: {calibration_report['brier_calibrated']:.4f} | "
        f"Brier calibrated+shrink: {calibration_report['brier_calibrated_shrunk']:.4f}"
    )
    print(
        f"    ECE raw: {calibration_report['ece_raw']:.4f} | "
        f"ECE calibrated: {calibration_report['ece_calibrated']:.4f} | "
        f"MCE raw: {calibration_report['mce_raw']:.4f} | "
        f"MCE calibrated: {calibration_report['mce_calibrated']:.4f}"
    )
    print(f"    Calibrator scelto: {calibration_report['calibrator_name']}")
    print(f"    Reliability plot: {calibration_report['plot_file']}")
    print(f"    Reliability table: {calibration_report['table_file']}")
    print("    Top bin summary:")
    print(
        calibration_table[
            ["bin", "n", "calibrated_pred", "avg_pred_shrunk", "win_rate", "gap_calibrated", "ece_component"]
        ].head(10).to_string(index=False)
    )

    # 7. TEMPORAL VALIDATION (H1 tuning -> H2 test)
    temporal_report = run_temporal_validation(df_sim, BASELINE_CONFIG)

    # 8. STRESS TEST
    stress_report = run_stress_tests(
        df_sim,
        BASELINE_CONFIG,
        calibrator=global_calibrator,
        baseline_selection=baseline_result.get("selected_decisions", {}),
    )

    # 9. WALK-FORWARD MULTI-SPLIT
    walkforward_report = run_walk_forward_validation(df_sim, BASELINE_CONFIG)

    # Consolidated report
    consolidated = {
        "baseline_config": BASELINE_CONFIG,
        "merge_stats": merge_stats,
        "baseline_result": {
            "bankroll": baseline_result["bankroll"],
            "roi": baseline_result["roi"],
            "bets": baseline_result["bets"],
            "wins": baseline_result["wins"],
            "win_rate": baseline_result["win_rate"],
            "max_drawdown_pct": baseline_result["max_drawdown_pct"],
            "profit_factor": baseline_result["profit_factor"],
            "bootstrap_roi_ci": baseline_result["bootstrap_roi_ci"],
        },
        "calibration": {
            "brier_raw": calibration_report["brier_raw"],
            "brier_calibrated": calibration_report["brier_calibrated"],
            "brier_calibrated_shrunk": calibration_report["brier_calibrated_shrunk"],
            "brier_shrunk": calibration_report["brier_calibrated_shrunk"],
            "plot_file": calibration_report["plot_file"],
            "table_file": calibration_report["table_file"],
        },
        "calibration_metrics": {
            "calibrator_name": calibration_report["calibrator_name"],
            "brier_raw": calibration_report["brier_raw"],
            "brier_calibrated": calibration_report["brier_calibrated"],
            "brier_calibrated_shrunk": calibration_report["brier_calibrated_shrunk"],
            "ece_raw": calibration_report["ece_raw"],
            "ece_calibrated": calibration_report["ece_calibrated"],
            "mce_raw": calibration_report["mce_raw"],
            "mce_calibrated": calibration_report["mce_calibrated"],
            "selection_primary_metric": global_calibrator.get("selection_primary_metric", "ece"),
            "selection_reason": global_calibrator.get("selection_reason", "n/a"),
            "ece_raw_val": (global_calibrator.get("metrics") or {}).get("ece_raw_val"),
            "ece_sigmoid_val": (global_calibrator.get("metrics") or {}).get("ece_sigmoid_val"),
            "ece_isotonic_val": (global_calibrator.get("metrics") or {}).get("ece_isotonic_val"),
            "calibrator_validation_metrics": global_calibrator.get("metrics", {}),
        },
        "temporal_validation": temporal_report,
        "tuning_diagnostics": (temporal_report or {}).get("tuning_diagnostics"),
        "stress_tests": stress_report,
        "walk_forward": walkforward_report,
    }
    oos_gate = evaluate_oos_gate(consolidated)
    consolidated["oos_gate"] = oos_gate
    with open(VALIDATION_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\nReport consolidato salvato in: {VALIDATION_REPORT_FILE}")
    print("\n[10] OOS Gate Decision")
    print(f"    Status: {oos_gate['status']} | Pass: {oos_gate['pass']}")
    if oos_gate["reasons"]:
        print(f"    Reasons: {', '.join(oos_gate['reasons'])}")


if __name__ == "__main__":
    run_backtest_v7()
