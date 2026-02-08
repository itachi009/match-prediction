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
from sklearn.metrics import roc_auc_score

try:
    from utils import DataNormalizer, standardize_dates
except ImportError:
    print("[ERROR] utils.py not found. Please ensure it exists.")
    sys.exit(1)


# --- FILE CONFIG ---
FEATURES_FILE = "processed_features.csv"
METADATA_FILE = "clean_matches.csv"
ODDS_FILE_LOCAL = "2024_test.csv"
ACTIVE_MODEL_FILE = "active_model.json"
MODEL_FILE_DEFAULT = "model_v9_balanced.pkl"
MODEL_FEATURES_FILE_DEFAULT = "model_features.pkl"
PLOT_FILE = "real_backtest.png"
BASELINE_CONFIG_FILE = "backtest_baseline_config.json"
VALIDATION_REPORT_FILE = "backtest_validation_report.json"
WALKFORWARD_REPORT_FILE = "backtest_walkforward_report.json"
STRESS_REPORT_FILE = "backtest_stress_report.json"
RELIABILITY_PLOT_FILE = "reliability_curve.png"
RELIABILITY_TABLE_FILE = "reliability_table.csv"
BACKTEST_TUNING_CONFIG_FILE = "configs/backtest_tuning.json"


DEFAULT_TUNING_RUNTIME = {
    "min_bets_for_tuning": 100,
    "objective_mode": "roi_minus_lambda_bets_ratio",
    "lambda_bets": 0.5,
    "max_bets_increase_pct": 0.20,
    "fallback_eps_roi_pct": 0.10,
    "restricted_tuning_mode": False,
    "restricted_fixed_min_ev": 0.06,
    "restricted_fixed_prob_shrink": 0.60,
    "restricted_min_edge_values": [0.06, 0.065],
    "restricted_min_confidence_values": [0.64, 0.66],
    "fold_freq": "quarter",
}


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _as_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_float_list(value, default_list):
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                continue
        return out if out else [float(v) for v in default_list]
    return [float(v) for v in default_list]


def load_backtest_tuning_runtime(path=BACKTEST_TUNING_CONFIG_FILE):
    runtime = dict(DEFAULT_TUNING_RUNTIME)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            if isinstance(payload, dict):
                for k in runtime:
                    if k in payload:
                        runtime[k] = payload[k]
        except Exception as e:
            print(f"[WARN] Impossibile leggere tuning config {path}: {e}. Uso default.")

    env_overrides = {
        "BT_MIN_BETS_FOR_TUNING": ("min_bets_for_tuning", _as_int),
        "BT_OBJECTIVE_MODE": ("objective_mode", str),
        "BT_LAMBDA_BETS": ("lambda_bets", _as_float),
        "BT_MAX_BETS_INCREASE_PCT": ("max_bets_increase_pct", _as_float),
        "BT_FALLBACK_EPS_ROI_PCT": ("fallback_eps_roi_pct", _as_float),
        "BT_RESTRICTED_TUNING_MODE": ("restricted_tuning_mode", _as_bool),
        "BT_FOLD_FREQ": ("fold_freq", str),
    }
    for env_key, (cfg_key, caster) in env_overrides.items():
        raw = os.getenv(env_key)
        if raw is None:
            continue
        if caster in {_as_int, _as_float, _as_bool}:
            runtime[cfg_key] = caster(raw, runtime[cfg_key])
        else:
            runtime[cfg_key] = caster(raw)

    runtime["min_bets_for_tuning"] = _as_int(runtime.get("min_bets_for_tuning"), 100)
    runtime["lambda_bets"] = _as_float(runtime.get("lambda_bets"), 0.5)
    runtime["max_bets_increase_pct"] = _as_float(runtime.get("max_bets_increase_pct"), 0.20)
    runtime["fallback_eps_roi_pct"] = _as_float(runtime.get("fallback_eps_roi_pct"), 0.10)
    runtime["restricted_tuning_mode"] = _as_bool(runtime.get("restricted_tuning_mode"), False)
    runtime["restricted_fixed_min_ev"] = _as_float(runtime.get("restricted_fixed_min_ev"), 0.06)
    runtime["restricted_fixed_prob_shrink"] = _as_float(runtime.get("restricted_fixed_prob_shrink"), 0.60)
    runtime["restricted_min_edge_values"] = _as_float_list(runtime.get("restricted_min_edge_values"), [0.06, 0.065])
    runtime["restricted_min_confidence_values"] = _as_float_list(
        runtime.get("restricted_min_confidence_values"), [0.64, 0.66]
    )

    objective_mode = str(runtime.get("objective_mode", "roi_minus_lambda_bets_ratio")).strip().lower()
    valid_objectives = {"roi_minus_lambda_bets_ratio", "roi_minus_lambda_bets"}
    runtime["objective_mode"] = objective_mode if objective_mode in valid_objectives else "roi_minus_lambda_bets_ratio"

    fold_freq = str(runtime.get("fold_freq", "quarter")).strip().lower()
    runtime["fold_freq"] = fold_freq if fold_freq in {"quarter", "halfyear"} else "quarter"
    return runtime


def resolve_active_model_registry(path=ACTIVE_MODEL_FILE):
    default_payload = {
        "run_id": "default_backtest_fallback",
        "model_family": "xgb",
        "model_path": MODEL_FILE_DEFAULT,
        "features_path": MODEL_FEATURES_FILE_DEFAULT,
        "metadata_path": None,
    }
    if not os.path.exists(path):
        print(f"[WARN] {path} non trovato. Uso fallback legacy.")
        return default_payload
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        merged = dict(default_payload)
        merged.update(payload or {})
        return merged
    except Exception as e:
        print(f"[WARN] Impossibile leggere {path}: {e}. Uso fallback legacy.")
        return default_payload


ACTIVE_MODEL = resolve_active_model_registry()
MODEL_FILE = ACTIVE_MODEL.get("model_path", MODEL_FILE_DEFAULT)
MODEL_FEATURES_FILE = ACTIVE_MODEL.get("features_path", MODEL_FEATURES_FILE_DEFAULT)


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
    "no_odds_eval_enabled": True,
    "no_odds_eval_year": 2024,
    "no_odds_eval_levels": ["C", "F"],
    "no_odds_gate_policy": "light",
    "no_odds_min_matches": 200,
}


# --- TEMPORAL VALIDATION (POINT 3) ---
H1_H2_SPLIT_DATE = "2024-07-01"
MIN_TRAIN_MATCHES_WF = 300
MIN_TEST_MATCHES_WF = 120
BOOTSTRAP_SAMPLES = 3000
WF_TUNED_TRAIN_ADV_THRESHOLD = 0.25
MIN_VALID_FOLDS_FOR_WF_GATE = 3
MIN_NOT_WORSE_FOLDS = 2
MAX_TUNING_EVALS = 2000
TUNING_RANDOM_SEED = 42
TUNING_REFINEMENT_TOPK = 30
# Full legacy grid (kept for backward compatibility and optional full mode).
TUNING_GRID = {
    "min_edge": [0.055, 0.06, 0.065],
    "min_ev": [0.05, 0.06],
    "min_confidence": [0.62, 0.64, 0.66],
    "prob_shrink": [0.60, 0.62, 0.64],
    "kelly_fraction": [0.01, 0.015, 0.02],
    "max_stake_pct": [0.01, 0.015, 0.02],
    "max_bet_share": [0.08, 0.10],
    "min_kelly_f": [0.05, 0.07, 0.10],
    "min_signal_score": [0.003, 0.004, 0.005],
    "edge_slope_by_odds": [0.015, 0.02, 0.03],
    "residual_shrink_odds_2_5": [0.96, 0.98],
    "residual_shrink_odds_3_0": [0.92, 0.95, 0.96],
}

BACKTEST_TUNING_RUNTIME = load_backtest_tuning_runtime()
MIN_BETS_FOR_TUNING = int(BACKTEST_TUNING_RUNTIME["min_bets_for_tuning"])
OBJECTIVE_MODE = str(BACKTEST_TUNING_RUNTIME["objective_mode"])
LAMBDA_BETS = float(BACKTEST_TUNING_RUNTIME["lambda_bets"])
MAX_BETS_INCREASE_PCT = float(BACKTEST_TUNING_RUNTIME["max_bets_increase_pct"])
FALLBACK_EPS_ROI_PCT = float(BACKTEST_TUNING_RUNTIME["fallback_eps_roi_pct"])
RESTRICTED_TUNING_MODE = bool(BACKTEST_TUNING_RUNTIME["restricted_tuning_mode"])
RESTRICTED_FIXED_MIN_EV = float(BACKTEST_TUNING_RUNTIME["restricted_fixed_min_ev"])
RESTRICTED_FIXED_PROB_SHRINK = float(BACKTEST_TUNING_RUNTIME["restricted_fixed_prob_shrink"])
RESTRICTED_MIN_EDGE_VALUES = [float(v) for v in BACKTEST_TUNING_RUNTIME["restricted_min_edge_values"]]
RESTRICTED_MIN_CONFIDENCE_VALUES = [float(v) for v in BACKTEST_TUNING_RUNTIME["restricted_min_confidence_values"]]
FOLD_FREQ = str(BACKTEST_TUNING_RUNTIME["fold_freq"])

STRESS_SCENARIOS = [
    {"name": "baseline", "payout_haircut_pct": 0.00, "commission_pct": 0.00, "slippage_pct": 0.00},
    {"name": "mild", "payout_haircut_pct": 0.01, "commission_pct": 0.002, "slippage_pct": 0.002},
    {"name": "medium", "payout_haircut_pct": 0.02, "commission_pct": 0.003, "slippage_pct": 0.003},
    {"name": "hard", "payout_haircut_pct": 0.03, "commission_pct": 0.005, "slippage_pct": 0.005},
]


def render_progress_bar(current, total, width=30):
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    ratio = current / total
    done = int(round(width * ratio))
    bar = "#" * done + "-" * (width - done)
    return f"[{bar}] {ratio * 100:6.2f}% ({current}/{total})"


def freeze_baseline_config(config, path=BASELINE_CONFIG_FILE):
    payload = {
        "frozen_at": pd.Timestamp.utcnow().isoformat(),
        "model_file": MODEL_FILE,
        "model_features_file": MODEL_FEATURES_FILE,
        "active_model_file": ACTIVE_MODEL_FILE,
        "active_model": ACTIVE_MODEL,
        "features_file": FEATURES_FILE,
        "odds_file": ODDS_FILE_LOCAL,
        "tuning_runtime": BACKTEST_TUNING_RUNTIME,
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
            "is_level_a",
            "is_level_c",
            "is_level_f",
            "is_level_cf",
            "p1_days_since_last_match",
            "p2_days_since_last_match",
            "days_since_last_diff",
            "inactive_bucket_diff",
            "matches_last_30d_diff",
            "long_stop_90d_diff",
            "no_prev_match_diff",
            "days_since_last_diff_cf",
            "long_stop_90d_diff_cf",
            "upset_win_rate_12_diff",
            "bad_loss_rate_12_diff",
            "confidence_rank_score_12_diff",
            "confidence_n_valid_12_diff",
            "confidence_rank_score_12_diff_cf",
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


def safe_float_or_none(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def infer_match_levels(df):
    if "level" in df.columns:
        level_raw = df["level"]
    elif "match_level" in df.columns:
        level_raw = df["match_level"]
    else:
        level_raw = pd.Series([None] * len(df), index=df.index, dtype="object")

    def norm_level(val):
        if pd.isna(val):
            return None
        s = str(val).strip().upper()
        if not s:
            return None
        if s in {"A", "C", "F"}:
            return s
        if s.startswith("ATP"):
            return "A"
        if "CHALL" in s:
            return "C"
        if "FUT" in s:
            return "F"
        first = s[0]
        return first if first in {"A", "C", "F"} else None

    levels = level_raw.apply(norm_level).astype("object")

    if "is_level_c" in df.columns:
        is_c = pd.to_numeric(df["is_level_c"], errors="coerce").fillna(0) > 0.5
        levels.loc[levels.isna() & is_c] = "C"
    if "is_level_f" in df.columns:
        is_f = pd.to_numeric(df["is_level_f"], errors="coerce").fillna(0) > 0.5
        levels.loc[levels.isna() & is_f] = "F"
    if "is_level_a" in df.columns:
        is_a = pd.to_numeric(df["is_level_a"], errors="coerce").fillna(0) > 0.5
        levels.loc[levels.isna() & is_a] = "A"

    return levels


def infer_match_years(df):
    date_col = None
    for c in ["tourney_date", "date"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    raw = df[date_col]
    years = pd.to_datetime(raw, errors="coerce").dt.year.astype("float64")

    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_year = np.floor(numeric / 10000.0)
    fallback = years.isna() & numeric_year.between(1900, 2100)
    years.loc[fallback] = numeric_year.loc[fallback]
    return years


def extract_binary_target(df):
    if "target" in df.columns:
        y = pd.to_numeric(df["target"], errors="coerce")
    elif "p1_is_real_winner" in df.columns:
        y = pd.to_numeric(df["p1_is_real_winner"], errors="coerce")
    else:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    y = y.where(y.isin([0, 1]))
    return y.astype("float64")


def compute_prob_metrics(y_true, prob, bins=10):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = np.clip(p[valid], 1e-9, 1.0 - 1e-9)
    n = int(len(y))
    if n == 0:
        return {
            "n_matches": 0,
            "logloss": None,
            "brier": None,
            "auc": None,
            "accuracy_0_5": None,
            "ece_10_bins": None,
            "mce_10_bins": None,
        }

    y_int = y.astype(int)
    logloss = -np.mean(y_int * np.log(p) + (1 - y_int) * np.log(1.0 - p))
    brier = np.mean((p - y_int) ** 2)
    acc = np.mean((p >= 0.5).astype(int) == y_int)
    ece, mce, _, _, _ = compute_ece_mce(p, y_int, bins=bins)

    auc = None
    if len(np.unique(y_int)) >= 2:
        try:
            auc = float(roc_auc_score(y_int, p))
        except Exception:
            auc = None

    return {
        "n_matches": n,
        "logloss": safe_float_or_none(logloss),
        "brier": safe_float_or_none(brier),
        "auc": safe_float_or_none(auc),
        "accuracy_0_5": safe_float_or_none(acc),
        "ece_10_bins": safe_float_or_none(ece),
        "mce_10_bins": safe_float_or_none(mce),
    }


def compute_no_odds_train_prior(eval_year, allowed_levels):
    try:
        header = pd.read_csv(FEATURES_FILE, nrows=0)
    except Exception:
        return 0.5, 0

    keep = [
        c
        for c in [
            "target",
            "p1_is_real_winner",
            "date",
            "tourney_date",
            "level",
            "match_level",
            "is_level_a",
            "is_level_c",
            "is_level_f",
        ]
        if c in header.columns
    ]
    if not keep:
        return 0.5, 0

    hist = pd.read_csv(FEATURES_FILE, usecols=keep, low_memory=False)
    years = infer_match_years(hist)
    levels = infer_match_levels(hist)
    y = extract_binary_target(hist)

    mask = years.lt(float(eval_year)) & levels.isin(allowed_levels) & y.notna()
    n_rows = int(mask.sum())
    if n_rows <= 0:
        return 0.5, 0

    prior = float(np.clip(y.loc[mask].mean(), 1e-6, 1.0 - 1e-6))
    if not np.isfinite(prior):
        return 0.5, 0
    return prior, n_rows


def build_no_odds_eval_report(df_feats, baseline_config):
    eval_year = int(baseline_config.get("no_odds_eval_year", 2024))
    min_matches = int(baseline_config.get("no_odds_min_matches", 200))
    gate_policy = str(baseline_config.get("no_odds_gate_policy", "light"))
    cfg_levels = baseline_config.get("no_odds_eval_levels", ["C", "F"])
    levels = [str(x).strip().upper() for x in cfg_levels if str(x).strip()]
    levels = list(dict.fromkeys(levels)) if levels else ["C", "F"]

    df = df_feats.copy()
    if "tourney_date" not in df.columns and "date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "tourney_date" in df.columns:
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    else:
        df["tourney_date"] = pd.NaT

    years = infer_match_years(df)
    match_levels = infer_match_levels(df)
    y = extract_binary_target(df)
    raw_prob = pd.to_numeric(df.get("prob_p1_win"), errors="coerce")

    mask = years.eq(float(eval_year)) & match_levels.isin(levels) & y.notna() & raw_prob.notna()
    subset = pd.DataFrame(
        {
            "tourney_date": df["tourney_date"],
            "level": match_levels,
            "y": y,
            "prob_raw": raw_prob,
        }
    ).loc[mask].copy()

    split_date = pd.Timestamp(H1_H2_SPLIT_DATE)
    calibrator_input = subset[subset["tourney_date"] < split_date][["tourney_date", "prob_raw", "y"]].copy()
    calibrator_input = calibrator_input.rename(columns={"prob_raw": "prob_p1_win", "y": "p1_is_real_winner"})
    calibrator = fit_probability_calibrator(calibrator_input)

    if len(subset) > 0:
        subset["prob_final"] = subset["prob_raw"].apply(
            lambda p: apply_probability_pipeline(p, 2.0, baseline_config, calibrator=calibrator)[2]
        )
    else:
        subset["prob_final"] = np.nan

    combined = compute_prob_metrics(subset["y"], subset["prob_final"], bins=10)
    by_level = {}
    for lvl in levels:
        lv = subset[subset["level"] == lvl]
        by_level[lvl] = compute_prob_metrics(lv["y"], lv["prob_final"], bins=10)

    prior_prob, prior_n_rows = compute_no_odds_train_prior(eval_year, levels)
    baseline_p50 = compute_prob_metrics(subset["y"], np.full(len(subset), 0.5, dtype=float), bins=10)
    baseline_prior = compute_prob_metrics(
        subset["y"],
        np.full(len(subset), prior_prob, dtype=float),
        bins=10,
    )

    warnings = []
    if combined["auc"] is not None and combined["auc"] < 0.50:
        warnings.append(f"auc_below_0_50:{combined['auc']:.4f}")
    if combined["ece_10_bins"] is not None and combined["ece_10_bins"] > 0.10:
        warnings.append(f"ece_above_0_10:{combined['ece_10_bins']:.4f}")

    reasons = []
    if combined["n_matches"] < min_matches:
        status = "insufficient_data"
        reasons.append(f"n_matches_below_min:{combined['n_matches']}<{min_matches}")
    else:
        improves_logloss = (
            combined["logloss"] is not None
            and baseline_prior["logloss"] is not None
            and combined["logloss"] < baseline_prior["logloss"]
        )
        improves_brier = (
            combined["brier"] is not None
            and baseline_prior["brier"] is not None
            and combined["brier"] < baseline_prior["brier"]
        )
        if improves_logloss and improves_brier:
            status = "pass"
            reasons.append("model_beats_train_prior_on_logloss_and_brier")
        else:
            status = "fail"
            if not improves_logloss:
                reasons.append("logloss_not_better_than_train_prior")
            if not improves_brier:
                reasons.append("brier_not_better_than_train_prior")

    return {
        "enabled": True,
        "source": FEATURES_FILE,
        "year": eval_year,
        "levels": levels,
        "combined": combined,
        "by_level": by_level,
        "baselines": {
            "p50": {
                "probability": 0.5,
                "combined": baseline_p50,
            },
            "train_prior": {
                "probability": safe_float_or_none(prior_prob),
                "n_train_rows": prior_n_rows,
                "combined": baseline_prior,
            },
        },
        "gate": {
            "policy": gate_policy,
            "status": status,
            "reasons": reasons,
        },
        "warnings": warnings,
        "calibrator_name": calibrator.get("name", "identity"),
        "odds_proxy_for_pipeline": 2.0,
    }


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


def compute_tuning_objective(train_roi, n_bets, n_matches_train):
    roi = float(train_roi)
    bets = float(n_bets)
    n_matches = max(1.0, float(n_matches_train))
    if OBJECTIVE_MODE == "roi_minus_lambda_bets":
        return roi - float(LAMBDA_BETS) * bets
    return roi - float(LAMBDA_BETS) * (bets / n_matches)


def build_tuning_search_space():
    if RESTRICTED_TUNING_MODE:
        keys = ["min_edge", "min_confidence"]
        combos = list(itertools.product(RESTRICTED_MIN_EDGE_VALUES, RESTRICTED_MIN_CONFIDENCE_VALUES))
        return keys, combos, True
    keys = list(TUNING_GRID.keys())
    combos = list(itertools.product(*[TUNING_GRID[k] for k in keys]))
    return keys, combos, False


def _guardrail_reason(bets, max_allowed):
    if float(bets) <= float(max_allowed) + 1e-9:
        return "ok"
    return f"bets_above_guardrail:{int(bets)}>{float(max_allowed):.2f}"


def build_fold_period_columns(date_series, fold_freq):
    freq = str(fold_freq).strip().lower()
    if freq == "halfyear":
        years = date_series.dt.year.astype(int)
        half = np.where(date_series.dt.month <= 6, 1, 2)
        half_s = pd.Series(half, index=date_series.index)
        labels = years.astype(str) + "H" + half_s.astype(str)
        sort_key = years * 2 + (half_s - 1)
        return labels.astype(str), sort_key.astype(int)

    q = date_series.dt.to_period("Q")
    labels = q.astype(str)
    sort_key = q.dt.year * 4 + q.dt.quarter
    return labels.astype(str), sort_key.astype(int)


def tune_strategy_config(df_train, baseline_config, calibrator=None, progress_label="Tuning", print_progress=False):
    objective_formula = f"{OBJECTIVE_MODE} (lambda={LAMBDA_BETS})"
    keys, full_combos, restricted_mode = build_tuning_search_space()
    sampled_mode = (not restricted_mode) and len(full_combos) > MAX_TUNING_EVALS
    if sampled_mode:
        rng = np.random.default_rng(TUNING_RANDOM_SEED)
        sampled_idx = rng.choice(len(full_combos), size=MAX_TUNING_EVALS, replace=False)
        combos = [full_combos[int(i)] for i in sampled_idx]
    else:
        combos = list(full_combos)

    if "odds_row_id" in df_train.columns and not df_train.empty:
        n_matches_train = int(df_train["odds_row_id"].nunique())
    else:
        n_matches_train = int(len(df_train))

    baseline_res = simulate_strategy(
        df_train,
        baseline_config,
        initial_bankroll=1000.0,
        compute_bootstrap=False,
        calibrator=calibrator,
    )
    baseline_objective = compute_tuning_objective(baseline_res["roi"], baseline_res["bets"], n_matches_train)
    baseline_bets = int(baseline_res["bets"])
    max_bets_allowed = float(baseline_bets) * (1.0 + float(MAX_BETS_INCREASE_PCT))

    rows = []
    candidates = []

    def evaluate_combo(values):
        cfg = dict(baseline_config)
        if restricted_mode:
            cfg["min_ev"] = float(RESTRICTED_FIXED_MIN_EV)
            cfg["prob_shrink"] = float(RESTRICTED_FIXED_PROB_SHRINK)
        for k, v in zip(keys, values):
            cfg[k] = float(v)

        res = simulate_strategy(df_train, cfg, initial_bankroll=1000.0, compute_bootstrap=False, calibrator=calibrator)
        objective = compute_tuning_objective(res["roi"], res["bets"], n_matches_train)
        cons = config_conservativeness_score(cfg)
        feasible_gate = float(res["max_drawdown_pct"]) <= 5.0 and int(res["bets"]) >= int(MIN_BETS_FOR_TUNING)
        guardrail_reason = _guardrail_reason(res["bets"], max_bets_allowed)
        guardrail_passed = guardrail_reason == "ok"

        row = {
            "score": float(objective),
            "objective": float(objective),
            "roi": float(res["roi"]),
            "max_dd": float(res["max_drawdown_pct"]),
            "bets": int(res["bets"]),
            "feasible_gate": bool(feasible_gate),
            "guardrail_passed": bool(guardrail_passed),
            "guardrail_reason": guardrail_reason,
            "objective_delta_vs_baseline": float(objective - baseline_objective),
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
            "objective": float(objective),
            "cons": float(cons),
            "feasible_gate": bool(feasible_gate),
            "guardrail_passed": bool(guardrail_passed),
            "guardrail_reason": guardrail_reason,
            "values": tuple(float(v) for v in values),
        }
        return row, cand

    stage1_step = max(1, len(combos) // 40)
    for i, values in enumerate(combos, start=1):
        row, cand = evaluate_combo(values)
        rows.append(row)
        candidates.append(cand)

        if print_progress and (i % stage1_step == 0 or i == len(combos)):
            print(f"    {progress_label} stage1 {render_progress_bar(i, len(combos))}")

    refinement_evals = 0
    if sampled_mode and candidates and TUNING_REFINEMENT_TOPK > 0:
        top = sorted(candidates, key=lambda x: (x["objective"], x["cons"]), reverse=True)[: int(TUNING_REFINEMENT_TOPK)]
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

        refine_step = max(1, len(refine_values) // 40)
        for i, values in enumerate(refine_values, start=1):
            row, cand = evaluate_combo(values)
            rows.append(row)
            candidates.append(cand)
            refinement_evals += 1
            if print_progress and (i % refine_step == 0 or i == len(refine_values)):
                print(f"    {progress_label} refine {render_progress_bar(i, len(refine_values))}")

    sorted_candidates = sorted(candidates, key=lambda x: (x["objective"], x["cons"]), reverse=True) if candidates else []
    feasible_pool = [c for c in sorted_candidates if c["feasible_gate"]]
    guardrail_pool = [c for c in feasible_pool if c["guardrail_passed"]]

    selected = None
    selection_reason = "baseline_fallback_no_candidates"
    selected_guardrail_passed = True
    selected_guardrail_reason = "baseline_fallback"

    if guardrail_pool:
        selected = guardrail_pool[0]
        selection_reason = "tuned_guardrail_pass"
        selected_guardrail_passed = True
        selected_guardrail_reason = selected["guardrail_reason"]
    elif feasible_pool:
        selection_reason = "baseline_fallback_guardrail_failed"
        selected_guardrail_passed = False
        selected_guardrail_reason = feasible_pool[0]["guardrail_reason"]
    elif sorted_candidates:
        selection_reason = "baseline_fallback_no_feasible_candidates"
        selected_guardrail_passed = False
        selected_guardrail_reason = "no_candidate_with_dd<=5_and_min_bets"

    if selected is not None:
        selected_obj = float(selected["objective"])
        selected_roi = float(selected["res"]["roi"])
        baseline_roi = float(baseline_res["roi"])
        if selected_obj > baseline_objective and selected_roi < (baseline_roi - float(FALLBACK_EPS_ROI_PCT)):
            selected = None
            selection_reason = "baseline_fallback_safety_roi"
            selected_guardrail_passed = False
            selected_guardrail_reason = (
                f"objective_gt_baseline_but_roi_lt_baseline_minus_eps:{selected_roi:.4f}<{baseline_roi - FALLBACK_EPS_ROI_PCT:.4f}"
            )

    if selected is None:
        best_cfg = dict(baseline_config)
        best_res = baseline_res
        best_objective = float(baseline_objective)
        selected_is_tuned = False
    else:
        best_cfg = selected["cfg"]
        best_res = selected["res"]
        best_objective = float(selected["objective"])
        selected_is_tuned = True

    df_tuning = pd.DataFrame(rows)
    if not df_tuning.empty:
        df_tuning = df_tuning.sort_values(["objective", "conservativeness"], ascending=[False, False]).reset_index(drop=True)

    diagnostics = {
        "objective_formula": objective_formula,
        "score_formula": objective_formula,
        "objective_mode": OBJECTIVE_MODE,
        "lambda_bets": float(LAMBDA_BETS),
        "num_matches_train": int(n_matches_train),
        "min_bets_for_tuning": int(MIN_BETS_FOR_TUNING),
        "max_bets_increase_pct": float(MAX_BETS_INCREASE_PCT),
        "fallback_eps_roi_pct": float(FALLBACK_EPS_ROI_PCT),
        "restricted_tuning_mode": bool(restricted_mode),
        "num_configs_total": int(len(full_combos)),
        "sampled_mode": bool(sampled_mode),
        "num_configs_evaluated_stage1": int(len(combos)),
        "num_configs_evaluated_refinement": int(refinement_evals),
        "num_configs_evaluated_total": int(len(rows)),
        "num_configs_dd_le_5": int(sum(1 for c in candidates if c["res"]["max_drawdown_pct"] <= 5.0)),
        "num_configs_feasible": int(len(feasible_pool)),
        "num_configs_guardrail_pass": int(len(guardrail_pool)),
        "max_bets_allowed_from_baseline": float(max_bets_allowed),
        "baseline_train_roi": float(baseline_res["roi"]),
        "baseline_train_bets": int(baseline_res["bets"]),
        "baseline_objective": float(baseline_objective),
        "selected_train_roi": float(best_res["roi"]),
        "selected_train_bets": int(best_res["bets"]),
        "selected_objective": float(best_objective),
        "selected_is_tuned": bool(selected_is_tuned),
        "selected_guardrail_passed": bool(selected_guardrail_passed),
        "selected_guardrail_reason": selected_guardrail_reason,
        "selected_from_feasible_pool": bool(selected_is_tuned),
        "selection_reason": selection_reason,
        "objective_delta_vs_baseline": float(best_objective - baseline_objective),
        "roi_delta_vs_baseline": float(best_res["roi"] - baseline_res["roi"]),
    }
    return best_cfg, best_res, float(best_objective), df_tuning, diagnostics


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
    print(f"\n[9] Walk-Forward Multi-Split ({FOLD_FREQ})...")
    if df_sim.empty:
        print("    [WARN] Dataset vuoto. Salto walk-forward.")
        return None

    tmp = df_sim.copy()
    tmp["test_period"], tmp["_fold_sort"] = build_fold_period_columns(tmp["tourney_date"], FOLD_FREQ)
    periods = (
        tmp[["test_period", "_fold_sort"]]
        .drop_duplicates()
        .sort_values("_fold_sort")
        .reset_index(drop=True)["test_period"]
        .tolist()
    )
    if len(periods) < 3:
        print(f"    [WARN] Non abbastanza periodi per walk-forward (freq={FOLD_FREQ}).")
        return None

    baseline_bank = 1000.0
    tuned_bank = 1000.0
    n_valid_folds = 0
    rows = []

    total_folds = len(periods) - 1
    for i in range(1, len(periods)):
        print(f"    WF progress {render_progress_bar(i, total_folds)}")
        train_periods = periods[:i]
        test_period = periods[i]

        df_train = tmp[tmp["test_period"].isin(train_periods)].copy()
        df_test = tmp[tmp["test_period"] == test_period].copy()

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
                    "test_period": str(test_period),
                    "test_quarter": str(test_period) if FOLD_FREQ == "quarter" else None,
                    "train_matches": train_matches,
                    "test_matches": test_matches,
                    "fold_valid": False,
                    "skip_reason": skip_reason,
                    "train_best_score": np.nan,
                    "train_best_roi": np.nan,
                    "train_baseline_roi": np.nan,
                    "train_tuned_adv_roi": np.nan,
                    "baseline_train_bets": np.nan,
                    "tuned_train_bets": np.nan,
                    "objective_best": np.nan,
                    "objective_baseline": np.nan,
                    "objective_delta_vs_baseline": np.nan,
                    "bets_increase_pct_vs_baseline_train": np.nan,
                    "guardrail_passed": False,
                    "guardrail_reason": f"fold_invalid:{skip_reason}",
                    "restricted_tuning_mode": bool(RESTRICTED_TUNING_MODE),
                    "selection_reason": "fold_invalid",
                    "use_tuned_policy": np.nan,
                    "tuned_not_worse": np.nan,
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
        best_cfg, best_h_train, best_score, _, tuning_diag = tune_strategy_config(
            df_train,
            baseline_config,
            calibrator=fold_calibrator,
            progress_label=f"WF {test_period}",
            print_progress=True,
        )
        base_train = simulate_strategy(
            df_train,
            baseline_config,
            initial_bankroll=1000.0,
            compute_bootstrap=False,
            calibrator=fold_calibrator,
        )
        base_objective = compute_tuning_objective(base_train["roi"], base_train["bets"], train_matches)
        best_objective = float(best_score)
        tuned_train_adv = float(best_h_train["roi"]) - float(base_train["roi"])
        selected_is_tuned = bool(tuning_diag.get("selected_is_tuned", False))
        use_tuned_policy = bool(selected_is_tuned and tuned_train_adv >= WF_TUNED_TRAIN_ADV_THRESHOLD)
        selected_cfg = best_cfg if use_tuned_policy else dict(baseline_config)
        baseline_train_bets = int(base_train["bets"])
        tuned_train_bets = int(best_h_train["bets"])
        if baseline_train_bets > 0:
            bets_increase_pct = (float(tuned_train_bets - baseline_train_bets) / float(baseline_train_bets)) * 100.0
        else:
            bets_increase_pct = np.nan

        objective_delta = float(best_objective - base_objective)
        roi_delta = float(tuned_train_adv)
        guardrail_passed = bool(tuning_diag.get("selected_guardrail_passed", True))
        guardrail_reason = str(tuning_diag.get("selected_guardrail_reason", "ok"))
        selection_reason = str(tuning_diag.get("selection_reason", "n/a"))

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

        bets_delta_txt = f"{bets_increase_pct:+.2f}%" if np.isfinite(bets_increase_pct) else "n/a"
        print(
            f"    Fold {test_period} | Train bets baseline/tuned: {baseline_train_bets}/{tuned_train_bets} "
            f"(delta {bets_delta_txt})"
        )
        print(
            f"    Fold {test_period} | Train ROI delta: {roi_delta:+.2f}% | "
            f"Objective delta: {objective_delta:+.4f} | Guardrail: {guardrail_passed} ({guardrail_reason})"
        )
        print(
            f"    Fold {test_period} | Selected policy: {'tuned' if use_tuned_policy else 'baseline'} "
            f"(selection_reason={selection_reason})"
        )

        rows.append(
            {
                "test_period": str(test_period),
                "test_quarter": str(test_period) if FOLD_FREQ == "quarter" else None,
                "train_matches": train_matches,
                "test_matches": test_matches,
                "fold_valid": True,
                "skip_reason": "",
                "train_best_score": best_score,
                "train_best_roi": best_h_train["roi"],
                "train_baseline_roi": base_train["roi"],
                "train_tuned_adv_roi": tuned_train_adv,
                "baseline_train_bets": baseline_train_bets,
                "tuned_train_bets": tuned_train_bets,
                "objective_best": float(best_objective),
                "objective_baseline": float(base_objective),
                "objective_delta_vs_baseline": float(objective_delta),
                "bets_increase_pct_vs_baseline_train": float(bets_increase_pct) if np.isfinite(bets_increase_pct) else np.nan,
                "guardrail_passed": bool(guardrail_passed),
                "guardrail_reason": guardrail_reason,
                "restricted_tuning_mode": bool(tuning_diag.get("restricted_tuning_mode", RESTRICTED_TUNING_MODE)),
                "selection_reason": selection_reason,
                "use_tuned_policy": bool(use_tuned_policy),
                "tuned_not_worse": bool(float(tuned_test["roi"]) >= float(base_test["roi"])),
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
    overall_tuned_vs_baseline_roi_diff = (
        float(overall_tuned_roi - overall_baseline_roi)
        if np.isfinite(overall_tuned_roi) and np.isfinite(overall_baseline_roi)
        else np.nan
    )
    valid_df = wf_df[wf_df["fold_valid"] == True].copy()
    tuned_not_worse_count = int(valid_df["tuned_not_worse"].fillna(False).astype(bool).sum()) if not valid_df.empty else 0
    tuned_not_worse_min_required = int(min(MIN_NOT_WORSE_FOLDS, n_valid_folds))
    tuned_not_worse_check = bool(
        n_valid_folds >= MIN_VALID_FOLDS_FOR_WF_GATE and tuned_not_worse_count >= tuned_not_worse_min_required
    )
    print(f"    Valid folds: {n_valid_folds}")
    print(f"    Walk-forward chained ROI baseline (valid folds): {overall_baseline_roi:.2f}%")
    print(f"    Walk-forward chained ROI tuned (valid folds): {overall_tuned_roi:.2f}%")
    print(f"    Tuned vs baseline ROI diff (valid folds): {overall_tuned_vs_baseline_roi_diff:.2f}%")
    print(
        f"    Tuned not worse folds: {tuned_not_worse_count}/{n_valid_folds} "
        f"(required>={tuned_not_worse_min_required}, gate_ready={tuned_not_worse_check})"
    )

    wf_report = {
        "fold_freq": FOLD_FREQ,
        "objective_mode": OBJECTIVE_MODE,
        "lambda_bets": float(LAMBDA_BETS),
        "overall_baseline_roi": float(overall_baseline_roi),
        "overall_tuned_roi": float(overall_tuned_roi),
        "overall_tuned_vs_baseline_roi_diff": float(overall_tuned_vs_baseline_roi_diff)
        if np.isfinite(overall_tuned_vs_baseline_roi_diff)
        else np.nan,
        "n_valid_folds": int(n_valid_folds),
        "min_valid_folds_required_for_gate": int(MIN_VALID_FOLDS_FOR_WF_GATE),
        "tuned_not_worse_count": int(tuned_not_worse_count),
        "tuned_not_worse_min_required": int(tuned_not_worse_min_required),
        "tuned_not_worse_check": bool(tuned_not_worse_check),
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

    if RESTRICTED_TUNING_MODE:
        combos_count = len(RESTRICTED_MIN_EDGE_VALUES) * len(RESTRICTED_MIN_CONFIDENCE_VALUES)
        print(
            "    Tuning mode: restricted | "
            f"grid size: {combos_count} (min_edge x min_confidence), "
            f"fixed min_ev={RESTRICTED_FIXED_MIN_EV}, fixed prob_shrink={RESTRICTED_FIXED_PROB_SHRINK}"
        )
    else:
        keys = list(TUNING_GRID.keys())
        combos_count = len(list(itertools.product(*[TUNING_GRID[k] for k in keys])))
        print(f"    Tuning mode: full | grid size: {combos_count} combinazioni")

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
    print(f"    Best H1 ROI: {best_h1['roi']:.2f}% | Bets: {best_h1['bets']} | Objective: {best_score:.4f}")
    print(f"    Tuned H2 ROI: {tuned_h2['roi']:.2f}% | Bets: {tuned_h2['bets']}")

    print("\n    Top 10 combinazioni H1:")
    cols = [
        "objective",
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
            "objective": best_score,
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
    n_valid_folds = walk.get("n_valid_folds")
    tuned_not_worse_check = walk.get("tuned_not_worse_check")
    mdd = baseline_result.get("max_drawdown_pct")
    ece_raw = calib.get("ece_raw")
    ece_cal = calib.get("ece_calibrated")

    c1 = wf_baseline is not None and np.isfinite(wf_baseline) and wf_baseline > 0
    c_folds = n_valid_folds is not None and int(n_valid_folds) >= MIN_VALID_FOLDS_FOR_WF_GATE
    c2 = (
        c_folds
        and wf_baseline is not None
        and wf_tuned is not None
        and np.isfinite(wf_baseline)
        and np.isfinite(wf_tuned)
        and (wf_tuned >= wf_baseline - 0.10)
    )
    c2b = bool(c_folds and bool(tuned_not_worse_check))
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
    if not c_folds:
        reasons.append("walkforward_not_enough_valid_folds")
    if not c2:
        reasons.append("walkforward_tuned_underperform_baseline_threshold")
    if not c2b:
        reasons.append("walkforward_tuned_not_worse_fold_count_failed")
    if not c3:
        reasons.append("max_drawdown_over_5pct")
    if not c4:
        reasons.append("ece_calibrated_worse_than_raw_tolerance")

    passed = c1 and c_folds and c2 and c2b and c3 and c4
    n_ok = sum([c1, c_folds, c2, c2b, c3, c4])
    if passed:
        status = "GO"
    elif n_ok >= 5:
        status = "GO_WITH_CAUTION"
    else:
        status = "NO_GO"

    return {
        "status": status,
        "pass": bool(passed),
        "checks": {
            "walkforward_baseline_roi_gt_0": bool(c1),
            "walkforward_min_valid_folds": bool(c_folds),
            "walkforward_tuned_vs_baseline_threshold": bool(c2),
            "walkforward_tuned_not_worse_fold_count": bool(c2b),
            "max_drawdown_le_5pct": bool(c3),
            "ece_calibrated_lt_ece_raw": bool(c4),
        },
        "reasons": reasons,
    }


def build_promotion_decision(consolidated):
    oos_gate = consolidated.get("oos_gate") or {}
    walk = consolidated.get("walk_forward") or {}
    baseline_result = consolidated.get("baseline_result") or {}
    no_odds = consolidated.get("no_odds_eval") or {}
    no_odds_gate = no_odds.get("gate") or {}

    wf_baseline = walk.get("overall_baseline_roi")
    wf_tuned = walk.get("overall_tuned_roi")
    mdd = baseline_result.get("max_drawdown_pct")
    no_odds_status = no_odds_gate.get("status")

    c1 = bool(oos_gate.get("pass", False))
    c2 = (
        wf_baseline is not None
        and wf_tuned is not None
        and np.isfinite(wf_baseline)
        and np.isfinite(wf_tuned)
        and float(wf_tuned) >= float(wf_baseline) - 0.10
    )
    c3 = mdd is not None and np.isfinite(mdd) and float(mdd) <= 5.0
    c4 = no_odds_status in {"pass", "insufficient_data"}

    reasons = []
    if not c1:
        reasons.append("oos_gate_not_passed")
    if not c2:
        reasons.append("walkforward_tuned_below_baseline_minus_0_10")
    if not c3:
        reasons.append("max_drawdown_over_5pct")
    if not c4:
        reasons.append("no_odds_eval_gate_not_pass_or_insufficient_data")

    status = "promote" if (c1 and c2 and c3 and c4) else "keep_baseline"
    return {
        "status": status,
        "reasons": reasons,
        "criteria_snapshot": {
            "oos_gate_pass": bool(c1),
            "walk_forward_overall_baseline_roi": float(wf_baseline) if wf_baseline is not None and np.isfinite(wf_baseline) else None,
            "walk_forward_overall_tuned_roi": float(wf_tuned) if wf_tuned is not None and np.isfinite(wf_tuned) else None,
            "max_drawdown_pct": float(mdd) if mdd is not None and np.isfinite(mdd) else None,
            "no_odds_eval_gate_status": no_odds_status,
        },
    }


def run_backtest_v7():
    print("============================================================")
    print("BACKTEST V10 - ORIENTED VALUE ENGINE")
    print("============================================================")
    print(f"[MODEL] Active run_id: {ACTIVE_MODEL.get('run_id')}")
    print(f"[MODEL] model_path: {MODEL_FILE}")
    print(f"[MODEL] features_path: {MODEL_FEATURES_FILE}")
    print(
        f"[TUNING] min_bets={MIN_BETS_FOR_TUNING} | objective_mode={OBJECTIVE_MODE} | "
        f"lambda_bets={LAMBDA_BETS} | guardrail_max_bets_inc={MAX_BETS_INCREASE_PCT:.2f} | "
        f"restricted_mode={RESTRICTED_TUNING_MODE} | fold_freq={FOLD_FREQ}"
    )

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
    no_odds_report = None
    if bool(BASELINE_CONFIG.get("no_odds_eval_enabled", False)):
        print("\n[2b] Valutazione Model-Only C/F (senza quote)...")
        try:
            no_odds_report = build_no_odds_eval_report(df_feats, BASELINE_CONFIG)
            gate = no_odds_report.get("gate", {})
            print(
                f"    C/F combined matches: {no_odds_report.get('combined', {}).get('n_matches', 0)} | "
                f"Gate: {gate.get('status', 'n/a')}"
            )
            if no_odds_report.get("warnings"):
                print(f"    Warnings: {', '.join(no_odds_report['warnings'])}")
        except Exception as e:
            print(f"    [WARN] no_odds_eval non disponibile: {e}")
            no_odds_report = {
                "enabled": True,
                "source": FEATURES_FILE,
                "year": int(BASELINE_CONFIG.get('no_odds_eval_year', 2024)),
                "levels": BASELINE_CONFIG.get("no_odds_eval_levels", ["C", "F"]),
                "error": str(e),
            }

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
        "tuning_runtime": BACKTEST_TUNING_RUNTIME,
        "active_model": ACTIVE_MODEL,
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
    if no_odds_report is not None:
        consolidated["no_odds_eval"] = no_odds_report
    oos_gate = evaluate_oos_gate(consolidated)
    consolidated["oos_gate"] = oos_gate
    promotion_decision = build_promotion_decision(consolidated)
    consolidated["promotion_decision"] = promotion_decision
    with open(VALIDATION_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\nReport consolidato salvato in: {VALIDATION_REPORT_FILE}")
    print("\n[10] OOS Gate Decision")
    print(f"    Status: {oos_gate['status']} | Pass: {oos_gate['pass']}")
    if oos_gate["reasons"]:
        print(f"    Reasons: {', '.join(oos_gate['reasons'])}")
    print("\n[11] Promotion Decision")
    print(f"    Status: {promotion_decision['status']}")
    if promotion_decision["reasons"]:
        print(f"    Reasons: {', '.join(promotion_decision['reasons'])}")


if __name__ == "__main__":
    run_backtest_v7()
