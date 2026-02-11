from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PERIOD_COLUMN_CANDIDATES = ["test_period", "fold", "fold_id", "test_quarter", "period"]
DATE_COLUMN_CANDIDATES = ["tourney_date", "date", "match_date", "event_date"]
CONFIDENCE_COLUMN_CANDIDATES = ["confidence_before_policy", "effective_confidence", "p_calibrated"]
ODDS_COLUMN_CANDIDATES = ["odds", "odd_sel_raw"]


def detect_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    lower_to_col = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        mapped = lower_to_col.get(candidate.lower())
        if mapped:
            return mapped
    return None


def ensure_period_column(
    df: pd.DataFrame,
    out_col: str = "test_period",
    candidates: Sequence[str] = PERIOD_COLUMN_CANDIDATES,
) -> Tuple[pd.DataFrame, str]:
    if df is None:
        return pd.DataFrame(columns=[out_col]), out_col
    work = df.copy()
    period_col = detect_column(work, candidates)
    if period_col is None:
        work[out_col] = "UNKNOWN"
        return work, out_col
    if period_col != out_col:
        work[out_col] = work[period_col].astype(str)
    else:
        work[out_col] = work[out_col].astype(str)
    return work, out_col


def load_thresholds_from_baseline_config(payload: Dict) -> Dict[str, Optional[float]]:
    cfg = payload.get("config", {}) if isinstance(payload, dict) else {}

    def _as_float(name: str) -> Optional[float]:
        try:
            v = cfg.get(name)
            return float(v) if v is not None else None
        except Exception:
            return None

    return {
        "min_confidence": _as_float("min_confidence"),
        "min_odds": _as_float("min_odds"),
        "max_odds": _as_float("max_odds"),
    }


def fold_universe_from_wf_rows(wf_rows_df: pd.DataFrame) -> List[str]:
    if wf_rows_df is None or wf_rows_df.empty:
        return []
    work, period_col = ensure_period_column(wf_rows_df, out_col="test_period")
    if period_col not in work.columns:
        return []
    return [str(x) for x in work["test_period"].dropna().astype(str).tolist()]


def build_fold_summary(
    baseline_bets_df: pd.DataFrame,
    fold_universe: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    work = baseline_bets_df.copy() if baseline_bets_df is not None else pd.DataFrame()
    work, _ = ensure_period_column(work, out_col="test_period")
    for col in ["stake", "pnl", "odd_sel_raw"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if work.empty:
        summary = pd.DataFrame(columns=["test_period", "n_bets", "stake_sum", "pnl_sum", "avg_odds", "roi_pct"])
    else:
        summary = (
            work.groupby("test_period", dropna=False)
            .agg(
                n_bets=("test_period", "size"),
                stake_sum=("stake", "sum"),
                pnl_sum=("pnl", "sum"),
                avg_odds=("odd_sel_raw", "mean"),
            )
            .reset_index()
        )
        summary["roi_pct"] = summary.apply(
            lambda r: (float(r["pnl_sum"]) / float(r["stake_sum"]) * 100.0) if float(r["stake_sum"]) > 0 else 0.0,
            axis=1,
        )

    universe = [str(x) for x in (fold_universe or [])]
    if universe:
        order = {p: i for i, p in enumerate(universe)}
        all_folds = pd.DataFrame({"test_period": universe})
        summary = all_folds.merge(summary, on="test_period", how="left")
        for col in ["n_bets", "stake_sum", "pnl_sum", "avg_odds", "roi_pct"]:
            if col in summary.columns:
                summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0.0)
        summary["n_bets"] = summary["n_bets"].astype(int)
        summary["__sort"] = summary["test_period"].map(lambda x: order.get(str(x), 10**9))
        summary = summary.sort_values(["__sort", "test_period"]).drop(columns=["__sort"]).reset_index(drop=True)
    else:
        summary = summary.sort_values("test_period").reset_index(drop=True)
    return summary


def build_coverage_tables(
    policy_df: pd.DataFrame,
    baseline_bets_df: pd.DataFrame,
    min_confidence: Optional[float],
    min_odds: Optional[float],
    max_odds: Optional[float],
    fold_universe: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    policy = policy_df.copy() if policy_df is not None else pd.DataFrame()
    policy, _ = ensure_period_column(policy, out_col="test_period")
    if "policy_variant" in policy.columns:
        policy = policy[policy["policy_variant"].astype(str) == "baseline"].copy()

    conf_col = detect_column(policy, CONFIDENCE_COLUMN_CANDIDATES)
    odds_col = detect_column(policy, ODDS_COLUMN_CANDIDATES)
    if conf_col is None:
        policy["__conf__"] = np.nan
    else:
        policy["__conf__"] = pd.to_numeric(policy[conf_col], errors="coerce")
    if odds_col is None:
        policy["__odds__"] = np.nan
    else:
        policy["__odds__"] = pd.to_numeric(policy[odds_col], errors="coerce")

    if min_confidence is None:
        policy["pass_conf"] = True
    else:
        policy["pass_conf"] = policy["__conf__"] >= float(min_confidence)
    if min_odds is None and max_odds is None:
        policy["pass_odds"] = True
    else:
        lo = -np.inf if min_odds is None else float(min_odds)
        hi = np.inf if max_odds is None else float(max_odds)
        policy["pass_odds"] = (policy["__odds__"] >= lo) & (policy["__odds__"] <= hi)
    policy["pass_conf_and_odds"] = policy["pass_conf"] & policy["pass_odds"]

    bets = baseline_bets_df.copy() if baseline_bets_df is not None else pd.DataFrame()
    bets, _ = ensure_period_column(bets, out_col="test_period")
    if "policy_variant" in bets.columns:
        bets = bets[bets["policy_variant"].astype(str) == "baseline"].copy()
    bet_counts = bets.groupby("test_period", dropna=False).size().reset_index(name="n_final_bets")

    if policy.empty:
        per_fold = pd.DataFrame(
            columns=[
                "test_period",
                "n_predictions",
                "n_pass_conf",
                "n_pass_odds",
                "n_final_bets",
                "pass_conf_rate",
                "pass_odds_rate",
                "bet_rate_vs_predictions",
            ]
        )
    else:
        per_fold = (
            policy.groupby("test_period", dropna=False)
            .agg(
                n_predictions=("test_period", "size"),
                n_pass_conf=("pass_conf", "sum"),
                n_pass_odds=("pass_conf_and_odds", "sum"),
            )
            .reset_index()
        )
        per_fold["n_pass_conf"] = per_fold["n_pass_conf"].astype(int)
        per_fold["n_pass_odds"] = per_fold["n_pass_odds"].astype(int)
        per_fold = per_fold.merge(bet_counts, on="test_period", how="left")
        per_fold["n_final_bets"] = pd.to_numeric(per_fold["n_final_bets"], errors="coerce").fillna(0).astype(int)
        per_fold["pass_conf_rate"] = per_fold.apply(
            lambda r: (float(r["n_pass_conf"]) / float(r["n_predictions"])) if float(r["n_predictions"]) > 0 else 0.0,
            axis=1,
        )
        per_fold["pass_odds_rate"] = per_fold.apply(
            lambda r: (float(r["n_pass_odds"]) / float(r["n_predictions"])) if float(r["n_predictions"]) > 0 else 0.0,
            axis=1,
        )
        per_fold["bet_rate_vs_predictions"] = per_fold.apply(
            lambda r: (float(r["n_final_bets"]) / float(r["n_predictions"])) if float(r["n_predictions"]) > 0 else 0.0,
            axis=1,
        )

    universe = [str(x) for x in (fold_universe or [])]
    if universe:
        order = {p: i for i, p in enumerate(universe)}
        per_fold = pd.DataFrame({"test_period": universe}).merge(per_fold, on="test_period", how="left")
        for col in ["n_predictions", "n_pass_conf", "n_pass_odds", "n_final_bets"]:
            per_fold[col] = pd.to_numeric(per_fold[col], errors="coerce").fillna(0).astype(int)
        for col in ["pass_conf_rate", "pass_odds_rate", "bet_rate_vs_predictions"]:
            per_fold[col] = pd.to_numeric(per_fold[col], errors="coerce").fillna(0.0)
        per_fold["__sort"] = per_fold["test_period"].map(lambda x: order.get(str(x), 10**9))
        per_fold = per_fold.sort_values(["__sort", "test_period"]).drop(columns=["__sort"]).reset_index(drop=True)
    elif not per_fold.empty:
        per_fold = per_fold.sort_values("test_period").reset_index(drop=True)

    totals = {
        "n_predictions": int(per_fold["n_predictions"].sum()) if not per_fold.empty else 0,
        "n_pass_conf": int(per_fold["n_pass_conf"].sum()) if not per_fold.empty else 0,
        "n_pass_odds": int(per_fold["n_pass_odds"].sum()) if not per_fold.empty else 0,
        "n_final_bets": int(per_fold["n_final_bets"].sum()) if not per_fold.empty else 0,
    }
    return per_fold, totals


def suggest_threshold_pairs(
    policy_df: pd.DataFrame,
    target_bets_per_fold: float,
    max_odds: Optional[float],
    top_k: int = 3,
    fold_universe: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    policy = policy_df.copy()
    policy, _ = ensure_period_column(policy, out_col="test_period")
    if "policy_variant" in policy.columns:
        policy = policy[policy["policy_variant"].astype(str) == "baseline"].copy()

    conf_col = detect_column(policy, CONFIDENCE_COLUMN_CANDIDATES)
    odds_col = detect_column(policy, ODDS_COLUMN_CANDIDATES)
    if conf_col is None or odds_col is None or policy.empty:
        return pd.DataFrame(columns=["min_confidence", "min_odds", "avg_candidates_per_fold", "target_gap"])

    policy["__conf__"] = pd.to_numeric(policy[conf_col], errors="coerce")
    policy["__odds__"] = pd.to_numeric(policy[odds_col], errors="coerce")
    if max_odds is not None:
        policy = policy[policy["__odds__"] <= float(max_odds)].copy()

    policy = policy.dropna(subset=["__conf__", "__odds__", "test_period"])
    if policy.empty:
        return pd.DataFrame(columns=["min_confidence", "min_odds", "avg_candidates_per_fold", "target_gap"])

    conf_q = np.unique(np.quantile(policy["__conf__"], [0.65, 0.75, 0.85]))
    odds_q = np.unique(np.quantile(policy["__odds__"], [0.05, 0.15, 0.25]))

    folds = [str(x) for x in (fold_universe or sorted(policy["test_period"].astype(str).unique().tolist()))]
    n_folds = max(1, len(folds))
    rows: List[Dict[str, float]] = []
    for min_conf in conf_q:
        for min_odds in odds_q:
            mask = (policy["__conf__"] >= float(min_conf)) & (policy["__odds__"] >= float(min_odds))
            n_selected = int(mask.sum())
            avg_candidates_per_fold = float(n_selected) / float(n_folds)
            rows.append(
                {
                    "min_confidence": float(min_conf),
                    "min_odds": float(min_odds),
                    "avg_candidates_per_fold": avg_candidates_per_fold,
                    "target_gap": abs(avg_candidates_per_fold - float(target_bets_per_fold)),
                }
            )
    out = pd.DataFrame(rows).sort_values(["target_gap", "min_confidence", "min_odds"]).head(int(top_k)).reset_index(drop=True)
    return out
