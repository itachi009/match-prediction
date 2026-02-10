import numpy as np
import pandas as pd


def make_probability_bucket_edges(step=0.05, start=0.50, end=1.00):
    s = max(0.01, float(step))
    start = float(start)
    end = float(end)
    edges = list(np.arange(start, end, s))
    if not edges or abs(edges[0] - start) > 1e-9:
        edges = [start] + edges
    if abs(edges[-1] - end) > 1e-9:
        edges.append(end)
    return np.asarray(edges, dtype=float)


def _safe_logloss(y_true, prob):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    if valid.sum() <= 0:
        return np.nan
    y = y[valid]
    p = np.clip(p[valid], 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def make_bucket_report(
    df,
    bucket_col,
    pred_col,
    y_col,
    stake_col=None,
    pnl_col=None,
):
    work = df.copy()
    if work.empty:
        cols = [
            bucket_col,
            "n_matches",
            "avg_pred_p",
            "empirical_win_rate",
            "calibration_gap",
            "brier",
            "logloss",
        ]
        if stake_col is not None and pnl_col is not None:
            cols.extend(["n_bets_bucket", "roi_bucket_pct"])
        return pd.DataFrame(columns=cols)

    grouped_rows = []
    for bucket_value, group in work.groupby(bucket_col, observed=False):
        g = group.copy()
        g = g[np.isfinite(pd.to_numeric(g[pred_col], errors="coerce")) & np.isfinite(pd.to_numeric(g[y_col], errors="coerce"))]
        if g.empty:
            continue

        pred = pd.to_numeric(g[pred_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)
        n = int(len(g))
        avg_pred = float(np.mean(pred))
        win_rate = float(np.mean(y))
        calibration_gap = float(win_rate - avg_pred)
        brier = float(np.mean((pred - y) ** 2))
        logloss = _safe_logloss(y, pred)

        row = {
            bucket_col: str(bucket_value),
            "n_matches": n,
            "avg_pred_p": avg_pred,
            "empirical_win_rate": win_rate,
            "calibration_gap": calibration_gap,
            "brier": brier,
            "logloss": logloss,
        }
        if stake_col is not None and pnl_col is not None:
            stake_arr = pd.to_numeric(group[stake_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            pnl_arr = pd.to_numeric(group[pnl_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            bet_mask = stake_arr > 0
            total_stake = float(stake_arr[bet_mask].sum())
            total_pnl = float(pnl_arr[bet_mask].sum())
            row["n_bets_bucket"] = int(bet_mask.sum())
            row["roi_bucket_pct"] = float((total_pnl / total_stake) * 100.0) if total_stake > 0 else np.nan
        grouped_rows.append(row)

    if not grouped_rows:
        return pd.DataFrame(
            columns=[
                bucket_col,
                "n_matches",
                "avg_pred_p",
                "empirical_win_rate",
                "calibration_gap",
                "brier",
                "logloss",
                "n_bets_bucket",
                "roi_bucket_pct",
            ]
        )

    out = pd.DataFrame(grouped_rows)
    out = out.sort_values(bucket_col).reset_index(drop=True)
    return out


def summarize_bucket_extremes(df_report, metric, n=3, prefer_small_abs=False):
    if df_report is None or df_report.empty or metric not in df_report.columns:
        return {"best": [], "worst": []}

    tmp = df_report.copy()
    tmp = tmp[np.isfinite(pd.to_numeric(tmp[metric], errors="coerce"))].copy()
    if tmp.empty:
        return {"best": [], "worst": []}

    if prefer_small_abs:
        tmp["_rank_metric"] = pd.to_numeric(tmp[metric], errors="coerce").abs()
        best_df = tmp.nsmallest(n, "_rank_metric")
        worst_df = tmp.nlargest(n, "_rank_metric")
    else:
        best_df = tmp.nlargest(n, metric)
        worst_df = tmp.nsmallest(n, metric)

    return {
        "best": best_df.to_dict(orient="records"),
        "worst": worst_df.to_dict(orient="records"),
    }
