import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from _validation_common import RUNS_DIR, extract_run_id, load_json, materialize_run_dir, resolve_run_source, utc_now_iso
from walkforward_coverage import (
    DATE_COLUMN_CANDIDATES,
    build_coverage_tables,
    build_fold_summary,
    detect_column,
    ensure_period_column,
    fold_universe_from_wf_rows,
    load_thresholds_from_baseline_config,
)


MIN_FOLDS_WARNING = 2
MIN_BETS_WARNING = 30


def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_Nessun dato_"
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")
    headers = list(work.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in work.iterrows():
        vals = [str(row.get(col, "")) for col in headers]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _bucket_report(
    df: pd.DataFrame,
    value_col: str,
    bins: List[float],
    label: str,
) -> pd.DataFrame:
    if value_col not in df.columns or df.empty:
        return pd.DataFrame(columns=["test_period", label, "n_bets", "stake_sum", "pnl_sum", "roi_pct"])
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work[label] = pd.cut(work[value_col], bins=bins, include_lowest=True, right=False)
    grouped = (
        work.groupby(["test_period", label], dropna=False, observed=False)
        .agg(
            n_bets=("odds_row_id", "size"),
            stake_sum=("stake", "sum"),
            pnl_sum=("pnl", "sum"),
        )
        .reset_index()
    )
    grouped["roi_pct"] = grouped.apply(
        lambda r: (float(r["pnl_sum"]) / float(r["stake_sum"]) * 100.0) if float(r["stake_sum"]) > 0 else 0.0,
        axis=1,
    )
    return grouped.sort_values(["test_period", "pnl_sum"], ascending=[True, True]).reset_index(drop=True)


def _top_loss_segments(
    odds_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    unc_df: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> pd.DataFrame:
    q_folds = sorted([x for x in fold_summary["test_period"].astype(str).tolist() if x.endswith("Q3") or x.endswith("Q4")])
    if not q_folds:
        worst = fold_summary.sort_values("pnl_sum", ascending=True).head(2)
        q_folds = sorted(worst["test_period"].astype(str).tolist())

    items = []
    for bucket_name, source, bucket_col in [
        ("odds_bucket", odds_df, "odds_bucket"),
        ("confidence_bucket", conf_df, "confidence_bucket"),
        ("uncertainty_bucket", unc_df, "uncertainty_bucket"),
    ]:
        if source.empty:
            continue
        for _, row in source.iterrows():
            period = str(row.get("test_period", ""))
            if period not in q_folds:
                continue
            pnl = float(row.get("pnl_sum", 0.0))
            if pnl >= 0:
                continue
            items.append(
                {
                    "test_period": period,
                    "bucket_type": bucket_name,
                    "bucket": str(row.get(bucket_col)),
                    "n_bets": int(row.get("n_bets", 0)),
                    "pnl_sum": pnl,
                    "roi_pct": float(row.get("roi_pct", 0.0)),
                }
            )
    if not items:
        return pd.DataFrame(columns=["test_period", "bucket_type", "bucket", "n_bets", "pnl_sum", "roi_pct"])
    out = pd.DataFrame(items).sort_values("pnl_sum", ascending=True).head(3).reset_index(drop=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analisi diagnostica fold-level walk-forward.")
    parser.add_argument("--run-path", default=None, help="Percorso run (runs/ o runs/<run_id>).")
    parser.add_argument("--run-id", default=None, help="Run id da analizzare.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source_dir, validation_path = resolve_run_source(run_path=args.run_path, run_id=args.run_id, runs_dir=RUNS_DIR)
    validation = load_json(validation_path)
    run_id = extract_run_id(validation, source_dir)
    run_dir, _ = materialize_run_dir(run_id=run_id, source_dir=source_dir, validation_payload=validation, runs_dir=RUNS_DIR)

    wf_report_path = run_dir / "backtest_walkforward_report.json"
    baseline_cfg_path = run_dir / "backtest_baseline_config.json"
    bet_path = run_dir / "backtest_walkforward_bet_records.csv"
    policy_path = run_dir / "backtest_walkforward_policy_audit.csv"
    wf_report = load_json(wf_report_path) if wf_report_path.exists() else {}
    wf_rows_df = pd.DataFrame(wf_report.get("rows", []))
    if not wf_rows_df.empty:
        wf_rows_df, _ = ensure_period_column(wf_rows_df, out_col="test_period")
    fold_universe = fold_universe_from_wf_rows(wf_rows_df)

    md_lines: List[str] = []
    md_lines.append("# Walk-Forward Fold Analysis")
    md_lines.append("")
    md_lines.append(f"- generated_at_utc: `{utc_now_iso()}`")
    md_lines.append(f"- run_id: `{run_id}`")
    md_lines.append(f"- run_dir: `{run_dir}`")
    md_lines.append("")

    missing: List[str] = [p.name for p in [wf_report_path, bet_path, policy_path] if not p.exists()]
    if not wf_report_path.exists() and not bet_path.exists() and not policy_path.exists():
        md_lines.append("## Missing Inputs")
        md_lines.append("")
        for item in missing:
            md_lines.append(f"- `{item}`")
        md_lines.append("")
        out_path = run_dir / "analysis_walkforward.md"
        out_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
        print(f"[analyze-folds] report: {out_path}")
        print("[analyze-folds] done (graceful degradation)")
        return 0

    bet_df = pd.read_csv(bet_path) if bet_path.exists() else pd.DataFrame()
    if not bet_df.empty:
        bet_df, _ = ensure_period_column(bet_df, out_col="test_period")
    policy_df = pd.read_csv(policy_path) if policy_path.exists() else pd.DataFrame()
    if not policy_df.empty:
        policy_df, _ = ensure_period_column(policy_df, out_col="test_period")
    baseline_cfg = load_json(baseline_cfg_path) if baseline_cfg_path.exists() else {}
    thresholds = load_thresholds_from_baseline_config(baseline_cfg)

    for col in ["stake", "pnl", "odd_sel_raw", "effective_confidence", "uncertainty_score"]:
        if col in bet_df.columns:
            bet_df[col] = pd.to_numeric(bet_df[col], errors="coerce")

    if "policy_variant" in bet_df.columns:
        baseline_df = bet_df[bet_df["policy_variant"].astype(str) == "baseline"].copy()
    else:
        baseline_df = bet_df.copy()
    if baseline_df.empty:
        baseline_df = bet_df.copy()

    fold_summary = build_fold_summary(baseline_df, fold_universe=fold_universe)
    fold_count = int(len(fold_universe)) if fold_universe else int(fold_summary["test_period"].nunique() if not fold_summary.empty else 0)
    total_bets = int(fold_summary["n_bets"].sum()) if not fold_summary.empty else 0

    warnings: List[str] = []
    if fold_count < MIN_FOLDS_WARNING:
        warnings.append(
            f"Folds trovati insufficienti: {fold_count} (<{MIN_FOLDS_WARNING}). Verifica input walk-forward e period column."
        )
    if total_bets < MIN_BETS_WARNING:
        warnings.append(
            f"Bet totali bassi: {total_bets} (<{MIN_BETS_WARNING}). Q3/Q4 potrebbero non essere statisticamente diagnostici."
        )

    per_fold_coverage, coverage_totals = build_coverage_tables(
        policy_df=policy_df,
        baseline_bets_df=baseline_df,
        min_confidence=thresholds.get("min_confidence"),
        min_odds=thresholds.get("min_odds"),
        max_odds=thresholds.get("max_odds"),
        fold_universe=fold_universe,
    )

    bet_folds = sorted(baseline_df["test_period"].astype(str).unique().tolist()) if not baseline_df.empty else []
    policy_folds = sorted(policy_df["test_period"].astype(str).unique().tolist()) if not policy_df.empty else []
    wf_folds = list(fold_universe)
    date_col = detect_column(baseline_df, DATE_COLUMN_CANDIDATES)
    date_min = ""
    date_max = ""
    if date_col and date_col in baseline_df.columns and not baseline_df.empty:
        dates = pd.to_datetime(baseline_df[date_col], errors="coerce").dropna()
        if not dates.empty:
            date_min = str(dates.min().date())
            date_max = str(dates.max().date())

    odds_bucket = _bucket_report(
        baseline_df,
        value_col="odd_sel_raw",
        bins=[1.4, 1.7, 2.1, 2.7, 100.0],
        label="odds_bucket",
    )
    conf_col = "effective_confidence" if "effective_confidence" in baseline_df.columns else "p_model"
    confidence_bucket = _bucket_report(
        baseline_df,
        value_col=conf_col,
        bins=[0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.000001],
        label="confidence_bucket",
    )
    uncertainty_bucket = _bucket_report(
        baseline_df,
        value_col="uncertainty_score",
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.000001],
        label="uncertainty_bucket",
    )
    top_losses = _top_loss_segments(odds_bucket, confidence_bucket, uncertainty_bucket, fold_summary)

    if warnings:
        md_lines.append("## Warnings")
        md_lines.append("")
        for item in warnings:
            md_lines.append(f"- {item}")
            print(f"[WARN] {item}")
        md_lines.append("")

    md_lines.append("## Data Coverage")
    md_lines.append("")
    md_lines.append(f"- fold_freq: `{wf_report.get('fold_freq', 'n/a')}`")
    md_lines.append(f"- n_valid_folds: `{wf_report.get('n_valid_folds', 'n/a')}`")
    md_lines.append(f"- folds_from_wf_rows: `{', '.join(wf_folds) if wf_folds else 'n/a'}`")
    md_lines.append(f"- folds_from_bet_records: `{', '.join(bet_folds) if bet_folds else 'n/a'}`")
    md_lines.append(f"- folds_from_policy_audit: `{', '.join(policy_folds) if policy_folds else 'n/a'}`")
    md_lines.append(f"- date_coverage_min: `{date_min or 'n/a'}`")
    md_lines.append(f"- date_coverage_max: `{date_max or 'n/a'}`")
    md_lines.append(
        f"- thresholds_used: `min_confidence={thresholds.get('min_confidence')}`, "
        f"`min_odds={thresholds.get('min_odds')}`, `max_odds={thresholds.get('max_odds')}`"
    )
    md_lines.append("")
    md_lines.append("### Coverage Drop-Off (baseline)")
    md_lines.append("")
    md_lines.append(
        f"- totals: predictions={coverage_totals.get('n_predictions', 0)}, "
        f"pass_conf={coverage_totals.get('n_pass_conf', 0)}, "
        f"pass_conf_and_odds={coverage_totals.get('n_pass_odds', 0)}, "
        f"final_bets={coverage_totals.get('n_final_bets', 0)}"
    )
    md_lines.append("")
    md_lines.append(_md_table(per_fold_coverage))
    md_lines.append("")

    md_lines.append("## Fold Summary")
    md_lines.append("")
    md_lines.append(_md_table(fold_summary))
    md_lines.append("")

    md_lines.append("## Odds Bucket Distribution")
    md_lines.append("")
    md_lines.append(_md_table(odds_bucket))
    md_lines.append("")

    md_lines.append("## Confidence Bucket Distribution")
    md_lines.append("")
    md_lines.append(_md_table(confidence_bucket))
    md_lines.append("")

    md_lines.append("## Uncertainty Bucket Distribution")
    md_lines.append("")
    md_lines.append(_md_table(uncertainty_bucket))
    md_lines.append("")

    md_lines.append("## Top Loss Segments (Q3/Q4 o fold peggiori)")
    md_lines.append("")
    md_lines.append(_md_table(top_losses))
    md_lines.append("")

    if not policy_df.empty:
        if not policy_df.empty and {"test_period", "policy_reason", "policy_allowed"}.issubset(policy_df.columns):
            blocked = policy_df[policy_df["policy_allowed"].astype(str).str.lower().isin(["false", "0", "no"])].copy()
            blocked_summary = (
                blocked.groupby(["test_period", "policy_reason"], dropna=False)
                .size()
                .reset_index(name="blocked_count")
                .sort_values(["test_period", "blocked_count"], ascending=[True, False])
            )
            md_lines.append("## Policy Audit (Blocked by Reason)")
            md_lines.append("")
            md_lines.append(_md_table(blocked_summary))
            md_lines.append("")

    out_path = run_dir / "analysis_walkforward.md"
    out_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    print(f"[analyze-folds] report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
