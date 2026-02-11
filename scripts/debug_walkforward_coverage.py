import argparse
from pathlib import Path
from typing import List

import pandas as pd

from _validation_common import RUNS_DIR, extract_run_id, load_json, materialize_run_dir, resolve_run_source, utc_now_iso
from walkforward_coverage import (
    build_coverage_tables,
    build_fold_summary,
    ensure_period_column,
    fold_universe_from_wf_rows,
    load_thresholds_from_baseline_config,
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnostica coverage walk-forward e drop-off filtri.")
    parser.add_argument("--run-id", default=None, help="Run id da analizzare.")
    parser.add_argument("--run-dir", default=None, help="Percorso run (runs/<run_id>).")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source_dir, validation_path = resolve_run_source(run_path=args.run_dir, run_id=args.run_id, runs_dir=RUNS_DIR)
    validation = load_json(validation_path)
    run_id = extract_run_id(validation, source_dir)
    run_dir, _ = materialize_run_dir(run_id=run_id, source_dir=source_dir, validation_payload=validation, runs_dir=RUNS_DIR)

    wf_report_path = run_dir / "backtest_walkforward_report.json"
    bet_path = run_dir / "backtest_walkforward_bet_records.csv"
    policy_path = run_dir / "backtest_walkforward_policy_audit.csv"
    baseline_cfg_path = run_dir / "backtest_baseline_config.json"
    out_path = run_dir / "debug_walkforward_coverage.md"

    missing: List[str] = [p.name for p in [wf_report_path, bet_path, policy_path, baseline_cfg_path] if not p.exists()]
    wf_report = load_json(wf_report_path) if wf_report_path.exists() else {}
    wf_rows_df = pd.DataFrame(wf_report.get("rows", []))
    if not wf_rows_df.empty:
        wf_rows_df, _ = ensure_period_column(wf_rows_df, out_col="test_period")
    fold_universe = fold_universe_from_wf_rows(wf_rows_df)

    bet_df = pd.read_csv(bet_path) if bet_path.exists() else pd.DataFrame()
    if not bet_df.empty:
        bet_df, _ = ensure_period_column(bet_df, out_col="test_period")
    if "policy_variant" in bet_df.columns:
        baseline_bets_df = bet_df[bet_df["policy_variant"].astype(str) == "baseline"].copy()
    else:
        baseline_bets_df = bet_df.copy()
    if baseline_bets_df.empty:
        baseline_bets_df = bet_df.copy()

    policy_df = pd.read_csv(policy_path) if policy_path.exists() else pd.DataFrame()
    if not policy_df.empty:
        policy_df, _ = ensure_period_column(policy_df, out_col="test_period")

    baseline_cfg = load_json(baseline_cfg_path) if baseline_cfg_path.exists() else {}
    thresholds = load_thresholds_from_baseline_config(baseline_cfg)

    fold_summary = build_fold_summary(baseline_bets_df, fold_universe=fold_universe)
    coverage_per_fold, coverage_totals = build_coverage_tables(
        policy_df=policy_df,
        baseline_bets_df=baseline_bets_df,
        min_confidence=thresholds.get("min_confidence"),
        min_odds=thresholds.get("min_odds"),
        max_odds=thresholds.get("max_odds"),
        fold_universe=fold_universe,
    )

    print(f"[debug-wf] run_id={run_id}")
    print(f"[debug-wf] run_dir={run_dir}")
    print(f"[debug-wf] folds={', '.join(fold_universe) if fold_universe else 'n/a'}")
    if not fold_summary.empty:
        print("[debug-wf] n_bets per fold:")
        print(fold_summary[["test_period", "n_bets", "stake_sum", "pnl_sum", "roi_pct"]].to_string(index=False))
    print(
        "[debug-wf] drop-off totals: "
        f"pred={coverage_totals.get('n_predictions', 0)}, "
        f"pass_conf={coverage_totals.get('n_pass_conf', 0)}, "
        f"pass_conf_and_odds={coverage_totals.get('n_pass_odds', 0)}, "
        f"final_bets={coverage_totals.get('n_final_bets', 0)}"
    )

    md_lines: List[str] = []
    md_lines.append("# Debug Walk-Forward Coverage")
    md_lines.append("")
    md_lines.append(f"- generated_at_utc: `{utc_now_iso()}`")
    md_lines.append(f"- run_id: `{run_id}`")
    md_lines.append(f"- run_dir: `{run_dir}`")
    md_lines.append("")
    if missing:
        md_lines.append("## Missing Inputs")
        md_lines.append("")
        for item in missing:
            md_lines.append(f"- `{item}`")
        md_lines.append("")
        md_lines.append("Per rigenerare i file mancanti:")
        md_lines.append("- `python backtest.py`")
        md_lines.append("- `python scripts/analyze_walkforward_folds.py --run-id <run_id>`")
        md_lines.append("")

    md_lines.append("## Folds Found")
    md_lines.append("")
    md_lines.append(f"- fold_freq: `{wf_report.get('fold_freq', 'n/a')}`")
    md_lines.append(f"- n_valid_folds: `{wf_report.get('n_valid_folds', 'n/a')}`")
    md_lines.append(f"- folds: `{', '.join(fold_universe) if fold_universe else 'n/a'}`")
    md_lines.append("")

    md_lines.append("## Bets per Fold")
    md_lines.append("")
    if fold_summary.empty:
        md_lines.append("_Nessun bet record disponibile._")
    else:
        md_lines.append(_md_table(fold_summary[["test_period", "n_bets", "stake_sum", "pnl_sum", "roi_pct"]]))
    md_lines.append("")

    md_lines.append("## Drop-Off Totals")
    md_lines.append("")
    md_lines.append(
        f"- thresholds: `min_confidence={thresholds.get('min_confidence')}`, "
        f"`min_odds={thresholds.get('min_odds')}`, `max_odds={thresholds.get('max_odds')}`"
    )
    md_lines.append(
        f"- totals: pred={coverage_totals.get('n_predictions', 0)}, "
        f"pass_conf={coverage_totals.get('n_pass_conf', 0)}, "
        f"pass_conf_and_odds={coverage_totals.get('n_pass_odds', 0)}, "
        f"final_bets={coverage_totals.get('n_final_bets', 0)}"
    )
    md_lines.append("")

    md_lines.append("## Drop-Off by Fold")
    md_lines.append("")
    md_lines.append(_md_table(coverage_per_fold))
    md_lines.append("")

    out_path.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")
    print(f"[debug-wf] markdown={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
