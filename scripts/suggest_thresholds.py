import argparse
from pathlib import Path

import pandas as pd

from _validation_common import RUNS_DIR, extract_run_id, load_json, materialize_run_dir, resolve_run_source
from walkforward_coverage import (
    ensure_period_column,
    load_thresholds_from_baseline_config,
    suggest_threshold_pairs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Suggerisce soglie min_confidence/min_odds per aumentare coverage fold.")
    parser.add_argument("--run-id", default=None, help="Run id di riferimento.")
    parser.add_argument("--run-dir", default=None, help="Percorso run (runs/<run_id>).")
    parser.add_argument("--target-bets-per-fold", type=float, default=50.0, help="Target medio di candidate bet per fold.")
    parser.add_argument("--top-k", type=int, default=3, help="Numero combinazioni proposte.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source_dir, validation_path = resolve_run_source(run_path=args.run_dir, run_id=args.run_id, runs_dir=RUNS_DIR)
    validation = load_json(validation_path)
    run_id = extract_run_id(validation, source_dir)
    run_dir, _ = materialize_run_dir(run_id=run_id, source_dir=source_dir, validation_payload=validation, runs_dir=RUNS_DIR)

    policy_path = run_dir / "backtest_walkforward_policy_audit.csv"
    if not policy_path.exists():
        print(f"[suggest-thresholds] missing: {policy_path}")
        print("[suggest-thresholds] genera prima una run con backtest walk-forward.")
        return 0

    policy_df = pd.read_csv(policy_path)
    if policy_df.empty:
        print(f"[suggest-thresholds] policy audit vuoto: {policy_path}")
        return 0
    policy_df, _ = ensure_period_column(policy_df, out_col="test_period")

    baseline_cfg = load_json(run_dir / "backtest_baseline_config.json")
    thresholds = load_thresholds_from_baseline_config(baseline_cfg)
    candidates = suggest_threshold_pairs(
        policy_df=policy_df,
        target_bets_per_fold=args.target_bets_per_fold,
        max_odds=thresholds.get("max_odds"),
        top_k=max(1, int(args.top_k)),
    )

    print(f"[suggest-thresholds] run_id={run_id}")
    print(
        f"[suggest-thresholds] current thresholds: min_confidence={thresholds.get('min_confidence')}, "
        f"min_odds={thresholds.get('min_odds')}, max_odds={thresholds.get('max_odds')}"
    )
    print(f"[suggest-thresholds] target_bets_per_fold={args.target_bets_per_fold}")
    if candidates.empty:
        print("[suggest-thresholds] nessuna proposta disponibile (dati insufficienti).")
        return 0

    print(candidates.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
