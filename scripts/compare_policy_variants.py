import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _validation_common import (
    RUNS_DIR,
    extract_run_id,
    load_json,
    materialize_run_dir,
    resolve_run_source,
    save_json,
    table_str,
    utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

SOFT_POLICY_PRESET = {
    "MP_POLICY_ENABLED": "true",
    "MP_POLICY_SOFT_MODE": "true",
    "MP_POLICY_SKIP_ODDS_BUCKET_17_21": "false",
    "MP_POLICY_MAX_UNCERTAINTY_FOR_BET": "0.55",
}

OFF_POLICY_PRESET = {
    "MP_POLICY_ENABLED": "false",
}


def _parse_env_pairs(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not raw:
        return out
    for chunk in raw.split(","):
        piece = chunk.strip()
        if not piece or "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _run_backtest(env_overrides: Dict[str, str], backtest_args: str) -> int:
    env = dict(os.environ)
    env.update(env_overrides)
    cmd = [sys.executable, str(REPO_ROOT / "backtest.py")]
    if backtest_args.strip():
        cmd.extend(shlex.split(backtest_args.strip()))
    print(f"[compare-policy] cmd: {' '.join(cmd)}")
    print(f"[compare-policy] env overrides: {env_overrides}")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return int(result.returncode)


def _resolve_run_dir_from_latest() -> Tuple[str, Path]:
    validation_path = RUNS_DIR / "backtest_validation_report.json"
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing validation report: {validation_path}")
    payload = load_json(validation_path)
    run_id = extract_run_id(payload, RUNS_DIR)
    run_dir, _ = materialize_run_dir(run_id=run_id, source_dir=RUNS_DIR, validation_payload=payload, runs_dir=RUNS_DIR)
    return run_id, run_dir


def _resolve_run_dir(run_id: str) -> Path:
    source_dir, validation_path = resolve_run_source(run_id=run_id, run_path=None, runs_dir=RUNS_DIR)
    payload = load_json(validation_path)
    resolved_run_id = extract_run_id(payload, source_dir)
    run_dir, _ = materialize_run_dir(
        run_id=resolved_run_id,
        source_dir=source_dir,
        validation_payload=payload,
        runs_dir=RUNS_DIR,
    )
    return run_dir


def _extract_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    baseline = report.get("baseline_result") or {}
    wf = report.get("walk_forward") or {}
    oos = report.get("oos_gate") or {}
    return {
        "run_id": report.get("backtest_run_id"),
        "active_model_run_id": report.get("active_model_run_id"),
        "policy_layer_enabled": bool(report.get("policy_layer_enabled", False)),
        "baseline_roi_pct": baseline.get("roi"),
        "baseline_bets": baseline.get("bets"),
        "baseline_max_drawdown_pct": baseline.get("max_drawdown_pct"),
        "walkforward_overall_baseline_roi_pct": wf.get("overall_baseline_roi"),
        "walkforward_n_valid_folds": wf.get("n_valid_folds"),
        "blocked_rate": report.get("blocked_rate"),
        "oos_status": oos.get("status"),
        "oos_pass": oos.get("pass"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Confronto policy OFF vs policy ON soft.")
    parser.add_argument("--run-id-off", default=None, help="Run id baseline OFF (diff-only mode).")
    parser.add_argument("--run-id-soft", default=None, help="Run id policy SOFT (diff-only mode).")
    parser.add_argument("--backtest-args", default="", help="Argomenti extra passati a backtest.py.")
    parser.add_argument("--soft-env", default="", help="Override env addizionali per preset SOFT (k=v,k=v).")
    parser.add_argument("--off-env", default="", help="Override env addizionali per preset OFF (k=v,k=v).")
    parser.add_argument("--skip-runs", action="store_true", help="Non lanciare nuove run; confronta run-id forniti.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.skip_runs and (not args.run_id_off or not args.run_id_soft):
        parser.error("--skip-runs richiede --run-id-off e --run-id-soft.")

    if args.skip_runs:
        off_run_dir = _resolve_run_dir(args.run_id_off)
        soft_run_dir = _resolve_run_dir(args.run_id_soft)
    else:
        off_env = dict(OFF_POLICY_PRESET)
        off_env.update(_parse_env_pairs(args.off_env))
        soft_env = dict(SOFT_POLICY_PRESET)
        soft_env.update(_parse_env_pairs(args.soft_env))

        rc = _run_backtest(off_env, args.backtest_args)
        if rc != 0:
            print(f"[compare-policy] FAIL run OFF (rc={rc})")
            return rc
        off_run_id, off_run_dir = _resolve_run_dir_from_latest()
        print(f"[compare-policy] OFF run_id={off_run_id}")

        rc = _run_backtest(soft_env, args.backtest_args)
        if rc != 0:
            print(f"[compare-policy] FAIL run SOFT (rc={rc})")
            return rc
        soft_run_id, soft_run_dir = _resolve_run_dir_from_latest()
        print(f"[compare-policy] SOFT run_id={soft_run_id}")

    off_report = load_json(off_run_dir / "backtest_validation_report.json")
    soft_report = load_json(soft_run_dir / "backtest_validation_report.json")
    off_metrics = _extract_metrics(off_report)
    soft_metrics = _extract_metrics(soft_report)

    comparison = {
        "generated_at_utc": utc_now_iso(),
        "off_run_dir": str(off_run_dir),
        "soft_run_dir": str(soft_run_dir),
        "off": off_metrics,
        "soft": soft_metrics,
        "deltas": {
            "walkforward_overall_baseline_roi_pct_soft_minus_off": (
                (float(soft_metrics["walkforward_overall_baseline_roi_pct"]) - float(off_metrics["walkforward_overall_baseline_roi_pct"]))
                if off_metrics.get("walkforward_overall_baseline_roi_pct") is not None
                and soft_metrics.get("walkforward_overall_baseline_roi_pct") is not None
                else None
            ),
            "baseline_max_drawdown_pct_soft_minus_off": (
                (float(soft_metrics["baseline_max_drawdown_pct"]) - float(off_metrics["baseline_max_drawdown_pct"]))
                if off_metrics.get("baseline_max_drawdown_pct") is not None
                and soft_metrics.get("baseline_max_drawdown_pct") is not None
                else None
            ),
            "baseline_bets_soft_minus_off": (
                (int(soft_metrics["baseline_bets"]) - int(off_metrics["baseline_bets"]))
                if off_metrics.get("baseline_bets") is not None and soft_metrics.get("baseline_bets") is not None
                else None
            ),
            "blocked_rate_soft_minus_off": (
                (float(soft_metrics["blocked_rate"]) - float(off_metrics["blocked_rate"]))
                if off_metrics.get("blocked_rate") is not None and soft_metrics.get("blocked_rate") is not None
                else None
            ),
        },
    }

    table_rows = [
        {"metric": "run_id", "off": off_metrics.get("run_id"), "soft": soft_metrics.get("run_id")},
        {"metric": "walkforward_overall_baseline_roi_pct", "off": off_metrics.get("walkforward_overall_baseline_roi_pct"), "soft": soft_metrics.get("walkforward_overall_baseline_roi_pct")},
        {"metric": "baseline_max_drawdown_pct", "off": off_metrics.get("baseline_max_drawdown_pct"), "soft": soft_metrics.get("baseline_max_drawdown_pct")},
        {"metric": "baseline_bets", "off": off_metrics.get("baseline_bets"), "soft": soft_metrics.get("baseline_bets")},
        {"metric": "blocked_rate", "off": off_metrics.get("blocked_rate"), "soft": soft_metrics.get("blocked_rate")},
        {"metric": "oos_status", "off": off_metrics.get("oos_status"), "soft": soft_metrics.get("oos_status")},
        {"metric": "oos_pass", "off": off_metrics.get("oos_pass"), "soft": soft_metrics.get("oos_pass")},
    ]
    print(table_str(table_rows, ["metric", "off", "soft"]))

    out_name = f"policy_variant_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = RUNS_DIR / out_name
    save_json(out_path, comparison)
    print(f"[compare-policy] json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
