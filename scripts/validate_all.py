import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from _validation_common import RUNS_DIR, extract_run_id, load_json, materialize_run_dir, resolve_run_source


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent


DEFAULT_SMOKE_ORDER = [
    "smoke_uncertainty_and_calib.py",
    "smoke_policy_layer.py",
    "smoke_policy_toggle.py",
]


def _run_command(cmd: List[str], cwd: Path) -> int:
    print(f"[validate-all] cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd))
    return int(result.returncode)


def _run_smokes(skip_smokes: bool) -> int:
    if skip_smokes:
        print("[validate-all] smoke tests skipped (--skip-smokes).")
        return 0
    for name in DEFAULT_SMOKE_ORDER:
        path = SCRIPTS_DIR / name
        if not path.exists():
            print(f"[validate-all] smoke non trovato, skip: {path.name}")
            continue
        rc = _run_command([sys.executable, str(path)], cwd=REPO_ROOT)
        if rc != 0:
            print(f"[validate-all] smoke FAIL: {path.name} (rc={rc})")
            return rc
    print("[validate-all] smoke tests PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation suite one-command: smoke + backtest + regression gate.")
    parser.add_argument("--golden", default=None, help="Path golden run (default: golden/latest).")
    parser.add_argument("--run-id", default=None, help="Run id target (utile con --no-backtest).")
    parser.add_argument("--no-backtest", action="store_true", help="Salta backtest e lancia solo regression gate.")
    parser.add_argument("--backtest-args", default="", help="Argomenti extra per backtest.py.")
    parser.add_argument("--skip-smokes", action="store_true", help="Salta smoke tests.")
    parser.add_argument("--tol-brier", type=float, default=0.002)
    parser.add_argument("--tol-logloss", type=float, default=0.002)
    parser.add_argument("--tol-ece", type=float, default=0.002)
    parser.add_argument("--tol-bucket-gap", type=float, default=0.01)
    parser.add_argument("--tol-bucket-brier", type=float, default=0.003)
    parser.add_argument("--tol-bucket-logloss", type=float, default=0.003)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    rc = _run_smokes(skip_smokes=args.skip_smokes)
    if rc != 0:
        print("[validate-all] RESULT: FAIL (smoke tests)")
        return rc

    if not args.no_backtest:
        backtest_cmd = [sys.executable, str(REPO_ROOT / "backtest.py")]
        if args.backtest_args.strip():
            backtest_cmd.extend(shlex.split(args.backtest_args.strip()))
        rc = _run_command(backtest_cmd, cwd=REPO_ROOT)
        if rc != 0:
            print("[validate-all] RESULT: FAIL (backtest)")
            return rc

    if not args.no_backtest:
        source_dir = RUNS_DIR
        validation_path = RUNS_DIR / "backtest_validation_report.json"
        if not validation_path.exists():
            print(f"[validate-all] missing report: {validation_path}")
            return 1
    else:
        source_dir, validation_path = resolve_run_source(
            run_id=args.run_id,
            run_path=None,
            runs_dir=RUNS_DIR,
        )
    validation_payload = load_json(validation_path)
    run_id = extract_run_id(validation_payload, source_dir)
    run_dir, copied = materialize_run_dir(
        run_id=run_id,
        source_dir=source_dir,
        validation_payload=validation_payload,
        runs_dir=RUNS_DIR,
    )
    print(f"[validate-all] run_id={run_id}")
    print(f"[validate-all] run_dir={run_dir}")
    print(f"[validate-all] files materialized={len(copied)}")

    regression_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "regression_gate.py"),
        "--run-id",
        run_id,
        "--tol-brier",
        str(args.tol_brier),
        "--tol-logloss",
        str(args.tol_logloss),
        "--tol-ece",
        str(args.tol_ece),
        "--tol-bucket-gap",
        str(args.tol_bucket_gap),
        "--tol-bucket-brier",
        str(args.tol_bucket_brier),
        "--tol-bucket-logloss",
        str(args.tol_bucket_logloss),
    ]
    if args.golden:
        regression_cmd.extend(["--golden", args.golden])
    rc = _run_command(regression_cmd, cwd=REPO_ROOT)

    gate_result_path = run_dir / "regression_gate_result.json"
    if gate_result_path.exists():
        gate_payload = load_json(gate_result_path)
        status = gate_payload.get("status", "UNKNOWN")
        hard_failures = gate_payload.get("hard_failures", []) or []
        warnings = gate_payload.get("warnings", []) or []
        print(f"[validate-all] gate_status={status}")
        if hard_failures:
            print("[validate-all] hard failures:")
            for item in hard_failures:
                print(f"- {item}")
        if warnings:
            print(f"[validate-all] warnings={len(warnings)}")

    final = "PASS" if rc == 0 else "FAIL"
    print(f"[validate-all] RESULT: {final}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
