import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _validation_common import (
    RUNS_DIR,
    extract_run_id,
    load_json,
    materialize_run_dir,
    resolve_golden_dir,
    resolve_run_source,
    save_json,
    table_str,
    utc_now_iso,
)


def _nested_get(payload: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = payload
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _first_float(payloads: List[Dict[str, Any]], paths: List[List[str]]) -> Optional[float]:
    for payload in payloads:
        for p in paths:
            v = _as_float(_nested_get(payload, p))
            if v is not None:
                return v
    return None


def _first_int(payloads: List[Dict[str, Any]], paths: List[List[str]]) -> Optional[int]:
    for payload in payloads:
        for p in paths:
            v = _as_int(_nested_get(payload, p))
            if v is not None:
                return v
    return None


def _load_bucket_summary(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return None

    w_sum = 0.0
    brier_sum = 0.0
    logloss_sum = 0.0
    abs_gap_sum = 0.0
    brier_n = 0.0
    logloss_n = 0.0
    gap_n = 0.0

    for row in rows:
        w = _as_float(row.get("n_matches"))
        weight = w if w is not None and w > 0 else 1.0
        w_sum += weight

        brier = _as_float(row.get("brier"))
        if brier is not None:
            brier_sum += brier * weight
            brier_n += weight

        logloss = _as_float(row.get("logloss"))
        if logloss is not None:
            logloss_sum += logloss * weight
            logloss_n += weight

        gap = _as_float(row.get("calibration_gap"))
        if gap is not None:
            abs_gap_sum += abs(gap) * weight
            gap_n += weight

    return {
        "rows": float(len(rows)),
        "weighted_brier": (brier_sum / brier_n) if brier_n > 0 else float("nan"),
        "weighted_logloss": (logloss_sum / logloss_n) if logloss_n > 0 else float("nan"),
        "weighted_abs_calibration_gap": (abs_gap_sum / gap_n) if gap_n > 0 else float("nan"),
        "weight_sum": w_sum,
    }


def _maybe_add_warn(
    warnings: List[Dict[str, Any]],
    key: str,
    current_value: Optional[float],
    golden_value: Optional[float],
    tolerance: float,
    higher_is_worse: bool = True,
) -> None:
    if current_value is None or golden_value is None:
        return
    delta = current_value - golden_value
    if not higher_is_worse:
        delta = -delta
    if delta > tolerance:
        warnings.append(
            {
                "key": key,
                "status": "WARN",
                "current": current_value,
                "golden": golden_value,
                "delta": (current_value - golden_value),
                "tolerance": tolerance,
            }
        )


def evaluate_gate(
    current_validation: Dict[str, Any],
    current_walkforward: Dict[str, Any],
    current_stress: Dict[str, Any],
    golden_validation: Dict[str, Any],
    golden_walkforward: Dict[str, Any],
    golden_stress: Dict[str, Any],
    current_run_dir: Path,
    golden_dir: Path,
    tol_brier: float,
    tol_logloss: float,
    tol_ece: float,
    tol_bucket_gap: float,
    tol_bucket_brier: float,
    tol_bucket_logloss: float,
) -> Dict[str, Any]:
    current_payloads = [current_validation, current_walkforward, current_stress]
    golden_payloads = [golden_validation, golden_walkforward, golden_stress]

    rows: List[Dict[str, Any]] = []
    hard_failures: List[str] = []
    warnings: List[Dict[str, Any]] = []

    oos_pass = _nested_get(current_validation, ["oos_gate", "pass"])
    if oos_pass is not None:
        ok = bool(oos_pass) is True
        rows.append(
            {
                "check": "oos_gate.pass",
                "status": "PASS" if ok else "FAIL",
                "current": oos_pass,
                "rule": "must be True",
            }
        )
        if not ok:
            hard_failures.append("oos_gate.pass != True")

    coherence_pass = _nested_get(current_validation, ["report_coherence", "pass"])
    if coherence_pass is not None:
        ok = bool(coherence_pass) is True
        rows.append(
            {
                "check": "report_coherence.pass",
                "status": "PASS" if ok else "FAIL",
                "current": coherence_pass,
                "rule": "must be True",
            }
        )
        if not ok:
            hard_failures.append("report_coherence.pass != True")

    max_drawdown = _first_float(
        current_payloads,
        [
            ["baseline_result", "max_drawdown_pct"],
            ["scorecard_row", "baseline_max_drawdown_pct"],
            ["promotion_decision", "criteria_snapshot", "max_drawdown_pct"],
        ],
    )
    if max_drawdown is not None:
        ok = max_drawdown <= 5.0
        rows.append(
            {
                "check": "max_drawdown_pct",
                "status": "PASS" if ok else "FAIL",
                "current": round(max_drawdown, 6),
                "rule": "<= 5.0",
            }
        )
        if not ok:
            hard_failures.append(f"max_drawdown_pct={max_drawdown:.6f} > 5")

    n_valid_folds = _first_int(
        current_payloads,
        [
            ["walk_forward", "n_valid_folds"],
            ["n_valid_folds"],
        ],
    )
    if n_valid_folds is not None:
        ok = n_valid_folds >= 3
        rows.append(
            {
                "check": "n_valid_folds",
                "status": "PASS" if ok else "FAIL",
                "current": n_valid_folds,
                "rule": ">= 3",
            }
        )
        if not ok:
            hard_failures.append(f"n_valid_folds={n_valid_folds} < 3")

    wf_baseline_roi = _first_float(
        current_payloads,
        [
            ["walk_forward", "overall_baseline_roi"],
            ["scorecard_row", "walkforward_baseline_roi_pct"],
            ["overall_baseline_roi"],
        ],
    )
    if wf_baseline_roi is not None:
        ok = wf_baseline_roi >= 0.0
        rows.append(
            {
                "check": "walkforward_overall_baseline_roi",
                "status": "PASS" if ok else "FAIL",
                "current": round(wf_baseline_roi, 6),
                "rule": ">= 0",
            }
        )
        if not ok:
            hard_failures.append(f"walkforward_overall_baseline_roi={wf_baseline_roi:.6f} < 0")

    curr_brier = _first_float(
        current_payloads,
        [
            ["calibration", "brier_calibrated_shrunk"],
            ["calibration", "brier_calibrated"],
        ],
    )
    gold_brier = _first_float(
        golden_payloads,
        [
            ["calibration", "brier_calibrated_shrunk"],
            ["calibration", "brier_calibrated"],
        ],
    )
    _maybe_add_warn(warnings, "calibration.brier", curr_brier, gold_brier, tol_brier, higher_is_worse=True)

    curr_ece = _first_float(current_payloads, [["calibration_metrics", "ece_calibrated"]])
    gold_ece = _first_float(golden_payloads, [["calibration_metrics", "ece_calibrated"]])
    _maybe_add_warn(warnings, "calibration.ece", curr_ece, gold_ece, tol_ece, higher_is_worse=True)

    curr_logloss = _first_float(current_payloads, [["no_odds_eval", "combined", "logloss"]])
    gold_logloss = _first_float(golden_payloads, [["no_odds_eval", "combined", "logloss"]])
    _maybe_add_warn(
        warnings,
        "no_odds_eval.combined.logloss",
        curr_logloss,
        gold_logloss,
        tol_logloss,
        higher_is_worse=True,
    )

    bucket_files = [
        "reliability_by_p_bucket.csv",
        "reliability_by_uncertainty_bucket.csv",
        "reliability_by_odds_bucket.csv",
    ]
    for name in bucket_files:
        curr_path = current_run_dir / name
        gold_path = golden_dir / name
        curr_summary = _load_bucket_summary(curr_path)
        gold_summary = _load_bucket_summary(gold_path)
        if curr_summary is None or gold_summary is None:
            continue
        _maybe_add_warn(
            warnings,
            f"{name}:weighted_abs_calibration_gap",
            _as_float(curr_summary.get("weighted_abs_calibration_gap")),
            _as_float(gold_summary.get("weighted_abs_calibration_gap")),
            tol_bucket_gap,
            higher_is_worse=True,
        )
        _maybe_add_warn(
            warnings,
            f"{name}:weighted_brier",
            _as_float(curr_summary.get("weighted_brier")),
            _as_float(gold_summary.get("weighted_brier")),
            tol_bucket_brier,
            higher_is_worse=True,
        )
        _maybe_add_warn(
            warnings,
            f"{name}:weighted_logloss",
            _as_float(curr_summary.get("weighted_logloss")),
            _as_float(gold_summary.get("weighted_logloss")),
            tol_bucket_logloss,
            higher_is_worse=True,
        )

    for warn in warnings:
        rows.append(
            {
                "check": warn["key"],
                "status": "WARN",
                "current": round(float(warn["current"]), 6),
                "rule": f"delta <= {warn['tolerance']}",
            }
        )

    status = "FAIL" if hard_failures else ("PASS_WITH_WARNINGS" if warnings else "PASS")
    return {
        "status": status,
        "pass": len(hard_failures) == 0,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "checks": rows,
    }


def _load_run_reports(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    validation = load_json(run_dir / "backtest_validation_report.json")
    walkforward_path = run_dir / "backtest_walkforward_report.json"
    stress_path = run_dir / "backtest_stress_report.json"
    walkforward = load_json(walkforward_path) if walkforward_path.exists() else {}
    stress = load_json(stress_path) if stress_path.exists() else {}
    return validation, walkforward, stress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regression gate tra run corrente e golden run.")
    parser.add_argument("--run-path", default=None, help="Percorso run corrente.")
    parser.add_argument("--run-id", default=None, help="Run id corrente.")
    parser.add_argument("--golden", default=None, help="Percorso golden run (default: golden/latest o piu recente).")
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

    source_dir, validation_path = resolve_run_source(run_path=args.run_path, run_id=args.run_id, runs_dir=RUNS_DIR)
    validation_payload = load_json(validation_path)
    run_id = extract_run_id(validation_payload, source_dir)
    current_run_dir, copied = materialize_run_dir(
        run_id=run_id,
        source_dir=source_dir,
        validation_payload=validation_payload,
        runs_dir=RUNS_DIR,
    )

    golden_dir = resolve_golden_dir(args.golden)
    if not (golden_dir / "backtest_validation_report.json").exists():
        raise FileNotFoundError(f"Golden run invalida: manca backtest_validation_report.json in {golden_dir}")

    curr_validation, curr_wf, curr_stress = _load_run_reports(current_run_dir)
    gold_validation, gold_wf, gold_stress = _load_run_reports(golden_dir)

    result = evaluate_gate(
        current_validation=curr_validation,
        current_walkforward=curr_wf,
        current_stress=curr_stress,
        golden_validation=gold_validation,
        golden_walkforward=gold_wf,
        golden_stress=gold_stress,
        current_run_dir=current_run_dir,
        golden_dir=golden_dir,
        tol_brier=args.tol_brier,
        tol_logloss=args.tol_logloss,
        tol_ece=args.tol_ece,
        tol_bucket_gap=args.tol_bucket_gap,
        tol_bucket_brier=args.tol_bucket_brier,
        tol_bucket_logloss=args.tol_bucket_logloss,
    )
    result["run_id"] = run_id
    result["run_dir"] = str(current_run_dir)
    result["golden_dir"] = str(golden_dir)
    result["generated_at_utc"] = utc_now_iso()
    result["materialized_files"] = copied

    out_path = current_run_dir / "regression_gate_result.json"
    save_json(out_path, result)

    print(f"[regression-gate] run_id={run_id}")
    print(f"[regression-gate] current_run_dir={current_run_dir}")
    print(f"[regression-gate] golden_dir={golden_dir}")
    print(f"[regression-gate] result={result['status']}")
    print(table_str(result["checks"], ["check", "status", "current", "rule"]))
    if result["hard_failures"]:
        print("[regression-gate] hard failures:")
        for item in result["hard_failures"]:
            print(f"- {item}")
    if result["warnings"]:
        print("[regression-gate] warnings:")
        for item in result["warnings"]:
            print(
                f"- {item['key']} delta={item['delta']:.6f} "
                f"(current={item['current']:.6f}, golden={item['golden']:.6f}, tol={item['tolerance']})"
            )
    print(f"[regression-gate] json={out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
