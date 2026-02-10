import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import get_paths

LEGACY_POLICY_REASONS = {"skip:legacy_recommendation_gate"}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _load_validation_report() -> Tuple[Dict[str, Any], Path]:
    paths = get_paths()
    candidates = [
        paths["runs_dir"] / "backtest_validation_report.json",
        REPO_ROOT / "backtest_validation_report.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as fh:
                return json.load(fh), candidate
    raise FileNotFoundError("Impossibile trovare backtest_validation_report.json in runs/ o root.")


def _extract_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    baseline = report.get("baseline_result") or {}
    policy = baseline.get("policy") or {}
    oos_gate = report.get("oos_gate") or {}
    promotion = report.get("promotion_decision") or {}

    blocked_rate = report.get("blocked_rate")
    if blocked_rate is None:
        blocked_rate = policy.get("blocked_rate")

    blocked_counts = report.get("blocked_counts_by_reason")
    if not blocked_counts:
        blocked_counts = policy.get("blocked_counts_by_reason") or {}

    return {
        "backtest_run_id": report.get("backtest_run_id"),
        "active_model_run_id": report.get("active_model_run_id"),
        "policy_layer_enabled_reported": bool(report.get("policy_layer_enabled", True)),
        "roi_pct": _to_float(baseline.get("roi"), 0.0),
        "bets": _to_int(baseline.get("bets"), 0),
        "blocked_rate": _to_float(blocked_rate, 0.0),
        "blocked_counts_by_reason": blocked_counts,
        "oos_gate_status": str(oos_gate.get("status")),
        "oos_gate_pass": bool(oos_gate.get("pass", False)),
        "promotion_status": str(promotion.get("status")),
    }


def _run_backtest_with_policy(policy_enabled: bool) -> Dict[str, Any]:
    env = os.environ.copy()
    env["MP_POLICY_ENABLED"] = "true" if policy_enabled else "false"
    # Keep smoke runtime bounded while preserving ON/OFF comparability.
    env.setdefault("BT_RESTRICTED_TUNING_MODE", "true")
    env.setdefault("BT_FOLD_FREQ", "halfyear")
    label = "ON" if policy_enabled else "OFF"
    cmd = [sys.executable, str(REPO_ROOT / "backtest.py")]

    print(f"[smoke-policy-toggle] run policy={label}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

    report, report_path = _load_validation_report()
    metrics = _extract_metrics(report)
    metrics["report_path"] = str(report_path)
    metrics["policy_enabled_env"] = policy_enabled
    return metrics


def _split_blocked_reasons(blocked_counts: Any) -> Tuple[Dict[str, int], Dict[str, int]]:
    counts = blocked_counts if isinstance(blocked_counts, dict) else {}
    legacy: Dict[str, int] = {}
    policy_specific: Dict[str, int] = {}
    for key, val in counts.items():
        reason = str(key)
        count = _to_int(val, 0)
        if reason in LEGACY_POLICY_REASONS:
            legacy[reason] = count
        else:
            policy_specific[reason] = count
    return legacy, policy_specific


def main() -> None:
    paths = get_paths()
    runs_dir = paths["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    on_metrics = _run_backtest_with_policy(policy_enabled=True)
    off_metrics = _run_backtest_with_policy(policy_enabled=False)

    on_legacy, on_policy_specific = _split_blocked_reasons(on_metrics.get("blocked_counts_by_reason"))
    off_legacy, off_policy_specific = _split_blocked_reasons(off_metrics.get("blocked_counts_by_reason"))

    on_metrics["blocked_counts_legacy"] = on_legacy
    on_metrics["blocked_counts_policy_specific"] = on_policy_specific
    off_metrics["blocked_counts_legacy"] = off_legacy
    off_metrics["blocked_counts_policy_specific"] = off_policy_specific

    checks = {
        "policy_on_reported_enabled": bool(on_metrics.get("policy_layer_enabled_reported") is True),
        "policy_off_reported_disabled": bool(off_metrics.get("policy_layer_enabled_reported") is False),
        "policy_off_no_policy_specific_blocks": len(off_policy_specific) == 0,
        "policy_off_blocked_rate_lte_policy_on": _to_float(off_metrics.get("blocked_rate"))
        <= (_to_float(on_metrics.get("blocked_rate")) + 1e-12),
    }
    passed = all(checks.values())

    comparison = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenarios": {
            "policy_on": on_metrics,
            "policy_off": off_metrics,
        },
        "deltas": {
            "roi_pct_on_minus_off": _to_float(on_metrics.get("roi_pct")) - _to_float(off_metrics.get("roi_pct")),
            "bets_on_minus_off": _to_int(on_metrics.get("bets")) - _to_int(off_metrics.get("bets")),
            "blocked_rate_on_minus_off": _to_float(on_metrics.get("blocked_rate"))
            - _to_float(off_metrics.get("blocked_rate")),
        },
        "checks": checks,
        "pass": passed,
    }

    out_path = runs_dir / "policy_toggle_comparison.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, ensure_ascii=False, indent=2)

    print(f"[smoke-policy-toggle] output: {out_path}")
    print(
        "[smoke-policy-toggle] policy ON "
        f"roi={_to_float(on_metrics.get('roi_pct')):.4f}% "
        f"bets={_to_int(on_metrics.get('bets'))} "
        f"blocked_rate={_to_float(on_metrics.get('blocked_rate')) * 100:.2f}%"
    )
    print(
        "[smoke-policy-toggle] policy OFF "
        f"roi={_to_float(off_metrics.get('roi_pct')):.4f}% "
        f"bets={_to_int(off_metrics.get('bets'))} "
        f"blocked_rate={_to_float(off_metrics.get('blocked_rate')) * 100:.2f}%"
    )
    print(f"[smoke-policy-toggle] checks: {checks}")
    print(f"[smoke-policy-toggle] PASS={passed}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
