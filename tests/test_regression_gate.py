import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from regression_gate import evaluate_gate  # noqa: E402


FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _load_fixture(name: str):
    with (FIXTURES / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_eval(current_name: str, golden_name: str = "golden_base.json"):
    current = _load_fixture(current_name)
    golden = _load_fixture(golden_name)
    return evaluate_gate(
        current_validation=current,
        current_walkforward={},
        current_stress={},
        golden_validation=golden,
        golden_walkforward={},
        golden_stress={},
        current_run_dir=Path("runs"),
        golden_dir=Path("golden"),
        tol_brier=0.002,
        tol_logloss=0.002,
        tol_ece=0.002,
        tol_bucket_gap=0.01,
        tol_bucket_brier=0.003,
        tol_bucket_logloss=0.003,
    )


def test_regression_gate_pass():
    result = _run_eval("current_pass.json")
    assert result["pass"] is True
    assert result["status"] == "PASS"
    assert result["hard_failures"] == []


def test_regression_gate_hard_fail_on_oos():
    result = _run_eval("current_fail_oos.json")
    assert result["pass"] is False
    assert any("oos_gate.pass" in item for item in result["hard_failures"])


def test_regression_gate_hard_fail_on_valid_folds():
    result = _run_eval("current_fail_folds.json")
    assert result["pass"] is False
    assert any("n_valid_folds" in item for item in result["hard_failures"])


def test_regression_gate_warning_on_brier_regression():
    result = _run_eval("current_warn_brier.json")
    assert result["pass"] is True
    assert result["status"] == "PASS_WITH_WARNINGS"
    assert any(w["key"] == "calibration.brier" for w in result["warnings"])
