import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _validation_common import (  # noqa: E402
    BASELINE_CONFIG_NAME,
    STRESS_REPORT_NAME,
    VALIDATION_REPORT_NAME,
    WALKFORWARD_REPORT_NAME,
    materialize_run_dir,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_materialize_run_dir_does_not_overwrite_historical_run(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    old_run_id = "bt_old"
    new_run_id = "bt_new"
    old_run_dir = runs_dir / old_run_id
    old_run_dir.mkdir(parents=True, exist_ok=True)

    root_validation = runs_dir / VALIDATION_REPORT_NAME
    root_walkforward = runs_dir / WALKFORWARD_REPORT_NAME
    root_stress = runs_dir / STRESS_REPORT_NAME
    root_baseline = runs_dir / BASELINE_CONFIG_NAME

    _write_json(root_validation, {"backtest_run_id": new_run_id, "marker": "NEW"})
    _write_json(root_walkforward, {"marker": "NEW_WF"})
    _write_json(root_stress, {"marker": "NEW_STRESS"})
    _write_json(root_baseline, {"marker": "NEW_CFG"})

    old_validation_payload = {
        "backtest_run_id": old_run_id,
        "marker": "OLD",
        "report_files": {
            "validation": str(root_validation.resolve()),
            "walkforward": str(root_walkforward.resolve()),
            "stress": str(root_stress.resolve()),
            "baseline_config": str(root_baseline.resolve()),
        },
    }
    _write_json(old_run_dir / VALIDATION_REPORT_NAME, old_validation_payload)
    _write_json(old_run_dir / WALKFORWARD_REPORT_NAME, {"marker": "OLD_WF"})
    _write_json(old_run_dir / STRESS_REPORT_NAME, {"marker": "OLD_STRESS"})
    _write_json(old_run_dir / BASELINE_CONFIG_NAME, {"marker": "OLD_CFG"})

    run_dir, _ = materialize_run_dir(
        run_id=old_run_id,
        source_dir=old_run_dir,
        validation_payload=old_validation_payload,
        runs_dir=runs_dir,
        repo_root=repo_root,
    )

    assert run_dir == old_run_dir.resolve()
    validation_after = json.loads((old_run_dir / VALIDATION_REPORT_NAME).read_text(encoding="utf-8"))
    assert validation_after["backtest_run_id"] == old_run_id
    assert validation_after["marker"] == "OLD"

    root_after = json.loads(root_validation.read_text(encoding="utf-8"))
    assert root_after["backtest_run_id"] == new_run_id
    assert root_after["marker"] == "NEW"


def test_materialize_run_dir_raises_on_run_id_mismatch(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    runs_dir = repo_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="Run id mismatch"):
        materialize_run_dir(
            run_id="bt_requested",
            source_dir=runs_dir,
            validation_payload={"backtest_run_id": "bt_payload"},
            runs_dir=runs_dir,
            repo_root=repo_root,
        )
