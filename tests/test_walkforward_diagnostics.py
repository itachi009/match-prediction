import shutil
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_walkforward_folds  # noqa: E402
from walkforward_coverage import suggest_threshold_pairs  # noqa: E402


FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _copy_fixture(src_name: str, dst: Path) -> None:
    shutil.copy2(FIXTURES / src_name, dst)


def test_analyze_walkforward_includes_all_wf_folds(tmp_path, monkeypatch):
    source_dir = tmp_path / "source_run"
    source_dir.mkdir(parents=True, exist_ok=True)

    _copy_fixture("wf_validation_minimal.json", source_dir / "backtest_validation_report.json")
    _copy_fixture("wf_report_two_folds.json", source_dir / "backtest_walkforward_report.json")
    _copy_fixture("wf_bet_records_two_folds.csv", source_dir / "backtest_walkforward_bet_records.csv")
    _copy_fixture("wf_policy_audit_two_folds.csv", source_dir / "backtest_walkforward_policy_audit.csv")
    _copy_fixture("wf_baseline_config_two_folds.json", source_dir / "backtest_baseline_config.json")

    test_runs_dir = tmp_path / "runs"
    test_runs_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(analyze_walkforward_folds, "RUNS_DIR", test_runs_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_walkforward_folds.py", "--run-path", str(source_dir)],
    )

    rc = analyze_walkforward_folds.main()
    assert rc == 0

    out_path = test_runs_dir / "bt_fixture_two_folds" / "analysis_walkforward.md"
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "## Fold Summary" in text
    assert "2024Q2" in text
    assert "2024Q3" in text
    assert "| 2024Q3 | 0 |" in text
    assert "Bet totali bassi" in text
    assert "## Data Coverage" in text


def test_suggest_threshold_pairs_returns_candidates():
    policy_df = pd.read_csv(FIXTURES / "wf_policy_audit_two_folds.csv")
    out = suggest_threshold_pairs(
        policy_df=policy_df,
        target_bets_per_fold=2,
        max_odds=3.0,
        top_k=3,
        fold_universe=["2024Q2", "2024Q3"],
    )
    assert not out.empty
    assert len(out) <= 3
    assert {"min_confidence", "min_odds", "avg_candidates_per_fold", "target_gap"}.issubset(out.columns)
