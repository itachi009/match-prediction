import copy
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backtest


def _snapshot_runtime():
    keys = [
        "MIN_BETS_FOR_TUNING",
        "OBJECTIVE_MODE",
        "LAMBDA_BETS",
        "MAX_BETS_INCREASE_PCT",
        "FALLBACK_EPS_ROI_PCT",
        "RESTRICTED_TUNING_MODE",
        "RESTRICTED_FIXED_MIN_EV",
        "RESTRICTED_FIXED_PROB_SHRINK",
        "RESTRICTED_MIN_EDGE_VALUES",
        "RESTRICTED_MIN_CONFIDENCE_VALUES",
    ]
    snap = {}
    for k in keys:
        v = getattr(backtest, k)
        snap[k] = list(v) if isinstance(v, list) else v
    return snap


def _restore_runtime(snapshot):
    for k, v in snapshot.items():
        setattr(backtest, k, v)


def _make_synthetic_side_df(n_rows=260, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="7D")

    base_prob = np.where(np.arange(n_rows) % 5 == 0, 0.78, 0.68).astype(float)
    noise = rng.normal(0.0, 0.015, size=n_rows)
    prob = np.clip(base_prob + noise, 0.52, 0.90)

    odd_p1 = np.where(np.arange(n_rows) % 4 == 0, 1.90, 2.05).astype(float)
    odd_p2 = np.where(np.arange(n_rows) % 4 == 0, 2.05, 1.90).astype(float)
    y = (rng.random(n_rows) < prob).astype(int)

    return pd.DataFrame(
        {
            "odds_row_id": np.arange(n_rows, dtype=int),
            "tourney_date": dates,
            "odd_p1": odd_p1,
            "odd_p2": odd_p2,
            "prob_p1_win": prob,
            "p1_is_real_winner": y,
        }
    )


def _base_cfg():
    cfg = copy.deepcopy(backtest.BASELINE_CONFIG)
    cfg.update(
        {
            "max_overround": 2.00,
            "min_odds": 1.40,
            "max_odds": 3.20,
            "edge_slope_by_odds": 0.00,
            "max_bet_share": 1.00,
            "min_signal_score": 0.0,
            "min_kelly_f": 0.0,
            "min_ev": 0.0,
        }
    )
    return cfg


def run_smoke():
    snapshot = _snapshot_runtime()
    try:
        df = _make_synthetic_side_df()
        df_train = df.iloc[:180].copy()

        # Scenario A: guardrail active -> tuned should fallback if it exceeds allowed bet count.
        backtest.RESTRICTED_TUNING_MODE = True
        backtest.RESTRICTED_FIXED_MIN_EV = 0.06
        backtest.RESTRICTED_FIXED_PROB_SHRINK = 0.60
        backtest.RESTRICTED_MIN_EDGE_VALUES = [0.06, 0.065]
        backtest.RESTRICTED_MIN_CONFIDENCE_VALUES = [0.64, 0.66]
        backtest.MIN_BETS_FOR_TUNING = 1
        backtest.MAX_BETS_INCREASE_PCT = 0.0
        backtest.LAMBDA_BETS = 0.0
        backtest.OBJECTIVE_MODE = "roi_minus_lambda_bets_ratio"

        cfg_guardrail = _base_cfg()
        cfg_guardrail.update({"min_confidence": 0.74, "min_edge": 0.07})
        _, _, _, _, diag_guardrail = backtest.tune_strategy_config(
            df_train,
            cfg_guardrail,
            calibrator=None,
            progress_label="smoke_guardrail",
            print_progress=False,
        )

        assert not bool(diag_guardrail["selected_is_tuned"]), (
            "Guardrail test failed: tuned policy should be rejected when it exceeds allowed bets."
        )
        assert "guardrail" in str(diag_guardrail["selection_reason"]), (
            "Guardrail test failed: selection reason should mention guardrail fallback."
        )

        # Scenario B: MIN_BETS_FOR_TUNING active -> no feasible tuned candidate, fallback to baseline.
        backtest.RESTRICTED_TUNING_MODE = False
        backtest.MIN_BETS_FOR_TUNING = 10000
        backtest.MAX_BETS_INCREASE_PCT = 1.0
        backtest.LAMBDA_BETS = 0.5
        backtest.OBJECTIVE_MODE = "roi_minus_lambda_bets_ratio"

        cfg_min_bets = _base_cfg()
        _, _, _, _, diag_min_bets = backtest.tune_strategy_config(
            df_train,
            cfg_min_bets,
            calibrator=None,
            progress_label="smoke_min_bets",
            print_progress=False,
        )

        assert int(diag_min_bets["num_configs_feasible"]) == 0, (
            "MIN_BETS test failed: expected zero feasible tuned configs."
        )
        assert not bool(diag_min_bets["selected_is_tuned"]), (
            "MIN_BETS test failed: expected baseline fallback when feasible set is empty."
        )

        print(
            {
                "guardrail_selection_reason": diag_guardrail["selection_reason"],
                "guardrail_selected_is_tuned": diag_guardrail["selected_is_tuned"],
                "min_bets_selection_reason": diag_min_bets["selection_reason"],
                "min_bets_selected_is_tuned": diag_min_bets["selected_is_tuned"],
            }
        )
        print("SMOKE_OK")
    finally:
        _restore_runtime(snapshot)


if __name__ == "__main__":
    run_smoke()
