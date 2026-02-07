import backtest


# Fast tuning grid for Colab Free iterative runs.
backtest.TUNING_GRID = {
    "min_edge": [0.055, 0.06],
    "min_ev": [0.05, 0.06],
    "min_confidence": [0.62, 0.64],
    "prob_shrink": [0.62, 0.66],
    "kelly_fraction": [0.01, 0.015],
    "max_stake_pct": [0.01, 0.015],
    "max_bet_share": [0.08, 0.10],
    "min_kelly_f": [0.07, 0.10],
    "min_signal_score": [0.004, 0.005],
    "edge_slope_by_odds": [0.02],
    "residual_shrink_odds_2_5": [0.96],
    "residual_shrink_odds_3_0": [0.95],
}


if __name__ == "__main__":
    backtest.run_backtest_v7()
