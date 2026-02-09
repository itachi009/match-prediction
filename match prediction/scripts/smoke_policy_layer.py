import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import get_paths
from policy_layer import apply_policy_layer, load_policy_layer_config


def main():
    paths = get_paths()
    runs_dir = paths["runs_dir"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_policy_layer_config(REPO_ROOT / "configs" / "policy_layer.json")

    # Small deterministic sample useful to validate all policy branches.
    df = pd.DataFrame(
        {
            "p_calibrated": [0.64, 0.72, 0.73, 0.78, 0.68, 0.74, 0.71, 0.66],
            "confidence": [0.62, 0.70, 0.68, 0.77, 0.66, 0.71, 0.69, 0.63],
            "uncertainty_score": [0.10, 0.15, 0.32, 0.20, 0.58, 0.12, 0.27, 0.18],
            "odds": [1.65, 1.85, 1.95, 2.30, 1.90, 2.80, 1.75, 2.05],
            "level": ["A", "A", "A", "C", "A", "F", "A", "A"],
            "recommendation_allowed": [True, True, True, True, True, True, True, True],
        }
    )

    out = apply_policy_layer(df, cfg)

    total = int(len(out))
    allowed = int(out["policy_allowed"].astype(bool).sum())
    blocked = int(total - allowed)
    blocked_by_reason = (
        out.loc[~out["policy_allowed"], "policy_reason"].value_counts(dropna=False).to_dict()
        if blocked > 0
        else {}
    )

    preview = out[
        [
            "p_calibrated",
            "odds",
            "uncertainty_score",
            "policy_allowed",
            "policy_reason",
            "effective_confidence",
        ]
    ].copy()
    preview = preview.rename(columns={"p_calibrated": "p"})
    out_path = runs_dir / "policy_layer_preview.csv"
    preview.to_csv(out_path, index=False)

    print(f"total candidates: {total}")
    print(f"allowed: {allowed}")
    print(f"blocked: {blocked}")
    print(f"blocked by reason: {blocked_by_reason}")
    print(f"preview file: {out_path}")
    print(preview.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
