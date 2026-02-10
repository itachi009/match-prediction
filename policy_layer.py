import json
import os

import numpy as np
import pandas as pd


DEFAULT_POLICY_LAYER_CONFIG = {
    "enabled": True,
    "max_uncertainty_for_bet": 0.55,
    "skip_odds_bucket_17_21": True,
    "max_uncertainty_for_17_21": 0.20,
    "min_confidence_for_17_21": 0.70,
    "max_uncertainty_for_p_70_75": 0.25,
    "policy_soft_mode": False,
    "p_bucket_penalty": 0.15,
    "respect_legacy_recommendation_flag": True,
}


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _as_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def _clip01(value):
    return float(np.clip(float(value), 0.0, 1.0))


def _resolve_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default


def load_policy_layer_config(path):
    cfg = dict(DEFAULT_POLICY_LAYER_CONFIG)
    try:
        if path is not None and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            if isinstance(payload, dict):
                for k in cfg:
                    if k in payload:
                        cfg[k] = payload[k]
    except Exception as e:
        print(f"[WARN] Impossibile leggere policy config {path}: {e}. Uso default.")

    env_overrides = {
        "MP_POLICY_ENABLED": ("enabled", _as_bool),
        "MP_POLICY_MAX_UNCERTAINTY_FOR_BET": ("max_uncertainty_for_bet", _as_float),
        "MP_POLICY_SKIP_ODDS_BUCKET_17_21": ("skip_odds_bucket_17_21", _as_bool),
        "MP_POLICY_MAX_UNCERTAINTY_FOR_17_21": ("max_uncertainty_for_17_21", _as_float),
        "MP_POLICY_MIN_CONFIDENCE_FOR_17_21": ("min_confidence_for_17_21", _as_float),
        "MP_POLICY_MAX_UNCERTAINTY_FOR_P_70_75": ("max_uncertainty_for_p_70_75", _as_float),
        "MP_POLICY_SOFT_MODE": ("policy_soft_mode", _as_bool),
        "MP_POLICY_P_BUCKET_PENALTY": ("p_bucket_penalty", _as_float),
    }
    # Backward-friendly aliases.
    alias_env = {
        "MP_MAX_UNCERTAINTY_FOR_BET": ("max_uncertainty_for_bet", _as_float),
        "MP_SKIP_ODDS_BUCKET_17_21": ("skip_odds_bucket_17_21", _as_bool),
        "MP_MAX_UNCERTAINTY_FOR_17_21": ("max_uncertainty_for_17_21", _as_float),
        "MP_MIN_CONFIDENCE_FOR_17_21": ("min_confidence_for_17_21", _as_float),
        "MP_MAX_UNCERTAINTY_FOR_P_70_75": ("max_uncertainty_for_p_70_75", _as_float),
        "MP_POLICY_SOFT_MODE": ("policy_soft_mode", _as_bool),
        "MP_P_BUCKET_PENALTY": ("p_bucket_penalty", _as_float),
    }

    for env_name, (cfg_name, caster) in {**env_overrides, **alias_env}.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        if caster is _as_bool:
            cfg[cfg_name] = caster(raw, cfg[cfg_name])
        else:
            cfg[cfg_name] = caster(raw, cfg[cfg_name])

    cfg["enabled"] = _as_bool(cfg.get("enabled"), True)
    cfg["skip_odds_bucket_17_21"] = _as_bool(cfg.get("skip_odds_bucket_17_21"), True)
    cfg["policy_soft_mode"] = _as_bool(cfg.get("policy_soft_mode"), False)
    cfg["respect_legacy_recommendation_flag"] = _as_bool(cfg.get("respect_legacy_recommendation_flag"), True)
    cfg["max_uncertainty_for_bet"] = _clip01(_as_float(cfg.get("max_uncertainty_for_bet"), 0.55))
    cfg["max_uncertainty_for_17_21"] = _clip01(_as_float(cfg.get("max_uncertainty_for_17_21"), 0.20))
    cfg["min_confidence_for_17_21"] = _clip01(_as_float(cfg.get("min_confidence_for_17_21"), 0.70))
    cfg["max_uncertainty_for_p_70_75"] = _clip01(_as_float(cfg.get("max_uncertainty_for_p_70_75"), 0.25))
    cfg["p_bucket_penalty"] = _clip01(_as_float(cfg.get("p_bucket_penalty"), 0.15))
    return cfg


def apply_policy_layer(df_preds, cfg):
    if df_preds is None:
        return pd.DataFrame(columns=["policy_allowed", "policy_reason", "effective_confidence"])

    work = df_preds.copy()
    if work.empty:
        work["policy_allowed"] = []
        work["policy_reason"] = []
        work["effective_confidence"] = []
        return work

    p_col = _resolve_col(work, ["p_calibrated", "p", "probability"])
    conf_col = _resolve_col(work, ["confidence", "effective_confidence", "p_calibrated", "p"])
    unc_col = _resolve_col(work, ["uncertainty_score"], default=None)
    odds_col = _resolve_col(work, ["odds", "odd_p1", "bookmaker_odds_p1"], default=None)
    reco_col = _resolve_col(work, ["recommendation_allowed"], default=None)

    if p_col is None or conf_col is None:
        raise ValueError("apply_policy_layer richiede almeno colonne p_calibrated/p e confidence.")

    work[p_col] = pd.to_numeric(work[p_col], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    work[conf_col] = pd.to_numeric(work[conf_col], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    if unc_col is None:
        work["policy_unc"] = 0.0
    else:
        work["policy_unc"] = pd.to_numeric(work[unc_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if odds_col is None:
        work["policy_odds"] = np.nan
    else:
        work["policy_odds"] = pd.to_numeric(work[odds_col], errors="coerce")
    if reco_col is None:
        work["policy_reco"] = True
    else:
        work["policy_reco"] = pd.to_numeric(work[reco_col], errors="coerce").fillna(1.0) > 0.5

    policy_allowed = []
    policy_reason = []
    effective_conf = []

    enabled = bool(cfg.get("enabled", True))
    soft_mode = bool(cfg.get("policy_soft_mode", False))
    skip_odds_17_21 = bool(cfg.get("skip_odds_bucket_17_21", True))
    max_unc = float(cfg.get("max_uncertainty_for_bet", 0.55))
    max_unc_17_21 = float(cfg.get("max_uncertainty_for_17_21", 0.20))
    min_conf_17_21 = float(cfg.get("min_confidence_for_17_21", 0.70))
    max_unc_p_70_75 = float(cfg.get("max_uncertainty_for_p_70_75", 0.25))
    p_bucket_penalty = float(cfg.get("p_bucket_penalty", 0.15))
    respect_legacy = bool(cfg.get("respect_legacy_recommendation_flag", True))

    for row in work.itertuples(index=False):
        p_val = float(getattr(row, p_col))
        conf_val = float(getattr(row, conf_col))
        unc_val = float(getattr(row, "policy_unc"))
        odds_val = getattr(row, "policy_odds")
        reco_val = bool(getattr(row, "policy_reco"))

        allowed = True
        reason = "allow:ok"
        eff = float(np.clip(conf_val, 0.0, 1.0))

        if soft_mode:
            eff = float(np.clip(eff * (1.0 - 0.10 * unc_val), 0.0, 1.0))
            reason = "soft:uncertainty_penalty"

        if not enabled:
            if respect_legacy and (not reco_val):
                allowed = False
                reason = "skip:legacy_recommendation_gate"
            else:
                reason = "allow:policy_disabled"
            policy_allowed.append(bool(allowed))
            policy_reason.append(reason)
            effective_conf.append(eff if soft_mode else conf_val)
            continue

        if respect_legacy and (not reco_val):
            allowed = False
            reason = "skip:legacy_recommendation_gate"
        elif unc_val > max_unc:
            allowed = False
            reason = "skip:high_uncertainty"

        in_odds_17_21 = pd.notna(odds_val) and float(odds_val) >= 1.7 and float(odds_val) < 2.1
        if allowed and in_odds_17_21:
            if skip_odds_17_21:
                allowed = False
                reason = "skip:odds_bucket_guard"
            else:
                if not (unc_val <= max_unc_17_21 and eff >= min_conf_17_21):
                    allowed = False
                    reason = "skip:odds_bucket_guard"
                elif soft_mode:
                    eff = float(np.clip(eff * (1.0 - 0.10), 0.0, 1.0))
                    reason = "soft:odds_bucket_guard"

        in_p_70_75 = p_val >= 0.70 and p_val < 0.75
        if allowed and in_p_70_75 and unc_val > max_unc_p_70_75:
            if soft_mode:
                eff = float(np.clip(eff * (1.0 - p_bucket_penalty), 0.0, 1.0))
                reason = "soft:p_bucket_prudence"
            else:
                allowed = False
                reason = "skip:p_bucket_prudence"

        if allowed and reason.startswith("allow:"):
            reason = "allow:ok"

        policy_allowed.append(bool(allowed))
        policy_reason.append(reason)
        effective_conf.append(float(np.clip(eff, 0.0, 1.0)))

    work["policy_allowed"] = policy_allowed
    work["policy_reason"] = policy_reason
    work["effective_confidence"] = effective_conf
    work = work.drop(columns=["policy_unc", "policy_odds", "policy_reco"], errors="ignore")
    return work
