import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    import optuna
except Exception:
    optuna = None


INPUT_FEATURES_FILE = "processed_features.csv"
INPUT_FEATURES_PARQUET = "processed_features.parquet"
ACTIVE_MODEL_FILE = "active_model.json"
BENCHMARK_CONFIG_FILE = "configs/model_benchmark.json"

TRAIN_START_YEAR = 2018
TRAIN_END_YEAR = 2023
TEST_START_YEAR = 2024
CALIBRATION_SPLITS = 4


def infer_date_column(df):
    if "date" in df.columns:
        return "date"
    if "tourney_date" in df.columns:
        return "tourney_date"
    raise ValueError("Nessuna colonna data trovata: atteso 'date' o 'tourney_date'.")


def prepare_feature_columns(df, date_col):
    drop_cols = {"match_id", date_col, "p1_name", "p2_name", "target", "match_level"}
    cols = [c for c in df.columns if c not in drop_cols]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols


def build_inverse_rows(df):
    inv = df.copy()

    if "p1_name" in inv.columns and "p2_name" in inv.columns:
        inv[["p1_name", "p2_name"]] = inv[["p2_name", "p1_name"]]
    if "target" in inv.columns:
        inv["target"] = 1 - inv["target"]

    p1_cols = [c for c in inv.columns if c.startswith("p1_")]
    for p1_col in p1_cols:
        p2_col = "p2_" + p1_col[3:]
        if p2_col in inv.columns:
            tmp = inv[p1_col].copy()
            inv[p1_col] = inv[p2_col]
            inv[p2_col] = tmp

    diff_like_cols = []
    for c in inv.columns:
        if c.endswith("_diff") or "_diff_" in c:
            diff_like_cols.append(c)
    for explicit in ["rank_diff", "log_rank_diff", "h2h_diff"]:
        if explicit in inv.columns and explicit not in diff_like_cols:
            diff_like_cols.append(explicit)

    for c in diff_like_cols:
        if pd.api.types.is_numeric_dtype(inv[c]):
            inv[c] = -inv[c]

    return inv


def augment_symmetric(df):
    doubled = pd.concat([df, build_inverse_rows(df)], ignore_index=True)
    return doubled.drop_duplicates()


def compute_ece_mce(prob, y, bins=10):
    if len(prob) == 0:
        return np.nan, np.nan
    p = np.asarray(prob, dtype=float).clip(0.0, 1.0)
    t = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=False)

    ece = 0.0
    mce = 0.0
    n_total = len(p)
    for b in range(bins):
        mask = bin_ids == b
        n = int(mask.sum())
        if n == 0:
            continue
        avg_p = float(p[mask].mean())
        avg_t = float(t[mask].mean())
        gap = abs(avg_p - avg_t)
        ece += (n / n_total) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def compute_metrics(y_true, y_prob):
    y_prob = np.asarray(y_prob, dtype=float).clip(1e-9, 1.0 - 1e-9)
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = np.asarray(y_true, dtype=int)
    ece, mce = compute_ece_mce(y_prob, y_true, bins=10)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob)),
        "ece": float(ece),
        "mce": float(mce),
    }
    score_model = -metrics["logloss"] + 0.30 * metrics["auc"] - 0.20 * metrics["ece"] - 0.10 * metrics["mce"]
    metrics["score_model"] = float(score_model)
    return metrics


def model_family_available(model_family):
    if model_family == "xgb":
        return xgb is not None
    if model_family == "lgbm":
        return lgb is not None
    if model_family == "catboost":
        return CatBoostClassifier is not None
    if model_family == "logreg":
        return True
    return False


def build_default_params(model_family, seed):
    if model_family == "xgb":
        return {
            "n_estimators": 1400,
            "learning_rate": 0.02,
            "max_depth": 6,
            "min_child_weight": 2,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": seed,
        }
    if model_family == "lgbm":
        return {
            "n_estimators": 1600,
            "learning_rate": 0.02,
            "max_depth": -1,
            "num_leaves": 63,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "objective": "binary",
            "n_jobs": -1,
            "random_state": seed,
        }
    if model_family == "catboost":
        return {
            "iterations": 1200,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
        }
    if model_family == "logreg":
        return {
            "C": 1.0,
            "max_iter": 2000,
            "solver": "lbfgs",
            "random_state": seed,
        }
    raise ValueError(f"Model family non supportata: {model_family}")


def build_model(model_family, seed, params=None):
    cfg = build_default_params(model_family, seed)
    if params:
        cfg.update(params)
    if model_family == "xgb":
        return xgb.XGBClassifier(**cfg), cfg
    if model_family == "lgbm":
        return lgb.LGBMClassifier(**cfg), cfg
    if model_family == "catboost":
        return CatBoostClassifier(**cfg), cfg
    if model_family == "logreg":
        return LogisticRegression(**cfg), cfg
    raise ValueError(f"Model family non supportata: {model_family}")


def fit_base_model(model_family, model, X_train, y_train):
    n = len(X_train)
    if model_family in {"xgb", "lgbm"} and n >= 1200:
        split = int(n * 0.85)
        X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
        y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]
        if len(X_val) >= 200 and len(np.unique(y_val)) > 1:
            try:
                if model_family == "xgb":
                    model.set_params(early_stopping_rounds=100)
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                elif model_family == "lgbm":
                    callbacks = [lgb.early_stopping(100, verbose=False)] if lgb is not None else None
                    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=callbacks)
                return model, {"early_stopping_used": True, "train_rows_fit": int(len(X_tr)), "val_rows_fit": int(len(X_val))}
            except Exception:
                pass
    model.fit(X_train, y_train)
    return model, {"early_stopping_used": False, "train_rows_fit": int(len(X_train)), "val_rows_fit": 0}


def fit_model(model_family, calibration, X_train, y_train, seed, params=None):
    base_model, used_params = build_model(model_family, seed=seed, params=params)

    if calibration == "none":
        fitted_model, fit_details = fit_base_model(model_family, base_model, X_train, y_train)
        details = {
            "model_family": model_family,
            "calibration": calibration,
            "fit_mode": "direct",
            "fit_details": fit_details,
            "params": used_params,
        }
        return fitted_model, details

    cv = TimeSeriesSplit(n_splits=CALIBRATION_SPLITS)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method=calibration,
        cv=cv,
    )
    calibrated_model.fit(X_train, y_train)
    details = {
        "model_family": model_family,
        "calibration": calibration,
        "fit_mode": "calibrated_cv",
        "fit_details": {"cv_splits": CALIBRATION_SPLITS},
        "params": used_params,
    }
    return calibrated_model, details


def extract_importance(model_family, X_train, y_train, feature_cols, seed, params):
    try:
        imp_model, _ = build_model(model_family, seed=seed, params=params)
        imp_model, _ = fit_base_model(model_family, imp_model, X_train, y_train)
    except Exception:
        return []

    if hasattr(imp_model, "feature_importances_"):
        imps = np.asarray(imp_model.feature_importances_, dtype=float)
        feat_imp = pd.DataFrame({"feature": feature_cols, "importance": imps}).sort_values("importance", ascending=False)
        return feat_imp.head(20).to_dict(orient="records")

    if hasattr(imp_model, "coef_"):
        coef = np.asarray(imp_model.coef_).reshape(-1)
        feat_imp = pd.DataFrame({"feature": feature_cols, "importance": np.abs(coef)}).sort_values("importance", ascending=False)
        return feat_imp.head(20).to_dict(orient="records")

    return []


def load_features_frame():
    if os.path.exists(INPUT_FEATURES_PARQUET):
        try:
            print(f"[INFO] Loading parquet: {INPUT_FEATURES_PARQUET}")
            return pd.read_parquet(INPUT_FEATURES_PARQUET)
        except Exception as e:
            print(f"[WARN] Failed parquet load ({e}), fallback CSV.")
    print(f"[INFO] Loading CSV: {INPUT_FEATURES_FILE}")
    return pd.read_csv(INPUT_FEATURES_FILE, low_memory=False)


def prepare_dataset_cache():
    df = load_features_frame()
    date_col = infer_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, "target"]).sort_values(date_col).reset_index(drop=True)

    train_mask = (df[date_col].dt.year >= TRAIN_START_YEAR) & (df[date_col].dt.year <= TRAIN_END_YEAR)
    test_mask = df[date_col].dt.year >= TEST_START_YEAR
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Split train/test vuoto. Controlla anni e dati disponibili.")

    feature_cols = prepare_feature_columns(df, date_col)
    if not feature_cols:
        raise ValueError("Nessuna feature numerica disponibile per il training.")

    v12_tokens = [
        "days_since_last",
        "inactive_bucket",
        "matches_last_30d",
        "long_stop_90d",
        "no_prev_match",
        "upset_win_rate_12",
        "bad_loss_rate_12",
        "confidence_rank_score_12",
        "confidence_n_valid_12",
        "is_level_",
        "_diff_cf",
    ]
    v12_features = [c for c in feature_cols if any(t in c for t in v12_tokens)]

    train_aug = augment_symmetric(train_df)
    X_train = train_aug[feature_cols].fillna(0.0).astype(np.float32)
    y_train = train_aug["target"].astype(int)
    X_test = test_df[feature_cols].fillna(0.0).astype(np.float32)
    y_test = test_df["target"].astype(int)

    return {
        "date_col": date_col,
        "feature_cols": feature_cols,
        "v12_features": v12_features,
        "train_df": train_df,
        "train_aug": train_aug,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def ensure_run_dir(output_dir, run_id):
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_active_model(run_id, model_family, model_path, features_path, metadata_path):
    payload = {
        "run_id": run_id,
        "model_family": model_family,
        "model_path": str(model_path).replace("\\", "/"),
        "features_path": str(features_path).replace("\\", "/"),
        "metadata_path": str(metadata_path).replace("\\", "/"),
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    with open(ACTIVE_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Active model updated: {ACTIVE_MODEL_FILE}")


def format_run_id(model_family, calibration, suffix=None):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{ts}_{model_family}_{calibration}"
    if suffix:
        return f"{base}_{suffix}"
    return base


def train_once(model_family, calibration, seed, output_dir, set_active=True, dataset_cache=None, params=None, run_suffix=None):
    if not model_family_available(model_family):
        raise RuntimeError(f"Model family '{model_family}' non disponibile nell'ambiente corrente.")

    ds = dataset_cache or prepare_dataset_cache()
    X_train = ds["X_train"]
    y_train = ds["y_train"]
    X_test = ds["X_test"]
    y_test = ds["y_test"]
    feature_cols = ds["feature_cols"]

    model, fit_info = fit_model(model_family, calibration, X_train, y_train, seed=seed, params=params)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_prob)

    print("\n--- Validation (2024) ---")
    print(f"Model:    {model_family} | Calibration: {calibration}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC:      {metrics['auc']:.4f}")
    print(f"Brier:    {metrics['brier']:.4f} (piu basso e meglio)")
    print(f"LogLoss:  {metrics['logloss']:.4f} (piu basso e meglio)")
    print(f"ECE:      {metrics['ece']:.4f} (piu basso e meglio)")
    print(f"MCE:      {metrics['mce']:.4f} (piu basso e meglio)")
    print(f"Score:    {metrics['score_model']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    run_id = format_run_id(model_family, calibration, suffix=run_suffix)
    run_dir = ensure_run_dir(output_dir, run_id)
    model_path = run_dir / "model.pkl"
    features_path = run_dir / "model_features.pkl"
    report_path = run_dir / "training_report.json"

    joblib.dump(model, model_path)
    joblib.dump(feature_cols, features_path)

    top_features = extract_importance(model_family, X_train, y_train, feature_cols, seed=seed, params=fit_info.get("params"))
    report = {
        "run_id": run_id,
        "model_family": model_family,
        "calibration_method": calibration,
        "input_file": INPUT_FEATURES_PARQUET if os.path.exists(INPUT_FEATURES_PARQUET) else INPUT_FEATURES_FILE,
        "train_rows_original": int(len(ds["train_df"])),
        "train_rows_augmented": int(len(ds["train_aug"])),
        "test_rows": int(len(ds["test_df"])),
        "feature_count": int(len(feature_cols)),
        "v12_feature_count": int(len(ds["v12_features"])),
        "v12_feature_sample": ds["v12_features"][:30],
        "metrics_2024": metrics,
        "fit_info": fit_info,
        "top_features": top_features,
        "model_path": str(model_path).replace("\\", "/"),
        "features_path": str(features_path).replace("\\", "/"),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if set_active:
        write_active_model(
            run_id=run_id,
            model_family=model_family,
            model_path=model_path,
            features_path=features_path,
            metadata_path=report_path,
        )

    print(f"\n[OK] Model saved: {model_path}")
    print(f"[OK] Features saved: {features_path}")
    print(f"[OK] Report saved: {report_path}")
    return report


def evaluate_candidate(model_family, calibration, seed, dataset_cache, params=None):
    if not model_family_available(model_family):
        raise RuntimeError(f"Model family '{model_family}' non disponibile nell'ambiente corrente.")
    ds = dataset_cache
    model, fit_info = fit_model(
        model_family=model_family,
        calibration=calibration,
        X_train=ds["X_train"],
        y_train=ds["y_train"],
        seed=seed,
        params=params,
    )
    y_prob = model.predict_proba(ds["X_test"])[:, 1]
    metrics = compute_metrics(ds["y_test"], y_prob)
    return {
        "metrics": metrics,
        "fit_info": fit_info,
    }


def suggest_params(model_family, trial):
    if model_family == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 700, 1800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
    if model_family == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 700, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127, step=8),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120, step=10),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
    if model_family == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 600, 1800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
        }
    if model_family == "logreg":
        return {
            "C": trial.suggest_float("C", 0.1, 5.0, log=True),
        }
    return {}


def run_backtest_if_requested(run_backtest_top):
    if not run_backtest_top:
        return {"enabled": False}
    cmd = ["python", "backtest.py"]
    print(f"[INFO] Running backtest for active model: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "enabled": True,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout.splitlines()[-20:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
    }


def run_benchmark(args):
    if not os.path.exists(args.benchmark_config):
        raise FileNotFoundError(f"Benchmark config non trovato: {args.benchmark_config}")
    with open(args.benchmark_config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dataset_cache = prepare_dataset_cache()
    families_cfg = (cfg.get("families") or {})
    top_k = int(((cfg.get("selection") or {}).get("top_k_per_family", 2)))
    benchmark_rows = []
    skipped_families = {}

    for family, fcfg in families_cfg.items():
        if not bool((fcfg or {}).get("enabled", True)):
            continue
        if not model_family_available(family):
            skipped_families[family] = "library_not_available"
            continue

        n_trials = int((fcfg or {}).get("trials", 1))
        if n_trials <= 1 or optuna is None or family == "logreg":
            print(f"\n[Benchmark] Family={family} | fallback single run")
            single = train_once(
                model_family=family,
                calibration=args.calibration,
                seed=args.seed,
                output_dir=args.output_dir,
                set_active=False,
                dataset_cache=dataset_cache,
                params=None,
                run_suffix="benchmark_default",
            )
            benchmark_rows.append(
                {
                    "family": family,
                    "trial": 0,
                    "score_model": single["metrics_2024"]["score_model"],
                    "auc": single["metrics_2024"]["auc"],
                    "logloss": single["metrics_2024"]["logloss"],
                    "ece": single["metrics_2024"]["ece"],
                    "mce": single["metrics_2024"]["mce"],
                    "params": single["fit_info"]["params"],
                    "run_id": single["run_id"],
                }
            )
            continue

        print(f"\n[Benchmark] Family={family} | optuna trials={n_trials}")
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            params = suggest_params(family, trial)
            eval_result = evaluate_candidate(
                model_family=family,
                calibration=args.calibration,
                seed=args.seed,
                dataset_cache=dataset_cache,
                params=params,
            )
            m = eval_result["metrics"]
            trial.set_user_attr("metrics", m)
            trial.set_user_attr("params", eval_result["fit_info"]["params"])
            return float(m["score_model"])

        study.optimize(objective, n_trials=n_trials)
        completed = [t for t in study.trials if t.value is not None]
        completed = sorted(completed, key=lambda t: t.value, reverse=True)
        for t in completed:
            m = t.user_attrs.get("metrics", {})
            benchmark_rows.append(
                {
                    "family": family,
                    "trial": int(t.number),
                    "score_model": float(t.value),
                    "auc": m.get("auc"),
                    "logloss": m.get("logloss"),
                    "ece": m.get("ece"),
                    "mce": m.get("mce"),
                    "params": t.user_attrs.get("params", {}),
                    "run_id": None,
                }
            )
        top_family = completed[:top_k]
        for rank_idx, t in enumerate(top_family, start=1):
            params = t.user_attrs.get("params", {})
            persisted = train_once(
                model_family=family,
                calibration=args.calibration,
                seed=args.seed,
                output_dir=args.output_dir,
                set_active=False,
                dataset_cache=dataset_cache,
                params=params,
                run_suffix=f"benchmark_top{rank_idx}",
            )
            for row in benchmark_rows:
                if row["family"] == family and row["trial"] == int(t.number):
                    row["run_id"] = persisted["run_id"]
                    break

    if not benchmark_rows:
        raise RuntimeError("Benchmark non eseguito: nessuna famiglia disponibile.")

    bench_df = pd.DataFrame(benchmark_rows).sort_values("score_model", ascending=False).reset_index(drop=True)
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "calibration": args.calibration,
        "seed": int(args.seed),
        "benchmark_config": args.benchmark_config,
        "score_formula": "-logloss + 0.30*auc - 0.20*ece - 0.10*mce",
        "rows": bench_df.to_dict(orient="records"),
        "top_k_per_family": top_k,
        "skipped_families": skipped_families,
    }

    selected_df = bench_df.groupby("family", as_index=False).head(top_k).reset_index(drop=True)
    report["selected_candidates"] = selected_df.to_dict(orient="records")

    if args.set_active and len(selected_df) > 0:
        best = selected_df.iloc[0].to_dict()
        run_id = best.get("run_id")
        run_dir = Path(args.output_dir) / str(run_id)
        write_active_model(
            run_id=str(run_id),
            model_family=str(best.get("family")),
            model_path=run_dir / "model.pkl",
            features_path=run_dir / "model_features.pkl",
            metadata_path=run_dir / "training_report.json",
        )
        report["active_model_set_to"] = str(run_id)
        report["backtest_result"] = run_backtest_if_requested(args.run_backtest_top)
    else:
        report["active_model_set_to"] = None
        report["backtest_result"] = {"enabled": False}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Benchmark report saved: {report_path}")
    print("\nTop candidates:")
    print(selected_df[["family", "trial", "score_model", "auc", "logloss", "ece", "mce", "run_id"]].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Tennis model training and benchmarking.")
    parser.add_argument("--model-family", choices=["xgb", "lgbm", "catboost", "logreg"], default="xgb")
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic", "none"], default="sigmoid")
    parser.add_argument("--output-dir", default="artifacts/models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", action="store_true", help="Run multi-family benchmark using config.")
    parser.add_argument("--benchmark-config", default=BENCHMARK_CONFIG_FILE)
    parser.add_argument("--set-active", dest="set_active", action="store_true", default=True)
    parser.add_argument("--no-set-active", dest="set_active", action="store_false")
    parser.add_argument("--run-backtest-top", action="store_true", help="Run backtest.py on selected active benchmark model.")
    return parser.parse_args()


def main():
    print("============================================================")
    print("TRAIN V13 - MULTI-MODEL + CALIBRATION + REGISTRY")
    print("============================================================")
    args = parse_args()

    if args.benchmark:
        run_benchmark(args)
        return

    report = train_once(
        model_family=args.model_family,
        calibration=args.calibration,
        seed=args.seed,
        output_dir=args.output_dir,
        set_active=args.set_active,
        dataset_cache=None,
        params=None,
    )
    if args.run_backtest_top:
        bt = run_backtest_if_requested(True)
        print(f"[INFO] Backtest returncode: {bt.get('returncode')}")
    print(f"[DONE] Run completed: {report['run_id']}")


if __name__ == "__main__":
    main()
