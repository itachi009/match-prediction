import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


INPUT_FEATURES_FILE = "processed_features.csv"
OUTPUT_MODEL_FILE = "model_v9_balanced.pkl"
OUTPUT_FEATURE_LIST_FILE = "model_features.pkl"
OUTPUT_META_FILE = "model_v9_training_report.json"

# Split temporale:
# Train: 2018-2023 (con calibrazione CV interna)
# Test:  2024
TRAIN_START_YEAR = 2018
TRAIN_END_YEAR = 2023
TEST_START_YEAR = 2024

# Calibrazione probabilistica
CALIBRATION_METHOD = "sigmoid"  # "sigmoid" o "isotonic"
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

    # Swap names/target if present.
    if "p1_name" in inv.columns and "p2_name" in inv.columns:
        inv[["p1_name", "p2_name"]] = inv[["p2_name", "p1_name"]]
    if "target" in inv.columns:
        inv["target"] = 1 - inv["target"]

    # Swap p1_*/p2_* features.
    p1_cols = [c for c in inv.columns if c.startswith("p1_")]
    for p1_col in p1_cols:
        p2_col = "p2_" + p1_col[3:]
        if p2_col in inv.columns:
            tmp = inv[p1_col].copy()
            inv[p1_col] = inv[p2_col]
            inv[p2_col] = tmp

    # Invert differential features.
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
    original = df.copy()
    inverse = build_inverse_rows(df)
    doubled = pd.concat([original, inverse], ignore_index=True)

    # Avoid exact duplicates if dataset was already symmetric upstream.
    deduped = doubled.drop_duplicates()
    return deduped


def train_model():
    print("============================================================")
    print("TRAIN V9 - BALANCED + CALIBRATED")
    print("============================================================")

    try:
        df = pd.read_csv(INPUT_FEATURES_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"[ERROR] File non trovato: {INPUT_FEATURES_FILE}")
        return

    date_col = infer_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, "target"]).copy()
    df = df.sort_values(date_col).reset_index(drop=True)

    train_mask = (df[date_col].dt.year >= TRAIN_START_YEAR) & (df[date_col].dt.year <= TRAIN_END_YEAR)
    test_mask = df[date_col].dt.year >= TEST_START_YEAR

    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    if train_df.empty or test_df.empty:
        print("[ERROR] Split train/test vuoto. Controlla anni e dati disponibili.")
        return

    feature_cols = prepare_feature_columns(df, date_col)
    if not feature_cols:
        print("[ERROR] Nessuna feature numerica disponibile per il training.")
        return

    print(f"Dataset originale: train={len(train_df)} | test={len(test_df)}")
    print(f"Feature usate ({len(feature_cols)}): {feature_cols}")

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
    print(f"[CHECK] V12 feature rilevate: {len(v12_features)}")
    if v12_features:
        print(f"[CHECK] Esempio V12: {v12_features[:20]}")
    else:
        print("[CHECK] ATTENZIONE: nessuna feature V12 rilevata.")

    # Data doubling simmetrico solo sul train
    train_aug = augment_symmetric(train_df)
    print(f"Train dopo augmentation+dedup: {len(train_aug)} righe")

    y_train_dist = train_aug["target"].value_counts(normalize=True).sort_index()
    y_test_dist = test_df["target"].value_counts(normalize=True).sort_index()
    print(f"Distribuzione target train: {y_train_dist.to_dict()}")
    print(f"Distribuzione target test:  {y_test_dist.to_dict()}")

    X_train = train_aug[feature_cols].fillna(0.0)
    y_train = train_aug["target"].astype(int)
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["target"].astype(int)

    # Base learner (per calibrazione)
    base_model = xgb.XGBClassifier(
        n_estimators=1400,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )

    # Calibrazione CV con split temporale.
    cv = TimeSeriesSplit(n_splits=CALIBRATION_SPLITS)
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method=CALIBRATION_METHOD,
        cv=cv,
    )

    print(f"Training + calibrazione ({CALIBRATION_METHOD})...")
    calibrated_model.fit(X_train, y_train)

    # Valutazione 2024
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    print("\n--- Validation (2024) ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"Brier:    {brier:.4f} (piu basso e meglio)")
    print(f"LogLoss:  {ll:.4f} (piu basso e meglio)")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Feature importance: fit separato per interpretabilita.
    importance_model = xgb.XGBClassifier(
        n_estimators=1400,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    importance_model.fit(X_train, y_train)
    imps = importance_model.feature_importances_

    feat_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": imps}
    ).sort_values("importance", ascending=False)

    print("\nTop 15 feature importance:")
    print(feat_imp.head(15).to_string(index=False))

    rank_like = [c for c in feat_imp.head(3)["feature"].tolist() if c in {"rank_diff", "log_rank_diff"}]
    if rank_like:
        print(f"[CHECK] OK: feature rank nelle top3 -> {rank_like}")
    else:
        print("[CHECK] ATTENZIONE: nessuna tra 'rank_diff'/'log_rank_diff' e nelle top3.")

    # Save calibrated model + feature list
    joblib.dump(calibrated_model, OUTPUT_MODEL_FILE)
    joblib.dump(feature_cols, OUTPUT_FEATURE_LIST_FILE)

    report = {
        "model_file": OUTPUT_MODEL_FILE,
        "features_file": OUTPUT_FEATURE_LIST_FILE,
        "calibration_method": CALIBRATION_METHOD,
        "train_rows_original": int(len(train_df)),
        "train_rows_augmented": int(len(train_aug)),
        "test_rows": int(len(test_df)),
        "metrics_2024": {
            "accuracy": float(acc),
            "auc": float(auc),
            "brier": float(brier),
            "logloss": float(ll),
        },
        "v12_feature_count": int(len(v12_features)),
        "v12_feature_sample": v12_features[:30],
        "top15_features": feat_imp.head(15).to_dict(orient="records"),
    }
    with open(OUTPUT_META_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nModello calibrato salvato in: {OUTPUT_MODEL_FILE}")
    print(f"Feature list salvata in: {OUTPUT_FEATURE_LIST_FILE}")
    print(f"Report training salvato in: {OUTPUT_META_FILE}")


if __name__ == "__main__":
    train_model()
