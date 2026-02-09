# Tennis Match Prediction System - V13 (Research)

Stato progetto: **research/validation** (non production-ready).

Il sistema combina:
- modello ML probabilistico (`P1 win probability`),
- motore value betting con stake sizing,
- validazione robustezza OOS (temporal, walk-forward, stress),
- valutazione model-only su Challenger/Futures senza quote (`no_odds_eval`),
- model registry runtime (`active_model.json`),
- benchmark multi-modello (`xgb`, `lgbm`, `catboost`, `logreg`).

## Obiettivo

Obiettivo primario: massimizzare robustezza out-of-sample, non ROI spot.

Criteri guida:
- drawdown controllato,
- stabilita in walk-forward,
- sensibilita ai costi monitorata,
- calibrazione probabilistica non peggiorativa,
- regole esplicite di promozione baseline.

## Funzionalità

- Tuning anti-overfit configurabile via `configs/backtest_tuning.json` con override environment `BT_*`.
- Objective tuner penalizzato sui bets (`roi_minus_lambda_bets_ratio` o `roi_minus_lambda_bets`).
- Guardrail su numero bets tuned vs baseline con fallback automatico a policy piu robusta.
- Safety fallback: una policy con objective migliore non viene usata se degrada il ROI train oltre soglia (`fallback_eps_roi_pct`).
- Restricted tuning mode con griglia ridotta (`min_edge`, `min_confidence`) e valori fissi per `min_ev` e `prob_shrink`.
- Walk-forward configurabile `quarter|halfyear` con reporting fold-level esteso (`test_period`, objective e guardrail fields).
- Smoke test rapido `colab/smoke_walkforward_guardrail.py` per validare guardrail e soglia minima bets.
- Gestione path centralizzata via `paths.py` con `get_paths()` (`MP_DATA_DIR`, `MP_ARTIFACTS_DIR`, `MP_RUNS_DIR`) e creazione automatica directory.
- Migrazione one-shot `ensure_data_layout()` dei file dati verso `data/` con policy `_dup` anti-overwrite e smoke check dedicato `python scripts/smoke_paths.py`.

## Architettura

- `etl.py`, `utils.py`: ingestione e normalizzazione dati match.
- `features.py`: feature engineering match-level e snapshot stats.
- `train.py`: training/calibrazione/benchmark multi-modello + model registry.
- `backtest.py`: merge quote-feature, simulazione betting, gate OOS, promotion decision.
- `app.py`: dashboard Streamlit per predizione singolo match e value box.
- `predict.py`: CLI inference.
- `configs/model_benchmark.json`: budget benchmark e regole top-k.
- `active_model.json`: source of truth runtime per modello/features attivi.

## Dati e Vincoli

- Dati match: ATP + Challenger + Futures (storico Sackmann-like).
- Dati quote betting: `2024_test.csv` (principalmente ATP).
- Vincolo strutturale: senza quote C/F complete non esiste backtest betting C/F robusto.
- Soluzione: doppio binario
  - ATP betting con quote reali,
  - C/F model-only senza quote.

## Funzionalita App (`app.py`)

- Selezione match: `Player 1`, `Player 2`, `Surface`, `Level (A/C/F)`.
- Input bankroll + rischio (`kelly_fraction`).
- Input quota bookmaker lato P1.
- Predizione probabilistica `model.predict_proba`.
- Arricchimento opzionale live (`live_fetcher.py`) con fallback statico.
- Box decisione value:
  - `VALUE BET DETECTED` con stake suggerito,
  - `NO VALUE` se EV <= 0.
- Sezione analytics:
  - confronto Elo superficie,
  - radar tecnico.
- Runtime artifacts letti via `active_model.json` (`model_path`, `features_path`).

## Logica End-to-End

1. `etl.py`
- merge fonti, normalizzazione schema, parsing date esplicito (`%Y%m%d`, `%d/%m/%Y`, `%Y-%m-%d`),
- quality log: parse fail %, NaN critici, duplicati key/match_id,
- output `clean_matches.csv`.

2. `features.py`
- costruzione feature match-level (Elo, fatigue, rank/h2h, rolling stats, V12 prep/confidence),
- output `processed_features.csv` + `processed_features.parquet`,
- output `latest_stats.csv`.

3. `train.py`
- split temporale: train `2018-2023`, test `2024`,
- augmentation simmetrica solo sul train,
- feature cast a `float32`,
- training per famiglia modello (`xgb|lgbm|catboost|logreg`),
- calibrazione (`sigmoid|isotonic|none`),
- output artifact standard:
  - `artifacts/models/<run_id>/model.pkl`
  - `artifacts/models/<run_id>/model_features.pkl`
  - `artifacts/models/<run_id>/training_report.json`
- update `active_model.json` (se `--set-active`).

4. `backtest.py`
- carica modello attivo da `active_model.json`,
- predizioni 2024,
- merge orientato con quote (`directed_id = p1||p2`),
- probability pipeline:
  - raw -> calibrator -> residual shrink su quote alte,
- value filters + Kelly frazionale + cap rischio,
- report robustezza:
  - temporal H1/H2,
  - walk-forward quarterly/halfyear (configurabile),
  - stress test costi,
  - OOS gate,
  - promotion decision.

## Tuning Anti-Overfit (V13+)

Configurazione centralizzata: `configs/backtest_tuning.json`

Parametri chiave:
- `min_bets_for_tuning` (default `100`)
- `objective_mode`:
  - `roi_minus_lambda_bets_ratio` (default): `roi - lambda_bets * (bets / n_matches_train)`
  - `roi_minus_lambda_bets`
- `lambda_bets` (default `0.5`)
- `max_bets_increase_pct` (default `0.20`) per guardrail bets tuned vs baseline
- `fallback_eps_roi_pct` (default `0.10`) safety fallback
- `restricted_tuning_mode` (default `false`)
- `fold_freq` (default `quarter`, opzionale `halfyear`)

Override via env:
- `BT_MIN_BETS_FOR_TUNING`
- `BT_OBJECTIVE_MODE`
- `BT_LAMBDA_BETS`
- `BT_MAX_BETS_INCREASE_PCT`
- `BT_FALLBACK_EPS_ROI_PCT`
- `BT_RESTRICTED_TUNING_MODE`
- `BT_FOLD_FREQ`

Restricted tuning mode:
- fissa `min_ev=0.06` e `prob_shrink=0.60`,
- esplora solo:
  - `min_edge` in `{0.06, 0.065}`
  - `min_confidence` in `{0.64, 0.66}`

## Feature V12 (C/F Focus)

Blocchi principali:
- preparazione/inattivita:
  - `p*_days_since_last_match`,
  - bucket inattivita,
  - `matches_last_30d`,
  - `long_stop_90d`,
  - `no_prev_match`,
  - diff + interazioni `*_cf`.
- confidence ranking-aware:
  - upset wins, bad losses, confidence score su ultime 12 partite,
  - diff e interazioni `*_cf`.

## Doppio Binario ATP + C/F

Sezione `no_odds_eval` in `runs/backtest_validation_report.json`:
- filtro anno/livello configurabile (`no_odds_eval_*` in `BASELINE_CONFIG`),
- metriche model-only:
  - `n_matches`, `logloss`, `brier`, `auc`, `accuracy_0_5`, `ece_10_bins`, `mce_10_bins`,
- baseline confronto:
  - `p50`,
  - `train_prior` storico (`year < eval_year`),
- gate `light`:
  - `pass` se migliora logloss+brier vs `train_prior`,
  - `insufficient_data` se campione sotto soglia,
  - `fail` altrimenti.

## OOS Gate e Promotion

`oos_gate` controlla:
- walk-forward baseline ROI > 0,
- fold validi minimi rispettati,
- tuned non sotto baseline oltre soglia,
- tuned non peggiore in numero minimo fold,
- max drawdown <= 5%,
- calibrazione non peggiorativa vs raw.

`promotion_decision`:
- `promote` o `keep_baseline`,
- criteri espliciti su gate, walk-forward, drawdown, no_odds gate.

## Benchmark Multi-Modello

Configurato da `configs/model_benchmark.json`.

Score model-only:
- `score_model = -logloss + 0.30*AUC - 0.20*ECE - 0.10*MCE`

Policy benchmark:
- top-k candidati per famiglia,
- benchmark report in `artifacts/models/benchmark_report.json` (se run completo),
- fallback automatico se librerie non disponibili (es. `lgbm`/`catboost` assenti).

## Output Principali

- `runs/backtest_validation_report.json`
  - `oos_gate`
  - `promotion_decision`
  - `no_odds_eval`
- `runs/backtest_walkforward_report.json`
  - per fold include anche:
    - `test_period` (`YYYYQ*` o `YYYYH*`),
    - `baseline_train_bets`, `tuned_train_bets`,
    - `objective_best`, `objective_baseline`,
    - `guardrail_passed`, `guardrail_reason`,
    - `restricted_tuning_mode`
- `runs/backtest_stress_report.json`
- `runs/backtest_baseline_config.json`
- `runs/reliability_curve.png`
- `runs/reliability_table.csv`
- `runs/real_backtest.png`

## Path Persistenti (DATA / ARTIFACTS / RUNS)

Directory centralizzate configurabili via env:
- `MP_DATA_DIR` (default: `./data`)
- `MP_ARTIFACTS_DIR` (default: `./artifacts`)
- `MP_RUNS_DIR` (default: `./runs`)

Senza env variabili, il comportamento locale resta invariato ma i file vengono gestiti in `data/`, `artifacts/`, `runs/` invece che in root.

Esempio esecuzione persistente su Colab/Drive:
```bash
MP_DATA_DIR=/content/drive/MyDrive/match-prediction_persist/data \
MP_ARTIFACTS_DIR=/content/drive/MyDrive/match-prediction_persist/artifacts \
MP_RUNS_DIR=/content/drive/MyDrive/match-prediction_persist/runs \
python backtest.py
```

In Colab: carica i CSV in `data/` (non in root)
```python
from google.colab import files
import shutil
from pathlib import Path

repo = Path('/content/match-prediction')
data_dir = repo / 'data'
data_dir.mkdir(parents=True, exist_ok=True)

uploaded = files.upload()
for filename in uploaded:
    shutil.move(filename, data_dir / filename)
```

In Colab: mount Drive + symlink opzionale di `artifacts/` e `runs/`
```python
from google.colab import drive
import os
from pathlib import Path

drive.mount('/content/drive')
repo = Path('/content/match-prediction')
persist_root = Path('/content/drive/MyDrive/match-prediction_persist')

for name in ['artifacts', 'runs']:
    target = persist_root / name
    target.mkdir(parents=True, exist_ok=True)
    link = repo / name
    if link.is_symlink() or link.is_file():
        link.unlink()
    if not link.exists():
        os.symlink(target, link)
```

## Comandi Operativi

```bash
python etl.py
python features.py
python train.py --model-family xgb --calibration sigmoid
python backtest.py
streamlit run app.py
```

Walk-forward semestrale:
```bash
BT_FOLD_FREQ=halfyear python backtest.py
```

Smoke test anti-overfit:
```bash
python colab/smoke_walkforward_guardrail.py
```

Workflow di validazione consigliato (sequenziale):
```bash
python colab/smoke_walkforward_guardrail.py
python backtest.py
python -c "import json; r=json.load(open('runs/backtest_validation_report.json')); print({'oos_status':r['oos_gate']['status'],'oos_pass':r['oos_gate']['pass'],'promo':r['promotion_decision']['status'],'reasons':r['oos_gate']['reasons']})"
BT_FOLD_FREQ=halfyear python backtest.py
```

Coerenza report run:
- ogni esecuzione di `python backtest.py` genera `backtest_run_id` e `backtest_generated_at_utc` condivisi in:
  - `runs/backtest_validation_report.json`
  - `runs/backtest_walkforward_report.json`
  - `runs/backtest_stress_report.json`
  - `runs/backtest_baseline_config.json`
- prima di calcolare la `promotion_decision`, viene applicato un check di coerenza report (`report_coherence`).
- se il check fallisce, l'`oos_gate` viene forzato a `NO_GO` e la decisione resta `keep_baseline`.

## Operativita Post-GO

Quando `oos_gate.pass=True` e `promotion_decision=promote`:
- congela lo snapshot run (`runs/backtest_validation_report.json`, `runs/backtest_walkforward_report.json`, `runs/backtest_stress_report.json`, `runs/backtest_baseline_config.json`, `active_model.json`);
- traccia i KPI in `runs/scorecard_history.csv`;
- registra ogni bet live/paper in `runs/live_bets_log.csv` con `pnl` e `bankroll`.

Bootstrap automatico cartella `runs/`:
- `python backtest.py` crea automaticamente (se mancanti):
  - `runs/scorecard_history.csv`
  - `runs/live_bets_log.csv`
- a fine run viene aggiunta una riga KPI in `runs/scorecard_history.csv`.

Regole di monitoraggio continuo:
- `STOP` se `max_drawdown_pct > 5`;
- `RETRAIN` se `roi_last_100_pct < -2`;
- `OK` altrimenti.

Trigger retrain periodico:
- ogni 14 giorni oppure ogni 300 nuovi match.

Benchmark:
```bash
pip install -r requirements-ml.txt
python train.py --benchmark --calibration sigmoid
```

## KPI e Interpretazione

- betting:
  - `ROI`, `bets`, `win_rate`, `profit_factor`, `max_drawdown_pct`, `bootstrap_roi_ci`
- model-only:
  - `AUC`, `logloss`, `Brier`, `ECE`, `MCE`
- robustezza:
  - `walk_forward.overall_*`,
  - `stress_tests`,
  - `oos_gate`,
  - `promotion_decision`.
- live/paper monitoring:
  - `status` (`WAIT_FOR_DATA|OK|RETRAIN|STOP`),
  - `roi_last_100_pct`,
  - `max_drawdown_pct`.

## Limiti Noti

- Mancano quote C/F complete per betting backtest pieno su Challenger/Futures.
- Alcuni fold walk-forward possono restare campione-limitati.
- Il benchmark multi-famiglia dipende da dipendenze installate.
- Non e un sistema live-trading production (execution/risk governance incomplete).
