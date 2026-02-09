2026-02-08 - Sessione persistenza path Colab/Drive e manutenzione docs.
Introdotto `paths.py` con `get_paths()` e directory configurabili via `MP_DATA_DIR`, `MP_ARTIFACTS_DIR`, `MP_RUNS_DIR`.
Implementata `ensure_data_layout()` per migrazione one-shot dei CSV da root a `data/` con policy `_dup` anti-overwrite.
Aggiornati entrypoint (`etl.py`, `features.py`, `train.py`, `backtest.py`, `predict.py`, `app.py`, `download_data.py`) per usare path centralizzati.
Spostati input dati (`clean_matches.csv`, `processed_features.csv`, raw csv) su `DATA_DIR` e output run/report su `RUNS_DIR`.
Aggiunto `scripts/smoke_paths.py` per verifica path, creazione cartelle e migrazione.
Eseguire `python scripts/smoke_paths.py` su runtime nuovo e confermare `[OK] smoke_paths passed`.
Verificare in Colab che `MP_*` puntino a Drive e che `backtest_validation_report.json` finisca in `MP_RUNS_DIR`.
Portare nel repo la fix `json.dump(..., default=str)` in `backtest.py` se il crash `PosixPath` riappare.
Commit/push delle modifiche e test rapido end-to-end (`etl -> features -> train -> backtest`) con path persistenti.
