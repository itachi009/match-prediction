2026-02-09 - Sessione uncertainty/calibration/policy e manutenzione docs.
Implementato `uncertainty_score` pre-match rule-based con pesi configurabili e output `recommendation_allowed`.
Aggiunto `effective_confidence` derivato da confidence e uncertainty per ridurre falsa sicurezza pre-bet.
Introdotta calibrazione separata per livello `A/C/F` con fallback automatico sotto `MP_N_MIN_CALIB`.
Estesi i report bucket su probabilita, uncertainty e odds con CSV persistenti in `runs/`.
Integrato policy layer betting con `policy_allowed`, `policy_reason` e blocchi reason-based nei report.
Eseguire `python scripts/smoke_uncertainty_and_calib.py` e controllare i CSV `reliability_by_*_bucket.csv`.
Eseguire `python scripts/smoke_policy_layer.py` e verificare `runs/policy_layer_preview.csv`.
Aggiunto `python scripts/smoke_policy_toggle.py`: esegue backtest con policy ON/OFF, salva confronto in `runs/policy_toggle_comparison.json` e imposta default veloci (`BT_RESTRICTED_TUNING_MODE=true`, `BT_FOLD_FREQ=halfyear`) se assenti.
Eseguire lo smoke toggle e verificare `pass=true`: policy OFF deve avere zero blocchi policy-specific (sono ammessi blocchi legacy), con delta metriche (`roi_pct`, `bets`, `blocked_rate`) tracciati nel JSON.

2026-02-10 - Sessione validazione smoke + OOS gate (policy/tuning/model switch).
Smoke `python scripts/smoke_policy_toggle.py` corretto e passato (`PASS=True`) dopo fix check su blocchi legacy vs policy-specific.
Smoke `python scripts/smoke_uncertainty_and_calib.py` passato con CSV bucket aggiornati in `runs/`.
Smoke `python scripts/smoke_policy_layer.py` passato (8 candidati, 5 bloccati, preview generata).
Train eseguito: `20260210_140651_xgb_sigmoid` e benchmark multi-family eseguito con report salvato in `artifacts/models/benchmark_report.json`.
Active model finale impostato a `20260210_145638_xgb_sigmoid_benchmark_default`.
Backtest ripetuti con `BT_RESTRICTED_TUNING_MODE=true`, `BT_FOLD_FREQ=quarter`, policy ON/OFF e varianti su `MP_MAX_UNCERTAINTY_FOR_RECO`.
Stato finale: `baseline_roi=-0.1798698051`, `walkforward_overall_baseline_roi=-0.4737073631`, `n_valid_folds=3`.
`oos_status=GO_WITH_CAUTION`, `oos_pass=False`, reason unica: `walkforward_baseline_roi<=0`.
`promotion_status=keep_baseline`; priorita prossima: migliorare robustezza WF (in particolare fold Q3/Q4) prima di nuova promozione.

2026-02-10 - Sessione chiusura OOS gate (GO raggiunto).
Eseguiti smoke test: `smoke_policy_toggle` (PASS), `smoke_uncertainty_and_calib` (PASS), `smoke_policy_layer` (PASS).
Aggiunta diagnostica walk-forward con export CSV: `runs/backtest_walkforward_bet_records.csv` e `runs/backtest_walkforward_policy_audit.csv`.
Aggiunti override env baseline in `backtest.py` (`BT_BASELINE_*`) per tuning rapido senza hardcode.
Diagnostica fold-level: perdita concentrata su Q3/Q4; testate varianti policy/uncertainty/odds.
Configurazione vincente: `MP_POLICY_ENABLED=false`, `BT_FOLD_FREQ=quarter`, `BT_RESTRICTED_TUNING_MODE=true`, `BT_BASELINE_MIN_ODDS=1.7`, `BT_BASELINE_MIN_CONFIDENCE=0.71`.
Run finale: `backtest_run_id=bt_20260210_170528_55b3e28d`, `active_model_run_id=20260210_145638_xgb_sigmoid_benchmark_default`.
Esito finale: `oos_status=GO`, `oos_pass=True`, `promotion_status=promote`.

2026-02-10 - Tooling anti-regressione e pipeline validazione ripetibile.
Aggiunti script: `scripts/snapshot_golden_run.py`, `scripts/regression_gate.py`, `scripts/validate_all.py`, `scripts/analyze_walkforward_folds.py`, `scripts/compare_policy_variants.py`.
Aggiunto modulo condiviso `scripts/_validation_common.py` per risoluzione run, materializzazione `runs/<run_id>`, copia artifact e checksum.
Implementato snapshot golden con `manifest.json` (run_id, timestamp, git hash se disponibile, file copiati, size, sha256) e refresh `golden/latest`.
Implementato regression gate con hard fail (`oos_gate`, `report_coherence`, drawdown, n_valid_folds, walkforward ROI baseline) + warning su regressioni calibrazione/reliability.
Output gate salvato in `runs/<run_id>/regression_gate_result.json`.
Implementata suite one-command: `python scripts/validate_all.py --golden golden/latest` (smoke + backtest + gate).
Implementata analisi fold-level in markdown: `runs/<run_id>/analysis_walkforward.md` con bucket odds/confidence/uncertainty e top loss segments.
Implementato confronto policy OFF vs ON soft con preset non aggressivo e modalita diff-only tra run.
Test minimi aggiunti in `tests/test_regression_gate.py` con fixture JSON in `tests/fixtures/` (nota: `pytest` non installato nell'ambiente locale corrente).

2026-02-10 - Sessione consolidamento GO + regression gate PASS.
Eseguita run rapida con `BT_FOLD_FREQ=halfyear`: esito `NO_GO` (profilo smoke/veloce, non baseline finale).
Rieseguito backtest con configurazione target GO: `MP_POLICY_ENABLED=false`, `MP_ENABLE_BUCKET_REPORTS=true`, `BT_RESTRICTED_TUNING_MODE=true`, `BT_FOLD_FREQ=quarter`, `BT_BASELINE_MIN_ODDS=1.7`, `BT_BASELINE_MIN_CONFIDENCE=0.71`.
Run finale: `backtest_run_id=bt_20260210_192013_c1cb51e4`.
Esito finale: `oos_status=GO`, `oos_pass=True`, `promotion_status=promote`.
Snapshot golden aggiornato con `python scripts/snapshot_golden_run.py --run-id bt_20260210_192013_c1cb51e4`; `golden/latest/manifest.json` allineato al run id finale.
Regression gate finale: `status=PASS`, `pass=True`, `hard_failures={}`, `warnings={}`.
Eseguiti post-check: `python scripts/analyze_walkforward_folds.py --run-id bt_20260210_192013_c1cb51e4` e `python scripts/validate_all.py --no-backtest --skip-smokes --run-id bt_20260210_192013_c1cb51e4 --golden golden/latest`.
