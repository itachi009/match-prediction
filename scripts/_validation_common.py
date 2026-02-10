import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
GOLDEN_DIR = REPO_ROOT / "golden"

VALIDATION_REPORT_NAME = "backtest_validation_report.json"
WALKFORWARD_REPORT_NAME = "backtest_walkforward_report.json"
STRESS_REPORT_NAME = "backtest_stress_report.json"
BASELINE_CONFIG_NAME = "backtest_baseline_config.json"

OPTIONAL_ARTIFACT_NAMES = [
    "reliability_by_p_bucket.csv",
    "reliability_by_uncertainty_bucket.csv",
    "reliability_by_odds_bucket.csv",
    "reliability_curve.png",
    "reliability_table.csv",
    "backtest_walkforward_bet_records.csv",
    "backtest_walkforward_policy_audit.csv",
    "policy_toggle_comparison.json",
    "analysis_walkforward.md",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def git_commit_hash(repo_root: Path = REPO_ROOT) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _validation_path_from_dir(path: Path) -> Path:
    return path / VALIDATION_REPORT_NAME


def _list_run_dirs(runs_dir: Path) -> List[Path]:
    out: List[Path] = []
    if not runs_dir.exists():
        return out
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        if _validation_path_from_dir(child).exists():
            out.append(child)
    out.sort(key=lambda p: _validation_path_from_dir(p).stat().st_mtime, reverse=True)
    return out


def resolve_run_source(
    run_path: Optional[str] = None,
    run_id: Optional[str] = None,
    runs_dir: Path = RUNS_DIR,
) -> Tuple[Path, Path]:
    if run_path:
        p = Path(run_path).expanduser().resolve()
        if p.is_file():
            if p.name != VALIDATION_REPORT_NAME:
                raise FileNotFoundError(f"File non valido: {p}")
            return p.parent, p
        if p.is_dir():
            v = _validation_path_from_dir(p)
            if v.exists():
                return p, v
            raise FileNotFoundError(f"{VALIDATION_REPORT_NAME} non trovato in {p}")
        raise FileNotFoundError(f"Percorso run non trovato: {p}")

    root_validation = _validation_path_from_dir(runs_dir)
    if run_id:
        candidate = runs_dir / run_id
        candidate_validation = _validation_path_from_dir(candidate)
        if candidate_validation.exists():
            return candidate, candidate_validation
        if root_validation.exists():
            payload = load_json(root_validation)
            rid = str(payload.get("backtest_run_id", "")).strip()
            if rid == run_id:
                return runs_dir, root_validation
        raise FileNotFoundError(f"Run id non trovato: {run_id}")

    run_dirs = _list_run_dirs(runs_dir)
    if run_dirs:
        src = run_dirs[0]
        return src, _validation_path_from_dir(src)
    if root_validation.exists():
        return runs_dir, root_validation
    raise FileNotFoundError("Nessuna run trovata in runs/.")


def extract_run_id(validation_payload: Dict, source_dir: Path) -> str:
    rid = str(validation_payload.get("backtest_run_id", "")).strip()
    if rid:
        return rid
    return source_dir.name if source_dir.name else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _safe_path(value: object) -> Optional[Path]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except Exception:
        return None


def _is_child_of(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def collect_run_artifacts(
    source_dir: Path,
    validation_payload: Dict,
    runs_dir: Path = RUNS_DIR,
    repo_root: Path = REPO_ROOT,
) -> List[Path]:
    source_dir = source_dir.resolve()
    runs_dir = runs_dir.resolve()
    source_is_run_subdir = source_dir != runs_dir and source_dir.parent == runs_dir
    candidates: List[Path] = []

    required_names = [
        VALIDATION_REPORT_NAME,
        WALKFORWARD_REPORT_NAME,
        STRESS_REPORT_NAME,
        BASELINE_CONFIG_NAME,
    ]
    for name in required_names:
        p = source_dir / name
        if p.exists():
            candidates.append(p)

    report_files = validation_payload.get("report_files", {})
    if isinstance(report_files, dict):
        for value in report_files.values():
            p = _safe_path(value)
            if not p or not p.exists():
                continue
            # For historical materialization, trust only paths already under the run directory.
            if source_is_run_subdir and not _is_child_of(p, source_dir):
                continue
            candidates.append(p)

    for name in OPTIONAL_ARTIFACT_NAMES:
        p = source_dir / name
        if p.exists():
            candidates.append(p)

    active_model = repo_root / "active_model.json"
    if active_model.exists():
        candidates.append(active_model)

    unique: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p.resolve())
    return unique


def copy_artifacts_to_dir(files: Iterable[Path], target_dir: Path, allow_overwrite: bool = False) -> List[Dict]:
    import shutil

    target_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Dict] = []
    for src in files:
        if not src.exists():
            continue
        dst = target_dir / src.name
        if src.resolve() != dst.resolve():
            if dst.exists() and not allow_overwrite:
                continue
            shutil.copy2(src, dst)
        copied.append(
            {
                "name": dst.name,
                "path": str(dst.resolve()),
                "size_bytes": int(dst.stat().st_size),
                "sha256": sha256_file(dst),
            }
        )
    return copied


def materialize_run_dir(
    run_id: str,
    source_dir: Path,
    validation_payload: Dict,
    runs_dir: Path = RUNS_DIR,
    repo_root: Path = REPO_ROOT,
) -> Tuple[Path, List[Dict]]:
    source_dir = source_dir.resolve()
    runs_dir = runs_dir.resolve()
    run_dir = (runs_dir / run_id).resolve()

    payload_run_id = str(validation_payload.get("backtest_run_id", "")).strip()
    if payload_run_id and payload_run_id != run_id:
        raise ValueError(f"Run id mismatch: requested={run_id} payload={payload_run_id}")

    artifacts = collect_run_artifacts(
        source_dir=source_dir,
        validation_payload=validation_payload,
        runs_dir=runs_dir,
        repo_root=repo_root,
    )
    copied = copy_artifacts_to_dir(artifacts, run_dir, allow_overwrite=(source_dir == runs_dir))
    return run_dir, copied


def resolve_golden_dir(golden_arg: Optional[str] = None, golden_root: Path = GOLDEN_DIR) -> Path:
    if golden_arg:
        p = Path(golden_arg).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Golden run non trovata: {p}")

    latest = golden_root / "latest"
    if latest.exists():
        return latest.resolve()

    if not golden_root.exists():
        raise FileNotFoundError("Cartella golden/ non trovata e nessun --golden fornito.")

    candidates = [p for p in golden_root.iterdir() if p.is_dir() and p.name != "latest"]
    if not candidates:
        raise FileNotFoundError("Nessuna golden run disponibile in golden/.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def table_str(rows: Sequence[Dict], columns: Sequence[str]) -> str:
    if not rows:
        return "(nessuna riga)"
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    body = [" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns) for row in rows]
    return "\n".join([header, sep, *body])
