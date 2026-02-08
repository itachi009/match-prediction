from __future__ import annotations

import os
import shutil
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parent
RAW_CSV_PATTERNS = [
    "atp_matches*.csv",
    "*futures*.csv",
    "*chall*.csv",
    "*qual*.csv",
]
CORE_DATA_FILES = [
    "clean_matches.csv",
    "processed_features.csv",
]
OPTIONAL_DATA_FILES = [
    "processed_features.parquet",
    "latest_stats.csv",
    "atp_players.csv",
    "2024_test.csv",
]


def _resolve_path(value: str | None, default: Path, repo_root: Path) -> Path:
    raw = str(value).strip() if value is not None else ""
    if not raw:
        target = default
    else:
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = repo_root / target
    return target.resolve()


def get_paths() -> Dict[str, Path]:
    repo_root = REPO_ROOT
    data_dir = _resolve_path(os.getenv("MP_DATA_DIR"), repo_root / "data", repo_root)
    artifacts_dir = _resolve_path(os.getenv("MP_ARTIFACTS_DIR"), repo_root / "artifacts", repo_root)
    runs_dir = _resolve_path(os.getenv("MP_RUNS_DIR"), repo_root / "runs", repo_root)

    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "repo_root": repo_root,
        "data_dir": data_dir,
        "artifacts_dir": artifacts_dir,
        "runs_dir": runs_dir,
    }


def resolve_repo_path(path_like: str | Path, repo_root: Path | None = None) -> Path:
    root = repo_root or get_paths()["repo_root"]
    p = Path(path_like).expanduser()
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def _next_dup_name(path: Path) -> Path:
    base = path.with_name(f"{path.stem}_dup{path.suffix}")
    if not base.exists():
        return base
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem}_dup{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def _move_with_dup_policy(src: Path, dst: Path) -> Dict[str, str]:
    if not src.exists():
        return {"action": "missing", "src": str(src), "dst": str(dst)}

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dup = _next_dup_name(src)
        src.rename(dup)
        return {"action": "renamed_dup", "src": str(src), "dst": str(dup)}

    shutil.move(str(src), str(dst))
    return {"action": "moved", "src": str(src), "dst": str(dst)}


def ensure_data_layout() -> Dict[str, object]:
    paths = get_paths()
    repo_root = paths["repo_root"]
    data_dir = paths["data_dir"]

    summary: Dict[str, object] = {
        "repo_root": str(repo_root),
        "data_dir": str(data_dir),
        "moved": [],
        "renamed_dup": [],
        "missing": [],
    }

    explicit_files = CORE_DATA_FILES + OPTIONAL_DATA_FILES
    for name in explicit_files:
        src = repo_root / name
        dst = data_dir / name
        result = _move_with_dup_policy(src, dst)
        action = result["action"]
        if action == "moved":
            summary["moved"].append(result)
        elif action == "renamed_dup":
            summary["renamed_dup"].append(result)
        else:
            summary["missing"].append(result)

    moved_explicit_names = set(explicit_files)
    csv_candidates = sorted(repo_root.glob("*.csv"))
    for src in csv_candidates:
        name = src.name
        if name in moved_explicit_names:
            continue
        if not any(fnmatch(name.lower(), pattern.lower()) for pattern in RAW_CSV_PATTERNS):
            continue
        dst = data_dir / name
        result = _move_with_dup_policy(src, dst)
        action = result["action"]
        if action == "moved":
            summary["moved"].append(result)
        elif action == "renamed_dup":
            summary["renamed_dup"].append(result)
        else:
            summary["missing"].append(result)

    return summary
