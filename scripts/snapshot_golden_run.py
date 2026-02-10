import argparse
import shutil
from pathlib import Path

from _validation_common import (
    GOLDEN_DIR,
    REPO_ROOT,
    RUNS_DIR,
    collect_run_artifacts,
    copy_artifacts_to_dir,
    extract_run_id,
    git_commit_hash,
    load_json,
    resolve_run_source,
    save_json,
    utc_now_iso,
)


def _unique_target_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    i = 2
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_dup{i}")
        if not candidate.exists():
            return candidate
        i += 1


def _refresh_latest_link(target_dir: Path, golden_root: Path) -> Path:
    latest = golden_root / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    try:
        latest.symlink_to(target_dir, target_is_directory=True)
    except Exception:
        shutil.copytree(target_dir, latest)
    return latest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Crea snapshot golden run con manifest/checksum.")
    parser.add_argument("--run-path", default=None, help="Percorso run (runs/ o runs/<run_id>).")
    parser.add_argument("--run-id", default=None, help="Run id da snapshottare (se presente in runs/<run_id>).")
    parser.add_argument("--golden-root", default=str(GOLDEN_DIR), help="Cartella root golden.")
    parser.add_argument(
        "--no-latest",
        action="store_true",
        help="Non aggiornare golden/latest.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    golden_root = Path(args.golden_root).expanduser().resolve()
    golden_root.mkdir(parents=True, exist_ok=True)

    source_dir, validation_path = resolve_run_source(
        run_path=args.run_path,
        run_id=args.run_id,
        runs_dir=RUNS_DIR,
    )
    validation_payload = load_json(validation_path)
    run_id = extract_run_id(validation_payload, source_dir)

    target_dir = _unique_target_dir((golden_root / run_id).resolve())
    artifacts = collect_run_artifacts(source_dir=source_dir, validation_payload=validation_payload, repo_root=REPO_ROOT)
    copied_files = copy_artifacts_to_dir(artifacts, target_dir=target_dir)

    manifest = {
        "run_id": run_id,
        "snapshot_created_at_utc": utc_now_iso(),
        "source_run_path": str(source_dir.resolve()),
        "validation_report_source": str(validation_path.resolve()),
        "golden_dir": str(target_dir.resolve()),
        "git_commit_hash": git_commit_hash(REPO_ROOT),
        "files": copied_files,
        "files_count": len(copied_files),
    }
    manifest_path = target_dir / "manifest.json"
    save_json(manifest_path, manifest)

    latest_path = None
    if not args.no_latest:
        latest_path = _refresh_latest_link(target_dir=target_dir, golden_root=golden_root)

    print(f"[golden] source run: {source_dir}")
    print(f"[golden] run_id: {run_id}")
    print(f"[golden] files copied: {len(copied_files)}")
    print(f"[golden] manifest: {manifest_path}")
    if latest_path is not None:
        print(f"[golden] latest: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
