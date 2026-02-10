import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import ensure_data_layout, get_paths


def main():
    paths = get_paths()
    repo_root = paths["repo_root"]
    data_dir = paths["data_dir"]
    artifacts_dir = paths["artifacts_dir"]
    runs_dir = paths["runs_dir"]

    print("[PATHS]")
    print(f"repo_root={repo_root}")
    print(f"data_dir={data_dir}")
    print(f"artifacts_dir={artifacts_dir}")
    print(f"runs_dir={runs_dir}")

    migration = ensure_data_layout()
    print("[MIGRATION]")
    print(f"moved={len(migration.get('moved', []))}")
    print(f"renamed_dup={len(migration.get('renamed_dup', []))}")

    clean_path = data_dir / "clean_matches.csv"
    features_path = data_dir / "processed_features.csv"

    clean_ok = str(clean_path.resolve()).startswith(str(data_dir.resolve()))
    features_ok = str(features_path.resolve()).startswith(str(data_dir.resolve()))

    if not clean_ok:
        raise SystemExit("clean_matches path is not inside DATA_DIR")
    if not features_ok:
        raise SystemExit("processed_features path is not inside DATA_DIR")

    for folder in [data_dir, artifacts_dir, runs_dir]:
        if not folder.exists():
            raise SystemExit(f"Missing expected directory: {folder}")

    print("[OK] smoke_paths passed")
    print(f"clean_matches_expected={clean_path}")
    print(f"processed_features_expected={features_path}")


if __name__ == "__main__":
    main()