import requests

from paths import ensure_data_layout, get_paths

# Updated Sources based on Web Search
# Both ATP and Challenger/Qual matches seem to be in 'tennis_atp' repo.
# Filename: atp_matches_qual_chall_{year}.csv

SOURCES = [
    {
        "name": "ATP",
        "url_fmt": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{}.csv",
        "filename_fmt": "atp_matches_{}.csv",
    },
    {
        "name": "Challenger",
        "url_fmt": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_qual_chall_{}.csv",
        "filename_fmt": "atp_matches_qual_chall_{}.csv",
    },
    # Futures are often in a separate repo 'tennis_wta' (for wta) or 'tennis_atp' as 'atp_matches_futures_{year}.csv'
    {
        "name": "Futures",
        "url_fmt": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_futures_{}.csv",
        "filename_fmt": "atp_matches_futures_{}.csv",
    },
    {
        "name": "Players",
        "url_fmt": "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv",
        "filename_fmt": "atp_players.csv",
        "is_static": True,
    },
]

START_YEAR = 2015
END_YEAR = 2026

PATHS = get_paths()
DATA_DIR = PATHS["data_dir"]


def download_file(url, filename):
    filepath = DATA_DIR / filename
    print(f"Downloading {filename} -> {filepath}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            print("Success.")
            return True

        print(f"Failed: Status {response.status_code} at {url}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    migration = ensure_data_layout()
    moved = migration.get("moved", [])
    renamed = migration.get("renamed_dup", [])
    if moved or renamed:
        print(f"[DATA_LAYOUT] moved={len(moved)} renamed_dup={len(renamed)}")

    print("Starting Multi-Source Data Download...")
    success_count = 0

    for source in SOURCES:
        if source.get("is_static"):
            url = source["url_fmt"]
            filename = source["filename_fmt"]
            if download_file(url, filename):
                success_count += 1
            continue

        for year in range(START_YEAR, END_YEAR + 1):
            url = source["url_fmt"].format(year)
            filename = source["filename_fmt"].format(year)

            if download_file(url, filename):
                success_count += 1

    print(f"\nDownload complete. Successfully downloaded {success_count} files.")


if __name__ == "__main__":
    main()
