import pandas as pd
import numpy as np
from collections import defaultdict, deque
from datetime import timedelta
import tqdm

from paths import ensure_data_layout, get_paths


PATHS = get_paths()
DATA_DIR = PATHS["data_dir"]

# --- ELO CONFIG ---
START_RATING = 1500
K_FACTORS = {
    "G": 32,  # Grand Slam
    "M": 32,  # Masters
    "A": 28,  # ATP 250/500
    "C": 24,  # Challenger
    "F": 20,  # Futures
}
WEIGHTS = {"A": 1.0, "C": 0.7, "F": 0.5}


# --- V12 CONFIG ---
INACTIVITY_DEFAULT_DAYS = 30
CONFIDENCE_WINDOW = 12
CONFIDENCE_MIN_VALID = 3


def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating_a, rating_b, actual_a, k):
    expected_a = calculate_expected_score(rating_a, rating_b)
    new_a = rating_a + k * (actual_a - expected_a)
    new_b = rating_b + k * ((1 - actual_a) - (1 - expected_a))
    return new_a, new_b


def inactivity_bucket(days_since_last_match):
    d = float(days_since_last_match)
    if d <= 14:
        return 0
    if d <= 30:
        return 1
    if d <= 60:
        return 2
    if d <= 90:
        return 3
    return 4


def weighted_linear_mean(values):
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    w = np.arange(1, len(arr) + 1, dtype=float)
    return float(np.dot(arr, w) / w.sum())


def get_level_flags(level):
    lvl = str(level).upper() if pd.notna(level) else ""
    is_a = int(lvl == "A")
    is_c = int(lvl == "C")
    is_f = int(lvl == "F")
    is_cf = int(is_c or is_f)
    return {
        "is_level_a": is_a,
        "is_level_c": is_c,
        "is_level_f": is_f,
        "is_level_cf": is_cf,
    }


def process_matches(df):
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    df = df.sort_values(["tourney_date", "tourney_id", "match_num"])

    # ELO state
    elo_hard = defaultdict(lambda: START_RATING)
    elo_clay = defaultdict(lambda: START_RATING)
    elo_grass = defaultdict(lambda: START_RATING)
    elo_carpet = defaultdict(lambda: START_RATING)

    def get_dict(surface):
        s = str(surface).lower()
        if "clay" in s:
            return elo_clay
        if "grass" in s:
            return elo_grass
        if "carpet" in s:
            return elo_carpet
        return elo_hard

    # Rolling state
    player_history = defaultdict(list)
    fatigue_tracker = defaultdict(list)  # player -> list[(date, minutes)]
    h2h_tracker = defaultdict(lambda: defaultdict(int))

    # V12: inactivity/preparation state
    last_match_date = {}  # player -> last date
    recent_dates_30d = defaultdict(deque)  # player -> deque[date], pruned on read

    # V12: confidence ranking-aware state
    # each record: {"upset_win_flag", "bad_loss_flag", "quality_contrib"}
    recent_ranked_results = defaultdict(lambda: deque(maxlen=CONFIDENCE_WINDOW))

    features_list = []

    groups = df.groupby(["tourney_id", "match_num"], sort=False)
    print(f"Processing {len(groups)} matches features V12...")

    def calc_fatigue(player_name, current_date):
        hist = fatigue_tracker[player_name]
        cutoff = current_date - timedelta(days=3)
        recent_mins = [m for d, m in hist if d >= cutoff and d < current_date]
        return float(sum(recent_mins))

    def get_rolling(history, window=50):
        if not history:
            return {}
        recent = history[-window:]

        stats = {}

        # 1) Lefty performance
        vs_lefty = [m for m in recent if m["opponent_hand"] == "L"]
        if len(vs_lefty) >= 5:
            wins = sum(1 for m in vs_lefty if m["won"] == 1)
            stats["win_pct_vs_lefties"] = wins / len(vs_lefty)
        else:
            general_wins = sum(1 for m in recent if m["won"] == 1)
            stats["win_pct_vs_lefties"] = general_wins / len(recent) if recent else 0.5

        # 2) Decider performance
        deciders = [m for m in recent if m["is_decider"] == 1]
        if len(deciders) >= 3:
            d_wins = sum(1 for m in deciders if m["won"] == 1)
            stats["decider_win_pct"] = d_wins / len(deciders)
        else:
            stats["decider_win_pct"] = 0.5

        # 3) Weighted form
        w_sum = sum(m["weight"] * m["won"] for m in recent)
        stats["weighted_win_pct"] = w_sum / len(recent) if recent else 0.0

        # 4) Basic serve stats
        svpt = sum(m["svpt"] for m in recent)
        if svpt > 0:
            stats["ace_pct"] = sum(m["ace"] for m in recent) / svpt
            first_in = sum(m["1stIn"] for m in recent)
            stats["1st_won_pct"] = sum(m["1stWon"] for m in recent) / first_in if first_in > 0 else 0.0
        else:
            stats["ace_pct"] = 0.05
            stats["1st_won_pct"] = 0.70

        return stats

    def get_inactivity_features(player_name, current_date):
        if player_name not in last_match_date:
            return {
                "days_since_last_match": float(INACTIVITY_DEFAULT_DAYS),
                "inactive_bucket": float(inactivity_bucket(INACTIVITY_DEFAULT_DAYS)),
                "matches_last_30d": 0.0,
                "long_stop_90d": 0.0,
                "no_prev_match": 1.0,
            }

        prev_date = last_match_date[player_name]
        days_since = max(0, int((current_date - prev_date).days))

        dq = recent_dates_30d[player_name]
        cutoff = current_date - timedelta(days=30)
        while dq and dq[0] < cutoff:
            dq.popleft()

        return {
            "days_since_last_match": float(days_since),
            "inactive_bucket": float(inactivity_bucket(days_since)),
            "matches_last_30d": float(len(dq)),
            "long_stop_90d": float(int(days_since > 90)),
            "no_prev_match": 0.0,
        }

    def get_confidence_features(player_name):
        records = list(recent_ranked_results[player_name])
        n_valid = len(records)
        if n_valid < CONFIDENCE_MIN_VALID:
            return {
                "upset_win_rate_12": 0.0,
                "bad_loss_rate_12": 0.0,
                "confidence_rank_score_12": 0.0,
                "confidence_n_valid_12": float(n_valid),
            }

        upset_vals = [r["upset_win_flag"] for r in records]
        bad_loss_vals = [r["bad_loss_flag"] for r in records]
        quality_vals = [r["quality_contrib"] for r in records]

        return {
            "upset_win_rate_12": weighted_linear_mean(upset_vals),
            "bad_loss_rate_12": weighted_linear_mean(bad_loss_vals),
            "confidence_rank_score_12": weighted_linear_mean(quality_vals),
            "confidence_n_valid_12": float(n_valid),
        }

    def maybe_push_ranked_result(player_name, won, self_rank_raw, opp_rank_raw):
        if pd.isna(self_rank_raw) or pd.isna(opp_rank_raw):
            return
        if self_rank_raw <= 0 or opp_rank_raw <= 0:
            return

        rank_gap = float(np.log(float(self_rank_raw)) - np.log(float(opp_rank_raw)))
        upset_win_flag = float(int(won == 1 and rank_gap > 0))
        bad_loss_flag = float(int(won == 0 and rank_gap < 0))
        quality_contrib = max(rank_gap, 0.0) if won == 1 else -max(-rank_gap, 0.0)

        recent_ranked_results[player_name].append(
            {
                "upset_win_flag": upset_win_flag,
                "bad_loss_flag": bad_loss_flag,
                "quality_contrib": float(quality_contrib),
            }
        )

    for (_, _), match_rows in tqdm.tqdm(groups):
        if len(match_rows) != 2:
            continue

        row1 = match_rows.iloc[0]
        p1 = row1["p1_name"]
        p2 = row1["p2_name"]
        date = row1["tourney_date"]
        surface = row1["surface"]
        level = row1["match_level"]
        tourney_lvl = row1["tourney_level"]

        # --- 1) Pre-match metrics ---
        # Elo
        elo_dict = get_dict(surface)
        elo1 = float(elo_dict[p1])
        elo2 = float(elo_dict[p2])

        # Ranking (raw for V12 confidence, filled for model stability)
        r1_raw = row1.get("p1_rank", np.nan)
        r2_raw = row1.get("p2_rank", np.nan)
        r1 = r1_raw
        r2 = r2_raw
        if pd.isna(r1) or r1 <= 0:
            r1 = 1500
        if pd.isna(r2) or r2 <= 0:
            r2 = 1500
        log_rank_diff = float(np.log(r1) - np.log(r2))

        # H2H
        h2h_key = tuple(sorted([p1, p2]))
        h2h_stats = h2h_tracker[h2h_key]
        p1_h2h = float(h2h_stats[p1])
        p2_h2h = float(h2h_stats[p2])
        h2h_diff = float(p1_h2h - p2_h2h)

        # Surface encoding
        surf_map = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}
        surface_val = float(surf_map.get(surface, 0))

        # Fatigue
        fatigue1 = calc_fatigue(p1, date)
        fatigue2 = calc_fatigue(p2, date)

        # Rolling
        stats1 = get_rolling(player_history[p1])
        stats2 = get_rolling(player_history[p2])

        # V12 inactivity/preparation
        prep1 = get_inactivity_features(p1, date)
        prep2 = get_inactivity_features(p2, date)

        # V12 confidence
        conf1 = get_confidence_features(p1)
        conf2 = get_confidence_features(p2)

        # Level flags
        level_flags = get_level_flags(level)

        # Common row props
        base_match = {
            "match_id": f"{row1['tourney_id']}_{row1['match_num']}",
            "date": date,
            "match_level": level,
            "p1_name": p1,
            "p2_name": p2,
            "target": int(row1["target"]),
        }

        # --- Row P1 vs P2 ---
        f1 = base_match.copy()

        # Existing core features
        f1["surface_elo_p1"] = elo1
        f1["surface_elo_p2"] = elo2
        f1["surface_elo_diff"] = elo1 - elo2
        f1["log_rank_diff"] = log_rank_diff
        f1["h2h_diff"] = h2h_diff
        f1["surface_val"] = surface_val
        f1["fatigue_p1"] = fatigue1
        f1["fatigue_p2"] = fatigue2
        f1["fatigue_diff"] = fatigue1 - fatigue2

        p2_hand = row1["p2_hand"]
        if p2_hand == "L":
            f1["p1_vs_hand_win_pct"] = stats1.get("win_pct_vs_lefties", 0.5)
        else:
            f1["p1_vs_hand_win_pct"] = stats1.get("weighted_win_pct", 0.5)

        f1["p1_decider_win_pct"] = stats1.get("decider_win_pct", 0.5)
        f1["p2_decider_win_pct"] = stats2.get("decider_win_pct", 0.5)
        f1["p1_ace_pct"] = stats1.get("ace_pct", 0.0)
        f1["p2_ace_pct"] = stats2.get("ace_pct", 0.0)
        f1["p1_1st_won_pct"] = stats1.get("1st_won_pct", 0.0)
        f1["p2_1st_won_pct"] = stats2.get("1st_won_pct", 0.0)

        # V12 level flags
        for k, v in level_flags.items():
            f1[k] = float(v)

        # V12 inactivity/preparation P1/P2
        f1["p1_days_since_last_match"] = prep1["days_since_last_match"]
        f1["p2_days_since_last_match"] = prep2["days_since_last_match"]
        f1["p1_inactive_bucket"] = prep1["inactive_bucket"]
        f1["p2_inactive_bucket"] = prep2["inactive_bucket"]
        f1["p1_matches_last_30d"] = prep1["matches_last_30d"]
        f1["p2_matches_last_30d"] = prep2["matches_last_30d"]
        f1["p1_long_stop_90d"] = prep1["long_stop_90d"]
        f1["p2_long_stop_90d"] = prep2["long_stop_90d"]
        f1["p1_no_prev_match"] = prep1["no_prev_match"]
        f1["p2_no_prev_match"] = prep2["no_prev_match"]

        f1["days_since_last_diff"] = f1["p1_days_since_last_match"] - f1["p2_days_since_last_match"]
        f1["inactive_bucket_diff"] = f1["p1_inactive_bucket"] - f1["p2_inactive_bucket"]
        f1["matches_last_30d_diff"] = f1["p1_matches_last_30d"] - f1["p2_matches_last_30d"]
        f1["long_stop_90d_diff"] = f1["p1_long_stop_90d"] - f1["p2_long_stop_90d"]
        f1["no_prev_match_diff"] = f1["p1_no_prev_match"] - f1["p2_no_prev_match"]
        f1["days_since_last_diff_cf"] = f1["days_since_last_diff"] * f1["is_level_cf"]
        f1["long_stop_90d_diff_cf"] = f1["long_stop_90d_diff"] * f1["is_level_cf"]

        # V12 confidence P1/P2
        f1["p1_upset_win_rate_12"] = conf1["upset_win_rate_12"]
        f1["p2_upset_win_rate_12"] = conf2["upset_win_rate_12"]
        f1["p1_bad_loss_rate_12"] = conf1["bad_loss_rate_12"]
        f1["p2_bad_loss_rate_12"] = conf2["bad_loss_rate_12"]
        f1["p1_confidence_rank_score_12"] = conf1["confidence_rank_score_12"]
        f1["p2_confidence_rank_score_12"] = conf2["confidence_rank_score_12"]
        f1["p1_confidence_n_valid_12"] = conf1["confidence_n_valid_12"]
        f1["p2_confidence_n_valid_12"] = conf2["confidence_n_valid_12"]

        f1["upset_win_rate_12_diff"] = f1["p1_upset_win_rate_12"] - f1["p2_upset_win_rate_12"]
        f1["bad_loss_rate_12_diff"] = f1["p1_bad_loss_rate_12"] - f1["p2_bad_loss_rate_12"]
        f1["confidence_rank_score_12_diff"] = f1["p1_confidence_rank_score_12"] - f1["p2_confidence_rank_score_12"]
        f1["confidence_n_valid_12_diff"] = f1["p1_confidence_n_valid_12"] - f1["p2_confidence_n_valid_12"]
        f1["confidence_rank_score_12_diff_cf"] = f1["confidence_rank_score_12_diff"] * f1["is_level_cf"]

        features_list.append(f1)

        # --- Row P2 vs P1 (symmetrical) ---
        f2 = base_match.copy()
        f2["p1_name"] = p2
        f2["p2_name"] = p1
        f2["target"] = 1 - int(row1["target"])

        f2["surface_elo_p1"] = elo2
        f2["surface_elo_p2"] = elo1
        f2["surface_elo_diff"] = elo2 - elo1
        f2["log_rank_diff"] = -log_rank_diff
        f2["h2h_diff"] = -h2h_diff
        f2["surface_val"] = surface_val
        f2["fatigue_p1"] = fatigue2
        f2["fatigue_p2"] = fatigue1
        f2["fatigue_diff"] = fatigue2 - fatigue1

        p1_hand = row1["p1_hand"]
        if p1_hand == "L":
            f2["p1_vs_hand_win_pct"] = stats2.get("win_pct_vs_lefties", 0.5)
        else:
            f2["p1_vs_hand_win_pct"] = stats2.get("weighted_win_pct", 0.5)

        f2["p1_decider_win_pct"] = stats2.get("decider_win_pct", 0.5)
        f2["p2_decider_win_pct"] = stats1.get("decider_win_pct", 0.5)
        f2["p1_ace_pct"] = stats2.get("ace_pct", 0.0)
        f2["p2_ace_pct"] = stats1.get("ace_pct", 0.0)
        f2["p1_1st_won_pct"] = stats2.get("1st_won_pct", 0.0)
        f2["p2_1st_won_pct"] = stats1.get("1st_won_pct", 0.0)

        for k, v in level_flags.items():
            f2[k] = float(v)

        f2["p1_days_since_last_match"] = prep2["days_since_last_match"]
        f2["p2_days_since_last_match"] = prep1["days_since_last_match"]
        f2["p1_inactive_bucket"] = prep2["inactive_bucket"]
        f2["p2_inactive_bucket"] = prep1["inactive_bucket"]
        f2["p1_matches_last_30d"] = prep2["matches_last_30d"]
        f2["p2_matches_last_30d"] = prep1["matches_last_30d"]
        f2["p1_long_stop_90d"] = prep2["long_stop_90d"]
        f2["p2_long_stop_90d"] = prep1["long_stop_90d"]
        f2["p1_no_prev_match"] = prep2["no_prev_match"]
        f2["p2_no_prev_match"] = prep1["no_prev_match"]

        f2["days_since_last_diff"] = f2["p1_days_since_last_match"] - f2["p2_days_since_last_match"]
        f2["inactive_bucket_diff"] = f2["p1_inactive_bucket"] - f2["p2_inactive_bucket"]
        f2["matches_last_30d_diff"] = f2["p1_matches_last_30d"] - f2["p2_matches_last_30d"]
        f2["long_stop_90d_diff"] = f2["p1_long_stop_90d"] - f2["p2_long_stop_90d"]
        f2["no_prev_match_diff"] = f2["p1_no_prev_match"] - f2["p2_no_prev_match"]
        f2["days_since_last_diff_cf"] = f2["days_since_last_diff"] * f2["is_level_cf"]
        f2["long_stop_90d_diff_cf"] = f2["long_stop_90d_diff"] * f2["is_level_cf"]

        f2["p1_upset_win_rate_12"] = conf2["upset_win_rate_12"]
        f2["p2_upset_win_rate_12"] = conf1["upset_win_rate_12"]
        f2["p1_bad_loss_rate_12"] = conf2["bad_loss_rate_12"]
        f2["p2_bad_loss_rate_12"] = conf1["bad_loss_rate_12"]
        f2["p1_confidence_rank_score_12"] = conf2["confidence_rank_score_12"]
        f2["p2_confidence_rank_score_12"] = conf1["confidence_rank_score_12"]
        f2["p1_confidence_n_valid_12"] = conf2["confidence_n_valid_12"]
        f2["p2_confidence_n_valid_12"] = conf1["confidence_n_valid_12"]

        f2["upset_win_rate_12_diff"] = f2["p1_upset_win_rate_12"] - f2["p2_upset_win_rate_12"]
        f2["bad_loss_rate_12_diff"] = f2["p1_bad_loss_rate_12"] - f2["p2_bad_loss_rate_12"]
        f2["confidence_rank_score_12_diff"] = f2["p1_confidence_rank_score_12"] - f2["p2_confidence_rank_score_12"]
        f2["confidence_n_valid_12_diff"] = f2["p1_confidence_n_valid_12"] - f2["p2_confidence_n_valid_12"]
        f2["confidence_rank_score_12_diff_cf"] = f2["confidence_rank_score_12_diff"] * f2["is_level_cf"]

        features_list.append(f2)

        # --- 2) Update state post-match ---
        winner_name = p1 if int(row1["target"]) == 1 else p2
        h2h_tracker[h2h_key][winner_name] += 1

        kf = K_FACTORS.get(tourney_lvl, 24)
        result_p1 = int(row1["target"])
        new_elo1, new_elo2 = update_elo(elo1, elo2, result_p1, kf)
        elo_dict[p1] = new_elo1
        elo_dict[p2] = new_elo2

        mins = row1["minutes"] if not pd.isna(row1["minutes"]) else 90
        fatigue_tracker[p1].append((date, mins))
        fatigue_tracker[p2].append((date, mins))

        weight = WEIGHTS.get(level, 0.5)
        is_decider_val = int(row1["is_decider"]) if pd.notna(row1["is_decider"]) else 0

        h1_new = {
            "won": result_p1,
            "weight": weight,
            "opponent_hand": row1["p2_hand"],
            "is_decider": is_decider_val,
            "svpt": row1["p1_svpt"] if pd.notna(row1["p1_svpt"]) else 0,
            "ace": row1["p1_ace"] if pd.notna(row1["p1_ace"]) else 0,
            "1stIn": row1["p1_1stIn"] if pd.notna(row1["p1_1stIn"]) else 0,
            "1stWon": row1["p1_1stWon"] if pd.notna(row1["p1_1stWon"]) else 0,
        }
        player_history[p1].append(h1_new)

        h2_new = {
            "won": 1 - result_p1,
            "weight": weight,
            "opponent_hand": row1["p1_hand"],
            "is_decider": is_decider_val,
            "svpt": row1["p2_svpt"] if pd.notna(row1["p2_svpt"]) else 0,
            "ace": row1["p2_ace"] if pd.notna(row1["p2_ace"]) else 0,
            "1stIn": row1["p2_1stIn"] if pd.notna(row1["p2_1stIn"]) else 0,
            "1stWon": row1["p2_1stWon"] if pd.notna(row1["p2_1stWon"]) else 0,
        }
        player_history[p2].append(h2_new)

        # V12 inactivity state update
        last_match_date[p1] = date
        last_match_date[p2] = date
        recent_dates_30d[p1].append(date)
        recent_dates_30d[p2].append(date)

        # V12 confidence state update
        maybe_push_ranked_result(p1, result_p1, r1_raw, r2_raw)
        maybe_push_ranked_result(p2, 1 - result_p1, r2_raw, r1_raw)

    feat_df = pd.DataFrame(features_list)
    processed_csv = DATA_DIR / "processed_features.csv"
    feat_df.to_csv(processed_csv, index=False)
    print(f"Saved {processed_csv}")

    try:
        processed_parquet = DATA_DIR / "processed_features.parquet"
        feat_df.to_parquet(processed_parquet, index=False)
        print(f"Saved {processed_parquet}")
    except Exception as e:
        print(f"[WARN] Could not save processed_features.parquet: {e}")

    # Snapshot latest stats for app
    print("Snapshotting latest stats V12...")
    latest_rows = []
    all_players = set(player_history.keys())

    for p in all_players:
        row = {"player": p}
        row["elo_hard"] = elo_hard[p]
        row["elo_clay"] = elo_clay[p]
        row["elo_grass"] = elo_grass[p]

        st_roll = get_rolling(player_history[p])
        for k, v in st_roll.items():
            row[f"stats_{k}"] = v

        latest_rows.append(row)

    latest_stats_path = DATA_DIR / "latest_stats.csv"
    pd.DataFrame(latest_rows).to_csv(latest_stats_path, index=False)
    print(f"Saved {latest_stats_path}")


if __name__ == "__main__":
    migration = ensure_data_layout()
    moved = migration.get("moved", [])
    renamed = migration.get("renamed_dup", [])
    if moved or renamed:
        print(f"[DATA_LAYOUT] moved={len(moved)} renamed_dup={len(renamed)}")

    clean_matches_path = DATA_DIR / "clean_matches.csv"
    df = pd.read_csv(clean_matches_path)
    process_matches(df)
