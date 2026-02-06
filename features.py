import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import timedelta
import tqdm

# --- ELO CONFIG ---
START_RATING = 1500
K_FACTORS = {
    'G': 32, # Grand Slam
    'M': 32, # Masters
    'A': 28, # ATP 250/500
    'C': 24, # Challenger
    'F': 20  # Futures
}
WEIGHTS = {'A': 1.0, 'C': 0.7, 'F': 0.5}

def calculate_expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating_a, rating_b, actual_a, k):
    expected_a = calculate_expected_score(rating_a, rating_b)
    new_a = rating_a + k * (actual_a - expected_a)
    new_b = rating_b + k * ((1 - actual_a) - (1 - expected_a))
    return new_a, new_b

# --- FEATURES SCRIPT ---

def process_matches(df):
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    df = df.sort_values(['tourney_date', 'tourney_id', 'match_num'])
    
    # ELO State Containers (Dict: Player -> Rating)
    elo_hard = defaultdict(lambda: START_RATING)
    elo_clay = defaultdict(lambda: START_RATING)
    elo_grass = defaultdict(lambda: START_RATING)
    # Generic for unknown surface? Use Hard as default or separate? 
    # Let's map Carpet to Hard? Or separate. JeffSackmann usually Carpet=Carpet.
    elo_carpet = defaultdict(lambda: START_RATING)
    
    def get_dict(surface):
        s = str(surface).lower()
        if 'clay' in s: return elo_clay
        if 'grass' in s: return elo_grass
        if 'carpet' in s: return elo_carpet
        return elo_hard # Default
    
    # History Container for Rolling Stats
    player_history = defaultdict(list)
    
    # FATIGUE Container: Player -> List of (Date, Minutes) tuples
    fatigue_tracker = defaultdict(list)
    
    # H2H Container: tuple(sorted_names) -> {p1_name: wins, p2_name: wins}
    h2h_tracker = defaultdict(lambda: defaultdict(int))
    
    features_list = []
    
    # We must iterate match by match strictly to update Elo correctly
    # Grouping by match_id (2 rows per match: P1-P2 and P2-P1)
    # Issue: We update Elo once per match.
    # Logic: Iterate unique matches?
    # Or iterate the symmetrical DF but only process "Primary" rows?
    # Symmetrical DF: Row 0 is P1 vs P2. Row 1 is P2 vs P1.
    # They have same date/id.
    # Group by (tourney_id, match_num)
    
    groups = df.groupby(['tourney_id', 'match_num'], sort=False)
    
    print(f"Processing {len(groups)} matches features V4...")
    
    for (tid, mnum), match_rows in tqdm.tqdm(groups):
        if len(match_rows) != 2: continue
        
        row1 = match_rows.iloc[0] # P1 vs P2
        
        p1 = row1['p1_name']
        p2 = row1['p2_name']
        date = row1['tourney_date']
        surface = row1['surface']
        level = row1['match_level'] # A, C, F used for WEIGHTS?
        tourney_lvl = row1['tourney_level'] # G, M, A, C used for K-FACTOR
        
        # --- 1. GET PRE-MATCH METRICS (Features) ---
        
        # A. ELO
        elo_dict = get_dict(surface)
        elo1 = elo_dict[p1]
        elo2 = elo_dict[p2]
        
        # B. RANKING
        # Sackmann has 'p1_rank' and 'p2_rank' columns (renamed from winner_rank in ETL maybe?)
        # Let's check headers. ETL renamed 'winner_rank' -> 'p1_rank' ? 
        # No, ETL restructure creates p1_rank/p2_rank.
        r1 = row1.get('p1_rank', np.nan)
        r2 = row1.get('p2_rank', np.nan)
         
        # Handle NaN or <= 0
        if pd.isna(r1) or r1 <= 0: r1 = 1500 
        if pd.isna(r2) or r2 <= 0: r2 = 1500
        
        # Log Rank Diff (P1 - P2) usually negative if P1 better
        log_rank_diff = np.log(r1) - np.log(r2)
        
        # C. H2H
        # Key: Sorted tuple
        h2h_key = tuple(sorted([p1, p2]))
        h2h_stats = h2h_tracker[h2h_key]
        p1_h2h = h2h_stats[p1]
        p2_h2h = h2h_stats[p2]
        h2h_diff = p1_h2h - p2_h2h
        
        # D. SURFACE ENCODING
        surf_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
        surface_val = surf_map.get(surface, 0)
        
        # E. FATIGUE (Last 3 Days)
        # Sum minutes where date > (match_date - 3 days) AND date < match_date
        def calc_fatigue(p_name, current_date):
            hist = fatigue_tracker[p_name]
            # Filter
            cutoff = current_date - timedelta(days=3)
            recent_mins = [m for d, m in hist if d >= cutoff and d < current_date]
            return sum(recent_mins)
            
        fatigue1 = calc_fatigue(p1, date)
        fatigue2 = calc_fatigue(p2, date)
        
        # E. LEFTY / DECIDER / FORM (Rolling)
        # Helper to calc rolling
        def get_rolling(history, current_date, window=50):
            if not history: return {}
            # Filter (Assuming sorted append)
            # Actually history is list of dicts.
            # Slice last 50
            recent = history[-window:]
            
            stats = {}
            
            # 1. Lefty Performance
            # Filter matches where opponent was 'L'
            vs_lefty = [m for m in recent if m['opponent_hand'] == 'L']
            if len(vs_lefty) >= 5:
                wins = sum(1 for m in vs_lefty if m['won'] == 1)
                stats['win_pct_vs_lefties'] = wins / len(vs_lefty)
            else:
                # Fallback to general win %
                general_wins = sum(1 for m in recent if m['won'] == 1)
                stats['win_pct_vs_lefties'] = general_wins / len(recent) if recent else 0.5

            # 2. Decider Performance
            # matches where is_decider=1
            deciders = [m for m in recent if m['is_decider'] == 1]
            if len(deciders) >= 3:
                d_wins = sum(1 for m in deciders if m['won'] == 1)
                stats['decider_win_pct'] = d_wins / len(deciders)
            else:
                stats['decider_win_pct'] = 0.5
                
            # 3. Form (Weighted) - Kept from V3
            w_sum = sum(m['weight'] * m['won'] for m in recent)
            # w_count = sum(m['weight'] for m in recent) # No, divide by match count
            stats['weighted_win_pct'] = w_sum / len(recent) if recent else 0
            
            # 4. Standard Serve/Return
            # ... kept simple for speed or V3 logic.
            # Let's keep minimal heavy logic here to ensure speed.
            # We can rely on Elo/Fatigue as main new drivers.
            # But we should keep the basic physical stats (Serve %)
            
            svpt = sum(m['svpt'] for m in recent)
            if svpt > 0:
                stats['ace_pct'] = sum(m['ace'] for m in recent) / svpt
                stats['1st_won_pct'] = sum(m['1stWon'] for m in recent) / sum(m['1stIn'] for m in recent) if sum(m['1stIn']for m in recent) > 0 else 0
            else:
                stats['ace_pct'] = 0.05
                stats['1st_won_pct'] = 0.70
                
            return stats

        stats1 = get_rolling(player_history[p1], date)
        stats2 = get_rolling(player_history[p2], date)
        
        # --- BUILD FEATURE ROWS ---
        
        # Common Row Props
        base_match = {
            'match_id': f"{tid}_{mnum}",
            'date': date,
            'match_level': level,
            'p1_name': p1,
            'p2_name': p2,
            'target': row1['target'], # P1 win?
        }
        
        # Features P1 vs P2
        f1 = base_match.copy()
        # Elo
        f1['surface_elo_p1'] = elo1
        f1['surface_elo_p2'] = elo2
        f1['surface_elo_diff'] = elo1 - elo2
        # Ranking + H2H
        f1['log_rank_diff'] = log_rank_diff
        f1['h2h_diff'] = h2h_diff
        f1['surface_val'] = surface_val
        # Fatigue
        f1['fatigue_p1'] = fatigue1
        f1['fatigue_p2'] = fatigue2
        f1['fatigue_diff'] = fatigue1 - fatigue2
        # Lefty (P1 vs P2's hand)
        # Check P2 hand
        p2_hand = row1['p2_hand'] # from ETL
        if p2_hand == 'L':
            f1['p1_vs_hand_win_pct'] = stats1.get('win_pct_vs_lefties', 0.5)
        else:
            f1['p1_vs_hand_win_pct'] = stats1.get('weighted_win_pct', 0.5)
            
        f1['p1_decider_win_pct'] = stats1.get('decider_win_pct', 0.5)
        f1['p2_decider_win_pct'] = stats2.get('decider_win_pct', 0.5)
        
        # Add basic stats
        f1['p1_ace_pct'] = stats1.get('ace_pct', 0)
        f1['p2_ace_pct'] = stats2.get('ace_pct', 0)
        f1['p1_1st_won_pct'] = stats1.get('1st_won_pct', 0)
        f1['p2_1st_won_pct'] = stats2.get('1st_won_pct', 0)
        
        features_list.append(f1)
        
        # Symmetrical Row (P2 vs P1)
        f2 = base_match.copy()
        f2['p1_name'] = p2
        f2['p2_name'] = p1
        f2['target'] = 1 - row1['target']
        
        f2['surface_elo_p1'] = elo2
        f2['surface_elo_p2'] = elo1
        f2['surface_elo_diff'] = elo2 - elo1 # Diff inverted
        
        f2['fatigue_p1'] = fatigue2
        f2['fatigue_p2'] = fatigue1
        f2['fatigue_diff'] = fatigue2 - fatigue1
        
        p1_hand = row1['p1_hand']
        if p1_hand == 'L':
             f2['p1_vs_hand_win_pct'] = stats2.get('win_pct_vs_lefties', 0.5)
        else:
             f2['p1_vs_hand_win_pct'] = stats2.get('weighted_win_pct', 0.5)
             
        f2['p1_decider_win_pct'] = stats2.get('decider_win_pct', 0.5)
        f2['p2_decider_win_pct'] = stats1.get('decider_win_pct', 0.5)
        
        f2['p1_ace_pct'] = stats2.get('ace_pct', 0)
        f2['p2_ace_pct'] = stats1.get('ace_pct', 0)
        f2['p1_1st_won_pct'] = stats2.get('1st_won_pct', 0)
        f2['p2_1st_won_pct'] = stats1.get('1st_won_pct', 0)
        
        # Ranking + H2H (Symmetrical)
        f2['log_rank_diff'] = -log_rank_diff
        f2['h2h_diff'] = -h2h_diff
        f2['surface_val'] = surface_val
        
        features_list.append(f2)
        
        # --- UPDATE STATE (Post-Match) ---
        # Update H2H Tracker for next matches
        winner_name = p1 if row1['target'] == 1 else p2
        h2h_tracker[h2h_key][winner_name] += 1

        # --- 2. UPDATE STATE (Post-Match) ---
        
        # K-Factor
        # G=32, A=28 etc.
        # Fallback 24 if unknown
        kf = K_FACTORS.get(tourney_lvl, 24)
        
        # P1 Result (row1['target']) 
        # If P1 won (1), P2 lost (0)
        result_p1 = row1['target']
        new_elo1, new_elo2 = update_elo(elo1, elo2, result_p1, kf)
        
        elo_dict[p1] = new_elo1
        elo_dict[p2] = new_elo2
        
        # Fatigue Update
        # Add match minutes to histories
        mins = row1['minutes'] if not pd.isna(row1['minutes']) else 90
        fatigue_tracker[p1].append((date, mins))
        fatigue_tracker[p2].append((date, mins))
        
        # History Update (for Rolling)
        # Needs: won, weight, opponent_hand, is_decider, ace, svpt...
        weight = WEIGHTS.get(level, 0.5)
        is_decider_val = row1['is_decider']
        
        # P1 Hist
        h1_new = {
            'won': row1['target'], 'weight': weight, 
            'opponent_hand': row1['p2_hand'], 'is_decider': is_decider_val,
            'svpt': row1['p1_svpt'] if pd.notna(row1['p1_svpt']) else 0,
            'ace': row1['p1_ace'] if pd.notna(row1['p1_ace']) else 0,
            '1stIn': row1['p1_1stIn'] if pd.notna(row1['p1_1stIn']) else 0,
            '1stWon': row1['p1_1stWon'] if pd.notna(row1['p1_1stWon']) else 0
        }
        player_history[p1].append(h1_new)
        
        # P2 Hist
        h2_new = {
            'won': 1 - row1['target'], 'weight': weight,
            'opponent_hand': row1['p1_hand'], 'is_decider': is_decider_val,
            'svpt': row1['p2_svpt'] if pd.notna(row1['p2_svpt']) else 0,
            'ace': row1['p2_ace'] if pd.notna(row1['p2_ace']) else 0,
            '1stIn': row1['p2_1stIn'] if pd.notna(row1['p2_1stIn']) else 0,
            '1stWon': row1['p2_1stWon'] if pd.notna(row1['p2_1stWon']) else 0
        }
        player_history[p2].append(h2_new)

    # Save
    feat_df = pd.DataFrame(features_list)
    feat_df.to_csv("processed_features.csv", index=False)
    print("Saved processed_features.csv")
    
    # --- SAVE LATEST STATS (For App) ---
    print("Snapshotting latest stats V4...")
    latest_rows = []
    fake_date = pd.Timestamp.now()
    
    # Iterate all known players
    # We need a list of all players seen
    all_players = set(player_history.keys())
    
    for p in all_players:
        row = {'player': p}
        
        # Elo
        row['elo_hard'] = elo_hard[p]
        row['elo_clay'] = elo_clay[p]
        row['elo_grass'] = elo_grass[p]
        
        # General Stats (Using Hard Elo as proxy for generic 'rank' logic if needed? No, store all)
        # Rolling
        st_roll = get_rolling(player_history[p], fake_date)
        for k, v in st_roll.items(): row[f"stats_{k}"] = v
        
        # Handedness?
        # We didn't store player's own hand in history or helper.
        # But we can infer it or load it?
        # App might need it.
        # For now, minimal payload.
        
        latest_rows.append(row)
        
    pd.DataFrame(latest_rows).to_csv("latest_stats.csv", index=False)
    print("Saved latest_stats.csv")

if __name__ == "__main__":
    df = pd.read_csv("clean_matches.csv")
    process_matches(df)
