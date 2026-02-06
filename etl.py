import pandas as pd
import glob
import os

# Config
START_YEAR = 2015
END_YEAR = 2026

def load_data():
    all_files = []
    
    # Sources
    file_groups = [
        {"pattern": "atp_matches_{}.csv", "source_tag": "ATP"},
        {"pattern": "atp_matches_qual_chall_{}.csv", "source_tag": "CHALL"},
        {"pattern": "atp_matches_futures_{}.csv", "source_tag": "FUTURES"},
    ]
    
    df_list = []
    
    print("Loading data files...")
    for year in range(START_YEAR, END_YEAR + 1):
        for group in file_groups:
            filename = group["pattern"].format(year)
            if os.path.exists(filename):
                # Robust Loader Logic
                loaded = False
                for enc in ['utf-8', 'ISO-8859-1', 'cp1252', 'latin1']:
                    for sep in [',', ';']:
                        try:
                            df = pd.read_csv(filename, encoding=enc, sep=sep, low_memory=False, on_bad_lines='skip')
                            
                            # Validation: Check if we have expected columns (more than 1)
                            if len(df.columns) > 1:
                                # Clean columns
                                df.columns = df.columns.str.strip()
                                
                                # HEADER NORMALIZATION (European -> Sackmann)
                                header_map = {
                                    'Winner': 'winner_name', 'Loser': 'loser_name',
                                    'Date': 'tourney_date', 'Surface': 'surface',
                                    'Round': 'round', 'Series': 'tourney_level',
                                    'Tournament': 'tourney_name', 'B365W': 'B365W', 'B365L': 'B365L'
                                }
                                df = df.rename(columns=header_map)
    
                                
                                # FILL MISSING KEYS (for European Data)
                                if 'tourney_id' not in df.columns:
                                    df['tourney_id'] = np.nan
                                if 'match_num' not in df.columns:
                                    df['match_num'] = np.nan
                                    
                                # Synthesize tourney_id if missing
                                mask_no_id = df['tourney_id'].isna()
                                if mask_no_id.any():
                                    print(f"Synthesizing tourney_id for {mask_no_id.sum()} matches...")
                                    if 'tourney_name' not in df.columns: df['tourney_name'] = 'Unknown'
                                    df.loc[mask_no_id, 'tourney_name'] = df.loc[mask_no_id, 'tourney_name'].fillna('Unknown')
                                    df.loc[mask_no_id, 'tourney_id'] = 'eu_' + df.loc[mask_no_id, 'tourney_name'].astype(str).str.replace(' ', '_').str.lower()
                                    
                                # Synthesize match_num if missing
                                mask_no_num = df['match_num'].isna()
                                if mask_no_num.any():
                                    print(f"Synthesizing match_num for {mask_no_num.sum()} matches...")
                                    df.loc[mask_no_num, 'match_num'] = df[mask_no_num].groupby('tourney_id').cumcount() + 1
                                
                                df['source_file'] = filename
                                df_list.append(df)
                                print(f"Loaded {filename} (enc={enc}, sep='{sep}')")
                                loaded = True
                                break
                        except:
                            continue
                    if loaded:
                        break
                
                if not loaded:
                    print(f"Error reading {filename}: Failed with all encodings/separators.")
            
    if not df_list:
        raise FileNotFoundError("No match CSV files found!")
        
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Total raw rows: {len(full_df)}")
    return full_df

def load_players():
    # Load Handedness
    if os.path.exists("atp_players.csv"):
        print("Loading atp_players.csv...")
        players = pd.read_csv("atp_players.csv", usecols=['player_id', 'hand'])
        
        # Fill missing hands with 'R' (Right is majority)
        players['hand'] = players['hand'].fillna('R')
        players['hand'] = players['hand'].replace('U', 'R')
        
        return players
    else:
        print("Warning: atp_players.csv not found. Handedness will be missing.")
        return None

def restructure_data(df, players_df):
    print("Restructuring (Symmetry + Handedness)...")
    
    # 0. Deduplicate columns explicitly
    df = df.loc[:, ~df.columns.duplicated()]
    
    # 1. Merge Handedness (Winner & Loser)
    if players_df is not None:
        # Merge Winner Hand
        df = df.merge(players_df, left_on='winner_id', right_on='player_id', how='left')
        df = df.rename(columns={'hand': 'winner_hand'})
        df = df.drop(columns=['player_id'])
        
        # Merge Loser Hand
        df = df.merge(players_df, left_on='loser_id', right_on='player_id', how='left')
        df = df.rename(columns={'hand': 'loser_hand'})
        df = df.drop(columns=['player_id'])
        
        # Impute remaining NaNs
        df['winner_hand'] = df['winner_hand'].fillna('R')
        df['loser_hand'] = df['loser_hand'].fillna('R')
    else:
        df['winner_hand'] = 'R'
        df['loser_hand'] = 'R'
    
    # 2. Match Level Normalization
    def map_level(l):
        l = str(l).upper()
        if l in ['G', 'F', 'M', 'A', 'D']: return 'A'
        if l == 'C': return 'C'
        return 'F'
    
    df['match_level'] = df['tourney_level'].apply(map_level)
    
    # 3. Decider Set Detection
    # Logic: 
    # Best of 3: Decider is 3rd set.
    # Best of 5: Decider is 5th set.
    # JeffSackmann data has 'score' string e.g. "6-4 4-6 6-3" or "7-6(4) 6-2".
    # Best of 5 usually only for Grand Slams (tourney_level='G') and Davis Cup ('D').
    
    def is_decider(row):
        score = str(row['score'])
        if pd.isna(score) or score == 'nan': return 0
        
        # Simple set count based on spaces or hyphens?
        # Standard format: "6-4 6-3" (2 sets)
        # "6-4 4-6 6-3" (3 sets)
        # RET/W/O cases?
        if "RET" in score or "W/O" in score: return 0
        
        # Count sets by counting spaces + 1? No. "7-6(4)" has no space internal but space between sets.
        sets = score.strip().split()
        # Filter out junk like "RET"
        sets = [s for s in sets if s[0].isdigit()] 
        n_sets = len(sets)
        
        level = row['tourney_level']
        
        # Grand Slam / Davis => Best of 5
        if level in ['G', 'D']:
            return 1 if n_sets == 5 else 0
        else:
            # Best of 3
            return 1 if n_sets == 3 else 0

    df['is_decider'] = df.apply(is_decider, axis=1)

    # 4. Symmetry Transformation
    # P1 = Winner
    w_to_p1 = {col: col.replace('winner_', 'p1_').replace('w_', 'p1_') for col in df.columns if 'winner_' in col or col.startswith('w_')}
    l_to_p2 = {col: col.replace('loser_', 'p2_').replace('l_', 'p2_') for col in df.columns if 'loser_' in col or col.startswith('l_')}
    
    df_win = df.copy().rename(columns=w_to_p1).rename(columns=l_to_p2)
    df_win = df_win.loc[:, ~df_win.columns.duplicated()]
    df_win['target'] = 1
    
    # P1 = Loser
    l_to_p1 = {col: col.replace('loser_', 'p1_').replace('l_', 'p1_') for col in df.columns if 'loser_' in col or col.startswith('l_')}
    w_to_p2 = {col: col.replace('winner_', 'p2_').replace('w_', 'p2_') for col in df.columns if 'winner_' in col or col.startswith('w_')}
    
    df_loss = df.copy().rename(columns=l_to_p1).rename(columns=w_to_p2)
    df_loss = df_loss.loc[:, ~df_loss.columns.duplicated()]
    df_loss['target'] = 0
    
    combined = pd.concat([df_win, df_loss], ignore_index=True)
    return combined

def clean_data(df):
    print("Cleaning data...")
    
    # Robust Date Parsing (Sackmann YYYYMMDD vs European DD/MM/YYYY)
    dates = df['tourney_date'].astype(str)
    
    # 1. Try Sackmann Format
    d1 = pd.to_datetime(dates, format='%Y%m%d', errors='coerce')
    
    # 2. Try European/Standard Format
    d2 = pd.to_datetime(dates, dayfirst=True, errors='coerce')
    
    # Combine (prefer d1, fill with d2)
    df['tourney_date'] = d1.fillna(d2)
    
    # Drop NaNs
    df = df.dropna(subset=['p1_name', 'p2_name', 'tourney_date'])
    df = df.sort_values(by=['tourney_date', 'tourney_id', 'match_num']).reset_index(drop=True)
    
    # Impute minutes if missing (Step 1 requirement)
    # df['minutes']
    # If minutes NaN, sets_played * 40
    # We need sets_played. We calculated sets in is_decider but didn't save it. 
    # Let's do a quick apply.
    def estimate_minutes(row):
        if pd.notna(row['minutes']): return row['minutes']
        
        score = str(row['score'])
        if pd.isna(score): return 90 # Default
        sets = [s for s in score.split() if s and s[0].isdigit()]
        return len(sets) * 40
        
    df['minutes'] = df.apply(estimate_minutes, axis=1)
    
    return df

if __name__ == "__main__":
    raw = load_data()
    players = load_players()
    processed = restructure_data(raw, players)
    clean = clean_data(processed)
    
    print(f"Final dataset shape: {clean.shape}")
    clean.to_csv("clean_matches.csv", index=False)
    print("Saved clean_matches.csv")
