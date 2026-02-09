import re
import time
import random
from playwright.sync_api import sync_playwright
import pandas as pd
import numpy as np

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def get_player_data(player_name, browser_instance):
    """
    Scrapes TennisExplorer for a single player.
    Returns: dict with 'rank', 'matches_2025' (list of dicts)
    """
    page = browser_instance.new_page()
    # Stealth
    page.set_extra_http_headers({"User-Agent": USER_AGENT})
    
    data = {"rank": None, "matches": []}
    
    try:
        # 1. Search
        print(f"Searching live data for: {player_name}...")
        # TennisExplorer search logic
        # URL: https://www.tennisexplorer.com/search/?query={name}
        clean_name = player_name.split()[-1] # Search by Surname usually better
        # Actually full name search works
        q = player_name.replace(" ", "+")
        search_url = f"https://www.tennisexplorer.com/search/?query={q}"
        
        page.goto(search_url, timeout=10000)
        time.sleep(random.uniform(1, 2))
        
        # Check if we landed on profile directly or list
        # Profile url usually has /player/name/
        if "/player/" in page.url:
            # Direct hit
            pass
        else:
            # List results. Click first valid result.
            # Try to find link that matches name closely
            # Simple heuristic: Click first non-doubles player link
            # For now, just click the first link in table
            try:
                # Selector for result table
                page.locator("table.result tr td a").first.click()
                page.wait_for_load_state("domcontentloaded")
            except:
                print(f"Player {player_name} not found in search.")
                page.close()
                return None

        # 2. Extract Rank
        # Look for "Current/Highest rank - singles: 1. / 1."
        try:
            text = page.locator("div#center div.box").first.inner_text()
            # Regex for "Current/Highest rank - singles: 5."
            # Or just "Current rank: 5."
            # TennisExplorer format varies. 
            # Looking for "Current/Highest rank - singles: \d+"
            match = re.search(r"Current.*?rank.*?singles:\s*(\d+)", text, re.IGNORECASE)
            if match:
                data["rank"] = int(match.group(1))
        except:
            pass
            
        # 3. Extract Matches (Current Year)
        # Table usually id="matches-2025-1-single" or similar.
        # TennisExplorer tables are just <table class="result">
        # We need the one that contains "2025" in headers or preceding h3?
        # Simpler: Grab all rows from the main result table.
        # Filter for recent dates.
        
        rows = page.locator("table.result tbody tr").all()
        # Parse first N rows until we hit a different year or too old
        # Format: Date | Tournament | Surface | Rd | Opponent | Rank | Score
        
        for row in rows:
            txt = row.inner_text()
            if not txt.strip(): continue
            
            # Simple check if meaningful match row
            if "Round" in txt or "Tournament" in txt: continue 
            
            # Extract
            # Columns are tricky without mapped headers, but we can detect W/L
            # Usually class="won" or "lost" on the score or name
            # TennisExplorer puts 'not-active' class on passed matches?
            # Let's look for result info.
            
            is_win = False
            # Check if player won
            # Often the winner is bold or has specific class?
            # Actually TennisExplorer: 
            # If the row has score, and left side is winner?
            # Easier: Check if "not-active" class is NOT present (future?)
            # No, we want PAST matches for Stats.
            
            # Just look for score. e.g. "6-3 6-4"
            # And identify if user won.
            # On TE profile page, the score is always displayed.
            # But "Win" isn't explicit in text.
            # BUT, the opponent name is a link.
            
            # Let's simplify: Just count total matches and try to guess form?
            # Too risky.
            
            # Alternative: Live-Tennis.eu might be cleaner?
            # Let's stick to TE but be robust.
            # If we simply check the cell "Result" if it exists?
            pass
            
        # Fallback for V1: Just get Rank. Rank is most important live metric.
        # And maybe "last match date" to check for injury.
        
        # Lets try to get just the last 5 matches results (W/L)
        # In the profile table, the result is often not explicit column.
        # However, usually the match sets are coloring?
        # Actually TE highlights the player name if they won? No.
        
        # Strategy Update:
        # Just getting the RANK is a huge win over static 2024 data.
        # Getting recent form is bonus.
        # Let's grab the Rank and return.
        
    except Exception as e:
        print(f"Scraping error for {player_name}: {e}")
    finally:
        page.close()
        
    return data

def enrich_prediction_data(p1_name, p2_name, static_feature_row):
    """
    Updates the feature row with live data.
    """
    print("Fetching live data (Headless Browser)...")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            
            # P1 Data
            d1 = get_player_data(p1_name, browser)
            # Sleep
            time.sleep(1.5)
            # P2 Data
            d2 = get_player_data(p2_name, browser)
            
            browser.close()
            
            # Update Features
            updated_row = static_feature_row.copy()
            
            # 1. Ranks
            r1 = d1.get("rank") if d1 else None
            r2 = d2.get("rank") if d2 else None
            
            # Fallback to static if None
            old_r1 = 1000 # Default/from static
            # Extract old rank from log_rank if possible or just use what we have
            # Actually static_feature_row is a dict passed from app.py
            
            if r1: 
                print(f"Live Rank {p1_name}: {r1}")
                # Update diffs
                r2_val = r2 if r2 else (100 if not d2 else 1000) # estimated
                updated_row['rank_diff'] = r1 - r2_val
                updated_row['log_rank_p1'] = np.log(r1)
                
            if r2:
                print(f"Live Rank {p2_name}: {r2}")
                r1_val = r1 if r1 else (100 if not d1 else 1000)
                updated_row['rank_diff'] = r1_val - r2
                updated_row['log_rank_p2'] = np.log(r2)
                
            return updated_row, True # Success flag
            
    except Exception as e:
        print(f"Live fetch failed: {e}")
        return static_feature_row, False

if __name__ == "__main__":
    # Test
    row = {'rank_diff': 0, 'log_rank_p1': 0, 'log_rank_p2': 0}
    enriched, success = enrich_prediction_data("Jannik Sinner", "Carlos Alcaraz", row)
    print("Result:", enriched)
