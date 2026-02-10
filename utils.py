import pandas as pd
import numpy as np
from unidecode import unidecode
from thefuzz import process, fuzz
import datetime

def standardize_dates(series, source_type='sackmann'):
    """
    Standardize dates to datetime objects.
    dataset: 'sackmann' (YYYYMMDD) or 'tennis_data' (dd/mm/yyyy).
    """
    if source_type == 'sackmann':
        return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')
    elif source_type == 'tennis_data':
        # European format typical of Tennis-Data.co.uk
        return pd.to_datetime(series, dayfirst=True, errors='coerce')
    else:
        return pd.to_datetime(series, errors='coerce')

import json
import os

class DataNormalizer:
    def __init__(self, valid_names, cache_file='player_mapping.json'):
        """
        valid_names: List of "Nome Cognome" keys from training set.
        cache_file: Path to JSON file for persistent caching.
        """
        self.valid_names = set(valid_names)
        self.valid_names_clean = {self.clean_string(n): n for n in valid_names}
        self.valid_name_choices = list(self.valid_names_clean.keys())
        self.alias_map = self._build_alias_map(valid_names)
        self.cache_file = cache_file
        self.cache = self.load_cache()
        
    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_to_cache(self, key, value):
        if value is None:
            return
        self.cache[key] = value
        # Append/Write to file (simple approach: rewrite file)
        # For localized usage, this is acceptable.
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=4)
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")

    def clean_string(self, s):
        if pd.isna(s): return ""
        s = unidecode(str(s)).lower()
        s = s.replace('.', '').replace('-', ' ').replace("'", " ")
        s = " ".join(s.split())
        return s

    def _build_alias_map(self, valid_names):
        # Tennis-Data names are often "surname initial" (e.g. "sinner j").
        # Build a deterministic alias index to avoid broad fuzzy mistakes.
        alias_candidates = {}
        for full_name in valid_names:
            cleaned = self.clean_string(full_name)
            tokens = cleaned.split()
            if len(tokens) < 2:
                continue

            # Alias 1: "<full surname> <first initial>"
            first_name = tokens[0]
            full_surname = " ".join(tokens[1:])
            alias_candidates.setdefault(f"{full_surname} {first_name[0]}", set()).add(full_name)

            # Alias 2: "<last surname token> <all given-name initials>"
            last_surname = tokens[-1]
            given_tokens = tokens[:-1]
            given_initials = "".join(t[0] for t in given_tokens if t)
            if given_initials:
                alias_candidates.setdefault(f"{last_surname} {given_initials}", set()).add(full_name)
                alias_candidates.setdefault(f"{last_surname} {given_initials[0]}", set()).add(full_name)

            # Alias 3: "<last two surname tokens> <given initials>" for compound surnames
            if len(tokens) >= 3:
                compound_surname = " ".join(tokens[-2:])
                given_tokens_2 = tokens[:-2]
                if given_tokens_2:
                    given_initials_2 = "".join(t[0] for t in given_tokens_2 if t)
                    if given_initials_2:
                        alias_candidates.setdefault(f"{compound_surname} {given_initials_2}", set()).add(full_name)
                        alias_candidates.setdefault(f"{compound_surname} {given_initials_2[0]}", set()).add(full_name)

        alias_map = {}
        for alias, names in alias_candidates.items():
            if len(names) == 1:
                alias_map[alias] = next(iter(names))
        return alias_map

    def _is_compatible_mapping(self, short_cleaned, full_name):
        full_cleaned = self.clean_string(full_name)
        if not short_cleaned or not full_cleaned:
            return False
        if short_cleaned == full_cleaned:
            return True

        short_tokens = short_cleaned.split()
        full_tokens = full_cleaned.split()
        if len(short_tokens) < 2 or len(full_tokens) < 2:
            return False

        short_initials = short_tokens[-1]
        short_surname = " ".join(short_tokens[:-1])
        if not short_initials.isalpha() or len(short_initials) > 3:
            return False

        candidates = []
        # Treat last token as surname
        candidates.append((full_tokens[:-1], full_tokens[-1]))
        # Treat last two tokens as compound surname
        if len(full_tokens) >= 3:
            candidates.append((full_tokens[:-2], " ".join(full_tokens[-2:])))

        for given_tokens, surname in candidates:
            if not given_tokens:
                continue
            if short_surname != surname:
                continue

            initials = "".join(t[0] for t in given_tokens if t)
            if short_initials == initials:
                return True
            if len(short_initials) == 1 and initials.startswith(short_initials):
                return True
            if len(initials) == 1 and short_initials.startswith(initials):
                return True

            # Handle compact given names like "soonwoo" -> "sw"
            if len(given_tokens) == 1 and len(short_initials) > 1:
                name = given_tokens[0]
                i = 0
                for ch in name:
                    if i < len(short_initials) and ch == short_initials[i]:
                        i += 1
                if i == len(short_initials):
                    return True

        return False

    def convert_name(self, short_name):
        cleaned = self.clean_string(short_name)
        if not cleaned: return None
        
        # 1. Cache Lookup (O(1))
        if cleaned in self.cache:
            cached = self.cache[cleaned]
            if cached in self.valid_names and self._is_compatible_mapping(cleaned, cached):
                return cached
        
        # 2. Exact Match
        if cleaned in self.valid_names_clean:
            res = self.valid_names_clean[cleaned]
            self.save_to_cache(cleaned, res)
            return res

        # 2b. Deterministic alias map (surname + first initial)
        if cleaned in self.alias_map:
            res = self.alias_map[cleaned]
            self.save_to_cache(cleaned, res)
            return res
            
        # 3. Fuzzy Match with stricter threshold
        best = process.extractOne(cleaned, self.valid_name_choices, scorer=fuzz.token_set_ratio)
        if not best:
            return None

        best_match_clean, score = best
        
        if score > 70:
             print(f"[DEBUG_NORM] '{short_name}' -> '{self.valid_names_clean[best_match_clean]}' (Score: {score})")
        
        if score >= 92:
            full_name = self.valid_names_clean[best_match_clean]
            self.save_to_cache(cleaned, full_name)
            return full_name
            
        return None
