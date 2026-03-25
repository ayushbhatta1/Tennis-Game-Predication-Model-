"""
Player Identity Resolution — bridge API teamIDs to Sackmann player_ids.

API format: teamID = "JASMINE_PAOLINI_WTA", with names/birthday
Sackmann: player_id = "216347", with name_first/name_last/dob

Multi-pass matching:
  1. Exact first+last name match
  2. Normalized name match (strip diacritics, lowercase)
  3. Birthday match (when names fail)
  4. Fuzzy last name + first initial match
"""

import csv
import json
import os
import re
import unicodedata
from difflib import SequenceMatcher

BASE_DIR = os.path.dirname(__file__)
HIST_DIR = os.path.join(BASE_DIR, "historical")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
MODEL_DIR = os.path.join(BASE_DIR, "model")

MAPPING_FILE = os.path.join(MODEL_DIR, "player_mapping.json")


def strip_diacritics(s):
    """Remove diacritics and normalize unicode."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def normalize_name(name):
    """Normalize a name for matching: strip diacritics, hyphens, periods."""
    name = strip_diacritics(name)
    name = re.sub(r"[.\-']", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_sackmann_players():
    """Load all players from atp_players.csv and wta_players.csv."""
    players = {}  # player_id -> {first, last, hand, dob, ioc, height, league}
    for filename, league in [("atp_players.csv", "ATP"), ("wta_players.csv", "WTA")]:
        filepath = os.path.join(HIST_DIR, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("player_id", "").strip()
                if not pid:
                    continue
                players[pid] = {
                    "first": row.get("name_first", "").strip(),
                    "last": row.get("name_last", "").strip(),
                    "hand": row.get("hand", "").strip(),
                    "dob": row.get("dob", "").strip(),
                    "ioc": row.get("ioc", "").strip(),
                    "height": row.get("height", "").strip(),
                    "league": league,
                }
    return players


def load_api_teams():
    """Load teams from cached API data."""
    teams = {}  # teamID -> {first, last, birthday, league}
    for filename in ["all_teams_full.json", "all_teams.json"]:
        filepath = os.path.join(CACHE_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)
            for t in data:
                tid = t.get("teamID", "")
                if not tid:
                    continue
                names = t.get("names", {})
                sp = t.get("singlePlayer", {})
                teams[tid] = {
                    "first": names.get("firstName", ""),
                    "last": names.get("lastName", ""),
                    "long": names.get("long", ""),
                    "birthday": sp.get("birthday", ""),
                    "league": t.get("leagueID", ""),
                }
            break
    return teams


def build_name_index(sackmann_players):
    """Build lookup indices for Sackmann players."""
    # Exact name -> list of (pid, player)
    exact = {}
    # Normalized name -> list of (pid, player)
    normalized = {}
    # DOB -> list of (pid, player)
    by_dob = {}
    # Normalized last name -> list of (pid, player)
    by_last = {}

    for pid, p in sackmann_players.items():
        first = p["first"]
        last = p["last"]
        if not first or not last:
            continue

        # Exact
        key = f"{first.lower()}|{last.lower()}"
        exact.setdefault(key, []).append((pid, p))

        # Normalized
        nkey = f"{normalize_name(first)}|{normalize_name(last)}"
        normalized.setdefault(nkey, []).append((pid, p))

        # DOB
        dob = p["dob"]
        if dob and len(dob) >= 8:
            by_dob.setdefault(dob, []).append((pid, p))

        # Last name
        nlast = normalize_name(last)
        by_last.setdefault(nlast, []).append((pid, p))

    return exact, normalized, by_dob, by_last


def convert_birthday(api_bday):
    """Convert API birthday '2003-11-16' to Sackmann DOB format '20031116'."""
    if not api_bday:
        return ""
    return api_bday.replace("-", "")


def resolve_players():
    """Build mapping from API teamID -> Sackmann player_id."""
    print("Loading Sackmann players...")
    sackmann = load_sackmann_players()
    print(f"  {len(sackmann)} players")

    print("Loading API teams...")
    api_teams = load_api_teams()
    print(f"  {len(api_teams)} teams")

    exact_idx, norm_idx, dob_idx, last_idx = build_name_index(sackmann)

    mapping = {}  # teamID -> {"sackmann_id": str, "method": str}
    stats = {"exact": 0, "normalized": 0, "birthday": 0, "fuzzy": 0, "unmatched": 0}

    for tid, team in api_teams.items():
        first = team["first"]
        last = team["last"]
        bday = convert_birthday(team["birthday"])
        league = team["league"]

        matched_pid = None
        method = None

        # Pass 1: Exact name match (filter by league)
        key = f"{first.lower()}|{last.lower()}"
        if key in exact_idx:
            candidates = exact_idx[key]
            # Prefer same league
            same_league = [c for c in candidates if c[1]["league"] == league]
            if same_league:
                matched_pid = same_league[0][0]
                method = "exact"
            elif len(candidates) == 1:
                matched_pid = candidates[0][0]
                method = "exact"

        # Pass 2: Normalized name match
        if not matched_pid:
            nkey = f"{normalize_name(first)}|{normalize_name(last)}"
            if nkey in norm_idx:
                candidates = norm_idx[nkey]
                same_league = [c for c in candidates if c[1]["league"] == league]
                if same_league:
                    matched_pid = same_league[0][0]
                    method = "normalized"
                elif len(candidates) == 1:
                    matched_pid = candidates[0][0]
                    method = "normalized"

        # Pass 3: Birthday match (with last name confirmation)
        if not matched_pid and bday and bday in dob_idx:
            candidates = dob_idx[bday]
            # Confirm with fuzzy last name
            nlast = normalize_name(last)
            for pid, p in candidates:
                plast = normalize_name(p["last"])
                if plast == nlast or SequenceMatcher(None, plast, nlast).ratio() > 0.8:
                    matched_pid = pid
                    method = "birthday"
                    break
            # If only one candidate with same DOB, accept
            if not matched_pid and len(candidates) == 1:
                matched_pid = candidates[0][0]
                method = "birthday"

        # Pass 4: Fuzzy last name + first initial
        if not matched_pid:
            nlast = normalize_name(last)
            nfirst = normalize_name(first)
            if nlast in last_idx:
                candidates = last_idx[nlast]
                # Filter by first initial
                initial_matches = [
                    (pid, p) for pid, p in candidates
                    if normalize_name(p["first"])[:1] == nfirst[:1]
                    and p["league"] == league
                ]
                if len(initial_matches) == 1:
                    matched_pid = initial_matches[0][0]
                    method = "fuzzy"
                elif len(initial_matches) > 1 and bday:
                    # Try birthday tiebreaker
                    for pid, p in initial_matches:
                        if p["dob"] == bday:
                            matched_pid = pid
                            method = "fuzzy+birthday"
                            break

        if matched_pid:
            mapping[tid] = {
                "sackmann_id": matched_pid,
                "method": method,
                "name": f"{first} {last}",
            }
            stats[method.split("+")[0]] += 1
        else:
            stats["unmatched"] += 1

    # Save mapping
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nResolution stats:")
    for method, count in stats.items():
        print(f"  {method}: {count}")
    print(f"  Total mapped: {len(mapping)} / {len(api_teams)} ({len(mapping)/len(api_teams)*100:.1f}%)")
    print(f"Saved to {MAPPING_FILE}")

    return mapping


def load_mapping():
    """Load the player mapping from disk."""
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE) as f:
            return json.load(f)
    return {}


def get_sackmann_id(team_id, mapping=None):
    """Resolve an API teamID to a Sackmann player_id."""
    if mapping is None:
        mapping = load_mapping()
    entry = mapping.get(team_id)
    if entry:
        return entry["sackmann_id"]
    return None


if __name__ == "__main__":
    resolve_players()
