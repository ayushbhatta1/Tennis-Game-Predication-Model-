"""
Props Predictor — Underdog Fantasy player prop predictions.

Predicts Higher/Lower on 10 prop types:
  - Aces, Double Faults, Games Played, Games Won
  - 1st Set Games, 1st Set Games Won
  - Sets Played, Sets Won, Breakpoints Won, Tiebreakers

Architecture:
  1. Parse historical scores ("6-4 7-6(5)") for games/sets/tiebreaks
  2. Build per-player rolling prop averages from raw CSVs
  3. Predict expected values given matchup context (ELO, surface)
  4. Compare to Underdog lines → find edges
  5. Select optimal 8-leg parlay with correlation-aware diversification

Output: model/props_store.pkl
"""

import csv
import glob
import math
import os
import pickle
import re
from collections import defaultdict
from datetime import datetime

import numpy as np

BASE_DIR = os.path.dirname(__file__)
HIST_DIR = os.path.join(BASE_DIR, "historical")
MODEL_DIR = os.path.join(BASE_DIR, "model")

PROPS_STORE_FILE = os.path.join(MODEL_DIR, "props_store.pkl")
FEATURE_STORE_FILE = os.path.join(MODEL_DIR, "feature_store.pkl")

ROLLING_WINDOW = 30  # matches of history per player


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Score Parser
# ---------------------------------------------------------------------------

def parse_score(score_str):
    """Parse a tennis score string into structured data.

    Examples:
        "6-4 7-6(5) 6-2" → normal completed match
        "6-4 3-6 7-6(3)" → 3-setter
        "1-6 6-4 3-1 RET" → retirement (incomplete final set)
        "W/O" → walkover (no games played)

    Returns dict or None if unparseable / walkover.
    """
    if not score_str or not isinstance(score_str, str):
        return None

    score_str = score_str.strip()

    # Walkovers — no match played
    if "W/O" in score_str.upper() or "walkover" in score_str.lower():
        return None

    is_retirement = "RET" in score_str.upper() or "DEF" in score_str.upper()

    # Strip retirement/default markers
    clean = re.sub(r'\s*(RET|DEF|Def|ret|Retired|ABN|ABD|ABS|Abandoned|Default|Unfinished|In Progress|SUSP)\.?\s*$',
                   '', score_str, flags=re.IGNORECASE).strip()

    if not clean:
        return None

    # Split into sets
    set_parts = clean.split()
    if not set_parts:
        return None

    winner_games = 0
    loser_games = 0
    winner_sets = 0
    loser_sets = 0
    sets_played = 0
    tiebreaks_played = 0
    first_set_winner_games = 0
    first_set_loser_games = 0
    first_set_total = 0
    completed_sets = 0

    for i, part in enumerate(set_parts):
        # Match "6-4" or "7-6(5)" or "6-7(8)"
        m = re.match(r'^(\d+)-(\d+)(?:\((\d+)\))?$', part)
        if not m:
            continue

        w_g = int(m.group(1))
        l_g = int(m.group(2))
        tb = m.group(3)

        sets_played += 1
        winner_games += w_g
        loser_games += l_g

        if tb is not None:
            tiebreaks_played += 1

        # Determine who won this set
        if w_g > l_g:
            winner_sets += 1
            completed_sets += 1
        elif l_g > w_g:
            loser_sets += 1
            completed_sets += 1
        else:
            # Incomplete set (retirement mid-set) — don't count as completed
            # but still count games
            pass

        # First set data
        if i == 0:
            first_set_winner_games = w_g
            first_set_loser_games = l_g
            first_set_total = w_g + l_g

    if sets_played == 0:
        return None

    total_games = winner_games + loser_games

    return {
        "total_games": total_games,
        "winner_games": winner_games,
        "loser_games": loser_games,
        "sets_played": sets_played,
        "winner_sets": winner_sets,
        "loser_sets": loser_sets,
        "tiebreaks_played": tiebreaks_played,
        "first_set_total": first_set_total,
        "first_set_winner_games": first_set_winner_games,
        "first_set_loser_games": first_set_loser_games,
        "is_retirement": is_retirement,
        "completed_sets": completed_sets,
    }


# ---------------------------------------------------------------------------
# Props Store Builder
# ---------------------------------------------------------------------------

def load_all_matches():
    """Load all matches sorted by date (mirrors build_feature_store.py)."""
    matches = []
    for pattern in ["atp_matches_*.csv", "wta_matches_*.csv"]:
        files = sorted(glob.glob(os.path.join(HIST_DIR, pattern)))
        for filepath in files:
            if "futures" in filepath:
                continue
            league = "ATP" if "atp_" in filepath else "WTA"
            with open(filepath, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["_league"] = league
                    matches.append(row)

    matches.sort(key=lambda m: m.get("tourney_date", ""))
    return matches


def build_props_store():
    """Process all historical CSVs to build per-player rolling prop averages.

    For each player, stores last ROLLING_WINDOW matches with:
      - Raw serve counts (aces, DFs, svpt, bp_saved, bp_faced)
      - Opponent BP data (for breakpoints won calculation)
      - Parsed score data (games, sets, tiebreaks)
      - Context (surface, best_of, date, opponent, won)
    """
    print("Loading matches...")
    matches = load_all_matches()
    print(f"  {len(matches)} total matches")

    # Per-player rolling history: player_id -> list of match dicts
    player_history = defaultdict(list)

    # Aggregation accumulators for surface factors and game-total calibration
    surface_ace_counts = defaultdict(list)     # surface -> [ace_rate per match]
    surface_df_counts = defaultdict(list)
    surface_tb_counts = defaultdict(list)      # surface -> [tiebreaks per set]

    # Game totals by elo-diff bucket × surface × format for competitiveness lookup
    # Key: (surface, best_of, elo_bucket) → list of total_games
    game_totals_by_bucket = defaultdict(list)

    # We need ELO for competitiveness bucketing — load from feature store if available
    elo = {}
    feature_store_path = FEATURE_STORE_FILE
    if os.path.exists(feature_store_path):
        print("Loading ELO ratings from feature_store.pkl...")
        with open(feature_store_path, "rb") as f:
            fs = pickle.load(f)
        elo = fs.get("elo", {})
        surface_elo = fs.get("surface_elo", {})
        print(f"  {len(elo)} player ELO ratings loaded")
    else:
        surface_elo = {}
        print("  No feature store found — will use default ELO")

    processed = 0
    skipped = 0
    retirement_count = 0

    # Tour-wide stat accumulators for Bayesian priors
    tour_svpt_per_game = []  # serve points per service game across all matches

    for i, match in enumerate(matches):
        w_id = match.get("winner_id", "").strip()
        l_id = match.get("loser_id", "").strip()
        date_str = match.get("tourney_date", "").strip()
        surface = match.get("surface", "Hard").strip()
        best_of = safe_int(match.get("best_of"), 3)
        league = match.get("_league", "ATP")
        score_str = match.get("score", "")

        if not w_id or not l_id or not date_str:
            skipped += 1
            continue

        # Parse score
        score_data = parse_score(score_str)

        # Parse raw serve stats
        w_ace = safe_float(match.get("w_ace"), -1)
        w_df = safe_float(match.get("w_df"), -1)
        w_svpt = safe_float(match.get("w_svpt"), -1)
        w_bp_saved = safe_float(match.get("w_bpSaved"), -1)
        w_bp_faced = safe_float(match.get("w_bpFaced"), -1)

        l_ace = safe_float(match.get("l_ace"), -1)
        l_df = safe_float(match.get("l_df"), -1)
        l_svpt = safe_float(match.get("l_svpt"), -1)
        l_bp_saved = safe_float(match.get("l_bpSaved"), -1)
        l_bp_faced = safe_float(match.get("l_bpFaced"), -1)

        has_serve = w_svpt > 0 and l_svpt > 0

        # Skip if no useful data at all
        if not has_serve and score_data is None:
            skipped += 1
            continue

        is_retirement = score_data["is_retirement"] if score_data else False
        if is_retirement:
            retirement_count += 1

        # ELO-diff bucket for game total calibration
        w_elo = elo.get(w_id, 1500.0)
        l_elo = elo.get(l_id, 1500.0)
        elo_diff = abs(w_elo - l_elo)
        # Buckets: 0-50, 50-100, 100-200, 200-400, 400+
        if elo_diff < 50:
            elo_bucket = "0-50"
        elif elo_diff < 100:
            elo_bucket = "50-100"
        elif elo_diff < 200:
            elo_bucket = "100-200"
        elif elo_diff < 400:
            elo_bucket = "200-400"
        else:
            elo_bucket = "400+"

        # Compute service games for svpt-per-game
        if has_serve and score_data and not is_retirement:
            total_games = score_data["total_games"]
            if total_games > 0:
                # Each player serves ~half the games
                w_svc_games = total_games / 2.0
                l_svc_games = total_games / 2.0
                if w_svc_games > 0:
                    tour_svpt_per_game.append(w_svpt / w_svc_games)
                if l_svc_games > 0:
                    tour_svpt_per_game.append(l_svpt / l_svc_games)

        # Build match record for winner
        w_record = {
            "date": date_str,
            "surface": surface,
            "best_of": best_of,
            "opponent_id": l_id,
            "won": True,
            "league": league,
        }
        if has_serve:
            w_record["aces"] = int(w_ace)
            w_record["double_faults"] = int(w_df)
            w_record["serve_points"] = int(w_svpt)
            w_record["bp_saved"] = int(w_bp_saved)
            w_record["bp_faced"] = int(w_bp_faced)
            # Opponent data for breakpoints won calc
            w_record["opp_bp_faced"] = int(l_bp_faced)
            w_record["opp_bp_saved"] = int(l_bp_saved)
            w_record["opp_svpt"] = int(l_svpt)
            w_record["opp_aces"] = int(l_ace)

        if score_data and not is_retirement:
            w_record["total_games"] = score_data["total_games"]
            w_record["player_games"] = score_data["winner_games"]
            w_record["sets_played"] = score_data["sets_played"]
            w_record["sets_won"] = score_data["winner_sets"]
            w_record["sets_lost"] = score_data["loser_sets"]
            w_record["tiebreaks"] = score_data["tiebreaks_played"]
            w_record["first_set_total"] = score_data["first_set_total"]
            w_record["first_set_games"] = score_data["first_set_winner_games"]
        elif score_data and is_retirement:
            # Still store what we have for serve stats, but mark games incomplete
            w_record["is_retirement"] = True

        # Build match record for loser
        l_record = {
            "date": date_str,
            "surface": surface,
            "best_of": best_of,
            "opponent_id": w_id,
            "won": False,
            "league": league,
        }
        if has_serve:
            l_record["aces"] = int(l_ace)
            l_record["double_faults"] = int(l_df)
            l_record["serve_points"] = int(l_svpt)
            l_record["bp_saved"] = int(l_bp_saved)
            l_record["bp_faced"] = int(l_bp_faced)
            l_record["opp_bp_faced"] = int(w_bp_faced)
            l_record["opp_bp_saved"] = int(w_bp_saved)
            l_record["opp_svpt"] = int(w_svpt)
            l_record["opp_aces"] = int(w_ace)

        if score_data and not is_retirement:
            l_record["total_games"] = score_data["total_games"]
            l_record["player_games"] = score_data["loser_games"]
            l_record["sets_played"] = score_data["sets_played"]
            l_record["sets_won"] = score_data["loser_sets"]
            l_record["sets_lost"] = score_data["winner_sets"]
            l_record["tiebreaks"] = score_data["tiebreaks_played"]
            l_record["first_set_total"] = score_data["first_set_total"]
            l_record["first_set_games"] = score_data["first_set_loser_games"]
        elif score_data and is_retirement:
            l_record["is_retirement"] = True

        # Append to rolling history
        player_history[w_id].append(w_record)
        player_history[l_id].append(l_record)

        # Trim to keep memory bounded (keep 2× window for safety)
        for pid in [w_id, l_id]:
            if len(player_history[pid]) > ROLLING_WINDOW * 3:
                player_history[pid] = player_history[pid][-ROLLING_WINDOW * 2:]

        # Surface stat accumulation (non-retirement only)
        if has_serve and score_data and not is_retirement and w_svpt > 0 and l_svpt > 0:
            surface_ace_counts[surface].append(w_ace / w_svpt)
            surface_ace_counts[surface].append(l_ace / l_svpt)
            surface_df_counts[surface].append(w_df / w_svpt)
            surface_df_counts[surface].append(l_df / l_svpt)

        if score_data and not is_retirement:
            sets = score_data["sets_played"]
            tb = score_data["tiebreaks_played"]
            if sets > 0:
                surface_tb_counts[surface].append(tb / sets)

            # Game total bucketing
            bucket_key = (surface, best_of, elo_bucket)
            game_totals_by_bucket[bucket_key].append(score_data["total_games"])

        processed += 1
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1}/{len(matches)} processed")

    print(f"  Done: {processed} matches processed ({skipped} skipped, {retirement_count} retirements)")
    print(f"  Players with history: {len(player_history)}")

    # Trim all histories to final ROLLING_WINDOW
    for pid in player_history:
        player_history[pid] = player_history[pid][-ROLLING_WINDOW:]

    # Compute surface adjustment factors
    # Baseline = Hard court average, factors are relative multipliers
    surface_factors = _compute_surface_factors(surface_ace_counts, surface_df_counts, surface_tb_counts)

    # Compute game-total lookup table
    game_total_lookup = {}
    for key, totals in game_totals_by_bucket.items():
        if len(totals) >= 10:
            game_total_lookup[key] = {
                "mean": float(np.mean(totals)),
                "std": float(np.std(totals)),
                "n": len(totals),
            }

    # Tour-average serve points per service game
    avg_svpt_per_game = float(np.mean(tour_svpt_per_game)) if tour_svpt_per_game else 6.2

    # Compute tour-wide averages for Bayesian priors
    tour_averages = _compute_tour_averages(player_history)

    store = {
        "player_history": dict(player_history),
        "surface_factors": surface_factors,
        "game_total_lookup": game_total_lookup,
        "avg_svpt_per_game": avg_svpt_per_game,
        "tour_averages": tour_averages,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(PROPS_STORE_FILE, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(PROPS_STORE_FILE) / 1024 / 1024
    print(f"\nProps store saved to {PROPS_STORE_FILE} ({size_mb:.1f} MB)")
    _print_surface_stats(surface_factors, game_total_lookup, avg_svpt_per_game)

    return store


def _compute_surface_factors(ace_counts, df_counts, tb_counts):
    """Compute surface multipliers relative to Hard court baseline."""
    factors = {}

    # Hard court baseline
    hard_ace = np.mean(ace_counts.get("Hard", [0.05])) if ace_counts.get("Hard") else 0.05
    hard_df = np.mean(df_counts.get("Hard", [0.03])) if df_counts.get("Hard") else 0.03
    hard_tb = np.mean(tb_counts.get("Hard", [0.15])) if tb_counts.get("Hard") else 0.15

    for surface in ["Hard", "Clay", "Grass", "Carpet"]:
        ace_avg = np.mean(ace_counts.get(surface, [hard_ace])) if ace_counts.get(surface) else hard_ace
        df_avg = np.mean(df_counts.get(surface, [hard_df])) if df_counts.get(surface) else hard_df
        tb_avg = np.mean(tb_counts.get(surface, [hard_tb])) if tb_counts.get(surface) else hard_tb

        factors[surface] = {
            "ace_mult": float(ace_avg / hard_ace) if hard_ace > 0 else 1.0,
            "df_mult": float(df_avg / hard_df) if hard_df > 0 else 1.0,
            "tb_rate_per_set": float(tb_avg),
            "ace_rate_avg": float(ace_avg),
            "df_rate_avg": float(df_avg),
        }

    return factors


def _compute_tour_averages(player_history):
    """Compute tour-wide stat averages for Bayesian shrinkage."""
    all_aces = []
    all_dfs = []
    all_games = []
    all_player_games = []
    all_first_set = []
    all_bp_won = []
    all_sets = []

    for pid, history in player_history.items():
        for m in history:
            svpt = m.get("serve_points", 0)
            if svpt > 0:
                all_aces.append(m.get("aces", 0))
                all_dfs.append(m.get("double_faults", 0))
            if "total_games" in m:
                all_games.append(m["total_games"])
                all_player_games.append(m["player_games"])
                all_sets.append(m["sets_played"])
            if "first_set_total" in m:
                all_first_set.append(m["first_set_total"])
            opp_bp_faced = m.get("opp_bp_faced", 0)
            opp_bp_saved = m.get("opp_bp_saved", 0)
            if opp_bp_faced >= 0 and opp_bp_saved >= 0:
                all_bp_won.append(opp_bp_faced - opp_bp_saved)

    return {
        "aces": float(np.mean(all_aces)) if all_aces else 3.0,
        "double_faults": float(np.mean(all_dfs)) if all_dfs else 2.5,
        "total_games": float(np.mean(all_games)) if all_games else 21.0,
        "player_games": float(np.mean(all_player_games)) if all_player_games else 10.5,
        "first_set_total": float(np.mean(all_first_set)) if all_first_set else 10.0,
        "sets_played": float(np.mean(all_sets)) if all_sets else 2.3,
        "breakpoints_won": float(np.mean(all_bp_won)) if all_bp_won else 2.0,
    }


def _print_surface_stats(surface_factors, game_total_lookup, avg_svpt):
    """Print calibration summary."""
    print(f"\n  Surface Factors (relative to Hard):")
    for surface, f in sorted(surface_factors.items()):
        print(f"    {surface:8s}: ace_mult={f['ace_mult']:.3f}  df_mult={f['df_mult']:.3f}  "
              f"tb_rate/set={f['tb_rate_per_set']:.3f}")

    print(f"\n  Avg serve points per service game: {avg_svpt:.2f}")

    print(f"\n  Game Total Lookup (sample):")
    for key in sorted(game_total_lookup.keys())[:12]:
        v = game_total_lookup[key]
        print(f"    {str(key):35s}: mean={v['mean']:.1f}  std={v['std']:.1f}  n={v['n']}")


# ---------------------------------------------------------------------------
# Prediction Engine
# ---------------------------------------------------------------------------

# Lazy-loaded store
_props_store = None
_feature_store_elo = None


def _load_props_store():
    """Lazy-load the props store."""
    global _props_store, _feature_store_elo
    if _props_store is not None:
        return

    if not os.path.exists(PROPS_STORE_FILE):
        raise FileNotFoundError(f"Props store not found at {PROPS_STORE_FILE}. Run build_props_store() first.")

    with open(PROPS_STORE_FILE, "rb") as f:
        _props_store = pickle.load(f)

    # Also load ELO from feature store for win probability estimates
    if os.path.exists(FEATURE_STORE_FILE):
        with open(FEATURE_STORE_FILE, "rb") as f:
            fs = pickle.load(f)
        _feature_store_elo = {
            "elo": fs.get("elo", {}),
            "surface_elo": fs.get("surface_elo", {}),
            "players": fs.get("players", {}),
        }
    else:
        _feature_store_elo = {"elo": {}, "surface_elo": {}, "players": {}}


def _get_player_history(player_id):
    """Get rolling match history for a player."""
    _load_props_store()
    return _props_store["player_history"].get(player_id, [])


def _get_player_rolling_stats(player_id, surface=None):
    """Compute rolling averages from player's recent history.

    Returns dict with per-match averages for all prop-relevant stats.
    If surface is specified, also returns surface-specific stats.
    """
    _load_props_store()
    history = _props_store["player_history"].get(player_id, [])
    tour_avg = _props_store["tour_averages"]

    if not history:
        return _default_rolling_stats(tour_avg)

    # Filter to non-retirement matches for game-related stats
    full_matches = [m for m in history if "total_games" in m and not m.get("is_retirement")]
    serve_matches = [m for m in history if m.get("serve_points", 0) > 0]

    n_full = len(full_matches)
    n_serve = len(serve_matches)

    stats = {}

    # Serve-based stats (raw counts)
    if n_serve > 0:
        stats["avg_aces"] = np.mean([m["aces"] for m in serve_matches])
        stats["avg_dfs"] = np.mean([m["double_faults"] for m in serve_matches])
        stats["avg_svpt"] = np.mean([m["serve_points"] for m in serve_matches])
        stats["ace_rate"] = sum(m["aces"] for m in serve_matches) / sum(m["serve_points"] for m in serve_matches)
        stats["df_rate"] = sum(m["double_faults"] for m in serve_matches) / sum(m["serve_points"] for m in serve_matches)

        # BP stats
        bp_faced_matches = [m for m in serve_matches if m.get("bp_faced", 0) >= 0]
        if bp_faced_matches:
            stats["avg_bp_faced"] = np.mean([m["bp_faced"] for m in bp_faced_matches])
            stats["avg_bp_saved"] = np.mean([m["bp_saved"] for m in bp_faced_matches])
            total_f = sum(m["bp_faced"] for m in bp_faced_matches)
            total_s = sum(m["bp_saved"] for m in bp_faced_matches)
            stats["bp_save_rate"] = total_s / total_f if total_f > 0 else 0.6

        # Opponent BP data (for breakpoints won = opp_bp_faced - opp_bp_saved)
        bp_won_matches = [m for m in serve_matches if m.get("opp_bp_faced", 0) >= 0]
        if bp_won_matches:
            stats["avg_bp_won"] = np.mean([m["opp_bp_faced"] - m["opp_bp_saved"]
                                           for m in bp_won_matches])
            total_opp_f = sum(m["opp_bp_faced"] for m in bp_won_matches)
            total_opp_s = sum(m["opp_bp_saved"] for m in bp_won_matches)
            stats["opp_bp_faced_per_match"] = total_opp_f / len(bp_won_matches) if bp_won_matches else 3.0
            stats["opp_bp_save_rate"] = total_opp_s / total_opp_f if total_opp_f > 0 else 0.6

        # Aces allowed (opponent's ace rate against this player)
        opp_ace_matches = [m for m in serve_matches if m.get("opp_aces", 0) >= 0 and m.get("opp_svpt", 0) > 0]
        if opp_ace_matches:
            total_opp_aces = sum(m["opp_aces"] for m in opp_ace_matches)
            total_opp_svpt = sum(m["opp_svpt"] for m in opp_ace_matches)
            stats["opp_ace_rate_against"] = total_opp_aces / total_opp_svpt if total_opp_svpt > 0 else 0.05

        # Serve points per service game
        svpt_per_game_list = []
        for m in serve_matches:
            if "total_games" in m and m["total_games"] > 0:
                svc_games = m["total_games"] / 2.0
                if svc_games > 0:
                    svpt_per_game_list.append(m["serve_points"] / svc_games)
        stats["svpt_per_game"] = float(np.mean(svpt_per_game_list)) if svpt_per_game_list else _props_store["avg_svpt_per_game"]
    else:
        stats["avg_aces"] = tour_avg["aces"]
        stats["avg_dfs"] = tour_avg["double_faults"]
        stats["ace_rate"] = 0.05
        stats["df_rate"] = 0.03
        stats["svpt_per_game"] = _props_store["avg_svpt_per_game"]
        stats["avg_bp_won"] = tour_avg["breakpoints_won"]

    # Game-based stats
    if n_full > 0:
        stats["avg_total_games"] = np.mean([m["total_games"] for m in full_matches])
        stats["avg_player_games"] = np.mean([m["player_games"] for m in full_matches])
        stats["avg_sets"] = np.mean([m["sets_played"] for m in full_matches])
        stats["avg_sets_won"] = np.mean([m["sets_won"] for m in full_matches])
        stats["avg_tiebreaks"] = np.mean([m["tiebreaks"] for m in full_matches])

        first_set_matches = [m for m in full_matches if m.get("first_set_total", 0) > 0]
        if first_set_matches:
            stats["avg_first_set_total"] = np.mean([m["first_set_total"] for m in first_set_matches])
            stats["avg_first_set_games"] = np.mean([m["first_set_games"] for m in first_set_matches])
        else:
            stats["avg_first_set_total"] = tour_avg["first_set_total"]
            stats["avg_first_set_games"] = tour_avg["first_set_total"] / 2.0

        # Surface-specific game averages
        if surface:
            surf_matches = [m for m in full_matches if m.get("surface") == surface]
            if len(surf_matches) >= 3:
                stats["surface_avg_total_games"] = np.mean([m["total_games"] for m in surf_matches])
                stats["surface_avg_aces"] = np.mean([m.get("aces", 0) for m in surf_matches if m.get("aces") is not None])
    else:
        stats["avg_total_games"] = tour_avg["total_games"]
        stats["avg_player_games"] = tour_avg["player_games"]
        stats["avg_sets"] = tour_avg["sets_played"]
        stats["avg_sets_won"] = 1.2
        stats["avg_tiebreaks"] = 0.3
        stats["avg_first_set_total"] = tour_avg["first_set_total"]
        stats["avg_first_set_games"] = tour_avg["first_set_total"] / 2.0

    stats["n_matches"] = len(history)
    stats["n_full"] = n_full
    stats["n_serve"] = n_serve

    return stats


def _default_rolling_stats(tour_avg):
    """Return tour-average defaults for unknown players."""
    return {
        "avg_aces": tour_avg["aces"],
        "avg_dfs": tour_avg["double_faults"],
        "ace_rate": 0.05,
        "df_rate": 0.03,
        "avg_total_games": tour_avg["total_games"],
        "avg_player_games": tour_avg["player_games"],
        "avg_sets": tour_avg["sets_played"],
        "avg_sets_won": 1.2,
        "avg_tiebreaks": 0.3,
        "avg_first_set_total": tour_avg["first_set_total"],
        "avg_first_set_games": tour_avg["first_set_total"] / 2.0,
        "avg_bp_won": tour_avg["breakpoints_won"],
        "svpt_per_game": 6.2,
        "n_matches": 0,
        "n_full": 0,
        "n_serve": 0,
    }


def _bayesian_shrink(player_val, tour_val, n_matches, min_matches=10):
    """Shrink player estimate toward tour average when sample is small.

    At n_matches=0 → pure tour average.
    At n_matches=min_matches → 50/50 blend.
    At n_matches=2*min_matches → ~75% player.
    """
    weight = n_matches / (n_matches + min_matches)
    return weight * player_val + (1 - weight) * tour_val


def _win_probability(p1_elo, p2_elo):
    """ELO-based win probability for player 1."""
    return 1.0 / (1.0 + 10.0 ** ((p2_elo - p1_elo) / 400.0))


def _get_elo(player_id, surface=None):
    """Get player's ELO (overall or surface-specific)."""
    _load_props_store()
    if surface and _feature_store_elo:
        selo = _feature_store_elo.get("surface_elo", {}).get(player_id, {}).get(surface)
        if selo:
            return selo
    if _feature_store_elo:
        return _feature_store_elo.get("elo", {}).get(player_id, 1500.0)
    return 1500.0


# ---------------------------------------------------------------------------
# Individual Prop Predictors
# ---------------------------------------------------------------------------

def predict_aces(p_stats, opp_stats, surface, expected_svpt):
    """Predict aces: rate-based estimate blended with raw average.

    Rate path: (85% player ace rate + 15% opponent aces-allowed rate) × svpt × surface.
    Raw path: player's rolling average aces per match.
    Final: 50/50 blend of rate-based and raw average (corrects for match-length effects).
    """
    _load_props_store()
    sf = _props_store["surface_factors"].get(surface, _props_store["surface_factors"].get("Hard", {}))
    ace_mult = sf.get("ace_mult", 1.0)
    tour_avg = _props_store["tour_averages"]

    player_ace_rate = p_stats.get("ace_rate", 0.05)
    opp_ace_allowed = opp_stats.get("opp_ace_rate_against", sf.get("ace_rate_avg", 0.05))

    # Lighter opponent blend — aces are primarily about the server
    blended_rate = 0.85 * player_ace_rate + 0.15 * opp_ace_allowed

    # Bayesian shrinkage only for small samples (min_matches=5)
    n = p_stats.get("n_serve", 0)
    shrunk_rate = _bayesian_shrink(blended_rate, sf.get("ace_rate_avg", 0.05), n, min_matches=5)

    rate_based = shrunk_rate * expected_svpt * ace_mult

    # Raw average path: player's actual ace counts per match
    raw_avg = p_stats.get("avg_aces", tour_avg["aces"])
    raw_avg_shrunk = _bayesian_shrink(raw_avg, tour_avg["aces"], n, min_matches=5)

    # Blend: rate-based (matchup-adjusted) with raw average (captures match-length effects)
    if n >= 10:
        predicted = 0.50 * rate_based + 0.50 * raw_avg_shrunk
    else:
        predicted = 0.70 * rate_based + 0.30 * raw_avg_shrunk

    std_dev = max(1.0, predicted * 0.35)

    return predicted, std_dev


def predict_double_faults(p_stats, surface, expected_svpt):
    """Predict double faults: rate-based blended with raw average.

    Self-generated, no opponent adjustment.
    """
    _load_props_store()
    sf = _props_store["surface_factors"].get(surface, _props_store["surface_factors"].get("Hard", {}))
    df_mult = sf.get("df_mult", 1.0)
    tour_avg = _props_store["tour_averages"]

    player_df_rate = p_stats.get("df_rate", 0.03)
    n = p_stats.get("n_serve", 0)
    shrunk_rate = _bayesian_shrink(player_df_rate, sf.get("df_rate_avg", 0.03), n, min_matches=5)

    rate_based = shrunk_rate * expected_svpt * df_mult

    raw_avg = p_stats.get("avg_dfs", tour_avg["double_faults"])
    raw_avg_shrunk = _bayesian_shrink(raw_avg, tour_avg["double_faults"], n, min_matches=5)

    if n >= 10:
        predicted = 0.50 * rate_based + 0.50 * raw_avg_shrunk
    else:
        predicted = 0.70 * rate_based + 0.30 * raw_avg_shrunk

    std_dev = max(0.8, predicted * 0.40)

    return predicted, std_dev


def predict_total_games(p1_stats, p2_stats, p1_elo, p2_elo, surface, best_of):
    """Predict total games in the match.

    ELO-diff bucket lookup blended with both players' recent game averages.
    """
    _load_props_store()

    elo_diff = abs(p1_elo - p2_elo)
    if elo_diff < 50:
        bucket = "0-50"
    elif elo_diff < 100:
        bucket = "50-100"
    elif elo_diff < 200:
        bucket = "100-200"
    elif elo_diff < 400:
        bucket = "200-400"
    else:
        bucket = "400+"

    lookup_key = (surface, best_of, bucket)
    lookup = _props_store["game_total_lookup"].get(lookup_key)

    if lookup and lookup["n"] >= 20:
        structural_est = lookup["mean"]
        structural_std = lookup["std"]
    else:
        # Fallback: use nearby buckets
        fallback_keys = [(surface, best_of, b) for b in ["0-50", "50-100", "100-200", "200-400", "400+"]]
        vals = []
        for k in fallback_keys:
            v = _props_store["game_total_lookup"].get(k)
            if v and v["n"] >= 10:
                vals.append(v["mean"])
        structural_est = np.mean(vals) if vals else (21.0 if best_of == 3 else 33.0)
        structural_std = 4.0

    # Player recent averages
    p1_avg = p_stats_game_avg(p1_stats, best_of)
    p2_avg = p_stats_game_avg(p2_stats, best_of)
    player_avg = (p1_avg + p2_avg) / 2.0

    # Blend: 60% structural (elo-bucket), 40% player recent
    n_min = min(p1_stats.get("n_full", 0), p2_stats.get("n_full", 0))
    player_weight = min(0.4, n_min / 30.0 * 0.4)
    predicted = (1 - player_weight) * structural_est + player_weight * player_avg

    std_dev = structural_std if lookup else 4.5

    return predicted, std_dev


def p_stats_game_avg(stats, best_of):
    """Get player's average total games, with format awareness."""
    avg = stats.get("avg_total_games", 21.0)
    # If we have surface-specific, prefer it
    if "surface_avg_total_games" in stats:
        avg = 0.6 * stats["surface_avg_total_games"] + 0.4 * avg
    return avg


def predict_games_won(p_stats, expected_total_games, p_win_prob):
    """Predict games won by this player.

    Game share dampened from win probability: 0.5 + (p_win - 0.5) × 0.35.
    """
    game_share = 0.5 + (p_win_prob - 0.5) * 0.35
    predicted = expected_total_games * game_share
    std_dev = max(1.5, expected_total_games * 0.08)
    return predicted, std_dev


def predict_first_set_games(p1_stats, p2_stats, surface, best_of):
    """Predict total games in the first set."""
    _load_props_store()
    tour_avg = _props_store["tour_averages"]

    p1_avg = p1_stats.get("avg_first_set_total", tour_avg["first_set_total"])
    p2_avg = p2_stats.get("avg_first_set_total", tour_avg["first_set_total"])

    # Bayesian shrink each
    n1 = p1_stats.get("n_full", 0)
    n2 = p2_stats.get("n_full", 0)
    p1_shrunk = _bayesian_shrink(p1_avg, tour_avg["first_set_total"], n1)
    p2_shrunk = _bayesian_shrink(p2_avg, tour_avg["first_set_total"], n2)

    predicted = (p1_shrunk + p2_shrunk) / 2.0
    std_dev = 1.5
    return predicted, std_dev


def predict_first_set_games_won(p_stats, expected_first_set_total, p_win_prob):
    """Predict first-set games won by this player."""
    # Dampened game share — first set is more competitive than overall match
    game_share = 0.5 + (p_win_prob - 0.5) * 0.25
    predicted = expected_first_set_total * game_share
    std_dev = max(1.0, expected_first_set_total * 0.10)
    return predicted, std_dev


def predict_sets_played(p1_elo, p2_elo, best_of):
    """Predict sets played.

    Derived from P(straight sets) which depends on match win probability
    and competitiveness.
    """
    p_win = _win_probability(p1_elo, p2_elo)
    fav_prob = max(p_win, 1 - p_win)

    if best_of == 3:
        # P(straight sets) ≈ fav_prob^2 + (1-fav_prob)^2 + correction for close matches
        p_straight = fav_prob ** 2 + (1 - fav_prob) ** 2
        # Expected sets = 2 * P(straight) + 3 * P(3sets)
        predicted = 2 * p_straight + 3 * (1 - p_straight)
        std_dev = 0.45
    else:  # best_of == 5
        # More complex for 5 sets but similar idea
        p3 = fav_prob ** 3 + (1 - fav_prob) ** 3
        p_finish_3 = p3
        p_finish_4 = 3 * (fav_prob ** 3 * (1 - fav_prob) + (1 - fav_prob) ** 3 * fav_prob)
        p_finish_5 = 1 - p_finish_3 - p_finish_4
        predicted = 3 * p_finish_3 + 4 * p_finish_4 + 5 * max(0, p_finish_5)
        std_dev = 0.7

    return predicted, std_dev


def predict_sets_won(p_win_prob, best_of):
    """Predict sets won by a player.

    For Underdog line 0.5: this is P(player takes at least 1 set).
    For line 1.5: P(player wins match in BO3) or P(takes 2+ sets in BO5).
    We return expected value.
    """
    if best_of == 3:
        # P(win 0 sets) = P(lose in straight sets) ≈ (1-p)^2
        p_win_0 = (1 - p_win_prob) ** 2
        p_win_1 = 2 * p_win_prob * (1 - p_win_prob)  # lose in 3
        p_win_2 = p_win_prob  # win the match (2 sets needed)
        # But winning requires winning 2 sets:
        # Actually: P(win 2 sets) = p_win_prob, P(win 1 set) = 1-p_win_prob * P(go 3 sets)
        # Simpler: E[sets won] = 2*p_win + 1*(1-p_win)*(1-P(straight set loss))
        p_straight_loss = (1 - p_win_prob) ** 2
        expected = 2 * p_win_prob + 1 * (1 - p_win_prob) * (1 - (1 - p_win_prob)) + 0 * p_straight_loss
        # Simplified: expected = 2*p + (1-p)*p = p(2 + 1 - p) = p(3-p)... let me just be precise:
        # In BO3: win → 2 sets. Lose in 3 → 1 set. Lose in 2 → 0 sets.
        p_win_match = p_win_prob
        p_lose_3 = (1 - p_win_prob) - (1 - p_win_prob) ** 2  # lose match but take a set
        p_lose_2 = (1 - p_win_prob) ** 2
        expected = 2 * p_win_match + 1 * p_lose_3 + 0 * p_lose_2
    else:
        # BO5: win → 3 sets. Lose in 5 → 2 sets. Lose in 4 → 1. Lose in 3 → 0.
        p_win_match = p_win_prob
        p_lose_3 = (1 - p_win_prob) ** 3
        p_lose_4 = 3 * p_win_prob * (1 - p_win_prob) ** 3
        p_lose_5 = max(0, (1 - p_win_prob) - p_lose_3 - p_lose_4)
        expected = 3 * p_win_match + 0 * p_lose_3 + 1 * p_lose_4 + 2 * p_lose_5

    std_dev = 0.6
    return expected, std_dev


def predict_breakpoints_won(p_stats, opp_stats, expected_total_games, surface):
    """Predict breakpoints won (= opponent's break points faced - saved).

    Method: expected_opp_service_games × opp_bp_faced_per_game × (1 - opp_bp_save_rate),
    blended with player's return quality.
    """
    _load_props_store()
    tour_avg = _props_store["tour_averages"]

    # Opponent's service games ≈ total_games / 2
    opp_svc_games = expected_total_games / 2.0

    # How many BPs does the opponent face per service game?
    opp_bp_per_match = opp_stats.get("avg_bp_faced", 3.5)
    opp_total_games_avg = opp_stats.get("avg_total_games", 21.0)
    opp_svc_games_avg = opp_total_games_avg / 2.0
    opp_bp_per_svc_game = opp_bp_per_match / opp_svc_games_avg if opp_svc_games_avg > 0 else 0.35

    # What % of BPs does opponent save?
    opp_bp_save = opp_stats.get("bp_save_rate", 0.6)

    # Structural estimate from opponent's vulnerability
    structural_bp_won = opp_svc_games * opp_bp_per_svc_game * (1 - opp_bp_save)

    # Player's own historical BP won average
    player_bp_won = p_stats.get("avg_bp_won", tour_avg["breakpoints_won"])

    # Blend: 55% structural (opponent-driven), 45% player history
    n_p = p_stats.get("n_serve", 0)
    n_o = opp_stats.get("n_serve", 0)
    player_weight = min(0.45, n_p / 30.0 * 0.45)
    predicted = (1 - player_weight) * structural_bp_won + player_weight * player_bp_won

    # Bayesian shrinkage for small samples
    predicted = _bayesian_shrink(predicted, tour_avg["breakpoints_won"],
                                 min(n_p, n_o))

    std_dev = max(1.0, predicted * 0.45)
    return predicted, std_dev


def predict_tiebreakers(p1_stats, p2_stats, expected_sets, surface):
    """Predict tiebreakers in the match.

    Rate driven by combined hold rates + surface tiebreak frequency.
    """
    _load_props_store()
    sf = _props_store["surface_factors"].get(surface, _props_store["surface_factors"].get("Hard", {}))
    surface_tb_rate = sf.get("tb_rate_per_set", 0.15)

    # Player-specific tiebreak tendencies
    p1_avg_tb = p1_stats.get("avg_tiebreaks", 0.3)
    p1_avg_sets = p1_stats.get("avg_sets", 2.3)
    p2_avg_tb = p2_stats.get("avg_tiebreaks", 0.3)
    p2_avg_sets = p2_stats.get("avg_sets", 2.3)

    p1_tb_rate = p1_avg_tb / p1_avg_sets if p1_avg_sets > 0 else surface_tb_rate
    p2_tb_rate = p2_avg_tb / p2_avg_sets if p2_avg_sets > 0 else surface_tb_rate

    # Blend player rates with surface baseline
    n1 = p1_stats.get("n_full", 0)
    n2 = p2_stats.get("n_full", 0)
    p1_shrunk = _bayesian_shrink(p1_tb_rate, surface_tb_rate, n1)
    p2_shrunk = _bayesian_shrink(p2_tb_rate, surface_tb_rate, n2)

    blended_rate = (p1_shrunk + p2_shrunk) / 2.0
    predicted = expected_sets * blended_rate

    std_dev = max(0.4, predicted * 0.6)
    return predicted, std_dev


# ---------------------------------------------------------------------------
# Unified Prediction Entry Point
# ---------------------------------------------------------------------------

def predict_player_props(player_id, opponent_id, surface="Hard", best_of=3):
    """Predict all 10 prop types for a single player in a matchup.

    Returns dict of {prop_type: {"predicted": float, "std_dev": float}}.
    """
    _load_props_store()

    p_stats = _get_player_rolling_stats(player_id, surface)
    opp_stats = _get_player_rolling_stats(opponent_id, surface)

    p_elo = _get_elo(player_id, surface)
    opp_elo = _get_elo(opponent_id, surface)
    p_win = _win_probability(p_elo, opp_elo)

    # Expected serve points for this player
    expected_total_games, games_std = predict_total_games(
        p_stats, opp_stats, p_elo, opp_elo, surface, best_of
    )
    expected_svc_games = expected_total_games / 2.0
    svpt_per_game = p_stats.get("svpt_per_game", _props_store["avg_svpt_per_game"])
    expected_svpt = expected_svc_games * svpt_per_game

    # Expected sets
    expected_sets, sets_std = predict_sets_played(p_elo, opp_elo, best_of)

    # First set
    expected_fs_total, fs_std = predict_first_set_games(p_stats, opp_stats, surface, best_of)

    props = {}

    # 1. Aces
    val, std = predict_aces(p_stats, opp_stats, surface, expected_svpt)
    props["aces"] = {"predicted": round(val, 2), "std_dev": round(std, 2)}

    # 2. Double Faults
    val, std = predict_double_faults(p_stats, surface, expected_svpt)
    props["double_faults"] = {"predicted": round(val, 2), "std_dev": round(std, 2)}

    # 3. Games Played (total)
    props["total_games"] = {"predicted": round(expected_total_games, 2), "std_dev": round(games_std, 2)}

    # 4. Games Won
    val, std = predict_games_won(p_stats, expected_total_games, p_win)
    props["games_won"] = {"predicted": round(val, 2), "std_dev": round(std, 2)}

    # 5. 1st Set Games (total)
    props["first_set_games"] = {"predicted": round(expected_fs_total, 2), "std_dev": round(fs_std, 2)}

    # 6. 1st Set Games Won
    val, std = predict_first_set_games_won(p_stats, expected_fs_total, p_win)
    props["first_set_games_won"] = {"predicted": round(val, 2), "std_dev": round(std, 2)}

    # 7. Sets Played
    props["sets_played"] = {"predicted": round(expected_sets, 2), "std_dev": round(sets_std, 2)}

    # 8. Sets Won
    val, std = predict_sets_won(p_win, best_of)
    props["sets_won"] = {"predicted": round(val, 2), "std_dev": round(std, 2)}

    # 9. Breakpoints Won
    val, std = predict_breakpoints_won(p_stats, opp_stats, expected_total_games, surface)
    props["breakpoints_won"] = {"predicted": round(val, 2), "std_dev": round(std, 2)}

    # 10. Tiebreakers
    val, std = predict_tiebreakers(p_stats, opp_stats, expected_sets, surface)
    props["tiebreakers"] = {"predicted": round(val, 2), "std_dev": round(std, 2)}

    return {
        "player_id": player_id,
        "opponent_id": opponent_id,
        "surface": surface,
        "best_of": best_of,
        "p_win": round(p_win, 4),
        "p_elo": round(p_elo, 1),
        "opp_elo": round(opp_elo, 1),
        "n_matches": p_stats.get("n_matches", 0),
        "props": props,
    }


# ---------------------------------------------------------------------------
# Edge Detection
# ---------------------------------------------------------------------------

def compute_edge(predicted, line, std_dev, n_matches):
    """Compute edge and confidence for a single prop.

    Returns (direction, edge, confidence).
    """
    edge = predicted - line
    direction = "Higher" if edge > 0 else "Lower"

    # Confidence: based on |edge| relative to std_dev, with sample-size discount
    if std_dev > 0:
        z_score = abs(edge) / std_dev
    else:
        z_score = abs(edge)

    # Sample size factor: penalize when we have few matches
    sample_factor = min(1.0, n_matches / 15.0)

    # Base confidence from z-score (0 to 100 scale)
    confidence = min(95, z_score * 30 * sample_factor)

    return direction, round(edge, 2), round(confidence, 1)


# ---------------------------------------------------------------------------
# Player Resolution for Underdog Names
# ---------------------------------------------------------------------------

def _resolve_underdog_name(name, players_dict):
    """Resolve an Underdog Fantasy player name to a Sackmann player_id.

    Tries exact match, then fuzzy matching on the players database.
    """
    if not players_dict:
        return None

    name_lower = name.lower().strip()
    parts = name_lower.split()

    # Try to find by first + last name match
    best_match = None
    best_score = 0

    for pid, p in players_dict.items():
        first = p.get("first", "").lower()
        last = p.get("last", "").lower()
        full = f"{first} {last}"

        if full == name_lower:
            return pid

        # Check last name + first initial
        if len(parts) >= 2:
            if last == parts[-1] and first and first[0] == parts[0][0]:
                score = 0.8
                if len(parts[0]) > 1 and first.startswith(parts[0]):
                    score = 0.95
                if score > best_score:
                    best_score = score
                    best_match = pid

    if best_match and best_score >= 0.7:
        return best_match

    return None


# ---------------------------------------------------------------------------
# Slate Prediction — Main Entry Point
# ---------------------------------------------------------------------------

def predict_slate(matches):
    """Predict props for a slate of Underdog Fantasy matches.

    Args:
        matches: list of dicts, each with:
            - player: str (display name, e.g. "Iga Swiatek")
            - opponent: str (display name)
            - surface: str ("Hard", "Clay", "Grass")
            - best_of: int (default 3)
            - props: dict of {prop_type: line_value}
              e.g. {"aces": 2.5, "total_games": 17.5, "games_won": 12.5}
            - player_id: str (optional, Sackmann ID — auto-resolved if missing)
            - opponent_id: str (optional)

    Returns:
        dict with per-match predictions and recommended parlay.
    """
    _load_props_store()

    # Load player database for name resolution
    players_db = _feature_store_elo.get("players", {}) if _feature_store_elo else {}

    all_edges = []
    match_results = []

    for match in matches:
        player_name = match["player"]
        opponent_name = match["opponent"]
        surface = match.get("surface", "Hard")
        best_of = match.get("best_of", 3)
        lines = match.get("props", {})

        # Resolve player IDs
        p_id = match.get("player_id") or _resolve_underdog_name(player_name, players_db)
        opp_id = match.get("opponent_id") or _resolve_underdog_name(opponent_name, players_db)

        if not p_id:
            match_results.append({
                "player": player_name,
                "opponent": opponent_name,
                "error": f"Could not resolve player: {player_name}",
            })
            continue

        if not opp_id:
            # Can still predict, just without opponent context
            opp_id = ""

        # Get predictions
        pred = predict_player_props(p_id, opp_id, surface, best_of)

        # Compare to lines
        edges = []
        for prop_type, line_val in lines.items():
            prop_pred = pred["props"].get(prop_type)
            if prop_pred is None:
                continue

            direction, edge, confidence = compute_edge(
                prop_pred["predicted"], line_val, prop_pred["std_dev"],
                pred["n_matches"]
            )

            edge_info = {
                "player": player_name,
                "opponent": opponent_name,
                "match_key": f"{player_name} vs {opponent_name}",
                "prop_type": prop_type,
                "line": line_val,
                "predicted": prop_pred["predicted"],
                "edge": edge,
                "direction": direction,
                "confidence": confidence,
                "std_dev": prop_pred["std_dev"],
                "surface": surface,
            }
            edges.append(edge_info)
            all_edges.append(edge_info)

        match_results.append({
            "player": player_name,
            "opponent": opponent_name,
            "player_id": p_id,
            "p_win": pred["p_win"],
            "p_elo": pred["p_elo"],
            "opp_elo": pred["opp_elo"],
            "n_matches": pred["n_matches"],
            "predictions": pred["props"],
            "edges": edges,
        })

    # Build optimal parlay
    parlay = build_props_parlay(all_edges) if all_edges else None

    return {
        "matches": match_results,
        "all_edges": sorted(all_edges, key=lambda x: x["confidence"], reverse=True),
        "parlay": parlay,
    }


# ---------------------------------------------------------------------------
# Parlay Optimizer — 8-Leg Selection with Correlation Constraints
# ---------------------------------------------------------------------------

# Correlation groups within a match (correlated props that shouldn't stack)
CORRELATION_GROUPS = {
    "game_volume": {"total_games", "games_won"},
    "set_volume": {"sets_played", "sets_won"},
    "first_set_volume": {"first_set_games", "first_set_games_won"},
}

# Props that are player-specific and can be mixed freely
INDEPENDENT_PROPS = {"aces", "double_faults", "breakpoints_won", "tiebreakers"}


def build_props_parlay(all_edges, num_legs=8, max_per_match=2):
    """Select optimal N-leg parlay with correlation-aware diversification.

    Constraints:
      - Max `max_per_match` legs from the same match
      - Within a match, don't pick two props from the same correlation group
      - Rank by confidence, pick greedily with constraints
    """
    if not all_edges:
        return None

    # Sort by confidence descending
    sorted_edges = sorted(all_edges, key=lambda x: x["confidence"], reverse=True)

    selected = []
    match_counts = defaultdict(int)  # match_key -> count
    match_groups_used = defaultdict(set)  # match_key -> set of group names used

    for edge in sorted_edges:
        if len(selected) >= num_legs:
            break

        mk = edge["match_key"]
        pt = edge["prop_type"]

        # Check max-per-match constraint
        if match_counts[mk] >= max_per_match:
            continue

        # Check correlation group constraint
        prop_group = None
        for group_name, group_props in CORRELATION_GROUPS.items():
            if pt in group_props:
                prop_group = group_name
                break

        if prop_group and prop_group in match_groups_used[mk]:
            continue

        # Accept this leg
        selected.append(edge)
        match_counts[mk] += 1
        if prop_group:
            match_groups_used[mk].add(prop_group)

    if not selected:
        return None

    # Compute parlay stats
    avg_confidence = np.mean([s["confidence"] for s in selected])
    total_legs = len(selected)

    # Estimate combined probability (rough: product of individual directional probs)
    combined_prob = 1.0
    for s in selected:
        # Convert confidence to a rough probability
        leg_prob = 0.5 + s["confidence"] / 200.0  # 0% conf → 50%, 100% conf → 100%
        combined_prob *= leg_prob

    return {
        "legs": [{
            "player": s["player"],
            "opponent": s["opponent"],
            "prop": s["prop_type"],
            "line": s["line"],
            "direction": s["direction"],
            "predicted": s["predicted"],
            "edge": s["edge"],
            "confidence": s["confidence"],
        } for s in selected],
        "num_legs": total_legs,
        "avg_confidence": round(avg_confidence, 1),
        "combined_prob": round(combined_prob * 100, 2),
        "unique_matches": len(set(s["match_key"] for s in selected)),
    }


# ---------------------------------------------------------------------------
# Pretty Print
# ---------------------------------------------------------------------------

def print_slate_results(results):
    """Pretty-print prediction results."""
    print(f"\n{'='*80}")
    print(f"  UNDERDOG FANTASY PROPS PREDICTIONS")
    print(f"{'='*80}")

    for match in results["matches"]:
        if "error" in match:
            print(f"\n  {match['player']} vs {match['opponent']}: {match['error']}")
            continue

        print(f"\n  {match['player']} vs {match['opponent']}")
        print(f"  ELO: {match['p_elo']:.0f} vs {match['opp_elo']:.0f}  |  "
              f"Win Prob: {match['p_win']*100:.1f}%  |  "
              f"Sample: {match['n_matches']} matches")
        print(f"  {'Prop':<22s} {'Line':>6s} {'Pred':>6s} {'Edge':>6s} {'Dir':>6s} {'Conf':>5s}")
        print(f"  {'-'*55}")

        for edge in match.get("edges", []):
            marker = "*" if edge["confidence"] >= 50 else " "
            print(f" {marker}{edge['prop_type']:<22s} {edge['line']:>6.1f} "
                  f"{edge['predicted']:>6.1f} {edge['edge']:>+6.1f} "
                  f"{edge['direction']:>6s} {edge['confidence']:>5.1f}")

    # Parlay
    parlay = results.get("parlay")
    if parlay:
        print(f"\n{'='*80}")
        print(f"  RECOMMENDED {parlay['num_legs']}-LEG PARLAY")
        print(f"  Avg Confidence: {parlay['avg_confidence']:.1f}  |  "
              f"Combined Prob: {parlay['combined_prob']:.1f}%  |  "
              f"Matches: {parlay['unique_matches']}")
        print(f"{'='*80}")
        for i, leg in enumerate(parlay["legs"], 1):
            print(f"  {i}. {leg['player']:<22s} {leg['prop']:<22s} "
                  f"{leg['direction']:>6s} {leg['line']:>5.1f}  "
                  f"(pred={leg['predicted']:.1f}, edge={leg['edge']:+.1f}, "
                  f"conf={leg['confidence']:.0f})")

    # All edges ranked
    print(f"\n{'='*80}")
    print(f"  ALL EDGES RANKED BY CONFIDENCE")
    print(f"{'='*80}")
    for i, edge in enumerate(results["all_edges"][:20], 1):
        print(f"  {i:2d}. {edge['player']:<20s} {edge['prop_type']:<22s} "
              f"{edge['direction']:>6s} {edge['line']:>5.1f} → {edge['predicted']:>5.1f}  "
              f"(edge={edge['edge']:+.1f}, conf={edge['confidence']:.0f})")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_props_store()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test with sample data
        print("Testing score parser...")
        tests = [
            ("6-4 7-6(5) 6-2", False),
            ("6-4 6-4", False),
            ("7-6(3) 6-7(5) 7-5", False),
            ("1-6 6-4 3-1 RET", True),
            ("W/O", None),
            ("6-3 6-4 6-4", False),
            ("6-3 4-6 6-3 6-7(4) 6-3", False),
        ]
        for score, expected_ret in tests:
            result = parse_score(score)
            if result is None:
                print(f"  {score:30s} → (walkover/unparseable)")
            else:
                print(f"  {score:30s} → games={result['total_games']}, "
                      f"sets={result['sets_played']}, tb={result['tiebreaks_played']}, "
                      f"ret={result['is_retirement']}")
                if expected_ret is not None:
                    assert result["is_retirement"] == expected_ret, f"Retirement mismatch for {score}"
        print("  All tests passed!")
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        # Run predictions on sample Underdog lines
        sample_slate = [
            {
                "player": "Iga Swiatek",
                "opponent": "Magda Linette",
                "surface": "Hard",
                "props": {
                    "aces": 2.5,
                    "double_faults": 2.5,
                    "total_games": 17.5,
                    "games_won": 12.5,
                    "first_set_games": 9.5,
                    "first_set_games_won": 5.5,
                    "sets_played": 2.5,
                    "breakpoints_won": 3.5,
                },
            },
            {
                "player": "Reilly Opelka",
                "opponent": "Frances Tiafoe",
                "surface": "Hard",
                "props": {
                    "aces": 21.5,
                    "double_faults": 4.5,
                    "total_games": 24.5,
                    "games_won": 12.5,
                    "tiebreakers": 1.5,
                },
            },
        ]
        results = predict_slate(sample_slate)
        print_slate_results(results)
    else:
        print("Usage:")
        print("  python props_predictor.py build    — Build props store from historical data")
        print("  python props_predictor.py test     — Test score parser")
        print("  python props_predictor.py predict  — Run sample predictions")
