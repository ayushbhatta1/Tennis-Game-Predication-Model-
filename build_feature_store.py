"""
Build Feature Store — process all historical matches chronologically.

Computes rolling stats with no look-ahead:
  - ELO + surface ELO (K=32)
  - Serve stats (exponential decay, window=20)
  - Form (last 10/20, surface, weighted, upset rate, momentum)
  - H2H records (overall + surface-specific)
  - Rankings trajectories from rankings CSVs
  - Fatigue (days since last, match density)
  - Physical attributes (height, age, hand from players CSVs)

Output: model/feature_store.pkl
"""

import csv
import glob
import math
import os
import pickle
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

BASE_DIR = os.path.dirname(__file__)
HIST_DIR = os.path.join(BASE_DIR, "historical")
MODEL_DIR = os.path.join(BASE_DIR, "model")

STORE_FILE = os.path.join(MODEL_DIR, "feature_store.pkl")

ELO_K = 32
SERVE_DECAY = 0.95  # exponential decay factor
SERVE_WINDOW = 20


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


def parse_date(date_str):
    """Parse YYYYMMDD date string to datetime."""
    if not date_str or len(date_str) < 8:
        return None
    try:
        return datetime.strptime(date_str[:8], "%Y%m%d")
    except ValueError:
        return None


def load_players():
    """Load player physical attributes from CSVs."""
    players = {}  # player_id -> {hand, dob, height, ioc}
    for filename, league in [("atp_players.csv", "ATP"), ("wta_players.csv", "WTA")]:
        path = os.path.join(HIST_DIR, filename)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                pid = row.get("player_id", "").strip()
                if not pid:
                    continue
                players[pid] = {
                    "hand": row.get("hand", "U").strip(),
                    "dob": row.get("dob", "").strip(),
                    "height": safe_float(row.get("height"), 0),
                    "ioc": row.get("ioc", "").strip(),
                    "first": row.get("name_first", "").strip(),
                    "last": row.get("name_last", "").strip(),
                    "league": league,
                }
    return players


def load_rankings():
    """Load all rankings CSVs into a dict: player_id -> list of (date, rank, points)."""
    rankings = defaultdict(list)
    for pattern in ["atp_rankings_*.csv", "wta_rankings_*.csv"]:
        for filepath in sorted(glob.glob(os.path.join(HIST_DIR, pattern))):
            with open(filepath, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    date_str = row.get("ranking_date", "").strip()
                    pid = row.get("player", "").strip()
                    rank = safe_int(row.get("rank"), 9999)
                    pts = safe_int(row.get("points"), 0)
                    if date_str and pid:
                        rankings[pid].append((date_str, rank, pts))

    # Sort each player's rankings by date
    for pid in rankings:
        rankings[pid].sort(key=lambda x: x[0])

    return rankings


def get_ranking_at_date(rankings_list, date_str):
    """Binary search for most recent ranking before date_str."""
    if not rankings_list:
        return 500, 0
    # Find last entry with date <= date_str
    lo, hi = 0, len(rankings_list) - 1
    result = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if rankings_list[mid][0] <= date_str:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if result is not None:
        return rankings_list[result][1], rankings_list[result][2]
    return 500, 0


def get_rank_momentum(rankings_list, date_str, weeks=8):
    """Compute rank improvement over last N weeks. Positive = improving."""
    if not rankings_list:
        return 0.0
    current_rank, _ = get_ranking_at_date(rankings_list, date_str)
    # Find rank ~8 weeks ago
    dt = parse_date(date_str.replace("-", ""))
    if dt is None:
        return 0.0
    past_dt = dt - timedelta(weeks=weeks)
    past_str = past_dt.strftime("%Y%m%d")
    past_rank, _ = get_ranking_at_date(rankings_list, past_str)
    if past_rank == 0 or current_rank == 0:
        return 0.0
    # Log ratio: positive means improving (rank number decreased)
    return (math.log1p(past_rank) - math.log1p(current_rank)) / 3.0


def get_peak_rank(rankings_list, date_str):
    """Get best (lowest) rank achieved before date_str."""
    if not rankings_list:
        return 500
    best = 9999
    for d, r, p in rankings_list:
        if d > date_str:
            break
        if r < best:
            best = r
    return best if best < 9999 else 500


def load_all_matches():
    """Load all matches sorted by date, with parsed serve stats."""
    matches = []
    for pattern in ["atp_matches_*.csv", "wta_matches_*.csv"]:
        files = sorted(glob.glob(os.path.join(HIST_DIR, pattern)))
        for filepath in files:
            # Skip futures
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


class FeatureStoreBuilder:
    """Processes matches chronologically, building rolling stats."""

    def __init__(self, players, rankings):
        self.players = players
        self.rankings = rankings

        # Rolling state
        self.elo = defaultdict(lambda: 1500.0)
        self.surface_elo = defaultdict(lambda: defaultdict(lambda: 1500.0))

        # Serve stats: player -> list of recent serve dicts (window=SERVE_WINDOW)
        self.serve_history = defaultdict(list)

        # Form: player -> list of (date, won, surface, was_upset)
        self.form_history = defaultdict(list)

        # H2H: (p1, p2) sorted -> {p1_wins, p2_wins, surface: {p1, p2}}
        self.h2h = defaultdict(lambda: {
            "wins": defaultdict(int),
            "surface_wins": defaultdict(lambda: defaultdict(int)),
        })

        # Scheduling: player -> list of match dates
        self.schedule = defaultdict(list)

        # Store: snapshot of all stats at match time
        self.match_features = []

    def _compute_serve_stats(self, player_id):
        """Compute exponentially weighted serve averages."""
        history = self.serve_history[player_id]
        if not history:
            return {
                "ace_rate": 0.05, "first_serve_pct": 0.60,
                "first_serve_won": 0.70, "second_serve_won": 0.50,
                "bp_save_rate": 0.60, "df_rate": 0.03,
                "serve_dominance": 0.0,
            }

        # Exponentially weighted average (most recent = highest weight)
        total_weight = 0.0
        stats = defaultdict(float)
        for i, s in enumerate(reversed(history[-SERVE_WINDOW:])):
            w = SERVE_DECAY ** i
            total_weight += w
            for key in ["ace_rate", "first_serve_pct", "first_serve_won",
                        "second_serve_won", "bp_save_rate", "df_rate"]:
                stats[key] += s.get(key, 0.0) * w

        if total_weight > 0:
            for key in stats:
                stats[key] /= total_weight

        stats["serve_dominance"] = (
            stats["first_serve_won"] * 0.4 +
            stats["ace_rate"] * 0.3 -
            stats["df_rate"] * 0.3
        )
        return dict(stats)

    def _compute_form(self, player_id, surface=None):
        """Compute form metrics for a player."""
        history = self.form_history[player_id]
        if not history:
            return {
                "last_10": 0.5, "last_20": 0.5,
                "surface": {}, "weighted": 0.5,
                "upset_rate": 0.0, "momentum": 0.0,
            }

        recent_10 = history[-10:]
        recent_20 = history[-20:]

        last_10 = sum(h[1] for h in recent_10) / len(recent_10)
        last_20 = sum(h[1] for h in recent_20) / len(recent_20)

        # Surface form
        surface_form = {}
        for surf in ["Hard", "Clay", "Grass", "Carpet"]:
            surf_matches = [h for h in history if h[2] == surf][-20:]
            if surf_matches:
                surface_form[surf] = sum(h[1] for h in surf_matches) / len(surf_matches)
            else:
                surface_form[surf] = 0.5

        # Weighted form (more recent = higher weight)
        weighted = 0.0
        tw = 0.0
        for i, h in enumerate(reversed(recent_20)):
            w = 0.95 ** i
            weighted += h[1] * w
            tw += w
        weighted = weighted / tw if tw > 0 else 0.5

        # Upset rate (won as underdog, using rank)
        upsets = [h for h in history[-20:] if h[3]]
        upset_rate = len(upsets) / max(len(recent_20), 1)

        # Momentum (last 5 vs prev 5)
        if len(history) >= 10:
            last5 = sum(h[1] for h in history[-5:]) / 5
            prev5 = sum(h[1] for h in history[-10:-5]) / 5
            momentum = last5 - prev5
        else:
            momentum = 0.0

        return {
            "last_10": last_10, "last_20": last_20,
            "surface": surface_form, "weighted": weighted,
            "upset_rate": upset_rate, "momentum": momentum,
        }

    def _get_h2h(self, p1_id, p2_id):
        """Get H2H record from p1's perspective."""
        key = tuple(sorted([p1_id, p2_id]))
        record = self.h2h[key]
        return {
            "wins": record["wins"][p1_id],
            "losses": record["wins"][p2_id],
            "surface_wins": dict(record["surface_wins"][p1_id]),
            "surface_losses": dict(record["surface_wins"][p2_id]),
        }

    def _get_fatigue(self, player_id, match_date):
        """Compute fatigue metrics."""
        dates = self.schedule[player_id]
        if not dates or not match_date:
            return {"days_since_last": 14, "matches_14d": 3}

        dt = parse_date(match_date)
        if dt is None:
            return {"days_since_last": 14, "matches_14d": 3}

        # Days since last match
        last_dates = [parse_date(d) for d in dates if d < match_date]
        last_dates = [d for d in last_dates if d is not None]
        if last_dates:
            days_since = (dt - last_dates[-1]).days
        else:
            days_since = 30

        # Matches in last 14 days
        cutoff = dt - timedelta(days=14)
        matches_14d = sum(1 for d in last_dates if d and d >= cutoff)

        return {"days_since_last": days_since, "matches_14d": matches_14d}

    def _get_player_stats(self, player_id, opponent_id, date_str, surface):
        """Get all stats for a player at a specific point in time."""
        player = self.players.get(player_id, {})
        rankings_list = self.rankings.get(player_id, [])

        rank, points = get_ranking_at_date(rankings_list, date_str)
        momentum = get_rank_momentum(rankings_list, date_str)
        peak = get_peak_rank(rankings_list, date_str)

        # Compute age at match date
        dob = player.get("dob", "")
        match_dt = parse_date(date_str)
        if dob and match_dt and len(dob) >= 8:
            dob_dt = parse_date(dob)
            if dob_dt:
                age = (match_dt - dob_dt).days / 365.25
            else:
                age = 25.0
        else:
            age = 25.0

        height = player.get("height", 0)
        if height == 0:
            height = 180.0

        return {
            "player_id": player_id,
            "elo": self.elo[player_id],
            "surface_elo": dict(self.surface_elo[player_id]),
            "serve": self._compute_serve_stats(player_id),
            "form": self._compute_form(player_id, surface),
            "h2h": {opponent_id: self._get_h2h(player_id, opponent_id)},
            "physical": {
                "height": height,
                "age": age,
                "hand": player.get("hand", "R"),
            },
            "ranking": {
                "rank": rank,
                "points": points,
                "momentum": momentum,
                "peak": peak,
            },
            "fatigue": self._get_fatigue(player_id, date_str),
        }

    def _parse_serve_stats(self, match, prefix):
        """Parse serve stats from a match row (prefix = 'w_' or 'l_')."""
        svpt = safe_float(match.get(f"{prefix}svpt"), 0)
        if svpt < 1:
            return None

        ace = safe_float(match.get(f"{prefix}ace"), 0)
        df = safe_float(match.get(f"{prefix}df"), 0)
        first_in = safe_float(match.get(f"{prefix}1stIn"), 0)
        first_won = safe_float(match.get(f"{prefix}1stWon"), 0)
        second_won = safe_float(match.get(f"{prefix}2ndWon"), 0)
        bp_saved = safe_float(match.get(f"{prefix}bpSaved"), 0)
        bp_faced = safe_float(match.get(f"{prefix}bpFaced"), 0)

        second_pts = svpt - first_in
        return {
            "ace_rate": ace / svpt,
            "first_serve_pct": first_in / svpt if svpt > 0 else 0.6,
            "first_serve_won": first_won / first_in if first_in > 0 else 0.7,
            "second_serve_won": second_won / second_pts if second_pts > 0 else 0.5,
            "bp_save_rate": bp_saved / bp_faced if bp_faced > 0 else 0.6,
            "df_rate": df / svpt,
        }

    def _update_elo(self, w_id, l_id, surface):
        """Update ELO after a match. Returns (w_elo_before, l_elo_before)."""
        w_elo = self.elo[w_id]
        l_elo = self.elo[l_id]
        w_selo = self.surface_elo[w_id][surface]
        l_selo = self.surface_elo[l_id][surface]

        # Overall ELO
        exp_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        self.elo[w_id] = w_elo + ELO_K * (1 - exp_w)
        self.elo[l_id] = l_elo + ELO_K * (0 - (1 - exp_w))

        # Surface ELO
        exp_ws = 1 / (1 + 10 ** ((l_selo - w_selo) / 400))
        self.surface_elo[w_id][surface] = w_selo + ELO_K * (1 - exp_ws)
        self.surface_elo[l_id][surface] = l_selo + ELO_K * (0 - (1 - exp_ws))

        return w_elo, l_elo

    def process_match(self, match):
        """Process a single match: snapshot stats, then update."""
        w_id = match.get("winner_id", "").strip()
        l_id = match.get("loser_id", "").strip()
        date_str = match.get("tourney_date", "").strip()
        surface = match.get("surface", "Hard").strip()
        league = match.get("_league", "ATP")

        if not w_id or not l_id or not date_str:
            return None

        # Determine if this was an upset (loser was ranked higher)
        w_rank = safe_float(match.get("winner_rank"), 500)
        l_rank = safe_float(match.get("loser_rank"), 500)
        was_upset = l_rank < w_rank  # loser had better rank

        # 1. Snapshot stats BEFORE this match (no look-ahead)
        w_stats = self._get_player_stats(w_id, l_id, date_str, surface)
        l_stats = self._get_player_stats(l_id, w_id, date_str, surface)

        # 2. Store match feature record
        record = {
            "winner_id": w_id,
            "loser_id": l_id,
            "date": date_str,
            "surface": surface,
            "round": match.get("round", ""),
            "tourney_level": match.get("tourney_level", "A"),
            "best_of": safe_int(match.get("best_of"), 3),
            "league": league,
            "tourney_id": match.get("tourney_id", ""),
            "match_num": match.get("match_num", ""),
            "winner_stats": w_stats,
            "loser_stats": l_stats,
            "winner_name": match.get("winner_name", ""),
            "loser_name": match.get("loser_name", ""),
        }

        # 3. Update rolling stats AFTER snapshot
        self._update_elo(w_id, l_id, surface)

        # Serve stats
        w_serve = self._parse_serve_stats(match, "w_")
        l_serve = self._parse_serve_stats(match, "l_")
        if w_serve:
            self.serve_history[w_id].append(w_serve)
            if len(self.serve_history[w_id]) > SERVE_WINDOW * 2:
                self.serve_history[w_id] = self.serve_history[w_id][-SERVE_WINDOW * 2:]
        if l_serve:
            self.serve_history[l_id].append(l_serve)
            if len(self.serve_history[l_id]) > SERVE_WINDOW * 2:
                self.serve_history[l_id] = self.serve_history[l_id][-SERVE_WINDOW * 2:]

        # Form
        self.form_history[w_id].append((date_str, 1, surface, was_upset))
        self.form_history[l_id].append((date_str, 0, surface, False))
        # Keep bounded
        for pid in [w_id, l_id]:
            if len(self.form_history[pid]) > 100:
                self.form_history[pid] = self.form_history[pid][-100:]

        # H2H
        key = tuple(sorted([w_id, l_id]))
        self.h2h[key]["wins"][w_id] += 1
        self.h2h[key]["surface_wins"][w_id][surface] += 1

        # Schedule
        self.schedule[w_id].append(date_str)
        self.schedule[l_id].append(date_str)
        for pid in [w_id, l_id]:
            if len(self.schedule[pid]) > 50:
                self.schedule[pid] = self.schedule[pid][-50:]

        return record

    def get_current_stats(self, player_id, opponent_id, date_str, surface):
        """Get current stats for a player (for live predictions)."""
        return self._get_player_stats(player_id, opponent_id, date_str, surface)


def build_store():
    """Build the feature store from all historical matches."""
    print("Loading players...")
    players = load_players()
    print(f"  {len(players)} players")

    print("Loading rankings...")
    rankings = load_rankings()
    print(f"  {len(rankings)} players with rankings")

    print("Loading matches...")
    matches = load_all_matches()
    print(f"  {len(matches)} matches")

    print("Processing matches chronologically...")
    builder = FeatureStoreBuilder(players, rankings)
    records = []
    skipped = 0

    for i, match in enumerate(matches):
        record = builder.process_match(match)
        if record:
            records.append(record)
        else:
            skipped += 1

        if (i + 1) % 50000 == 0:
            print(f"  {i + 1}/{len(matches)} processed ({len(records)} valid)")

    print(f"  Done: {len(records)} match records ({skipped} skipped)")

    # Convert defaultdicts to regular dicts for pickle compatibility
    h2h_plain = {}
    for key, val in builder.h2h.items():
        h2h_plain[key] = {
            "wins": dict(val["wins"]),
            "surface_wins": {
                pid: dict(sw) for pid, sw in val["surface_wins"].items()
            },
        }

    # Save everything needed for training and live prediction
    store = {
        "records": records,
        "elo": dict(builder.elo),
        "surface_elo": {pid: dict(selos) for pid, selos in builder.surface_elo.items()},
        "serve_history": dict(builder.serve_history),
        "form_history": dict(builder.form_history),
        "h2h": h2h_plain,
        "schedule": dict(builder.schedule),
        "players": players,
        "rankings": rankings,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(STORE_FILE, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(STORE_FILE) / 1024 / 1024
    print(f"\nFeature store saved to {STORE_FILE} ({size_mb:.0f} MB)")
    print(f"  Match records: {len(records)}")
    print(f"  Players with ELO: {len(builder.elo)}")
    print(f"  Players with serve stats: {len(builder.serve_history)}")

    return store


if __name__ == "__main__":
    build_store()
