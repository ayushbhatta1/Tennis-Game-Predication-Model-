"""
Tennis Match Scraper — replaces SportsGameOdds API.

Fetches match data from ESPN's public API and odds from The Odds API (free tier).
Outputs events in the EXACT same JSON format the prediction pipeline expects,
so predictor.py, mega_predict.py, and feature_engine.py work unchanged.

Data sources:
  - ESPN public API — match schedules, scores, results (no key required)
    ATP: https://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard
    WTA: https://site.api.espn.com/apis/site/v2/sports/tennis/wta/scoreboard
  - The Odds API (the-odds-api.com) — moneyline + spread + totals odds
    Free tier: 500 requests/month. Set ODDS_API_KEY env var.
  - Sofascore / Flashscore as tertiary fallbacks

Usage:
    python scrape_matches.py                  # fetch today's matches
    python scrape_matches.py --date 2026-03-19
    python scrape_matches.py --completed      # fetch completed results
"""

import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# ESPN public API (primary data source — no key required, returns JSON)
ESPN_API_BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"
ESPN_ENDPOINTS = {
    "ATP": f"{ESPN_API_BASE}/atp/scoreboard",
    "WTA": f"{ESPN_API_BASE}/wta/scoreboard",
}

# Flashscore (tertiary fallback)
FLASHSCORE_API = "https://www.flashscore.com"

# Robust headers to avoid blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache",
}

# The Odds API sport keys for tennis
ODDS_SPORT_KEYS = [
    # ATP
    "tennis_atp_australian_open",
    "tennis_atp_french_open",
    "tennis_atp_us_open",
    "tennis_atp_wimbledon",
    # WTA
    "tennis_wta_australian_open",
    "tennis_wta_french_open",
    "tennis_wta_us_open",
    "tennis_wta_wimbledon",
]

# Session for connection pooling + retries
_session = None


def _get_session() -> requests.Session:
    """Return a shared session with retry logic."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update(HEADERS)
        adapter = requests.adapters.HTTPAdapter(
            max_retries=requests.adapters.Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        )
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


# ---------------------------------------------------------------------------
# Utility: odds format conversion
# ---------------------------------------------------------------------------

def decimal_to_american(decimal_odds: float) -> str:
    """Convert decimal odds (e.g. 1.85, 2.10) to American string (e.g. '-118', '+110')."""
    if decimal_odds is None or decimal_odds <= 1.0:
        return None
    if decimal_odds >= 2.0:
        american = round((decimal_odds - 1) * 100)
        return f"+{american}"
    else:
        american = round(-100 / (decimal_odds - 1))
        return str(american)


def american_to_decimal(american_str: str) -> Optional[float]:
    """Convert American odds string to decimal."""
    try:
        odds = int(american_str)
    except (ValueError, TypeError):
        return None
    if odds < 0:
        return 1 + (100 / abs(odds))
    else:
        return 1 + (odds / 100)


def _make_event_id(home: str, away: str, date_str: str) -> str:
    """Generate a deterministic event ID from player names and date."""
    raw = f"{home}|{away}|{date_str}".lower().strip()
    return hashlib.md5(raw.encode()).hexdigest()[:20]


def _make_team_id(name: str, league: str) -> str:
    """Generate a teamID matching the SportsGameOdds format: FIRST_LAST_LEAGUE."""
    parts = name.strip().split()
    if len(parts) >= 2:
        tid = "_".join(p.upper() for p in parts) + f"_{league}"
    else:
        tid = name.upper().replace(" ", "_") + f"_{league}"
    # Clean non-alphanumeric (except underscore)
    tid = re.sub(r"[^A-Z0-9_]", "", tid)
    return tid


def _make_names(full_name: str) -> Dict[str, str]:
    """Build the names dict matching SportsGameOdds format."""
    parts = full_name.strip().split()
    first = parts[0] if parts else full_name
    last = parts[-1] if len(parts) > 1 else full_name
    medium = f"{first[0]}. {last}" if len(parts) > 1 else full_name
    short = last[:3].upper() if last else "UNK"
    return {
        "long": full_name,
        "medium": medium,
        "short": short,
        "firstName": first,
        "lastName": last,
    }


def _normalize_name(name: str) -> str:
    """Normalize a player name for fuzzy matching."""
    name = name.strip().lower()
    # Remove accents/diacritics for matching
    replacements = {
        "a": "aàáâãäå", "c": "cçć", "e": "eèéêë", "i": "iìíîï",
        "n": "nñń", "o": "oòóôõö", "s": "sšś", "u": "uùúûü",
        "y": "yýÿ", "z": "zžź", "d": "dđ",
    }
    for ascii_char, unicode_chars in replacements.items():
        for uc in unicode_chars[1:]:
            name = name.replace(uc, ascii_char)
    return name


def _names_match(name1: str, name2: str) -> bool:
    """Fuzzy-match two player names."""
    n1 = _normalize_name(name1)
    n2 = _normalize_name(name2)
    # Exact
    if n1 == n2:
        return True
    # Last name match
    parts1 = n1.split()
    parts2 = n2.split()
    if parts1 and parts2 and parts1[-1] == parts2[-1]:
        # Also check first initial
        if parts1[0][0] == parts2[0][0]:
            return True
    return False


# ---------------------------------------------------------------------------
# The Odds API — fetch real odds data (free tier)
# ---------------------------------------------------------------------------

def _fetch_odds_api_sports() -> List[str]:
    """Fetch currently available tennis sport keys from The Odds API."""
    if not ODDS_API_KEY:
        return []
    try:
        resp = _get_session().get(
            f"{ODDS_API_BASE}/sports",
            params={"apiKey": ODDS_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        sports = resp.json()
        tennis_keys = [
            s["key"] for s in sports
            if s.get("group", "").lower() == "tennis" and s.get("active")
        ]
        return tennis_keys
    except Exception as e:
        print(f"  [OddsAPI] Failed to fetch sports: {e}")
        return []


def _fetch_odds_for_sport(sport_key: str) -> List[Dict]:
    """Fetch odds for a single sport key from The Odds API."""
    if not ODDS_API_KEY:
        return []
    try:
        resp = _get_session().get(
            f"{ODDS_API_BASE}/sports/{sport_key}/odds",
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us,eu",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "decimal",
            },
            timeout=15,
        )
        if resp.status_code == 401:
            print("  [OddsAPI] Invalid API key")
            return []
        if resp.status_code == 422:
            # Sport not currently active
            return []
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        print(f"  [OddsAPI] {sport_key}: {len(resp.json())} events (requests remaining: {remaining})")
        return resp.json()
    except Exception as e:
        print(f"  [OddsAPI] Error fetching {sport_key}: {e}")
        return []


def fetch_all_odds() -> Dict[str, Dict]:
    """
    Fetch all available tennis odds from The Odds API.

    Returns dict keyed by normalized "away_name|home_name" for matching.
    Each value contains the odds converted to the pipeline format.
    """
    if not ODDS_API_KEY:
        print("  [OddsAPI] No ODDS_API_KEY set — skipping odds fetch")
        return {}

    print("[OddsAPI] Fetching available tennis sports...")
    sport_keys = _fetch_odds_api_sports()
    if not sport_keys:
        # Fallback: try known keys
        sport_keys = ODDS_SPORT_KEYS

    all_odds = {}
    for key in sport_keys:
        events = _fetch_odds_for_sport(key)
        for event in events:
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            commence = event.get("commence_time", "")

            # Determine league from sport key
            league = "ATP"
            if "wta" in key.lower():
                league = "WTA"

            # Convert bookmaker odds to pipeline format
            odds_data = _convert_odds_api_to_pipeline(event, league)
            if odds_data:
                # Key for matching
                match_key = f"{_normalize_name(home_team)}|{_normalize_name(away_team)}"
                all_odds[match_key] = {
                    "odds": odds_data,
                    "league": league,
                    "commence_time": commence,
                    "home_team": home_team,
                    "away_team": away_team,
                    "sport_key": key,
                }

        time.sleep(0.2)  # Rate limit courtesy

    print(f"  [OddsAPI] Total events with odds: {len(all_odds)}")
    return all_odds


def _convert_odds_api_to_pipeline(event: Dict, league: str) -> Optional[Dict]:
    """
    Convert The Odds API event to the SportsGameOdds odds format.

    The Odds API provides decimal odds per bookmaker for markets:
      h2h (moneyline), spreads, totals

    We need to produce the pipeline format with keys like:
      points-home-game-ml-home, points-away-game-ml-away, etc.
    """
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None

    # Collect all h2h odds per bookmaker
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")

    home_ml_by_bk = {}
    away_ml_by_bk = {}
    home_spreads = []
    away_spreads = []
    totals_over = []
    totals_under = []

    for bk in bookmakers:
        bk_key = bk.get("key", "unknown")
        bk_markets = bk.get("markets", [])

        for market in bk_markets:
            mkey = market.get("key", "")
            outcomes = market.get("outcomes", [])

            if mkey == "h2h":
                for outcome in outcomes:
                    dec_odds = outcome.get("price")
                    if dec_odds is None:
                        continue
                    am = decimal_to_american(dec_odds)
                    if outcome.get("name") == home_team:
                        home_ml_by_bk[bk_key] = am
                    elif outcome.get("name") == away_team:
                        away_ml_by_bk[bk_key] = am

            elif mkey == "spreads":
                for outcome in outcomes:
                    dec_odds = outcome.get("price")
                    point = outcome.get("point")
                    if dec_odds is None:
                        continue
                    am = decimal_to_american(dec_odds)
                    if outcome.get("name") == home_team:
                        home_spreads.append({"odds": am, "spread": point, "bk": bk_key})
                    elif outcome.get("name") == away_team:
                        away_spreads.append({"odds": am, "spread": point, "bk": bk_key})

            elif mkey == "totals":
                for outcome in outcomes:
                    dec_odds = outcome.get("price")
                    point = outcome.get("point")
                    if dec_odds is None:
                        continue
                    am = decimal_to_american(dec_odds)
                    if outcome.get("name") == "Over":
                        totals_over.append({"odds": am, "total": point, "bk": bk_key})
                    elif outcome.get("name") == "Under":
                        totals_under.append({"odds": am, "total": point, "bk": bk_key})

    if not home_ml_by_bk:
        return None

    # Compute fair odds (average across bookmakers, vig-removed)
    home_fair = _compute_fair_odds(home_ml_by_bk, away_ml_by_bk)
    away_fair = _compute_fair_odds(away_ml_by_bk, home_ml_by_bk)

    # Pick "book odds" from the sharpest available book (pinnacle > betfair > first available)
    sharp_priority = ["pinnacle", "betfair_ex_eu", "betfair", "williamhill", "bet365"]
    home_book = _pick_sharp_odds(home_ml_by_bk, sharp_priority)
    away_book = _pick_sharp_odds(away_ml_by_bk, sharp_priority)

    odds = {}

    # Moneyline
    odds["points-home-game-ml-home"] = {
        "oddID": "points-home-game-ml-home",
        "opposingOddID": "points-away-game-ml-away",
        "marketName": "Moneyline",
        "statID": "points",
        "statEntityID": "home",
        "periodID": "game",
        "betTypeID": "ml",
        "sideID": "home",
        "fairOdds": home_fair,
        "bookOdds": home_book or home_fair,
        "openFairOdds": home_fair,  # Use current as open (no historical data)
        "openBookOdds": home_book or home_fair,
        "byBookmaker": {
            bk: {"odds": odds_val, "available": True}
            for bk, odds_val in home_ml_by_bk.items()
        },
    }

    odds["points-away-game-ml-away"] = {
        "oddID": "points-away-game-ml-away",
        "opposingOddID": "points-home-game-ml-home",
        "marketName": "Moneyline",
        "statID": "points",
        "statEntityID": "away",
        "periodID": "game",
        "betTypeID": "ml",
        "sideID": "away",
        "fairOdds": away_fair,
        "bookOdds": away_book or away_fair,
        "openFairOdds": away_fair,
        "openBookOdds": away_book or away_fair,
        "byBookmaker": {
            bk: {"odds": odds_val, "available": True}
            for bk, odds_val in away_ml_by_bk.items()
        },
    }

    # Game spread (from spreads market)
    if home_spreads:
        sp = home_spreads[0]
        odds["games-home-game-sp-home"] = {
            "oddID": "games-home-game-sp-home",
            "fairOdds": sp["odds"],
            "bookOdds": sp["odds"],
            "fairSpread": str(sp.get("spread", "")),
        }
    if away_spreads:
        sp = away_spreads[0]
        odds["games-away-game-sp-away"] = {
            "oddID": "games-away-game-sp-away",
            "fairOdds": sp["odds"],
            "bookOdds": sp["odds"],
            "fairSpread": str(sp.get("spread", "")),
        }

    # Totals (over/under)
    if totals_over:
        t = totals_over[0]
        odds["games-all-game-ou-over"] = {
            "oddID": "games-all-game-ou-over",
            "fairOdds": t["odds"],
            "bookOdds": t["odds"],
            "fairTotal": str(t.get("total", "")),
        }
    if totals_under:
        t = totals_under[0]
        odds["games-all-game-ou-under"] = {
            "oddID": "games-all-game-ou-under",
            "fairOdds": t["odds"],
            "bookOdds": t["odds"],
            "fairTotal": str(t.get("total", "")),
        }

    return odds


def _american_to_prob(odds_str: str) -> Optional[float]:
    """Convert American odds to implied probability."""
    try:
        odds = int(odds_str)
    except (ValueError, TypeError):
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    return 0.5


def _prob_to_american(prob: float) -> str:
    """Convert probability to American odds string."""
    if prob <= 0 or prob >= 1:
        return "-100"
    if prob >= 0.5:
        return str(int(-100 * prob / (1 - prob)))
    else:
        return f"+{int(100 * (1 - prob) / prob)}"


def _compute_fair_odds(side_bk: Dict[str, str], opp_bk: Dict[str, str]) -> str:
    """Compute vig-removed fair odds from bookmaker odds."""
    probs = []
    for bk, odds_str in side_bk.items():
        p = _american_to_prob(odds_str)
        if p:
            probs.append(p)

    opp_probs = []
    for bk, odds_str in opp_bk.items():
        p = _american_to_prob(odds_str)
        if p:
            opp_probs.append(p)

    if not probs:
        return "-100"

    avg_prob = sum(probs) / len(probs)
    avg_opp = sum(opp_probs) / len(opp_probs) if opp_probs else (1 - avg_prob)

    # Remove vig: normalize to sum to 1
    total = avg_prob + avg_opp
    if total > 0:
        fair_prob = avg_prob / total
    else:
        fair_prob = 0.5

    return _prob_to_american(fair_prob)


def _pick_sharp_odds(bk_odds: Dict[str, str], priority: List[str]) -> Optional[str]:
    """Pick odds from the sharpest available bookmaker."""
    for bk in priority:
        if bk in bk_odds:
            return bk_odds[bk]
    # Fallback: first available
    if bk_odds:
        return next(iter(bk_odds.values()))
    return None


# ---------------------------------------------------------------------------
# ESPN public API — primary data source (no key required)
# ---------------------------------------------------------------------------

def _fetch_espn_tennis(date_str: str) -> List[Dict]:
    """
    Fetch tennis matches from ESPN's public scoreboard API.

    This is the most reliable source: returns structured JSON, no auth needed,
    includes player names, set-by-set scores, match status, and start times.

    Args:
        date_str: Date in YYYY-MM-DD format (e.g. "2026-03-19")

    Returns:
        List of raw match dicts ready for _build_event().
    """
    matches = []
    seen_ids = set()  # Deduplicate across endpoints
    # ESPN date format: YYYYMMDD
    espn_date = date_str.replace("-", "")

    for endpoint_league, url in ESPN_ENDPOINTS.items():
        try:
            params = {"dates": espn_date}
            resp = _get_session().get(url, params=params, timeout=15)
            if resp.status_code != 200:
                print(f"  [ESPN] {endpoint_league} HTTP {resp.status_code}")
                continue

            data = resp.json()
            events = data.get("events", [])

            for event in events:
                # ESPN groups competitions inside "groupings" (by round/category)
                groupings = event.get("groupings", [])
                # Also check top-level competitions (some formats)
                top_competitions = event.get("competitions", [])

                tournament_name = event.get("name", "")

                # Process top-level competitions
                for comp in top_competitions:
                    comp_id = comp.get("id", "")
                    if comp_id in seen_ids:
                        continue
                    seen_ids.add(comp_id)
                    match = _parse_espn_competition(
                        comp, endpoint_league, tournament_name, grouping_name=""
                    )
                    if match and match.get("starts_at", "")[:10] == date_str:
                        matches.append(match)

                # Process grouped competitions (with category info)
                for grp in groupings:
                    # The grouping tells us Men's/Women's Singles
                    grp_name = grp.get("displayName", "")
                    grp_slug = grp.get("slug", "")
                    for comp in grp.get("competitions", []):
                        comp_id = comp.get("id", "")
                        if comp_id in seen_ids:
                            continue
                        seen_ids.add(comp_id)
                        match = _parse_espn_competition(
                            comp, endpoint_league, tournament_name, grouping_name=grp_name
                        )
                        if match and match.get("starts_at", "")[:10] == date_str:
                            matches.append(match)

        except Exception as e:
            print(f"  [ESPN] {endpoint_league} error: {e}")

    return matches


def _parse_espn_competition(
    comp: Dict, endpoint_league: str, tournament: str, grouping_name: str = ""
) -> Optional[Dict]:
    """Parse a single ESPN competition (match) into our raw match format."""
    # Detect actual league from grouping name (Women's Singles = WTA, Men's = ATP)
    grp_lower = grouping_name.lower()
    if "women" in grp_lower:
        league = "WTA"
    elif "men" in grp_lower:
        league = "ATP"
    else:
        # Also check the competition's own type field
        comp_type = comp.get("type", {})
        comp_text = comp_type.get("text", "").lower() if isinstance(comp_type, dict) else ""
        if "women" in comp_text:
            league = "WTA"
        elif "men" in comp_text:
            league = "ATP"
        else:
            league = endpoint_league  # Fallback to endpoint

    competitors = comp.get("competitors", [])
    if len(competitors) < 2:
        return None

    # ESPN uses homeAway field
    home_comp = None
    away_comp = None
    for c in competitors:
        if c.get("homeAway") == "home":
            home_comp = c
        elif c.get("homeAway") == "away":
            away_comp = c

    # Fallback: if no homeAway, use order (first=home, second=away)
    if not home_comp or not away_comp:
        home_comp = competitors[0]
        away_comp = competitors[1]

    # Player names
    home_athlete = home_comp.get("athlete", {})
    away_athlete = away_comp.get("athlete", {})
    home_name = home_athlete.get("displayName") or home_athlete.get("fullName", "Unknown")
    away_name = away_athlete.get("displayName") or away_athlete.get("fullName", "Unknown")

    # Status
    status_obj = comp.get("status", {})
    status_type = status_obj.get("type", {})
    status_name = status_type.get("name", "")
    completed = status_type.get("completed", False)
    state = status_type.get("state", "")  # "pre", "in", "post"
    started = state in ("in", "post")
    live = state == "in"

    # Scores (sets won)
    # In tennis, linescores contains individual set game counts
    home_linescores = home_comp.get("linescores", [])
    away_linescores = away_comp.get("linescores", [])

    # Match score = sets won (count sets where player won)
    home_sets_won = 0
    away_sets_won = 0
    sets = {}

    for i, (h_set, a_set) in enumerate(zip(home_linescores, away_linescores)):
        set_num = f"{i + 1}s"
        h_games = int(h_set.get("value", 0))
        a_games = int(a_set.get("value", 0))
        sets[set_num] = {
            "home_games": h_games,
            "away_games": a_games,
        }
        if h_games > a_games:
            home_sets_won += 1
        elif a_games > h_games:
            away_sets_won += 1

    # If winner flag is available, use it
    if completed:
        if home_comp.get("winner"):
            # Ensure winner has more sets
            if home_sets_won <= away_sets_won:
                home_sets_won = max(home_sets_won, away_sets_won + 1)
        elif away_comp.get("winner"):
            if away_sets_won <= home_sets_won:
                away_sets_won = max(away_sets_won, home_sets_won + 1)

    # Start time
    starts_at = comp.get("startDate") or comp.get("date", "")
    if starts_at and not starts_at.endswith("Z"):
        starts_at = starts_at + "Z"
    # Normalize to pipeline format
    if starts_at and "." not in starts_at:
        starts_at = starts_at.replace("Z", ".000Z")

    return {
        "home": home_name,
        "away": away_name,
        "league": league,
        "score_home": home_sets_won,
        "score_away": away_sets_won,
        "sets": sets,
        "started": started,
        "completed": completed,
        "live": live,
        "starts_at": starts_at,
        "tournament": tournament,
        "espn_id": comp.get("id", ""),
    }


# ---------------------------------------------------------------------------
# Flashscore scraping — tertiary fallback
# ---------------------------------------------------------------------------

def _fetch_flashscore_page(url: str, retries: int = 3) -> Optional[str]:
    """Fetch a Flashscore page with retries."""
    for attempt in range(retries):
        try:
            resp = _get_session().get(url, timeout=15)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 403:
                print(f"  [Flashscore] 403 Forbidden — may need different approach")
                return None
            print(f"  [Flashscore] HTTP {resp.status_code} (attempt {attempt + 1})")
        except Exception as e:
            print(f"  [Flashscore] Error (attempt {attempt + 1}): {e}")
        time.sleep(1 * (attempt + 1))
    return None


def _parse_flashscore_tennis(html: str, target_date: str = None) -> List[Dict]:
    """
    Parse Flashscore tennis page HTML into raw match dicts.
    Returns list of {home, away, league, time, score_home, score_away, sets, status}.
    """
    matches = []
    if not html:
        return matches

    soup = BeautifulSoup(html, "html.parser")

    # Flashscore uses dynamic JS rendering, so direct HTML parsing may not work.
    # We look for data embedded in script tags or structured divs.
    # The actual structure depends on what Flashscore serves to our user-agent.

    # Try to find match containers
    # Flashscore typically has div.event__match or similar structures
    match_divs = soup.find_all("div", class_=re.compile(r"event__match"))

    if not match_divs:
        # Try alternative selectors
        match_divs = soup.find_all("div", class_=re.compile(r"sportName tennis"))
        if not match_divs:
            # Try finding any structured data
            match_divs = soup.find_all("div", {"id": re.compile(r"g_\d+_")})

    for div in match_divs:
        try:
            # Extract player names
            home_el = div.find(class_=re.compile(r"event__participant--home"))
            away_el = div.find(class_=re.compile(r"event__participant--away"))
            if not home_el or not away_el:
                continue

            home_name = home_el.get_text(strip=True)
            away_name = away_el.get_text(strip=True)

            # Extract scores
            scores = div.find_all(class_=re.compile(r"event__score"))
            home_score = 0
            away_score = 0
            if len(scores) >= 2:
                try:
                    home_score = int(scores[0].get_text(strip=True))
                    away_score = int(scores[1].get_text(strip=True))
                except ValueError:
                    pass

            # Extract set scores
            sets = {}
            set_scores = div.find_all(class_=re.compile(r"event__part"))
            for i, part in enumerate(set_scores):
                set_num = f"{i // 2 + 1}s"
                vals = part.find_all("span")
                if len(vals) >= 2:
                    try:
                        if i % 2 == 0:
                            sets.setdefault(set_num, {})["home_games"] = int(vals[0].get_text(strip=True))
                        else:
                            sets.setdefault(set_num, {})["away_games"] = int(vals[0].get_text(strip=True))
                    except ValueError:
                        pass

            # Extract time/status
            time_el = div.find(class_=re.compile(r"event__time"))
            status_text = time_el.get_text(strip=True) if time_el else ""

            # Determine league from parent header
            league = "ATP"  # Default
            parent = div.find_previous(class_=re.compile(r"event__title"))
            if parent:
                title_text = parent.get_text(strip=True).upper()
                if "WTA" in title_text:
                    league = "WTA"

            matches.append({
                "home": home_name,
                "away": away_name,
                "league": league,
                "score_home": home_score,
                "score_away": away_score,
                "sets": sets,
                "status_text": status_text,
            })
        except Exception:
            continue

    return matches


# ---------------------------------------------------------------------------
# Alternative data source: API-Tennis or sofascore public API
# ---------------------------------------------------------------------------

def _fetch_sofascore_tennis(date_str: str) -> List[Dict]:
    """
    Fetch tennis matches from Sofascore's public API for a given date.
    This is more reliable than HTML scraping.
    """
    matches = []
    try:
        url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{date_str}"
        resp = _get_session().get(url, timeout=15)
        if resp.status_code != 200:
            return matches

        data = resp.json()
        events = data.get("events", [])

        for event in events:
            try:
                tournament = event.get("tournament", {})
                category = tournament.get("category", {})
                cat_name = category.get("name", "").upper()

                # Determine league
                league = "ATP"
                tourn_name = tournament.get("name", "").upper()
                if "WTA" in tourn_name or "WOMEN" in cat_name:
                    league = "WTA"

                home_team = event.get("homeTeam", {})
                away_team = event.get("awayTeam", {})
                home_name = home_team.get("name", "Unknown")
                away_name = away_team.get("name", "Unknown")

                # Status
                status_obj = event.get("status", {})
                status_code = status_obj.get("code", 0)
                # 0=not started, 6=in progress, 7=interrupted, 100=finished
                started = status_code >= 6
                completed = status_code == 100
                live = status_code in (6, 7)

                # Score
                home_score_obj = event.get("homeScore", {})
                away_score_obj = event.get("awayScore", {})
                home_score = home_score_obj.get("current", 0) or 0
                away_score = away_score_obj.get("current", 0) or 0

                # Set scores
                sets = {}
                for period_num in range(1, 6):
                    h_key = f"period{period_num}"
                    if h_key in home_score_obj and h_key in away_score_obj:
                        sets[f"{period_num}s"] = {
                            "home_games": home_score_obj[h_key],
                            "away_games": away_score_obj[h_key],
                        }

                # Start time
                start_timestamp = event.get("startTimestamp", 0)
                starts_at = ""
                if start_timestamp:
                    starts_at = datetime.fromtimestamp(
                        start_timestamp, tz=timezone.utc
                    ).strftime("%Y-%m-%dT%H:%M:%S.000Z")

                matches.append({
                    "home": home_name,
                    "away": away_name,
                    "league": league,
                    "score_home": home_score,
                    "score_away": away_score,
                    "sets": sets,
                    "started": started,
                    "completed": completed,
                    "live": live,
                    "starts_at": starts_at,
                    "tournament": tournament.get("name", ""),
                    "sofascore_id": event.get("id"),
                })
            except Exception:
                continue

    except Exception as e:
        print(f"  [Sofascore] Error: {e}")

    return matches


def _fetch_api_tennis_free(date_str: str) -> List[Dict]:
    """
    Fetch from a free tennis data endpoint as fallback.
    Uses the open Tennis Live Scores approach.
    """
    matches = []
    try:
        # Try the open Flashscore API (JSON endpoint)
        # This is an undocumented but commonly used endpoint
        url = f"https://flashlive-data.p.rapidapi.com/v1/events/list"
        # This requires a RapidAPI key, so skip if not available
        # Instead, try the basic Flashscore HTML
        pass
    except Exception:
        pass
    return matches


# ---------------------------------------------------------------------------
# Build pipeline-compatible events
# ---------------------------------------------------------------------------

def _build_event(
    match: Dict,
    odds_lookup: Dict[str, Dict],
    date_str: str,
) -> Dict:
    """
    Convert a raw scraped match + odds into the exact SportsGameOdds event format
    that the prediction pipeline expects.
    """
    home_name = match["home"]
    away_name = match["away"]
    league = match.get("league", "ATP")
    starts_at = match.get("starts_at", f"{date_str}T12:00:00.000Z")

    home_tid = _make_team_id(home_name, league)
    away_tid = _make_team_id(away_name, league)
    event_id = _make_event_id(home_name, away_name, date_str)

    # Match status
    started = match.get("started", False)
    completed = match.get("completed", False)
    live = match.get("live", False)

    if completed:
        display_short = "F"
    elif live:
        display_short = "Live"
    else:
        display_short = "NS"

    # Scores
    home_score = match.get("score_home", 0)
    away_score = match.get("score_away", 0)

    # Results (set-by-set)
    results = {}
    sets_data = match.get("sets", {})
    for set_key, set_val in sets_data.items():
        results[set_key] = {
            "home": {"games": set_val.get("home_games", 0)},
            "away": {"games": set_val.get("away_games", 0)},
        }

    # Add game-level results
    if completed or started:
        results["game"] = {
            "home": {"points": home_score},
            "away": {"points": away_score},
        }

    # Try to match odds
    odds = {}
    matched_odds = _match_odds(home_name, away_name, odds_lookup)
    if matched_odds:
        odds = matched_odds

    # Build the event in pipeline format
    event = {
        "eventID": event_id,
        "sportID": "TENNIS",
        "leagueID": league,
        "type": "match",
        "teams": {
            "home": {
                "teamID": home_tid,
                "names": _make_names(home_name),
                "colors": {
                    "primaryContrast": "#FFFFFF",
                    "primary": "#374DF5",
                },
                "statEntityID": "home",
                "score": home_score,
            },
            "away": {
                "teamID": away_tid,
                "names": _make_names(away_name),
                "colors": {
                    "primaryContrast": "#FFFFFF",
                    "primary": "#374DF5",
                },
                "statEntityID": "away",
                "score": away_score,
            },
        },
        "status": {
            "started": started,
            "completed": completed,
            "cancelled": False,
            "ended": completed,
            "live": live,
            "delayed": False,
            "displayShort": display_short,
            "displayLong": "Final" if completed else ("Live" if live else "Not Started"),
            "startsAt": starts_at,
            "hardStart": True,
            "oddsPresent": bool(odds),
            "oddsAvailable": bool(odds) and not completed,
            "finalized": completed,
        },
        "odds": odds,
        "results": results,
        "info": {},
        "links": {},
        "players": {},
        "_source": "scraper",
        "_tournament": match.get("tournament", ""),
    }

    return event


def _match_odds(home_name: str, away_name: str, odds_lookup: Dict) -> Optional[Dict]:
    """Try to match a scraped match to odds data using fuzzy name matching."""
    if not odds_lookup:
        return None

    # Direct match
    key = f"{_normalize_name(home_name)}|{_normalize_name(away_name)}"
    if key in odds_lookup:
        return odds_lookup[key].get("odds")

    # Reverse match (home/away might be swapped)
    rev_key = f"{_normalize_name(away_name)}|{_normalize_name(home_name)}"
    if rev_key in odds_lookup:
        # Need to swap home/away in the odds
        odds = odds_lookup[rev_key].get("odds", {})
        return _swap_odds_home_away(odds)

    # Fuzzy match: try matching by last name
    for okey, odata in odds_lookup.items():
        oh = odata.get("home_team", "")
        oa = odata.get("away_team", "")
        if _names_match(home_name, oh) and _names_match(away_name, oa):
            return odata.get("odds")
        if _names_match(home_name, oa) and _names_match(away_name, oh):
            return _swap_odds_home_away(odata.get("odds", {}))

    return None


def _swap_odds_home_away(odds: Dict) -> Dict:
    """Swap home and away in an odds dict."""
    swapped = {}
    for key, val in odds.items():
        new_key = key
        if "-home-" in key:
            new_key = key.replace("-home-", "-SWAP-").replace("-away-", "-home-").replace("-SWAP-", "-away-")
        elif "-away-" in key:
            new_key = key.replace("-away-", "-SWAP-").replace("-home-", "-away-").replace("-SWAP-", "-home-")

        new_val = dict(val) if isinstance(val, dict) else val
        if isinstance(new_val, dict):
            # Swap statEntityID and sideID
            if new_val.get("statEntityID") == "home":
                new_val["statEntityID"] = "away"
            elif new_val.get("statEntityID") == "away":
                new_val["statEntityID"] = "home"
            if new_val.get("sideID") == "home":
                new_val["sideID"] = "away"
            elif new_val.get("sideID") == "away":
                new_val["sideID"] = "home"

        swapped[new_key] = new_val
    return swapped


# ---------------------------------------------------------------------------
# Public API — main entry points
# ---------------------------------------------------------------------------

def fetch_today_matches() -> List[Dict]:
    """
    Fetch today's tennis matches with odds.

    Returns list of events in the exact SportsGameOdds format
    that the prediction pipeline expects.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return fetch_matches_for_date(today)


def fetch_matches_for_date(date_str: str) -> List[Dict]:
    """
    Fetch matches for a specific date.

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns:
        List of events in pipeline-compatible format.
    """
    print(f"\n{'='*60}")
    print(f"SCRAPING TENNIS MATCHES FOR {date_str}")
    print(f"{'='*60}")

    # Step 1: Fetch odds from The Odds API
    print("\n[1/3] Fetching odds...")
    odds_lookup = fetch_all_odds()

    # Step 2: Fetch match data from ESPN (primary source)
    print("\n[2/3] Fetching match data from ESPN...")
    raw_matches = _fetch_espn_tennis(date_str)
    print(f"  Found {len(raw_matches)} matches from ESPN")

    # Step 2b: Fallback to Sofascore if ESPN fails
    if not raw_matches:
        print("  ESPN returned no data, trying Sofascore...")
        raw_matches = _fetch_sofascore_tennis(date_str)
        print(f"  Found {len(raw_matches)} matches from Sofascore")

    # Step 2c: Fallback to Flashscore HTML
    if not raw_matches:
        print("  Sofascore returned no data, trying Flashscore...")
        html = _fetch_flashscore_page(f"{FLASHSCORE_API}/tennis/")
        if html:
            raw_matches = _parse_flashscore_tennis(html, date_str)
            print(f"  Found {len(raw_matches)} matches from Flashscore")

    # Step 2d: If still no match data, create events from odds data alone
    if not raw_matches and odds_lookup:
        print("  No scrape data — building events from odds data alone...")
        for key, odata in odds_lookup.items():
            raw_matches.append({
                "home": odata["home_team"],
                "away": odata["away_team"],
                "league": odata["league"],
                "score_home": 0,
                "score_away": 0,
                "sets": {},
                "started": False,
                "completed": False,
                "live": False,
                "starts_at": odata.get("commence_time", f"{date_str}T12:00:00.000Z"),
                "tournament": odata.get("sport_key", ""),
            })
        print(f"  Created {len(raw_matches)} events from odds data")

    # Step 3: Build pipeline-compatible events
    print("\n[3/3] Building pipeline events...")
    events = []
    matched_odds_count = 0
    for match in raw_matches:
        event = _build_event(match, odds_lookup, date_str)
        if event.get("odds"):
            matched_odds_count += 1
        events.append(event)

    # Save to cache
    cache_path = os.path.join(CACHE_DIR, f"scraped_events_{date_str}.json")
    with open(cache_path, "w") as f:
        json.dump(events, f, indent=2)

    # Summary
    atp = [e for e in events if e["leagueID"] == "ATP"]
    wta = [e for e in events if e["leagueID"] == "WTA"]
    with_odds = [e for e in events if e.get("odds")]

    print(f"\n{'='*60}")
    print(f"SCRAPE COMPLETE")
    print(f"  Total matches:  {len(events)}")
    print(f"  ATP:            {len(atp)}")
    print(f"  WTA:            {len(wta)}")
    print(f"  With odds:      {len(with_odds)}")
    print(f"  Cached to:      {cache_path}")
    print(f"{'='*60}")

    return events


def fetch_completed_matches(date_str: str) -> List[Dict]:
    """
    Fetch completed matches for a given date (for backtesting/results).

    Args:
        date_str: Date in YYYY-MM-DD format.

    Returns:
        List of completed events in pipeline format.
    """
    events = fetch_matches_for_date(date_str)
    completed = [e for e in events if e.get("status", {}).get("completed")]
    print(f"  Completed matches: {len(completed)} / {len(events)}")
    return completed


def load_cached_events(date_str: str) -> List[Dict]:
    """Load previously cached scraped events for a date."""
    cache_path = os.path.join(CACHE_DIR, f"scraped_events_{date_str}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Integration: drop-in replacement for fetch_upcoming_events in app.py
# ---------------------------------------------------------------------------

def fetch_upcoming_events(league: str = None, limit: int = 30) -> List[Dict]:
    """
    Drop-in replacement for app.py's fetch_upcoming_events().

    Fetches today's matches using the scraper, filters by league,
    and returns up to `limit` events.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Try cached first (avoid re-scraping within the same hour)
    cache_path = os.path.join(CACHE_DIR, f"scraped_events_{today}.json")
    events = []

    if os.path.exists(cache_path):
        mod_time = os.path.getmtime(cache_path)
        age_minutes = (time.time() - mod_time) / 60
        if age_minutes < 30:
            # Use cache if less than 30 minutes old
            with open(cache_path) as f:
                events = json.load(f)

    if not events:
        events = fetch_today_matches()

    # Filter by league
    if league and league != "ALL":
        events = [e for e in events if e.get("leagueID") == league]

    # Filter to upcoming (not completed)
    upcoming = [e for e in events if not e.get("status", {}).get("completed")]

    return upcoming[:limit]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Tennis Match Scraper")
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date to fetch matches for (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--completed",
        action="store_true",
        help="Fetch only completed matches (for results/backtesting).",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only load from cache, don't scrape.",
    )
    args = parser.parse_args()

    if args.cache_only:
        events = load_cached_events(args.date)
        if not events:
            print(f"No cached data for {args.date}")
            sys.exit(1)
    elif args.completed:
        events = fetch_completed_matches(args.date)
    else:
        events = fetch_matches_for_date(args.date)

    if not events:
        print("\nNo matches found.")
        sys.exit(0)

    # Print summary
    print(f"\n{'='*60}")
    print(f"MATCH SUMMARY — {args.date}")
    print(f"{'='*60}")

    for i, event in enumerate(events, 1):
        teams = event.get("teams", {})
        home = teams.get("home", {}).get("names", {}).get("long", "?")
        away = teams.get("away", {}).get("names", {}).get("long", "?")
        league = event.get("leagueID", "?")
        status = event.get("status", {}).get("displayShort", "?")
        has_odds = "Y" if event.get("odds") else "N"

        home_score = teams.get("home", {}).get("score", "")
        away_score = teams.get("away", {}).get("score", "")
        score_str = f"  [{home_score}-{away_score}]" if (home_score or away_score) else ""

        # Show odds if available
        odds_str = ""
        ml_home = event.get("odds", {}).get("points-home-game-ml-home", {})
        ml_away = event.get("odds", {}).get("points-away-game-ml-away", {})
        if ml_home and ml_away:
            odds_str = f"  ML: {ml_home.get('fairOdds', '?')}/{ml_away.get('fairOdds', '?')}"

        print(f"  {i:3}. [{league:3}] {home} vs {away}  ({status}){score_str}{odds_str}  [odds:{has_odds}]")

    print(f"\n  Total: {len(events)} matches")

    # Verify pipeline compatibility
    print(f"\n  Pipeline compatibility check:")
    good = 0
    for event in events:
        has_eid = bool(event.get("eventID"))
        has_teams = bool(event.get("teams", {}).get("home", {}).get("teamID"))
        has_status = bool(event.get("status", {}).get("startsAt"))
        if has_eid and has_teams and has_status:
            good += 1
    print(f"    {good}/{len(events)} events have all required fields")


if __name__ == "__main__":
    main()
