"""
Bulk data fetcher — download EVERYTHING from the API before the key expires.
Fetches all tennis events with full odds, paginating aggressively.
Also fetches teams/players data and any other useful metadata.
"""

import json
import os
import time
import requests

API_KEY = "052ad81a89543e7c59737853ec205d2c"
HEADERS = {"X-Api-Key": API_KEY}
BASE = "https://api.sportsgameodds.com/v2"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_all_month_events(year, month, max_pages=100):
    """Fetch ALL events for a month, no limits."""
    starts_after = f"{year}-{month:02d}-01T00:00:00.000Z"
    if month == 12:
        starts_before = f"{year + 1}-01-01T00:00:00.000Z"
    else:
        starts_before = f"{year}-{month + 1:02d}-01T00:00:00.000Z"

    all_events = []
    cursor = None
    page = 0

    while page < max_pages:
        params = {
            "sportID": "TENNIS",
            "startsAfter": starts_after,
            "startsBefore": starts_before,
            "includeOdds": "true",
            "limit": 50,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(f"{BASE}/events", headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 401:
                print("  API KEY EXPIRED!")
                return all_events
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break

        events = data.get("data", [])
        if not events:
            break

        for e in events:
            status = e.get("status", {})
            if status.get("cancelled"):
                continue
            if e.get("odds"):
                all_events.append(e)

        cursor = data.get("nextCursor")
        if not cursor:
            break

        page += 1
        time.sleep(0.15)

    return all_events


def fetch_teams():
    """Fetch all tennis teams/players."""
    all_teams = []
    for league in ["ATP", "WTA"]:
        cursor = None
        page = 0
        while page < 50:
            params = {"sportID": "TENNIS", "leagueID": league, "limit": 50}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = requests.get(f"{BASE}/teams", headers=HEADERS, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  Teams error: {e}")
                break

            teams = data.get("data", [])
            all_teams.extend(teams)
            cursor = data.get("nextCursor")
            if not cursor or not teams:
                break
            page += 1
            time.sleep(0.15)

    return all_teams


def fetch_markets():
    """Fetch all available tennis markets."""
    try:
        resp = requests.get(f"{BASE}/markets", headers=HEADERS,
                            params={"sportID": "TENNIS"}, timeout=20)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


def fetch_bookmakers():
    """Fetch all bookmakers."""
    try:
        resp = requests.get(f"{BASE}/bookmakers", headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


def main():
    print("=" * 60)
    print("BULK DATA FETCH — Downloading everything before key expires")
    print("=" * 60)

    # 1. Metadata
    print("\n[1/4] Fetching metadata...")
    teams = fetch_teams()
    with open(os.path.join(CACHE_DIR, "all_teams.json"), "w") as f:
        json.dump(teams, f)
    print(f"  Teams/Players: {len(teams)}")

    markets = fetch_markets()
    with open(os.path.join(CACHE_DIR, "all_markets.json"), "w") as f:
        json.dump(markets, f)
    print(f"  Markets: {len(markets)}")

    bookmakers = fetch_bookmakers()
    with open(os.path.join(CACHE_DIR, "all_bookmakers.json"), "w") as f:
        json.dump(bookmakers, f)
    print(f"  Bookmakers: {len(bookmakers)}")

    # 2. Re-fetch ALL months with no page limit
    print("\n[2/4] Re-fetching all monthly events (no page limit)...")
    total_events = 0
    months = []
    for year in [2024, 2025, 2026]:
        start_m = 2 if year == 2024 else 1
        end_m = 3 if year == 2026 else 12
        for month in range(start_m, end_m + 1):
            months.append((year, month))

    for year, month in months:
        cache_file = os.path.join(CACHE_DIR, f"events_{year}_{month:02d}.json")

        # Check existing
        existing = 0
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                existing = len(json.load(f))

        events = fetch_all_month_events(year, month)
        new_count = len(events)

        if new_count >= existing:
            with open(cache_file, "w") as f:
                json.dump(events, f)
            gained = new_count - existing
            status = f"+{gained} new" if gained > 0 else "same"
        else:
            status = f"kept existing ({existing})"
            new_count = existing

        total_events += new_count
        print(f"  {year}-{month:02d}: {new_count} events ({status})")

    # 3. Fetch upcoming events (next 2 weeks)
    print("\n[3/4] Fetching all upcoming events...")
    upcoming = []
    cursor = None
    for page in range(20):
        params = {
            "sportID": "TENNIS",
            "startsAfter": "2026-03-18T00:00:00.000Z",
            "startsBefore": "2026-04-01T00:00:00.000Z",
            "includeOdds": "true",
            "limit": 50,
        }
        if cursor:
            params["cursor"] = cursor
        try:
            resp = requests.get(f"{BASE}/events", headers=HEADERS, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            upcoming.extend(data.get("data", []))
            cursor = data.get("nextCursor")
            if not cursor:
                break
            time.sleep(0.15)
        except Exception:
            break

    with open(os.path.join(CACHE_DIR, "upcoming_snapshot.json"), "w") as f:
        json.dump(upcoming, f)
    print(f"  Upcoming events: {len(upcoming)}")

    # 4. Summary
    print("\n[4/4] Fetching sports & leagues reference data...")
    for endpoint in ["sports", "leagues"]:
        try:
            resp = requests.get(f"{BASE}/{endpoint}", headers=HEADERS, timeout=20)
            resp.raise_for_status()
            with open(os.path.join(CACHE_DIR, f"ref_{endpoint}.json"), "w") as f:
                json.dump(resp.json().get("data", []), f)
        except Exception:
            pass

    print(f"\n{'=' * 60}")
    print(f"DONE!")
    print(f"Total historical events: {total_events:,}")
    print(f"Upcoming events: {len(upcoming)}")
    print(f"Teams/Players: {len(teams)}")
    print(f"Markets: {len(markets)}")
    print(f"Bookmakers: {len(bookmakers)}")

    # Disk usage
    cache_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f))
                     for f in os.listdir(CACHE_DIR)) / 1024 / 1024
    print(f"Cache size: {cache_size:.0f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
