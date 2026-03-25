"""
Tennis Match Prediction Engine

Uses odds data from SportsGameOdds API to generate win probabilities,
confidence ratings, and value bet identification.
"""

import statistics


def american_to_probability(odds_str):
    """Convert American odds string to implied probability (0-1)."""
    try:
        odds = int(odds_str)
    except (ValueError, TypeError):
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    elif odds > 0:
        return 100 / (odds + 100)
    return 0.5


def probability_to_american(prob):
    """Convert probability (0-1) to American odds string."""
    if prob is None or prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return f"{int(-100 * prob / (1 - prob))}"
    else:
        return f"+{int(100 * (1 - prob) / prob)}"


def extract_odds_data(event, use_opening=False):
    """Extract structured odds from an event's odds object.

    Args:
        event: Event dict from API.
        use_opening: If True, use opening odds (for backtesting).
                     If False, use current/closing odds (for live predictions).
    """
    odds = event.get("odds", {})
    if not odds:
        return None

    home_ml_key = "points-home-game-ml-home"
    away_ml_key = "points-away-game-ml-away"

    home_odds = odds.get(home_ml_key)
    away_odds = odds.get(away_ml_key)

    if not home_odds or not away_odds:
        return None

    if use_opening:
        # For backtesting: use pre-match opening odds only
        home_fair = home_odds.get("openFairOdds")
        away_fair = away_odds.get("openFairOdds")
        home_book = home_odds.get("openBookOdds")
        away_book = away_odds.get("openBookOdds")
        # No movement data available in backtest mode (we only have open)
        open_home_fair = None
        open_away_fair = None
        open_home_book = None
        open_away_book = None
    else:
        home_fair = home_odds.get("fairOdds")
        away_fair = away_odds.get("fairOdds")
        home_book = home_odds.get("bookOdds")
        away_book = away_odds.get("bookOdds")
        open_home_fair = home_odds.get("openFairOdds")
        open_away_fair = away_odds.get("openFairOdds")
        open_home_book = home_odds.get("openBookOdds")
        open_away_book = away_odds.get("openBookOdds")

    if not home_fair or not away_fair:
        return None

    data = {
        "home": {
            "fair_odds": home_fair,
            "book_odds": home_book,
            "open_fair_odds": open_home_fair,
            "open_book_odds": open_home_book,
            "by_bookmaker": {},
        },
        "away": {
            "fair_odds": away_fair,
            "book_odds": away_book,
            "open_fair_odds": open_away_fair,
            "open_book_odds": open_away_book,
            "by_bookmaker": {},
        },
    }

    # Extract individual bookmaker odds
    for bk, bk_data in home_odds.get("byBookmaker", {}).items():
        if use_opening:
            odds_val = bk_data.get("openOdds")
        else:
            odds_val = bk_data.get("odds")
        if odds_val:
            data["home"]["by_bookmaker"][bk] = odds_val

    for bk, bk_data in away_odds.get("byBookmaker", {}).items():
        if use_opening:
            odds_val = bk_data.get("openOdds")
        else:
            odds_val = bk_data.get("odds")
        if odds_val:
            data["away"]["by_bookmaker"][bk] = odds_val

    return data


def calculate_prediction(event, use_opening=False):
    """
    Generate a match prediction from event data.

    Args:
        event: Event dict from API.
        use_opening: If True, use opening odds only (for backtesting).

    Returns dict with:
      - home/away player info
      - win probabilities
      - confidence rating
      - value bets
      - odds movement
    """
    teams = event.get("teams", {})
    home_team = teams.get("home", {})
    away_team = teams.get("away", {})

    home_names = home_team.get("names", {})
    away_names = away_team.get("names", {})

    prediction = {
        "event_id": event.get("eventID"),
        "league": event.get("leagueID"),
        "starts_at": event.get("status", {}).get("startsAt"),
        "home": {
            "name": home_names.get("long", "Unknown"),
            "short": home_names.get("short", "???"),
        },
        "away": {
            "name": away_names.get("long", "Unknown"),
            "short": away_names.get("short", "???"),
        },
        "has_odds": False,
    }

    odds_data = extract_odds_data(event, use_opening=use_opening)
    if not odds_data:
        return prediction

    prediction["has_odds"] = True

    # --- Signal 1: Fair Odds (vig-removed, best single signal) ---
    home_fair_prob = american_to_probability(odds_data["home"]["fair_odds"])
    away_fair_prob = american_to_probability(odds_data["away"]["fair_odds"])

    # --- Signal 2: Bookmaker Consensus ---
    home_bk_probs = []
    away_bk_probs = []
    for bk, odds_str in odds_data["home"]["by_bookmaker"].items():
        p = american_to_probability(odds_str)
        if p:
            home_bk_probs.append(p)
    for bk, odds_str in odds_data["away"]["by_bookmaker"].items():
        p = american_to_probability(odds_str)
        if p:
            away_bk_probs.append(p)

    home_consensus = statistics.mean(home_bk_probs) if home_bk_probs else None
    away_consensus = statistics.mean(away_bk_probs) if away_bk_probs else None

    # Normalize consensus to sum to 1
    if home_consensus and away_consensus:
        total = home_consensus + away_consensus
        home_consensus /= total
        away_consensus /= total

    # --- Signal 3: Odds Movement ---
    home_open_prob = american_to_probability(odds_data["home"]["open_fair_odds"])
    home_current_prob = home_fair_prob
    movement = None
    if home_open_prob and home_current_prob:
        movement = home_current_prob - home_open_prob  # positive = home strengthened

    # --- Composite Prediction ---
    # Weight: 50% fair odds, 35% consensus, 15% movement-adjusted
    signals = []
    weights = []

    if home_fair_prob:
        signals.append(home_fair_prob)
        weights.append(0.50)

    if home_consensus:
        signals.append(home_consensus)
        weights.append(0.35)

    if home_fair_prob and movement is not None:
        adjusted = home_fair_prob + (movement * 0.3)
        adjusted = max(0.02, min(0.98, adjusted))
        signals.append(adjusted)
        weights.append(0.15)

    if signals:
        total_weight = sum(weights)
        home_win_prob = sum(s * w for s, w in zip(signals, weights)) / total_weight
        away_win_prob = 1 - home_win_prob
    else:
        home_win_prob = 0.5
        away_win_prob = 0.5

    prediction["home"]["win_prob"] = round(home_win_prob * 100, 1)
    prediction["away"]["win_prob"] = round(away_win_prob * 100, 1)
    prediction["home"]["fair_odds"] = odds_data["home"]["fair_odds"]
    prediction["away"]["fair_odds"] = odds_data["away"]["fair_odds"]
    prediction["home"]["book_odds"] = odds_data["home"]["book_odds"]
    prediction["away"]["book_odds"] = odds_data["away"]["book_odds"]

    # --- Confidence ---
    if home_bk_probs and len(home_bk_probs) >= 3:
        # Live mode: based on bookmaker agreement + count
        spread = max(home_bk_probs) - min(home_bk_probs)
        book_count_factor = min(len(home_bk_probs) / 20, 1.0)
        agreement_factor = max(0, 1 - (spread * 3))
        confidence = (agreement_factor * 0.7 + book_count_factor * 0.3) * 100
        prediction["confidence"] = round(min(confidence, 99), 0)
        prediction["num_bookmakers"] = len(home_bk_probs)
    else:
        # Backtest mode or few bookmakers: derive confidence from odds strength
        # + agreement between fair odds and book odds
        winner_prob = max(home_win_prob, away_win_prob)
        prob_strength = min((winner_prob - 0.5) * 4, 1.0)  # 75%+ → full strength

        # Fair vs book agreement
        home_book_prob = american_to_probability(odds_data["home"]["book_odds"])
        fair_book_agreement = 1.0
        if home_book_prob and home_fair_prob:
            diff = abs(home_fair_prob - home_book_prob)
            fair_book_agreement = max(0, 1 - diff * 5)

        confidence = (prob_strength * 0.6 + fair_book_agreement * 0.4) * 100
        prediction["confidence"] = round(min(max(confidence, 5), 95), 0)
        prediction["num_bookmakers"] = len(home_bk_probs)

    # --- Predicted Winner ---
    if home_win_prob >= away_win_prob:
        prediction["predicted_winner"] = "home"
        prediction["winner_name"] = prediction["home"]["name"]
        prediction["winner_prob"] = prediction["home"]["win_prob"]
    else:
        prediction["predicted_winner"] = "away"
        prediction["winner_name"] = prediction["away"]["name"]
        prediction["winner_prob"] = prediction["away"]["win_prob"]

    # --- Value Bets ---
    prediction["value_bets"] = find_value_bets(odds_data, home_win_prob, away_win_prob, prediction)

    # --- Odds Movement ---
    if movement is not None:
        direction = "home" if movement > 0.01 else ("away" if movement < -0.01 else "stable")
        prediction["movement"] = {
            "direction": direction,
            "magnitude": round(abs(movement) * 100, 1),
        }

    # --- Bookmaker odds for comparison ---
    bk_comparison = []
    all_bookmakers = set(list(odds_data["home"]["by_bookmaker"].keys()) +
                         list(odds_data["away"]["by_bookmaker"].keys()))
    for bk in sorted(all_bookmakers):
        h = odds_data["home"]["by_bookmaker"].get(bk)
        a = odds_data["away"]["by_bookmaker"].get(bk)
        if h and a:
            bk_comparison.append({
                "name": bk,
                "home_odds": h,
                "away_odds": a,
            })
    prediction["bookmakers"] = bk_comparison

    return prediction


def find_value_bets(odds_data, home_prob, away_prob, prediction):
    """Identify value bets using mega XGBoost model with NN directional confirmation.

    Requirements (all must be met):
      1. Mega model (XGBoost) must favor this side at >58%
      2. NN model must agree directionally (same winner, >50%)
      3. Mega probability exceeds best bookmaker implied prob by 5%+
      4. Only the best-odds bookmaker shown per side (not all 30)

    Uses mega (XGBoost) as the probability source because the NN is poorly
    calibrated (outputs ~5% or ~95% for most matches). NN serves only as
    directional confirmation that a second model agrees on the winner.
    """
    value_bets = []

    mega_home = prediction.get("mega_home_prob")
    nn_home = prediction.get("nn_home_prob")

    # Need at least the mega model
    if mega_home is None:
        return value_bets

    for side, odds_prob, player_info in [
        ("home", home_prob, prediction["home"]),
        ("away", away_prob, prediction["away"]),
    ]:
        mega_side = mega_home / 100 if side == "home" else (100 - mega_home) / 100

        # Mega must strongly favor this side
        if mega_side <= 0.58:
            continue

        # NN directional confirmation (if available)
        confirmed_by = "mega"
        if nn_home is not None:
            nn_side = nn_home / 100 if side == "home" else (100 - nn_home) / 100
            if nn_side <= 0.50:
                continue  # NN disagrees on the winner — skip
            confirmed_by = "mega+nn"

        # Find best bookmaker odds (lowest implied prob = best payout)
        best_bk = None
        best_bk_prob = 1.0
        best_odds = None

        for bk, odds_str in odds_data[side]["by_bookmaker"].items():
            bk_prob = american_to_probability(odds_str)
            if bk_prob and bk_prob < best_bk_prob:
                best_bk_prob = bk_prob
                best_bk = bk
                best_odds = odds_str

        if best_bk is None:
            book_odds = odds_data[side]["book_odds"]
            if book_odds:
                bk_prob = american_to_probability(book_odds)
                if bk_prob:
                    best_bk = "market"
                    best_bk_prob = bk_prob
                    best_odds = book_odds

        if best_bk is None:
            continue

        edge = mega_side - best_bk_prob
        if edge > 0.05:  # 5%+ edge vs best available odds
            value_bets.append({
                "bookmaker": best_bk,
                "player": player_info["name"],
                "side": side,
                "book_odds": best_odds,
                "implied_prob": round(best_bk_prob * 100, 1),
                "predicted_prob": round(mega_side * 100, 1),
                "edge": round(edge * 100, 1),
                "confirmed_by": confirmed_by,
            })

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets
