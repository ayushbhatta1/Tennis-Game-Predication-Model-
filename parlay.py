"""
Parlay Builder & Optimizer

Builds realistic parlays using best available bookmaker odds,
honest expected value calculations, and bankroll management.
"""

from itertools import combinations
from predictor import american_to_probability


def odds_to_decimal(american_odds_str):
    """Convert American odds to decimal odds."""
    try:
        odds = int(american_odds_str)
    except (ValueError, TypeError):
        return None
    if odds > 0:
        return 1 + odds / 100
    elif odds < 0:
        return 1 + 100 / abs(odds)
    return 2.0


def decimal_to_american(decimal_odds):
    """Convert decimal odds to American odds string."""
    if decimal_odds >= 2.0:
        return f"+{int((decimal_odds - 1) * 100)}"
    else:
        return f"{int(-100 / (decimal_odds - 1))}"


def extract_parlay_legs(predictions):
    """
    Extract all viable parlay legs using BEST available bookmaker odds.
    This minimizes the vig on each leg.
    """
    legs = []

    for p in predictions:
        if not p.get("has_odds"):
            continue

        confidence = p.get("confidence", 0)
        winner = p.get("predicted_winner")
        if not winner:
            continue

        w = p["home"] if winner == "home" else p["away"]
        loser = p["away"] if winner == "home" else p["home"]

        model_prob = p.get("winner_prob", 50)
        consensus_book_odds = w.get("book_odds")
        fair_odds = w.get("fair_odds")

        if not consensus_book_odds or not fair_odds:
            continue

        # Find the BEST bookmaker odds for our pick (lowest vig)
        bks = p.get("bookmakers", [])
        best_odds = consensus_book_odds
        best_book = "consensus"

        side = "home" if winner == "home" else "away"
        for bk in bks:
            bk_odds_str = bk[f"{side}_odds"]
            try:
                bk_odds_val = int(bk_odds_str)
                best_val = int(best_odds)
                # Better odds = more positive (or less negative)
                if bk_odds_val > best_val:
                    best_odds = bk_odds_str
                    best_book = bk["name"]
            except (ValueError, TypeError):
                continue

        best_decimal = odds_to_decimal(best_odds)
        best_implied = american_to_probability(best_odds)
        fair_implied = american_to_probability(fair_odds)

        if not best_implied or not fair_implied:
            continue

        # Vig = how much the book inflates the implied prob
        vig = best_implied - fair_implied
        # Edge = model prob - book implied (usually slightly negative)
        edge = model_prob / 100 - best_implied

        leg = {
            "match": f"{p['home']['name']} vs {p['away']['name']}",
            "pick": w["name"],
            "opponent": loser["name"],
            "league": p.get("league", "?"),
            "starts_at": p.get("starts_at"),
            "model_prob": model_prob,
            "fair_odds": fair_odds,
            "book_odds": best_odds,
            "best_book": best_book,
            "decimal_odds": round(best_decimal, 3) if best_decimal else None,
            "book_implied": round(best_implied * 100, 1),
            "fair_implied": round(fair_implied * 100, 1),
            "edge": round(edge * 100, 1),
            "vig": round(vig * 100, 1),
            "confidence": confidence,
            "event_id": p.get("event_id"),
        }
        legs.append(leg)

    return legs


def build_parlays(legs, max_legs=4, bankroll=120, top_n=15):
    """
    Build parlay combinations ranked by risk/reward profile.
    """
    if len(legs) < 2:
        return []

    # Cap to top 12 by model probability (most likely to hit)
    sorted_legs = sorted(legs, key=lambda x: x["model_prob"], reverse=True)
    top_legs = sorted_legs[:12]

    parlays = []

    for num_legs in range(2, min(max_legs + 1, len(top_legs) + 1)):
        for combo in combinations(top_legs, num_legs):
            legs_list = list(combo)

            # Combined decimal odds
            parlay_decimal = 1.0
            for leg in legs_list:
                if leg["decimal_odds"]:
                    parlay_decimal *= leg["decimal_odds"]

            parlay_american = decimal_to_american(parlay_decimal)

            # True win probability (model-based, all legs must win)
            true_prob = 1.0
            for leg in legs_list:
                true_prob *= leg["model_prob"] / 100

            # Book implied probability
            implied_prob = 1.0
            for leg in legs_list:
                implied_prob *= leg["book_implied"] / 100

            # Expected value per $1 bet
            ev = (true_prob * parlay_decimal) - 1

            # Total vig on this parlay
            total_vig = sum(leg["vig"] for leg in legs_list)

            # Average model probability per leg
            avg_prob = sum(leg["model_prob"] for leg in legs_list) / len(legs_list)

            # Bet sizing: conservative fixed % of bankroll based on leg count
            # 2-leg: up to 3%, 3-leg: up to 2%, 4-leg: up to 1%
            max_pct = {2: 0.03, 3: 0.02, 4: 0.01}.get(num_legs, 0.01)
            # Scale by how close to +EV we are
            ev_factor = max(0.1, min(1.0, (ev + 0.15) / 0.15))  # 0.1 at -15% EV, 1.0 at 0% EV
            bet_pct = max_pct * ev_factor
            suggested_bet = round(bankroll * bet_pct, 2)
            suggested_bet = max(1, min(suggested_bet, bankroll * 0.05))  # floor $1, cap 5%

            payout = round(suggested_bet * parlay_decimal, 2)
            profit = round(payout - suggested_bet, 2)

            parlay = {
                "legs": legs_list,
                "num_legs": num_legs,
                "parlay_odds": parlay_american,
                "decimal_odds": round(parlay_decimal, 2),
                "true_prob": round(true_prob * 100, 1),
                "implied_prob": round(implied_prob * 100, 1),
                "ev_per_dollar": round(ev, 3),
                "ev_pct": round(ev * 100, 1),
                "total_vig": round(total_vig, 1),
                "avg_leg_prob": round(avg_prob, 1),
                "suggested_bet": suggested_bet,
                "potential_payout": payout,
                "potential_profit": profit,
            }
            parlays.append(parlay)

    return parlays


def find_best_parlays(predictions, max_legs=4, bankroll=120, top_n=15, **kwargs):
    """
    Main entry: find best parlay combinations.

    Returns structured results with:
    - Conservative picks (highest win prob)
    - Balanced picks (good prob + good payout)
    - Aggressive picks (highest payout)
    - All available legs
    """
    all_legs = extract_parlay_legs(predictions)

    if not all_legs:
        return {
            "viable_legs": [],
            "total_legs": 0,
            "filtered_legs": 0,
            "bankroll": bankroll,
            "top_parlays": [],
            "conservative": [],
            "balanced": [],
            "aggressive": [],
            "stats": {
                "total_combos": 0,
                "positive_ev_count": 0,
                "best_ev": 0,
                "avg_ev_top10": 0,
            },
        }

    # Sort legs by model prob (favorites first)
    all_legs.sort(key=lambda x: x["model_prob"], reverse=True)

    parlays = build_parlays(all_legs, max_legs=max_legs, bankroll=bankroll)

    # Categorize
    positive_ev = [p for p in parlays if p["ev_per_dollar"] > 0]

    # Conservative: highest win probability
    conservative = sorted(parlays, key=lambda x: x["true_prob"], reverse=True)[:5]

    # Balanced: best EV (closest to +EV or least -EV)
    balanced = sorted(parlays, key=lambda x: x["ev_per_dollar"], reverse=True)[:5]

    # Aggressive: best profit potential with reasonable win prob (>10%)
    aggressive_pool = [p for p in parlays if p["true_prob"] > 8]
    aggressive = sorted(aggressive_pool, key=lambda x: x["potential_profit"], reverse=True)[:5]

    # Top overall: blend of EV and win probability
    # Score = EV * 0.4 + normalized win prob * 0.6
    max_prob = max((p["true_prob"] for p in parlays), default=1)
    for p in parlays:
        p["_score"] = p["ev_per_dollar"] * 0.4 + (p["true_prob"] / max_prob) * 0.6
    top_parlays = sorted(parlays, key=lambda x: x["_score"], reverse=True)[:top_n]
    for p in parlays:
        p.pop("_score", None)

    return {
        "viable_legs": all_legs,
        "total_legs": len(all_legs),
        "filtered_legs": len(all_legs),
        "bankroll": bankroll,
        "top_parlays": top_parlays,
        "conservative": conservative,
        "balanced": balanced,
        "aggressive": aggressive,
        "stats": {
            "total_combos": len(parlays),
            "positive_ev_count": len(positive_ev),
            "best_ev": parlays[0]["ev_per_dollar"] if parlays else 0,
            "avg_ev_top10": round(
                sum(p["ev_per_dollar"] for p in balanced) / max(1, len(balanced)), 3
            ),
        },
    }
