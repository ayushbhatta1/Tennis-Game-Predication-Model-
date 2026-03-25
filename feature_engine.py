"""
Unified Feature Engine — 66 features for all models.

Feature groups:
  Odds (18):       0-17   fair prob, book prob, vig, spread, O/U, set1, consensus, movement
  ELO (4):         18-21  overall diff, surface diff, raw values
  Serve (10):      22-31  ace rate, 1st%, 1st won%, 2nd won%, BP save%, DF rate, dominance
  Form (6):        32-37  last 10/20 win rate, surface form, weighted form, upset rate, momentum
  H2H (3):         38-40  ratio, familiarity, surface H2H
  Physical (4):    41-44  height diff, age diff, handedness matchup, leftie flag
  Rankings (5):    45-49  log rank diff, pts ratio, rank momentum, peak rank diff, rank norm
  Tournament (6):  50-55  surface (3), round depth, tourney level, best_of
  Fatigue (4):     56-59  days since last (each), matches in 14d (each)
  Interactions (4):60-63  elo*surface_form, odds*elo_sign, serve*surface, form*rank_sign
  Flags (2):       64-65  has_odds, has_history
"""

import numpy as np

FEATURE_NAMES = [
    # Odds (18): 0-17
    "fair_prob_home", "fair_prob_away", "book_prob_home", "book_prob_away",
    "vig_home", "spread_prob", "ou_prob", "set1_home_prob", "set1_away_prob",
    "bk_count_norm", "bk_spread", "bk_consensus_home", "fav_margin",
    "open_fair_home", "movement", "sets_ou_prob", "player_games_diff",
    "league_is_atp",
    # ELO (4): 18-21
    "elo_diff", "surface_elo_diff", "elo_home_norm", "elo_away_norm",
    # Serve (10): 22-31
    "ace_rate_diff", "first_serve_pct_diff", "first_serve_won_diff",
    "second_serve_won_diff", "bp_save_rate_diff", "df_rate_diff",
    "serve_dominance_diff", "ace_rate_home", "ace_rate_away",
    "serve_rating_diff",
    # Form (6): 32-37
    "form_10_diff", "form_20_diff", "surface_form_diff",
    "weighted_form_diff", "upset_rate_diff", "momentum_diff",
    # H2H (3): 38-40
    "h2h_ratio", "h2h_familiarity", "h2h_surface_ratio",
    # Physical (4): 41-44
    "height_diff", "age_diff", "handedness_matchup", "leftie_flag",
    # Rankings (5): 45-49
    "log_rank_diff", "pts_ratio", "rank_momentum_diff", "peak_rank_diff",
    "rank_norm_diff",
    # Tournament (6): 50-55
    "surface_hard", "surface_clay", "surface_grass", "round_depth",
    "tourney_level", "best_of",
    # Fatigue (4): 56-59
    "days_since_last_home", "days_since_last_away",
    "matches_14d_home", "matches_14d_away",
    # Interactions (4): 60-63
    "elo_x_surface_form", "odds_x_elo_sign", "serve_x_surface",
    "form_x_rank_sign",
    # Flags (2): 64-65
    "has_odds", "has_history",
]

NUM_FEATURES = len(FEATURE_NAMES)
assert NUM_FEATURES == 66, f"Expected 66 features, got {NUM_FEATURES}"


def flip_features(features):
    """Flip feature vector to swap home/away perspective."""
    f = features.copy()

    # Swap pairs: (home, away)
    for a, b in [(0, 1), (2, 3), (7, 8), (20, 21), (29, 30), (56, 57), (58, 59)]:
        f[a], f[b] = features[b], features[a]

    # Negate diffs
    for i in [4, 12, 14, 16, 18, 19, 22, 23, 24, 25, 26, 27, 28, 31,
              32, 33, 34, 35, 36, 37, 41, 42, 45, 47, 48, 49, 60, 61, 62, 63]:
        f[i] = -f[i]

    # 1-x for ratios/probs (home-centric)
    for i in [11, 13, 38, 40, 46]:
        f[i] = 1.0 - f[i]

    return f


def build_feature_vector(
    odds_features=None,
    home_stats=None,
    away_stats=None,
    surface="Hard",
    round_val="R32",
    tourney_level="A",
    best_of=3,
    league="ATP",
    has_odds=False,
    has_history=False,
):
    """
    Build the unified 66-feature vector.

    Args:
        odds_features: dict with odds-derived features (or None)
        home_stats: dict with home player stats from feature store
        away_stats: dict with away player stats from feature store
        surface, round_val, tourney_level, best_of, league: match context
        has_odds: whether odds data is available
        has_history: whether historical stats are available
    """
    vec = np.zeros(NUM_FEATURES, dtype=np.float32)
    hs = home_stats or {}
    as_ = away_stats or {}

    # === Odds (18): indices 0-17 ===
    if odds_features and has_odds:
        of = odds_features
        vec[0] = of.get("fair_prob_home", 0.5)
        vec[1] = of.get("fair_prob_away", 0.5)
        vec[2] = of.get("book_prob_home", 0.5)
        vec[3] = of.get("book_prob_away", 0.5)
        vec[4] = of.get("vig_home", 0.0)
        vec[5] = of.get("spread_prob", 0.5)
        vec[6] = of.get("ou_prob", 0.5)
        vec[7] = of.get("set1_home_prob", 0.5)
        vec[8] = of.get("set1_away_prob", 0.5)
        vec[9] = of.get("bk_count_norm", 0.0)
        vec[10] = of.get("bk_spread", 0.0)
        vec[11] = of.get("bk_consensus_home", 0.5)
        vec[12] = of.get("fav_margin", 0.0)
        vec[13] = of.get("open_fair_home", 0.5)
        vec[14] = of.get("movement", 0.0)
        vec[15] = of.get("sets_ou_prob", 0.5)
        vec[16] = of.get("player_games_diff", 0.0)
    vec[17] = 1.0 if league == "ATP" else 0.0

    # === ELO (4): indices 18-21 ===
    h_elo = hs.get("elo", 1500.0)
    a_elo = as_.get("elo", 1500.0)
    h_selo = hs.get("surface_elo", {}).get(surface, 1500.0)
    a_selo = as_.get("surface_elo", {}).get(surface, 1500.0)
    vec[18] = (h_elo - a_elo) / 400.0
    vec[19] = (h_selo - a_selo) / 400.0
    vec[20] = (h_elo - 1500.0) / 400.0
    vec[21] = (a_elo - 1500.0) / 400.0

    # === Serve (10): indices 22-31 ===
    h_serve = hs.get("serve", {})
    a_serve = as_.get("serve", {})
    vec[22] = h_serve.get("ace_rate", 0.05) - a_serve.get("ace_rate", 0.05)
    vec[23] = h_serve.get("first_serve_pct", 0.6) - a_serve.get("first_serve_pct", 0.6)
    vec[24] = h_serve.get("first_serve_won", 0.7) - a_serve.get("first_serve_won", 0.7)
    vec[25] = h_serve.get("second_serve_won", 0.5) - a_serve.get("second_serve_won", 0.5)
    vec[26] = h_serve.get("bp_save_rate", 0.6) - a_serve.get("bp_save_rate", 0.6)
    vec[27] = h_serve.get("df_rate", 0.03) - a_serve.get("df_rate", 0.03)
    vec[28] = h_serve.get("serve_dominance", 0.0) - a_serve.get("serve_dominance", 0.0)
    vec[29] = h_serve.get("ace_rate", 0.05)
    vec[30] = a_serve.get("ace_rate", 0.05)
    h_srating = (h_serve.get("first_serve_won", 0.7) * 0.4 +
                 h_serve.get("ace_rate", 0.05) * 0.3 +
                 h_serve.get("bp_save_rate", 0.6) * 0.3)
    a_srating = (a_serve.get("first_serve_won", 0.7) * 0.4 +
                 a_serve.get("ace_rate", 0.05) * 0.3 +
                 a_serve.get("bp_save_rate", 0.6) * 0.3)
    vec[31] = h_srating - a_srating

    # === Form (6): indices 32-37 ===
    h_form = hs.get("form", {})
    a_form = as_.get("form", {})
    vec[32] = h_form.get("last_10", 0.5) - a_form.get("last_10", 0.5)
    vec[33] = h_form.get("last_20", 0.5) - a_form.get("last_20", 0.5)
    vec[34] = (h_form.get("surface", {}).get(surface, 0.5) -
               a_form.get("surface", {}).get(surface, 0.5))
    vec[35] = h_form.get("weighted", 0.5) - a_form.get("weighted", 0.5)
    vec[36] = h_form.get("upset_rate", 0.0) - a_form.get("upset_rate", 0.0)
    vec[37] = h_form.get("momentum", 0.0) - a_form.get("momentum", 0.0)

    # === H2H (3): indices 38-40 ===
    opp_id = as_.get("player_id", "")
    h2h = hs.get("h2h", {}).get(opp_id, {})
    total_h2h = h2h.get("wins", 0) + h2h.get("losses", 0)
    vec[38] = h2h.get("wins", 0) / total_h2h if total_h2h > 0 else 0.5
    vec[39] = min(total_h2h / 20.0, 1.0)
    surf_w = h2h.get("surface_wins", {}).get(surface, 0)
    surf_l = h2h.get("surface_losses", {}).get(surface, 0)
    surf_total = surf_w + surf_l
    vec[40] = surf_w / surf_total if surf_total > 0 else 0.5

    # === Physical (4): indices 41-44 ===
    h_phys = hs.get("physical", {})
    a_phys = as_.get("physical", {})
    vec[41] = (h_phys.get("height", 180) - a_phys.get("height", 180)) / 20.0
    vec[42] = (h_phys.get("age", 25) - a_phys.get("age", 25)) / 10.0
    h_hand = h_phys.get("hand", "R")
    a_hand = a_phys.get("hand", "R")
    if h_hand == "R" and a_hand == "L":
        vec[43] = 1.0
    elif h_hand == "L" and a_hand == "R":
        vec[43] = 0.0
    else:
        vec[43] = 0.5
    vec[44] = 1.0 if (h_hand == "L" or a_hand == "L") else 0.0

    # === Rankings (5): indices 45-49 ===
    h_rank = hs.get("ranking", {})
    a_rank = as_.get("ranking", {})
    hr = h_rank.get("rank", 500)
    ar = a_rank.get("rank", 500)
    hp = h_rank.get("points", 0)
    ap = a_rank.get("points", 0)
    vec[45] = np.log1p(ar) - np.log1p(hr)  # positive = home ranked higher
    vec[46] = hp / (hp + ap) if (hp + ap) > 0 else 0.5
    vec[47] = h_rank.get("momentum", 0.0) - a_rank.get("momentum", 0.0)
    h_peak = h_rank.get("peak", 500)
    a_peak = a_rank.get("peak", 500)
    vec[48] = (np.log1p(a_peak) - np.log1p(h_peak)) / 7.0
    vec[49] = (np.log1p(hr) - np.log1p(ar)) / 7.0

    # === Tournament (6): indices 50-55 ===
    vec[50] = 1.0 if surface == "Hard" else 0.0
    vec[51] = 1.0 if surface == "Clay" else 0.0
    vec[52] = 1.0 if surface == "Grass" else 0.0
    round_map = {"F": 7, "SF": 6, "QF": 5, "R16": 4, "R32": 3,
                 "R64": 2, "R128": 1, "RR": 3}
    vec[53] = round_map.get(round_val, 3) / 7.0
    level_map = {"G": 1.0, "M": 0.75, "A": 0.5, "D": 0.25, "F": 0.1, "C": 0.1}
    vec[54] = level_map.get(tourney_level, 0.5)
    vec[55] = best_of / 5.0

    # === Fatigue (4): indices 56-59 ===
    h_fat = hs.get("fatigue", {})
    a_fat = as_.get("fatigue", {})
    vec[56] = min(h_fat.get("days_since_last", 14) / 30.0, 1.0)
    vec[57] = min(a_fat.get("days_since_last", 14) / 30.0, 1.0)
    vec[58] = h_fat.get("matches_14d", 3) / 10.0
    vec[59] = a_fat.get("matches_14d", 3) / 10.0

    # === Interactions (4): indices 60-63 ===
    vec[60] = vec[18] * vec[34]  # elo_diff * surface_form_diff
    if has_odds:
        vec[61] = vec[0] * np.sign(vec[18])  # fair_prob * sign(elo_diff)
    vec[62] = vec[28] * (vec[50] + vec[52] * 1.5 + vec[51] * 0.5)  # serve*surface
    vec[63] = vec[32] * np.sign(vec[45])  # form_diff * sign(rank_diff)

    # === Flags (2): indices 64-65 ===
    vec[64] = 1.0 if has_odds else 0.0
    vec[65] = 1.0 if has_history else 0.0

    return vec


def extract_odds_features_from_event(event, use_opening=False):
    """Extract odds feature dict from an API event."""
    odds = event.get("odds", {})
    if not odds:
        return None

    hml = odds.get("points-home-game-ml-home", {})
    aml = odds.get("points-away-game-ml-away", {})
    if not hml or not aml:
        return None

    def atp(odds_str):
        try:
            o = int(odds_str)
        except (ValueError, TypeError):
            return None
        if o < 0:
            return abs(o) / (abs(o) + 100)
        elif o > 0:
            return 100 / (o + 100)
        return 0.5

    if use_opening:
        home_fair = atp(hml.get("openFairOdds"))
        away_fair = atp(aml.get("openFairOdds"))
        home_book = atp(hml.get("openBookOdds"))
        away_book = atp(aml.get("openBookOdds"))
    else:
        home_fair = atp(hml.get("fairOdds"))
        away_fair = atp(aml.get("fairOdds"))
        home_book = atp(hml.get("bookOdds"))
        away_book = atp(aml.get("bookOdds"))

    if home_fair is None or away_fair is None:
        return None

    total = home_fair + away_fair
    if total <= 0:
        return None
    hf = home_fair / total
    af = away_fair / total
    if home_book and away_book:
        bt = home_book + away_book
        hb = home_book / bt
        ab = away_book / bt
    else:
        hb, ab = hf, af

    vig = hb - hf

    # Opening / movement
    open_hf_raw = atp(hml.get("openFairOdds"))
    open_af_raw = atp(aml.get("openFairOdds"))
    if open_hf_raw and open_af_raw:
        ot = open_hf_raw + open_af_raw
        open_hf = open_hf_raw / ot
        movement = hf - open_hf
    else:
        open_hf = hf
        movement = 0.0

    # Bookmakers
    bk_data = hml.get("byBookmaker", {})
    bk_probs = []
    for bd in bk_data.values():
        key = "openOdds" if use_opening else "odds"
        p = atp(bd.get(key) or bd.get("odds"))
        if p:
            bk_probs.append(p)

    bk_spread = (max(bk_probs) - min(bk_probs)) if len(bk_probs) >= 2 else 0.0
    bk_consensus = (sum(bk_probs) / len(bk_probs)) if bk_probs else hf

    # Secondary markets
    def get_mkt(key):
        mkt = odds.get(key, {})
        if use_opening:
            return atp(mkt.get("openFairOdds"))
        return atp(mkt.get("fairOdds"))

    spread_prob = get_mkt("games-home-game-sp-home") or 0.5
    ou_prob = get_mkt("games-all-game-ou-over") or 0.5
    set1_home = get_mkt("points-home-1s-ml-home") or hf
    set1_away = get_mkt("points-away-1s-ml-away") or af
    s1t = set1_home + set1_away
    if s1t > 0:
        set1_home /= s1t
        set1_away /= s1t
    sets_ou = get_mkt("points-all-game-ou-over") or 0.5
    home_games = get_mkt("games-home-game-ou-over") or 0.5
    away_games = get_mkt("games-away-game-ou-over") or 0.5

    return {
        "fair_prob_home": hf,
        "fair_prob_away": af,
        "book_prob_home": hb,
        "book_prob_away": ab,
        "vig_home": vig,
        "spread_prob": spread_prob,
        "ou_prob": ou_prob,
        "set1_home_prob": set1_home,
        "set1_away_prob": set1_away,
        "bk_count_norm": min(len(bk_data) / 30.0, 1.0),
        "bk_spread": bk_spread,
        "bk_consensus_home": bk_consensus,
        "fav_margin": abs(hf - af),
        "open_fair_home": open_hf,
        "movement": movement,
        "sets_ou_prob": sets_ou,
        "player_games_diff": home_games - away_games,
    }
