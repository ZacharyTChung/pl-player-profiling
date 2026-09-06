"""Feature definitions.

Every feature listed here was verified to contain non-null data in BOTH the 2024-25 and
2025-26 seasons before being included. That check matters because FBref's Opta-derived
columns were deleted in January 2026: the site still serves the passing, possession,
defense, goal_shot_creation and keeper_adv tables, but they are empty shells. Selecting
features from FBref's published schema rather than from the data would silently produce
an all-NaN feature matrix.

Two sources are merged:

* **FBref** supplies what survived the deletion: shooting volume, goals and assists,
  discipline, fouls, offsides, crosses, interceptions, tackles won and playing time.
* **Understat** supplies the shot-quality and creation block that FBref lost: npxG, xA,
  key passes, xGChain and xGBuildup.

Names on the left of each mapping are canonical and are what the rest of the pipeline
and the paper use. Names on the right are the raw source columns.
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
# Source maps: canonical_name -> (raw table, raw column)
# --------------------------------------------------------------------------------------

FBREF_SOURCES: dict[str, tuple[str, str]] = {
    # Playing time and context
    "matches_played": ("standard", "Playing Time__MP"),
    "starts": ("standard", "Playing Time__Starts"),
    "minutes": ("standard", "Playing Time__Min"),
    "nineties": ("standard", "Playing Time__90s"),
    # Goal output
    "goals": ("standard", "Performance__Gls"),
    "assists": ("standard", "Performance__Ast"),
    "goals_non_penalty": ("standard", "Performance__G-PK"),
    "penalties_scored": ("standard", "Performance__PK"),
    "penalties_attempted": ("standard", "Performance__PKatt"),
    # Shooting
    "shots": ("shooting", "Standard__Sh"),
    "shots_on_target": ("shooting", "Standard__SoT"),
    "shot_accuracy_pct": ("shooting", "Standard__SoT%"),
    "goals_per_shot": ("shooting", "Standard__G/Sh"),
    # Duels, discipline and wide play. These come from the misc table, which survived.
    "cards_yellow": ("misc", "Performance__CrdY"),
    "cards_red": ("misc", "Performance__CrdR"),
    "fouls_committed": ("misc", "Performance__Fls"),
    "fouls_drawn": ("misc", "Performance__Fld"),
    "offsides": ("misc", "Performance__Off"),
    "crosses": ("misc", "Performance__Crs"),
    "interceptions": ("misc", "Performance__Int"),
    "tackles_won": ("misc", "Performance__TklW"),
    # Team context, used for interpretation only, never as a clustering feature.
    "team_points_per_match": ("playing_time", "Team Success__PPM"),
    "team_plus_minus_per90": ("playing_time", "Team Success__+/-90"),
    "on_off": ("playing_time", "Team Success__On-Off"),
    "pct_squad_minutes": ("playing_time", "Playing Time__Min%"),
    "subs_on": ("playing_time", "Subs__Subs"),
    "starts_completed": ("playing_time", "Starts__Compl"),
}

UNDERSTAT_SOURCES: dict[str, str] = {
    "us_minutes": "minutes",
    "np_xg": "np_xg",
    "xa": "xa",
    "us_shots": "shots",
    "key_passes": "key_passes",
    "xg_chain": "xg_chain",
    "xg_buildup": "xg_buildup",
    "us_goals_non_penalty": "np_goals",
    "us_assists": "assists",
}

# --------------------------------------------------------------------------------------
# Goalkeeper sources. keeper_adv was destroyed, so PSxG, sweeper actions, launch rate
# and pass length are all unavailable. Only the basic keeper table survived.
# --------------------------------------------------------------------------------------

KEEPER_SOURCES: dict[str, tuple[str, str]] = {
    "gk_matches": ("keeper", "Playing Time__MP"),
    "gk_starts": ("keeper", "Playing Time__Starts"),
    "gk_minutes": ("keeper", "Playing Time__Min"),
    "gk_nineties": ("keeper", "Playing Time__90s"),
    "gk_goals_against": ("keeper", "Performance__GA"),
    "gk_goals_against_per90": ("keeper", "Performance__GA90"),
    "gk_shots_on_target_against": ("keeper", "Performance__SoTA"),
    "gk_saves": ("keeper", "Performance__Saves"),
    "gk_save_pct": ("keeper", "Performance__Save%"),
    "gk_clean_sheets": ("keeper", "Performance__CS"),
    "gk_clean_sheet_pct": ("keeper", "Performance__CS%"),
    "gk_pens_faced": ("keeper", "Penalty Kicks__PKatt"),
    "gk_pens_allowed": ("keeper", "Penalty Kicks__PKA"),
    "gk_pens_saved": ("keeper", "Penalty Kicks__PKsv"),
}

# --------------------------------------------------------------------------------------
# Feature sets used for analysis
# --------------------------------------------------------------------------------------

# Counting stats, converted to per-90 before use.
OUTFIELD_COUNTS = [
    "np_xg",
    "xa",
    "key_passes",
    "xg_chain",
    "xg_buildup",
    "shots",
    "shots_on_target",
    "goals_non_penalty",
    "assists",
    "crosses",
    "interceptions",
    "tackles_won",
    "fouls_committed",
    "fouls_drawn",
    "offsides",
    "cards_yellow",
]

# Already rates or ratios, used as-is.
OUTFIELD_RATES = [
    "shot_accuracy_pct",
    "goals_per_shot",
]

#: The clustering feature set. Team-strength columns are deliberately excluded because
#: they describe the squad a player happens to be in, not the role he performs.
OUTFIELD_CORE = [f"{c}_p90" for c in OUTFIELD_COUNTS] + OUTFIELD_RATES

# Context columns retained alongside the features but never used as features.
CONTEXT_COLS = [
    "minutes",
    "nineties",
    "matches_played",
    "starts",
    "pct_squad_minutes",
    "team_points_per_match",
    "team_plus_minus_per90",
    "on_off",
    "primary_position",
    "position_group",
    "position_full",
    "age",
]

# Goalkeeper counting stats, converted to per-90.
KEEPER_COUNTS = [
    "gk_goals_against",
    "gk_shots_on_target_against",
    "gk_saves",
    "gk_clean_sheets",
]
KEEPER_RATES = [
    "gk_save_pct",
    "gk_clean_sheet_pct",
]
KEEPER_CORE = [f"{c}_p90" for c in KEEPER_COUNTS] + KEEPER_RATES

#: Headline metrics for the descriptive summary table.
HEADLINE_METRICS = [
    "np_xg_p90",
    "xa_p90",
    "key_passes_p90",
    "xg_chain_p90",
    "xg_buildup_p90",
    "shots_p90",
    "shot_accuracy_pct",
    "crosses_p90",
    "interceptions_p90",
    "tackles_won_p90",
    "fouls_committed_p90",
    "offsides_p90",
]

#: Radar axes per position group, fixed so clusters are visually comparable.
RADAR_AXES = [
    "np_xg_p90",
    "shots_p90",
    "xa_p90",
    "key_passes_p90",
    "xg_chain_p90",
    "xg_buildup_p90",
    "crosses_p90",
    "tackles_won_p90",
    "interceptions_p90",
    "fouls_drawn_p90",
]

#: Features used to predict npxG/90 in the "finishing above role" regression. Shooting
#: columns are excluded so the model can only use non-shooting behaviour.
#:
#: ``xg_chain_p90`` is deliberately absent even though it is not a shooting statistic by
#: name. Understat's xGChain credits a player for every possession he was involved in,
#: including the ones he finished himself, so it carries the target inside it: it
#: correlates with npxG per ninety at 0.904, and including it lifts the cross-validated
#: coefficient of determination from 0.536 to 0.892 by letting the model reconstruct the
#: outcome rather than predict it. xGBuildup is retained because it explicitly strips
#: shots and key passes, which is exactly the leak-free part of the same measure.
NON_SHOOTING_FEATURES = [
    "xa_p90",
    "key_passes_p90",
    "xg_buildup_p90",
    "crosses_p90",
    "interceptions_p90",
    "tackles_won_p90",
    "fouls_committed_p90",
    "fouls_drawn_p90",
    "offsides_p90",
    "cards_yellow_p90",
]


def all_canonical_names() -> list[str]:
    """Every canonical column the preprocessing step is expected to produce."""
    return sorted(set(FBREF_SOURCES) | set(UNDERSTAT_SOURCES) | set(KEEPER_SOURCES))
