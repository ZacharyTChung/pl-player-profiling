"""Does the two-mode structure relate to what happened on the pitch?

The paper finds two modes in role space, a defensive pole and an attacking pole with
midfield spread between them, and six within-position archetypes below that. Every one
of those results is internal to the feature matrix: silhouettes against clusterless
nulls, replication across scopes, agreement with listed positions. None of it says
whether a club's mix of the two modes has anything to do with the league table. This
module asks that question against the archive's real results for all 490 team-seasons.

The obvious test, that the attacking-mode share of a club's minutes correlates with its
points, is partly circular: good teams have good attackers, and a good attacker is an
attacking-mode player by construction. The informative comparison is what the
composition adds beyond team possession, which is the simplest summary of dominance and
is already inside the feature set through the exposure adjustment. So the composition
is set against possession in a ridge regression scored out of sample, holding out each
season in turn and each league in turn, with the penalty chosen inside the training
folds so nothing leaks. If the modes add nothing beyond possession out of sample, that
is the result and it is written down as such.

Three things about the outcome data are stated up front, because the archive does not
ship a league table. The two team tables carry matches, goals by the club's own players,
expected goals and possession, but no wins, draws, losses or points. Results are
recovered from the goalkeeper table, which credits every match to the keeper who was in
goal for it, and the sums reproduce the published results and goals conceded of the
champions exactly. Goals against are that table's goals conceded. Goals for are the
club's goals while each keeper was on the pitch, from the playing time table, which
include the own goals in the club's favour that a player goal total cannot, accepted
only where the same rows' goals against agree with the keeper table and replaced by the
player total where they do not. Each derivation is audited in the saved metrics rather
than trusted.
"""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src import archive as A
from src import cluster as C
from src import plotting as P

# --------------------------------------------------------------------------------------
# Declared choices
# --------------------------------------------------------------------------------------

#: The pooled partition under test, refitted exactly as ``archive_validation`` fits it:
#: the same standardised matrix, the same k and the same seed tag, so the labels are the
#: paper's labels. ``pooled_partition`` checks that by comparing the adjusted Rand index
#: against position group with the one recorded in ``archive_cluster_validation.json``.
K = 2
POOLED_TAG = "archive|All outfield"

#: The six within-position archetypes, as labelled in ``results/membership_primary.csv``.
ARCHETYPES = ["DF-0", "DF-1", "MF-0", "MF-1", "FW-0", "FW-1"]

#: A team-season is kept when the players who carry labels account for at least this
#: share of its minutes, so the composition describes the side rather than a fragment
#: of it. Applied to outfield minutes for the modes and to keeper minutes for keepers.
MIN_COVERAGE = 0.70

#: Minutes floor for a keeper's shot-stopping rate, the study's usual threshold.
KEEPER_MIN_MINUTES = config.MIN_MINUTES

#: Bootstrap resamples of team-seasons, and the coverage of the percentile interval.
N_BOOTSTRAP = 2000
CONFIDENCE = 0.95

#: Ridge penalties searched inside the training folds, and the inner fold count. The
#: penalty is chosen on the training folds alone, so the held-out season or league never
#: touches the fit that predicts it.
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
INNER_FOLDS = 5

#: League schedule. Every club plays the others home and away, so the match count follows
#: from the number of clubs. The 2019-20 Ligue 1 season was stopped after round 28 with
#: two clubs a match short, the only season in the archive that did not run its course.
CLUBS = {"Bundesliga": 18}
DEFAULT_CLUBS = 20
CURTAILED = {(2020, "Ligue 1"): 28}

#: A club whose credited results differ from the schedule by more than this many matches
#: is a data fault rather than a crediting quirk, and the build stops.
MAX_MATCH_DEVIATION = 1

#: On-pitch goal counts are accepted for a club only when its keepers' on-pitch goals
#: against agree with the keeper table's goals conceded to within this many goals, which
#: is the slack a goal scored in the minute of a keeper substitution can introduce.
ON_PITCH_TOLERANCE = 1

#: Two published records the derived table must reproduce before anything is fitted, as
#: wins, draws, losses, goals for and goals against: the two highest points totals in
#: the archive's window.
REFERENCE_RECORDS = {
    (2018, "Manchester City"): (32, 4, 2, 106, 27),
    (2020, "Liverpool"): (32, 3, 3, 85, 33),
}

#: Composition measures per team-season, all minutes weighted.
MEASURES = {
    "attacking_share": "Attacking-mode minute share",
    "axis_mean": "Mean position on the two-mode axis",
    **{f"archetype_share_{a}": f"Minute share in {a}" for a in ARCHETYPES},
}
TWO_MODE = ["attacking_share", "axis_mean"]
ARCHETYPE_SHARES = [f"archetype_share_{a}" for a in ARCHETYPES]

TARGETS = {
    "points_per_match": "Points per match",
    "goal_difference_per_match": "Goal difference per match",
    "xg_difference_per_match": "Expected goal difference per match",
}

#: The three models the question is about, then the decomposition of the composition
#: block into its two-mode and archetype halves so the paper can say which half carries
#: whatever the block adds.
MODELS = {
    "possession": ["possession"],
    "composition": list(MEASURES),
    "both": ["possession", *MEASURES],
    "two_mode": TWO_MODE,
    "archetypes": ARCHETYPE_SHARES,
    "possession_and_two_mode": ["possession", *TWO_MODE],
    "possession_and_archetypes": ["possession", *ARCHETYPE_SHARES],
}
HEADLINE_MODELS = ["possession", "composition", "both"]
MODEL_LABELS = {"possession": "Possession", "composition": "Composition", "both": "Both"}
INCREMENTS = {
    "both_over_possession": ("both", "possession"),
    "possession_and_two_mode_over_possession": ("possession_and_two_mode", "possession"),
    "possession_and_archetypes_over_possession": ("possession_and_archetypes", "possession"),
    "both_over_possession_and_two_mode": ("both", "possession_and_two_mode"),
    "both_over_composition": ("both", "composition"),
}
# The club holdout is the strict one: no held-out team-season shares a club with the
# training data, so a model cannot score by remembering which clubs are good.
HOLDOUTS = {
    "season": "leave-one-season-out",
    "league": "leave-one-league-out",
    "team": "leave-one-club-out",
}
#: The unit each holdout leaves out, as the figure and the verdict name it.
HOLDOUT_UNIT = {"season": "season", "league": "league", "team": "club"}

#: Short league names for the figure, keyed by the archive's competition label.
SHORT = {
    "Premier League": "England",
    "La Liga": "Spain",
    "Bundesliga": "Germany",
    "Serie A": "Italy",
    "Ligue 1": "France",
}

KEYS = ["season", "league", "team"]


def _r(value, digits: int = 4):
    """Round for JSON, mapping non-finite values to None."""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _load(name: str) -> pd.DataFrame:
    """An archive table restricted to the study's seasons."""
    df = pd.read_parquet(A.RAW / f"{name}.parquet")
    df["Season_End_Year"] = pd.to_numeric(df["Season_End_Year"], errors="coerce")
    return df[df["Season_End_Year"].isin(A.SEASONS)].copy()


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


# --------------------------------------------------------------------------------------
# Real outcomes
# --------------------------------------------------------------------------------------


def _expected_matches(season: int, league: str) -> int:
    if (season, league) in CURTAILED:
        return CURTAILED[(season, league)]
    return 2 * (CLUBS.get(league, DEFAULT_CLUBS) - 1)


def outcomes() -> tuple[pd.DataFrame, dict]:
    """One row per team-season with results, goals, expected goals and possession.

    Wins, draws and losses are summed over the keeper table, whose rows credit each
    match to whoever was in goal. That closes at the schedule for every team-season
    except two, which are listed in the audit rather than patched because the archive
    holds nothing that says what the uncredited or double-credited match was, and the
    two clubs a match short when the 2019-20 Ligue 1 season was stopped. Goals
    for are the sum over the club's keepers of the goals scored while each was on the
    pitch, which is the club's total because exactly one keeper is on the pitch at a
    time. The same rows carry goals against while on the pitch, and the keeper table's
    goals conceded are exact, so the two are compared: where they agree the on-pitch
    goals for are accepted, and where they do not the on-pitch windows are wrong, the
    goals for are inflated with them, and the player goal total, a lower bound because
    it omits own goals in the club's favour, is used instead. Per-match rates divide by
    credited results, except expected goals, which the team table accumulates over its
    own match count.
    """
    ids = ["Season_End_Year", "Comp", "Squad"]

    keepers = _load("keepers")
    _numeric(keepers, ["W", "D", "L", "GA", "Min_Playing"])
    results = keepers.groupby(ids, as_index=False).agg(
        wins=("W", "sum"),
        draws=("D", "sum"),
        losses=("L", "sum"),
        goals_against=("GA", "sum"),
        keeper_minutes=("Min_Playing", "sum"),
    )

    playing = _load("playing_time")
    _numeric(playing, ["onG_Team.Success", "onGA_Team.Success", "Min_Playing.Time"])
    in_goal = playing[playing["Pos"].fillna("").str.contains("GK")]
    on_pitch = in_goal.groupby(ids, as_index=False).agg(
        goals_on_pitch=("onG_Team.Success", "sum"),
        goals_against_on_pitch=("onGA_Team.Success", "sum"),
        keeper_minutes_logged=("Min_Playing.Time", "sum"),
    )

    team = _load("team_standard")
    _numeric(team, ["MP_Playing", "Gls", "xG_Expected", "Poss"])
    own = team[team["Team_or_Opponent"] == "team"][
        [*ids, "MP_Playing", "Gls", "xG_Expected", "Poss"]
    ].rename(
        columns={
            "MP_Playing": "matches_in_team_table",
            "Gls": "player_goals",
            "xG_Expected": "xg_for",
            "Poss": "possession",
        }
    )
    against = team[team["Team_or_Opponent"] == "opponent"][[*ids, "xG_Expected"]].rename(
        columns={"xG_Expected": "xg_against"}
    )

    frame = (
        own.merge(against, on=ids, how="inner")
        .merge(results, on=ids, how="inner")
        .merge(on_pitch, on=ids, how="left")
        .merge(A.team_possession(), on=["Season_End_Year", "Squad"], how="left")
    )
    if not np.allclose(frame["possession"], frame["team_possession"]):
        raise ValueError("team possession differs between the two team tables")
    frame = frame.drop(columns="team_possession")
    frame = frame.rename(columns={"Season_End_Year": "season", "Comp": "league", "Squad": "team"})
    frame["season"] = frame["season"].astype(int)
    frame["goals_on_pitch"] = frame["goals_on_pitch"].fillna(0.0)
    frame["goals_against_on_pitch"] = frame["goals_against_on_pitch"].fillna(0.0)
    frame["keeper_minutes_logged"] = frame["keeper_minutes_logged"].fillna(0.0)

    # The on-pitch windows are verified against an exact count: the same table's goals
    # against while on the pitch must agree with the keeper table's goals conceded to
    # within the one goal a substitution-minute double count can add. Where they do not,
    # the on-pitch goals for are inflated by a similar amount and the player goal total,
    # a lower bound, is the closer count.
    frame["on_pitch_verified"] = (
        frame["goals_against_on_pitch"] - frame["goals_against"]
    ).abs() <= ON_PITCH_TOLERANCE
    frame["goals_for"] = np.where(
        frame["on_pitch_verified"], frame["goals_on_pitch"], frame["player_goals"]
    )
    frame["matches"] = frame["wins"] + frame["draws"] + frame["losses"]
    frame["points"] = 3 * frame["wins"] + frame["draws"]
    frame["points_per_match"] = frame["points"] / frame["matches"]
    frame["goals_for_per_match"] = frame["goals_for"] / frame["matches"]
    frame["goals_against_per_match"] = frame["goals_against"] / frame["matches"]
    frame["goal_difference_per_match"] = (
        frame["goals_for_per_match"] - frame["goals_against_per_match"]
    )
    frame["xg_difference_per_match"] = (frame["xg_for"] - frame["xg_against"]) / frame[
        "matches_in_team_table"
    ]
    frame = _league_position(frame)

    audit = _audit(frame)
    columns = [
        *KEYS,
        "matches",
        "matches_in_team_table",
        "wins",
        "draws",
        "losses",
        "points",
        "goals_for",
        "goals_against",
        "xg_for",
        "xg_against",
        "possession",
        "points_per_match",
        "goals_for_per_match",
        "goals_against_per_match",
        "goal_difference_per_match",
        "xg_difference_per_match",
        "league_position",
    ]
    return frame[columns].sort_values(KEYS).reset_index(drop=True), audit


def _league_position(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank clubs within each league-season by points, then goal difference.

    Both are taken per match, which is the same ordering as raw points and goal
    difference wherever every club played the same number of matches and is the
    ordering the league itself used in the one curtailed season. Head-to-head rules
    are not applied, so a pair level on both can sit the other way round from the
    official table.
    """
    frame = frame.sort_values(
        [
            "season",
            "league",
            "points_per_match",
            "goal_difference_per_match",
            "goals_for_per_match",
            "team",
        ],
        ascending=[True, True, False, False, False, True],
    ).copy()
    frame["league_position"] = frame.groupby(["season", "league"]).cumcount() + 1
    return frame


def _audit(frame: pd.DataFrame) -> dict:
    """Check the derived table against the schedule, itself and two published records."""
    clubs = frame.groupby(["season", "league"]).size()
    wrong = {
        f"{league} {season}": int(n)
        for (season, league), n in clubs.items()
        if n != CLUBS.get(league, DEFAULT_CLUBS)
    }
    if wrong:
        raise ValueError(f"league-seasons with the wrong number of clubs: {wrong}")

    expected = frame.apply(lambda r: _expected_matches(r["season"], r["league"]), axis=1)
    deviation = frame["matches"] - expected
    off = frame[deviation != 0]
    if (deviation.abs() > MAX_MATCH_DEVIATION).any():
        raise ValueError(
            "credited results depart from the schedule by more than one match: "
            f"{off[[*KEYS, 'matches']].to_dict('records')}"
        )
    reference = []
    for (season, team), (w, d, losses, gf, ga) in REFERENCE_RECORDS.items():
        row = frame[(frame["season"] == season) & (frame["team"] == team)].iloc[0]
        found = tuple(
            int(row[c]) for c in ("wins", "draws", "losses", "goals_for", "goals_against")
        )
        # Results and goals against come from tables that record each once, so they must
        # match exactly. Goals for come from on-pitch counts, which credit a goal scored in
        # the minute of a keeper substitution to both keepers, so one goal of slack is
        # allowed there and the discrepancy is written down rather than hidden.
        if found[:3] != (w, d, losses) or found[4] != ga or abs(found[3] - gf) > 1:
            raise ValueError(
                f"{team} {season}: derived {found}, published {(w, d, losses, gf, ga)}"
            )
        reference.append(
            {
                "team": team,
                "season": season,
                "derived": list(found),
                "published": [w, d, losses, gf, ga],
                "goals_for_discrepancy": int(found[3] - gf),
            }
        )

    by_league_season = frame.groupby(["season", "league"]).agg(
        goals_for=("goals_for", "sum"), goals_against=("goals_against", "sum")
    )
    identity = {
        f"{league} {season}": int(r.goals_for - r.goals_against)
        for (season, league), r in by_league_season.iterrows()
    }
    short = frame["keeper_minutes_logged"] < frame["keeper_minutes"] - 10
    fallback = ~frame["on_pitch_verified"]
    playoffs = frame["matches_in_team_table"] > expected
    return {
        "clubs_per_league_season": "verified: 18 in the Bundesliga, 20 elsewhere, in every season",
        "results_source": (
            "wins, draws and losses summed over the keeper table, which credits each match to "
            "the keeper in goal; the team tables carry no results or points column, so points "
            "are 3W + D by construction"
        ),
        "credited_results_off_schedule": [
            {
                **{k: r[k] for k in KEYS},
                "credited": int(r["matches"]),
                "scheduled": int(e),
                "note": (
                    "curtailed season, one match unplayed"
                    if (r["season"], r["league"]) in CURTAILED
                    else "one match credited to two keepers"
                    if r["matches"] > e
                    else "one match credited to no keeper"
                ),
            }
            for (_, r), e in zip(off.iterrows(), expected[deviation != 0], strict=True)
        ],
        "team_table_matches_exceed_credited": [
            f"{r.team} {r.season} ({int(r.matches_in_team_table)} in the team table, "
            f"{int(r.matches)} credited; the team table includes a relegation play-off)"
            for r in frame[playoffs].itertuples()
        ],
        "reference_records": reference,
        "goals_for_source": (
            "club goals while each keeper was on the pitch, from the playing time table, "
            "accepted where the same table's goals against while on the pitch agree with the "
            "keeper table's goals conceded to within one goal; otherwise the player goal "
            "total, a lower bound that omits own goals in the club's favour, is used and the "
            "club is listed; a goal scored in the minute of a keeper substitution is credited "
            "to both keepers, so an accepted sum can exceed the published total by one"
        ),
        "on_pitch_tolerance": ON_PITCH_TOLERANCE,
        "n_goals_for_from_on_pitch": int(frame["on_pitch_verified"].sum()),
        "n_goals_for_from_player_total": int(fallback.sum()),
        "goals_for_fallback_to_player_total": [
            {
                "team": r.team,
                "season": int(r.season),
                "on_pitch_goals_for": int(r.goals_on_pitch),
                "on_pitch_goals_against_excess": int(r.goals_against_on_pitch - r.goals_against),
                "player_goals_used": int(r.player_goals),
            }
            for r in frame[fallback].itertuples()
        ],
        "keeper_playing_time_rows_short": [
            f"{r.team} {r.season} ({int(r.keeper_minutes - r.keeper_minutes_logged)} minutes)"
            for r in frame[short].itertuples()
        ],
        "goals_for_minus_goals_against_by_league_season": identity,
        "goals_for_minus_goals_against_max_abs": int(max(abs(v) for v in identity.values())),
        "league_position": (
            "points per match, then goal difference per match, then goals for per match; "
            "head-to-head tie-breaks not applied"
        ),
    }


# --------------------------------------------------------------------------------------
# Composition of each team-season
# --------------------------------------------------------------------------------------


def pooled_partition() -> tuple[pd.DataFrame, dict]:
    """Refit the paper's pooled two-mode partition and place every player on its axis.

    The axis is the unit vector from the defensive centroid to the attacking one, with
    the origin at their midpoint, so a player's signed distance along it is positive on
    the attacking side and is the continuous quantity the cluster label thresholds.
    """
    processed = config.DATA_PROCESSED
    eligible = pd.read_parquet(processed / "archive_eligible.parquet").reset_index(drop=True)
    z_global = pd.read_parquet(processed / "archive_z_global.parquet").reset_index(drop=True)
    features = [c for c in A.OUTFIELD_CORE if c in z_global.columns]
    if len(features) != len(A.OUTFIELD_CORE):
        missing = sorted(set(A.OUTFIELD_CORE) - set(features))
        raise KeyError(f"archive feature columns absent from the standardised table: {missing}")
    if not (eligible["player"].to_numpy() == z_global["player"].to_numpy()).all():
        raise ValueError("eligible and standardised tables are not row aligned")

    x = z_global[features].to_numpy(float)
    fit = C._kmeans(x, K, POOLED_TAG)
    labels = fit.labels_

    ari = float(adjusted_rand_score(eligible["position_group"], labels))
    with open(config.METRICS / "archive_cluster_validation.json") as fh:
        reference = json.load(fh)["answer"]["adjusted_rand_index_vs_position_group"]
    if abs(ari - reference) > 5e-4:
        raise ValueError(
            f"refitted partition disagrees with the paper's: ARI vs position group {ari:.4f} "
            f"against {reference:.4f} recorded in archive_cluster_validation.json"
        )

    table = pd.crosstab(pd.Series(labels, name="cluster"), eligible["position_group"])
    forward_share = table["FW"] / table["FW"].sum()
    attacking = int(forward_share.idxmax())
    if forward_share.max() < 0.9:
        raise ValueError("neither pooled cluster holds almost all forwards")

    centres = fit.cluster_centers_
    direction = centres[attacking] - centres[1 - attacking]
    unit = direction / np.linalg.norm(direction)
    axis = (x - centres.mean(axis=0)) @ unit

    players = eligible[[*KEYS, "player", "position_group", "minutes"]].copy()
    players["season"] = players["season"].astype(int)
    players["attacking"] = (labels == attacking).astype(int)
    players["axis"] = axis

    membership = pd.read_csv(config.RESULTS / "membership_primary.csv")
    membership["season"] = membership["season"].astype(int)
    players = players.merge(
        membership[[*KEYS, "player", "archetype"]], on=[*KEYS, "player"], how="left"
    )
    if players["archetype"].isna().any() or len(players) != len(eligible):
        raise ValueError("membership_primary.csv does not cover the eligible sample one to one")

    info = {
        "k": K,
        "seed_tag": POOLED_TAG,
        "n_players": int(len(players)),
        "n_features": len(features),
        "adjusted_rand_index_vs_position_group": _r(ari),
        "reference_adjusted_rand_index": reference,
        "attacking_cluster": attacking,
        "contingency": {str(int(r)): table.loc[r].to_dict() for r in table.index},
        "forward_share_in_attacking_cluster": _r(forward_share.max()),
        "centroid_separation": _r(float(np.linalg.norm(direction))),
        "axis": (
            "signed distance along the unit vector from the defensive centroid to the "
            "attacking centroid, origin at their midpoint, in the pooled standardised space"
        ),
    }
    return players, info


def composition(players: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Minutes-weighted composition of every team-season, with its coverage."""
    everyone = pd.read_parquet(config.DATA_PROCESSED / "archive_all.parquet")
    outfield = everyone[everyone["position_group"].isin(config.OUTFIELD_GROUPS)].copy()
    outfield["season"] = outfield["season"].astype(int)
    total = outfield.groupby(KEYS)["minutes"].sum().rename("outfield_minutes")

    weighted = players[[*KEYS, "minutes"]].copy()
    weighted["w_attacking"] = players["minutes"] * players["attacking"]
    weighted["w_axis"] = players["minutes"] * players["axis"]
    for archetype in ARCHETYPES:
        weighted[f"w_{archetype}"] = players["minutes"] * (players["archetype"] == archetype)
    grouped = weighted.groupby(KEYS)
    club = grouped.sum(numeric_only=True)
    club["eligible_players"] = grouped.size()
    club = club.rename(columns={"minutes": "eligible_minutes"}).join(total, how="left")

    club["coverage"] = club["eligible_minutes"] / club["outfield_minutes"]
    club["attacking_share"] = club["w_attacking"] / club["eligible_minutes"]
    club["axis_mean"] = club["w_axis"] / club["eligible_minutes"]
    for archetype in ARCHETYPES:
        club[f"archetype_share_{archetype}"] = club[f"w_{archetype}"] / club["eligible_minutes"]
    club = club.reset_index()

    kept = club[club["coverage"] >= MIN_COVERAGE].copy()
    dropped = club[club["coverage"] < MIN_COVERAGE]
    info = {
        "minutes_weighting": (
            "each measure is a minutes-weighted average over the team-season's eligible "
            "outfield players, so a share is the fraction of covered outfield minutes"
        ),
        "n_team_seasons": int(len(club)),
        "coverage_min": _r(club["coverage"].min()),
        "coverage_median": _r(club["coverage"].median()),
        "coverage_mean": _r(club["coverage"].mean()),
        "coverage_threshold": MIN_COVERAGE,
        "n_dropped_for_coverage": int(len(dropped)),
        "dropped_for_coverage": [
            f"{r.team} {r.season} ({r.coverage:.3f})" for r in dropped.itertuples()
        ],
        "n_retained": int(len(kept)),
        "attacking_share_min": _r(kept["attacking_share"].min()),
        "attacking_share_median": _r(kept["attacking_share"].median()),
        "attacking_share_max": _r(kept["attacking_share"].max()),
        "eligible_players_per_team_season_mean": _r(kept["eligible_players"].mean(), 2),
    }
    columns = [
        *KEYS,
        "eligible_players",
        "eligible_minutes",
        "outfield_minutes",
        "coverage",
        *MEASURES,
    ]
    return kept[columns].reset_index(drop=True), info


# --------------------------------------------------------------------------------------
# Bootstrap helpers
# --------------------------------------------------------------------------------------


def bootstrap_indices(n: int) -> np.ndarray:
    """The resample matrix, drawn once per scope so every measure shares the same draws."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    return rng.integers(0, n, size=(N_BOOTSTRAP, n))


def _interval(draws: np.ndarray) -> tuple[float | None, float | None]:
    lo_q, hi_q = 100.0 * (1.0 - CONFIDENCE) / 2.0, 100.0 * (1.0 + CONFIDENCE) / 2.0
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return None, None
    lo, hi = np.percentile(finite, [lo_q, hi_q])
    return _r(lo), _r(hi)


def _spearman_draws(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Spearman correlation on every resample at once: rank within each draw, then Pearson."""
    xr = rankdata(x[idx], axis=1)
    yr = rankdata(y[idx], axis=1)
    xr = xr - xr.mean(axis=1, keepdims=True)
    yr = yr - yr.mean(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xr * yr).sum(axis=1) / np.sqrt((xr**2).sum(axis=1) * (yr**2).sum(axis=1))


def spearman_with_interval(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> dict:
    rho, p = spearmanr(x, y)
    lo, hi = _interval(_spearman_draws(x, y, idx))
    return {
        "rho": _r(rho),
        "p_value": _r(p),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(len(x)),
        "interval_excludes_zero": bool(lo is not None and (lo > 0 or hi < 0)),
    }


# --------------------------------------------------------------------------------------
# 1. Correlations with real outcomes
# --------------------------------------------------------------------------------------


def correlations(frame: pd.DataFrame) -> dict:
    """Spearman of every composition measure with every target, pooled and by scope."""

    def _block(sub: pd.DataFrame) -> dict:
        idx = bootstrap_indices(len(sub))
        return {
            measure: {
                target: spearman_with_interval(
                    sub[measure].to_numpy(float), sub[target].to_numpy(float), idx
                )
                for target in TARGETS
            }
            for measure in MEASURES
        }

    out = {"pooled": _block(frame), "by_league": {}, "by_season": {}}
    for league in A.LEAGUES:
        out["by_league"][league] = _block(frame[frame["league"] == league])
    for season in A.SEASONS:
        out["by_season"][str(season)] = _block(frame[frame["season"] == season])

    # The pooled number the paper leads with, plus the range of the scope refits, so a
    # reader can see whether one league or one season carries the pooled figure.
    summary = {}
    for target in TARGETS:
        leagues = [out["by_league"][lg]["attacking_share"][target]["rho"] for lg in A.LEAGUES]
        seasons = [out["by_season"][str(s)]["attacking_share"][target]["rho"] for s in A.SEASONS]
        summary[target] = {
            "pooled": out["pooled"]["attacking_share"][target],
            "league_rho_min": min(leagues),
            "league_rho_max": max(leagues),
            "season_rho_min": min(seasons),
            "season_rho_max": max(seasons),
            "scopes_with_interval_excluding_zero": int(
                sum(
                    out["by_league"][lg]["attacking_share"][target]["interval_excludes_zero"]
                    for lg in A.LEAGUES
                )
                + sum(
                    out["by_season"][str(s)]["attacking_share"][target]["interval_excludes_zero"]
                    for s in A.SEASONS
                )
            ),
            "scopes": len(A.LEAGUES) + len(A.SEASONS),
        }
    out["attacking_share_summary"] = summary
    return out


# --------------------------------------------------------------------------------------
# 2. What the modes add beyond possession, out of sample
# --------------------------------------------------------------------------------------


def _ridge() -> Pipeline:
    inner = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    return Pipeline(
        [("scale", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=inner))]
    )


def _r2(y: np.ndarray, fitted: np.ndarray) -> float:
    residual = ((y - fitted) ** 2).sum()
    total = ((y - y.mean()) ** 2).sum()
    return float(1.0 - residual / total)


def _r2_draws(y: np.ndarray, fitted: np.ndarray, idx: np.ndarray) -> np.ndarray:
    ys, fs = y[idx], fitted[idx]
    residual = ((ys - fs) ** 2).sum(axis=1)
    total = ((ys - ys.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return 1.0 - residual / total


def out_of_sample(frame: pd.DataFrame, target: str) -> dict:
    """Ridge under every holdout for every model, scored on the held-out predictions.

    Out-of-sample R squared pools the held-out predictions over all folds and scores them
    against the target's overall mean, so a model that only learns each fold's level
    cannot score by it. The per-fold figure, scored within the held-out block, is kept
    alongside. Increments between models are bootstrapped over team-seasons by
    resampling the held-out prediction pairs, which holds the fitted models fixed and
    asks how much of the gap is the particular clubs in the sample.
    """
    y = frame[target].to_numpy(float)
    idx = bootstrap_indices(len(frame))
    out = {}
    for holdout, scheme in HOLDOUTS.items():
        groups = frame[holdout].to_numpy()
        levels = sorted(set(groups.tolist()))
        block: dict = {"scheme": scheme, "folds": [str(v) for v in levels], "models": {}}
        held_out = {}
        for model, columns in MODELS.items():
            x = frame[columns].to_numpy(float)
            fitted = np.empty_like(y)
            alphas, by_fold = {}, {}
            for level in levels:
                test = groups == level
                pipe = _ridge().fit(x[~test], y[~test])
                fitted[test] = pipe.predict(x[test])
                alphas[str(level)] = _r(pipe["ridge"].alpha_, 6)
                # A fold with one row has no variance to score against; the pooled
                # out-of-sample figure still includes its prediction.
                if test.sum() >= 2:
                    by_fold[str(level)] = _r(_r2(y[test], fitted[test]))
            held_out[model] = fitted
            lo, hi = _interval(_r2_draws(y, fitted, idx))
            block["models"][model] = {
                "features": columns,
                "r2_out_of_sample": _r(_r2(y, fitted)),
                "r2_ci_low": lo,
                "r2_ci_high": hi,
                "r2_by_fold": by_fold,
                "r2_by_fold_mean": _r(np.mean(list(by_fold.values()))),
                "rmse_out_of_sample": _r(np.sqrt(((y - fitted) ** 2).mean())),
                "alpha_by_fold": alphas,
            }
        block["increments"] = {}
        for name, (full, base) in INCREMENTS.items():
            draws = _r2_draws(y, held_out[full], idx) - _r2_draws(y, held_out[base], idx)
            lo, hi = _interval(draws)
            block["increments"][name] = {
                "delta_r2": _r(_r2(y, held_out[full]) - _r2(y, held_out[base])),
                "ci_low": lo,
                "ci_high": hi,
                "interval_excludes_zero": bool(lo is not None and (lo > 0 or hi < 0)),
                "share_of_draws_positive": _r(float(np.mean(draws > 0))),
            }
        out[holdout] = block
    out["target_sd"] = _r(float(y.std(ddof=0)))
    return out


# --------------------------------------------------------------------------------------
# 3. Goalkeepers
# --------------------------------------------------------------------------------------


def keeper_test(results: pd.DataFrame) -> dict:
    """Is the full-feature keeper partition a league table in disguise?

    Three club-level quantities are set against points per match: the minutes share of
    the club's keepers in cluster 1 of the full-feature partition, the same share for the
    technique partition, and shot stopping as post-shot expected goals minus goals
    conceded per ninety, summed over keepers with the study's minutes floor. The paper's
    reading predicts the first tracks points strongly and the other two weakly, because
    the full-feature space is dominated by how often the keeper was shot at.
    """
    with open(config.METRICS / "archive_keeper_clusters.json") as fh:
        payload = json.load(fh)
    assigned = pd.DataFrame(payload["assignments"]).rename(
        columns={"squad": "team", "season_end_year": "season"}
    )
    assigned["season"] = assigned["season"].astype(int)

    base = _load("keepers").rename(
        columns={
            "Season_End_Year": "season",
            "Comp": "league",
            "Squad": "team",
            "Player": "player",
            "Min_Playing": "minutes",
        }
    )
    _numeric(base, ["minutes"])
    base["season"] = base["season"].astype(int)
    assigned = assigned.merge(
        base[["season", "league", "team", "player", "minutes"]],
        on=["season", "team", "player"],
        how="left",
    )
    if assigned["minutes"].isna().any():
        raise ValueError("keeper assignments do not all match a keeper table row")

    all_minutes = base.groupby(KEYS)["minutes"].sum().rename("keeper_minutes")
    assigned["w_full"] = assigned["minutes"] * (assigned["cluster_full"] == 1)
    assigned["w_technique"] = assigned["minutes"] * (assigned["cluster_technique"] == 1)
    club = assigned.groupby(KEYS).agg(
        assigned_minutes=("minutes", "sum"),
        w_full=("w_full", "sum"),
        w_technique=("w_technique", "sum"),
    )
    club = club.join(all_minutes, how="left").reset_index()
    club["keeper_coverage"] = club["assigned_minutes"] / club["keeper_minutes"]
    club["keeper_share_full"] = club["w_full"] / club["assigned_minutes"]
    club["keeper_share_technique"] = club["w_technique"] / club["assigned_minutes"]

    adv = _load("keepers_adv").rename(
        columns={
            "Season_End_Year": "season",
            "Squad": "team",
            "Player": "player",
            "PSxG+_per__minus__Expected": "psxg_net",
        }
    )
    _numeric(adv, ["psxg_net"])
    adv["season"] = adv["season"].astype(int)
    stoppers = base[base["minutes"] >= KEEPER_MIN_MINUTES].merge(
        adv[["season", "team", "player", "psxg_net"]], on=["season", "team", "player"], how="left"
    )
    if stoppers["psxg_net"].isna().any():
        raise ValueError("a keeper above the minutes floor has no post-shot expected goals")
    rate = stoppers.groupby(KEYS).agg(psxg_net=("psxg_net", "sum"), minutes=("minutes", "sum"))
    rate["shot_stopping_per90"] = rate["psxg_net"] / (rate["minutes"] / 90.0)
    rate = rate.reset_index()[[*KEYS, "shot_stopping_per90"]]

    merged = (
        results[[*KEYS, "points_per_match", "goal_difference_per_match"]]
        .merge(club, on=KEYS, how="inner")
        .merge(rate, on=KEYS, how="left")
    )
    dropped = merged[merged["keeper_coverage"] < MIN_COVERAGE]
    kept = merged[merged["keeper_coverage"] >= MIN_COVERAGE].reset_index(drop=True)
    idx = bootstrap_indices(len(kept))

    quantities = {
        "cluster_full_share": "keeper_share_full",
        "cluster_technique_share": "keeper_share_technique",
        "shot_stopping_per90": "shot_stopping_per90",
    }
    tests = {}
    for name, column in quantities.items():
        valid = kept[column].notna().to_numpy()
        sub = kept[valid]
        sub_idx = idx if valid.all() else bootstrap_indices(len(sub))
        tests[name] = {
            target: spearman_with_interval(
                sub[column].to_numpy(float), sub[target].to_numpy(float), sub_idx
            )
            for target in ("points_per_match", "goal_difference_per_match")
        }

    profiles = payload["spaces"]["full"]["cluster_profiles"]
    full = tests["cluster_full_share"]["points_per_match"]
    technique = tests["cluster_technique_share"]["points_per_match"]
    stopping = tests["shot_stopping_per90"]["points_per_match"]
    ordering = abs(full["rho"]) > abs(technique["rho"]) and abs(full["rho"]) > abs(stopping["rho"])
    return {
        "question": "Does the full-feature keeper partition track the league table?",
        "n_team_seasons": int(len(kept)),
        "n_dropped_for_keeper_coverage": int(len(dropped)),
        "dropped_for_keeper_coverage": [
            f"{r.team} {r.season} ({r.keeper_coverage:.3f})" for r in dropped.itertuples()
        ],
        "keeper_coverage_min": _r(kept["keeper_coverage"].min()),
        "cluster_full_1_profile": {
            k: profiles["1"]["raw_means"][k]
            for k in (
                "gk_goals_against_per90",
                "gk_clean_sheet_pct",
                "gk_sota_p90",
                "gk_psxg_net_p90",
            )
        },
        "cluster_full_0_profile": {
            k: profiles["0"]["raw_means"][k]
            for k in (
                "gk_goals_against_per90",
                "gk_clean_sheet_pct",
                "gk_sota_p90",
                "gk_psxg_net_p90",
            )
        },
        "shares": (
            "minutes-weighted share of the club's assigned keeper minutes in cluster 1; "
            "shot stopping is FBref's PSxG+/- summed over keepers with at least "
            f"{KEEPER_MIN_MINUTES} minutes, per ninety"
        ),
        "tests": tests,
        "prediction": (
            "cluster_full share tracks points strongly, cluster_technique share weakly, shot "
            "stopping weakly"
        ),
        "prediction_holds": bool(ordering),
        "reading": (
            "The full-feature partition tracks points more closely than either the technique "
            "partition or shot stopping itself, which is what a partition driven by workload "
            "rather than technique should do."
            if ordering
            else "The full-feature partition does not track points more closely than the "
            "technique partition or shot stopping, so the league-table reading is not supported."
        ),
    }


# --------------------------------------------------------------------------------------
# 4. Sanity table
# --------------------------------------------------------------------------------------


def extremes(frame: pd.DataFrame, n: int = 5) -> dict:
    """The team-seasons at either end of attacking-mode share, for the reader to check."""

    def _rows(sub: pd.DataFrame) -> list[dict]:
        return [
            {
                "team": r.team,
                "league": r.league,
                "season": int(r.season),
                "attacking_share": _r(r.attacking_share),
                "possession": _r(r.possession, 1),
                "points": int(r.points),
                "matches": int(r.matches),
                "points_per_match": _r(r.points_per_match, 3),
                "league_position": int(r.league_position),
            }
            for r in sub.itertuples()
        ]

    ordered = frame.sort_values(["attacking_share", "team"], ascending=[False, True])
    return {
        "highest_attacking_share": _rows(ordered.head(n)),
        "lowest_attacking_share": _rows(ordered.tail(n).iloc[::-1]),
    }


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------


def figure_outcomes(frame: pd.DataFrame, ridge: dict, pooled: dict) -> None:
    """Share against points by league, and out-of-sample R squared by model and holdout."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(P.WIDTH_FULL, 3.1), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )
    colour = P.CATEGORICAL[0]
    for i, league in enumerate(A.LEAGUES):
        sub = frame[frame["league"] == league]
        left.scatter(
            sub["attacking_share"],
            sub["points_per_match"],
            marker=P.MARKERS[i],
            s=13,
            color=colour,
            alpha=0.7,
            linewidths=0.3,
            edgecolors=P.SURFACE,
            label=SHORT[league],
            zorder=3,
        )
    x = frame["attacking_share"].to_numpy(float)
    y = frame["points_per_match"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 50)
    left.plot(grid, intercept + slope * grid, color=P.INK_PRIMARY, linewidth=1.2, zorder=4)
    stat = pooled["attacking_share"]["points_per_match"]
    P.style_axis(
        left,
        "Attacking-mode share of outfield minutes",
        "Points per match",
        f"Spearman {stat['rho']:.2f} [{stat['ci_low']:.2f}, {stat['ci_high']:.2f}]",
    )
    left.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=5,
        fontsize=P.BASE_FONT_PT - 2,
        handletextpad=0.2,
        columnspacing=0.8,
    )

    holdouts = list(HOLDOUTS)
    width = 0.78 / len(holdouts)
    offset = (len(holdouts) - 1) / 2
    positions = np.arange(len(HEADLINE_MODELS))
    colours = P.categorical(len(holdouts))
    hatches = ["", "///", "..."]
    for j, holdout in enumerate(holdouts):
        values = [
            ridge["points_per_match"][holdout]["models"][m]["r2_out_of_sample"]
            for m in HEADLINE_MODELS
        ]
        bars = right.bar(
            positions + (j - offset) * width,
            values,
            width * 0.94,
            color=colours[j],
            hatch=hatches[j % len(hatches)],
            edgecolor=P.SURFACE,
            linewidth=0.6,
            label=f"Hold out one {HOLDOUT_UNIT[holdout]}",
            zorder=3,
        )
        # Three bars per model leave no room for a horizontal label, so the values
        # stand upright above their bars.
        for bar, value in zip(bars, values, strict=True):
            right.annotate(
                f"{value:.2f}",
                (bar.get_x() + bar.get_width() / 2, max(value, 0.0)),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=P.BASE_FONT_PT - 3,
                color=P.INK_SECONDARY,
            )
    right.axhline(0, color=P.INK_SECONDARY, linewidth=0.6)
    right.set_xticks(positions)
    right.set_xticklabels([MODEL_LABELS[m] for m in HEADLINE_MODELS])
    right.set_xlim(-0.6, len(HEADLINE_MODELS) - 0.4)
    low = min(
        ridge["points_per_match"][h]["models"][m]["r2_out_of_sample"]
        for h in holdouts
        for m in HEADLINE_MODELS
    )
    right.set_ylim(min(0.0, low - 0.05), 1.0)
    P.style_axis(right, "", "Out-of-sample R squared", "Ridge, points per match")
    right.grid(False, axis="x")
    right.tick_params(axis="x", labelsize=P.BASE_FONT_PT - 1)
    right.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(holdouts),
        fontsize=P.BASE_FONT_PT - 2,
        handletextpad=0.4,
        columnspacing=0.8,
    )

    P.panel_labels([left, right])
    P.save_figure(fig, "outcomes")


# --------------------------------------------------------------------------------------
# Verdict and driver
# --------------------------------------------------------------------------------------


def _verdict(corr: dict, ridge: dict) -> str:
    pooled = corr["pooled"]["attacking_share"]["points_per_match"]
    points = ridge["points_per_match"]
    pieces = [
        f"Attacking-mode minute share correlates with points per match at Spearman "
        f"{pooled['rho']:.2f} [{pooled['ci_low']:.2f}, {pooled['ci_high']:.2f}] over "
        f"{pooled['n']} team-seasons."
    ]
    adds = []
    count = {2: "two", 3: "three", 4: "four"}.get(len(HOLDOUTS), str(len(HOLDOUTS)))
    for holdout in HOLDOUTS:
        models = points[holdout]["models"]
        inc = points[holdout]["increments"]["both_over_possession"]
        pieces.append(
            f"Holding out one {HOLDOUT_UNIT[holdout]} at a time, possession alone predicts points out of "
            f"sample at R squared {models['possession']['r2_out_of_sample']:.2f}, the "
            f"composition alone at {models['composition']['r2_out_of_sample']:.2f}, and both "
            f"together at {models['both']['r2_out_of_sample']:.2f}, an increment of "
            f"{inc['delta_r2']:+.3f} [{inc['ci_low']:+.3f}, {inc['ci_high']:+.3f}]."
        )
        adds.append(inc["interval_excludes_zero"] and inc["delta_r2"] > 0)
    if all(adds):
        pieces.append(
            "The two-mode composition therefore carries information about real results "
            f"beyond possession under all {count} holdouts."
        )
    elif not any(adds):
        pieces.append(
            "The two-mode composition therefore adds no out-of-sample information about real "
            "results beyond possession: its correlation with points is possession seen "
            "through the players."
        )
    else:
        pieces.append(
            "The increment beyond possession clears zero under some holdouts and not others, "
            "so the composition's information beyond possession is not robust to how the "
            "sample is split."
        )
    finer = [points[h]["increments"]["both_over_possession_and_two_mode"] for h in HOLDOUTS]
    spelled = " and ".join(
        f"{f['delta_r2']:+.3f} [{f['ci_low']:+.3f}, {f['ci_high']:+.3f}]" for f in finer
    )
    if all(f["interval_excludes_zero"] and f["delta_r2"] > 0 for f in finer):
        pieces.append(
            "The six archetype shares add further information beyond possession and the "
            f"two-mode measures, {spelled} under the {count} holdouts."
        )
    else:
        pieces.append(
            "The six archetype shares add nothing beyond possession and the two-mode "
            f"measures, {spelled} under the {count} holdouts, so the finer taxonomy carries no "
            "information about results that the two modes do not."
        )
    return " ".join(pieces)


def main() -> None:
    # Embedding the sans-serif face into the PDF makes fontTools log a note about the
    # font's own post table once per glyph run, which buries the numbers this prints.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    P.use_style()

    results, audit = outcomes()
    print(
        f"outcomes: {len(results)} team-seasons, {results['season'].nunique()} seasons, "
        f"{results['league'].nunique()} leagues; results closed at the schedule for "
        f"{len(results) - len(audit['credited_results_off_schedule'])} of them"
    )

    players, partition = pooled_partition()
    print(
        f"  pooled partition refitted: ARI vs position group "
        f"{partition['adjusted_rand_index_vs_position_group']:.4f} (recorded "
        f"{partition['reference_adjusted_rand_index']:.4f}), attacking cluster "
        f"{partition['attacking_cluster']} holds "
        f"{partition['forward_share_in_attacking_cluster']:.1%} of forwards"
    )

    clubs, coverage = composition(players)
    frame = clubs.merge(results, on=KEYS, how="inner")
    if len(frame) != len(clubs):
        raise ValueError("composition and outcomes do not share the same team-seasons")
    print(
        f"  composition: coverage min {coverage['coverage_min']:.3f}, median "
        f"{coverage['coverage_median']:.3f}; {coverage['n_dropped_for_coverage']} team-seasons "
        f"dropped below {MIN_COVERAGE:.0%}, {coverage['n_retained']} retained"
    )

    print("\n1. correlations with real outcomes")
    corr = correlations(frame)
    for target in TARGETS:
        s = corr["attacking_share_summary"][target]
        print(
            f"  attacking share vs {target}: rho {s['pooled']['rho']:+.3f} "
            f"[{s['pooled']['ci_low']:+.3f}, {s['pooled']['ci_high']:+.3f}]; leagues "
            f"{s['league_rho_min']:+.3f} to {s['league_rho_max']:+.3f}, seasons "
            f"{s['season_rho_min']:+.3f} to {s['season_rho_max']:+.3f}"
        )

    axis = corr["pooled"]["axis_mean"]["points_per_match"]
    print(
        f"  axis mean vs points_per_match: rho {axis['rho']:+.3f} "
        f"[{axis['ci_low']:+.3f}, {axis['ci_high']:+.3f}]"
    )

    print("\n2. beyond possession, out of sample")
    ridge = {target: out_of_sample(frame, target) for target in TARGETS}
    for target in TARGETS:
        for holdout in HOLDOUTS:
            models = ridge[target][holdout]["models"]
            inc = ridge[target][holdout]["increments"]["both_over_possession"]
            print(
                f"  {target:28s} {holdout:6s}  possession {models['possession']['r2_out_of_sample']:.3f}  "
                f"composition {models['composition']['r2_out_of_sample']:.3f}  "
                f"both {models['both']['r2_out_of_sample']:.3f}  increment "
                f"{inc['delta_r2']:+.3f} [{inc['ci_low']:+.3f}, {inc['ci_high']:+.3f}]"
            )

    for holdout in HOLDOUTS:
        inc = ridge["points_per_match"][holdout]["increments"]
        two = inc["possession_and_two_mode_over_possession"]
        finer = inc["both_over_possession_and_two_mode"]
        print(
            f"  points, {holdout:6s}  two modes over possession {two['delta_r2']:+.3f} "
            f"[{two['ci_low']:+.3f}, {two['ci_high']:+.3f}]  archetypes over that "
            f"{finer['delta_r2']:+.3f} [{finer['ci_low']:+.3f}, {finer['ci_high']:+.3f}]"
        )

    print("\n3. goalkeepers")
    keepers = keeper_test(results)
    for name, block in keepers["tests"].items():
        t = block["points_per_match"]
        print(f"  {name:24s} rho {t['rho']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]")
    print(f"  prediction holds: {keepers['prediction_holds']}")

    print("\n4. extremes of attacking-mode share")
    ends = extremes(frame)
    for label in ("highest_attacking_share", "lowest_attacking_share"):
        for r in ends[label]:
            print(
                f"  {r['team']:18s} {r['league']:15s} {r['season']}  share {r['attacking_share']:.3f}  "
                f"{r['points_per_match']:.2f} ppm  position {r['league_position']}"
            )

    figure_outcomes(frame, ridge, corr["pooled"])

    keeper_columns = ["keeper_share_full", "keeper_share_technique", "shot_stopping_per90"]
    table = frame.copy()
    table = table.merge(_keeper_table(results)[[*KEYS, *keeper_columns]], on=KEYS, how="left")
    payload = {
        "question": "Does the two-mode composition of a club relate to its real results?",
        "method": {
            "outcomes": audit["results_source"],
            "goals_for": audit["goals_for_source"],
            "league_position": audit["league_position"],
            "composition": coverage["minutes_weighting"],
            "axis": partition["axis"],
            "correlation": (
                "Spearman, pooled and within each league and each season, with a percentile "
                f"interval from {N_BOOTSTRAP:,} bootstrap resamples of team-seasons seeded "
                f"from config.RANDOM_STATE"
            ),
            "ridge": (
                "ridge regression with standardised inputs, the penalty chosen by "
                f"{INNER_FOLDS}-fold cross-validation inside the training folds over "
                f"{len(RIDGE_ALPHAS)} values from {RIDGE_ALPHAS[0]:g} to {RIDGE_ALPHAS[-1]:g}; "
                "scored on the held-out predictions under leave-one-season-out and "
                "leave-one-league-out; increments bootstrapped over the held-out pairs"
            ),
            "keepers": keepers["shares"],
        },
        "n_bootstrap": N_BOOTSTRAP,
        "confidence": CONFIDENCE,
        "seed": config.RANDOM_STATE,
        "outcomes_audit": audit,
        "partition": partition,
        "composition": coverage,
        "measures": MEASURES,
        "targets": TARGETS,
        "correlations": corr,
        "ridge": ridge,
        "keepers": keepers,
        "extremes": ends,
        "team_seasons": _records(table, keeper_columns),
        "verdict": _verdict(corr, ridge),
    }
    path = config.METRICS / "outcomes.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_native)
    print(f"\n  wrote {path.relative_to(config.ROOT)}")
    print("  verdict:", payload["verdict"])
    print("outcomes complete")


def _keeper_table(results: pd.DataFrame) -> pd.DataFrame:
    """The club-level keeper quantities again, for the saved team-season table."""
    with open(config.METRICS / "archive_keeper_clusters.json") as fh:
        assigned = pd.DataFrame(json.load(fh)["assignments"])
    assigned = assigned.rename(columns={"squad": "team", "season_end_year": "season"})
    base = _load("keepers").rename(
        columns={"Season_End_Year": "season", "Comp": "league", "Squad": "team", "Player": "player"}
    )
    _numeric(base, ["Min_Playing"])
    base["season"] = base["season"].astype(int)
    assigned = assigned.merge(
        base[["season", "league", "team", "player", "Min_Playing"]],
        on=["season", "team", "player"],
        how="left",
    )
    assigned["w_full"] = assigned["Min_Playing"] * (assigned["cluster_full"] == 1)
    assigned["w_technique"] = assigned["Min_Playing"] * (assigned["cluster_technique"] == 1)
    club = assigned.groupby(KEYS).agg(
        minutes=("Min_Playing", "sum"), w_full=("w_full", "sum"), w_technique=("w_technique", "sum")
    )
    club["keeper_share_full"] = club["w_full"] / club["minutes"]
    club["keeper_share_technique"] = club["w_technique"] / club["minutes"]
    adv = _load("keepers_adv").rename(
        columns={"Season_End_Year": "season", "Squad": "team", "Player": "player"}
    )
    _numeric(adv, ["PSxG+_per__minus__Expected"])
    adv["season"] = adv["season"].astype(int)
    stoppers = base[base["Min_Playing"] >= KEEPER_MIN_MINUTES].merge(
        adv[["season", "team", "player", "PSxG+_per__minus__Expected"]],
        on=["season", "team", "player"],
        how="left",
    )
    rate = stoppers.groupby(KEYS).agg(
        net=("PSxG+_per__minus__Expected", "sum"), minutes=("Min_Playing", "sum")
    )
    rate["shot_stopping_per90"] = rate["net"] / (rate["minutes"] / 90.0)
    return club.join(rate[["shot_stopping_per90"]], how="left").reset_index()


def _records(table: pd.DataFrame, keeper_columns: list[str]) -> list[dict]:
    columns = [
        "matches",
        "points",
        "points_per_match",
        "goal_difference_per_match",
        "xg_difference_per_match",
        "possession",
        "league_position",
        "coverage",
        *MEASURES,
        *keeper_columns,
    ]
    out = []
    for row in table.sort_values(KEYS).to_dict("records"):
        record = {"season": int(row["season"]), "league": row["league"], "team": row["team"]}
        for column in columns:
            value = row[column]
            record[column] = (
                int(value) if column in ("matches", "points", "league_position") else _r(value)
            )
        out.append(record)
    return out


if __name__ == "__main__":
    main()
