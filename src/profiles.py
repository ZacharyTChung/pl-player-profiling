"""Phase 6. Archetype characterisation, radar profiles and outliers.

What this module does
---------------------
Phase 5 selected k=2 in every scope: globally and separately within DF, MF and FW. This
module takes the within-group KMeans partition as given and describes it. For each of
the six within-group clusters it computes the centroid in the within-group z-scored
feature space, so every centroid coordinate reads directly as a standardised deviation
from that position group's mean, extracts the eight features with the largest absolute
deviation, lists the ten members closest to the centroid and the three members closest
to the decision boundary between the two clusters, and draws the profile on the fixed
ten radar axes of ``features.RADAR_AXES``.

What this module deliberately does not do
-----------------------------------------
It does not construct a five or six role taxonomy. There are two clusters per position
group, and the honest reading of most of them is a single dominant contrast between high
and low attacking involvement rather than a set of distinct tactical instructions. The
data forces this. FBref removed every Opta-derived column in January 2026, so there is no
possession, passing volume, carrying, pressing or defensive detail left, and the eighteen
surviving features describe shooting, chance creation, crossing and a thin residue of
defensive counting stats. A defender cannot be described positively by this feature set,
only by how much attacking output he produces, and the DF-0 centroid shows exactly that:
every one of its distinguishing features is negative and none is materially positive.

Naming and justification
------------------------
Names were assigned only after the centroid tables and the member lists printed here were
inspected, in line with the project ground rule that cluster names are hypotheses. Every
number quoted inside a justification paragraph is interpolated from the computed profile
at run time rather than typed in, and the direction word ("above" or "below") is derived
from the sign of the computed deviation, so the prose cannot drift away from the data.
The example players named in each paragraph are checked against the computed
representative list and the module raises if a cited player is no longer a member.

Stability is stated inside each paragraph because it is not uniform. Bootstrap adjusted
Rand index is 0.91 globally and 0.84 for midfielders, but only 0.58 for defenders and
0.60 for forwards, and the forward minimum falls below zero across resamples. The
midfield split is well supported, the defender and forward splits are not, and the
paragraphs say so in plain words.

Outliers
--------
An isolation forest is fitted within each position group on the same eighteen
standardised features, and a Mahalanobis distance is computed on the same matrix with a
Ledoit-Wolf shrinkage covariance, which is needed because the forward group has 49
players against 18 features and the empirical covariance there is close to singular.
Both detectors are converted to a within-group percentile so the three groups are
comparable, and the reported top 15 is ranked on the mean of the two percentiles. Each
entry carries a factual explanation naming the three features with the largest absolute
deviation for that player.

Outputs
-------
``results/archetypes.json``, ``results/metrics/archetype_profiles.json``,
``results/metrics/outliers.json``, ``data/processed/archetypes_{season}.parquet`` and the
figures ``radar_panel_{group}``, ``archetype_profiles`` and
``outliers_scatter``.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest

import config
from src import features as F
from src import plotting

# --------------------------------------------------------------------------------------
# Declared choices
# --------------------------------------------------------------------------------------

#: Number of largest absolute centroid deviations reported as distinguishing features.
N_DISTINGUISHING = 8

#: Members closest to the centroid, reported as representative of the archetype.
N_REPRESENTATIVE = 10

#: Members closest to the two-cluster decision boundary, reported as borderline cases.
N_BOUNDARY = 3

#: Length of the pooled outlier table.
N_OUTLIERS = 15

#: Statistics quoted in each outlier explanation.
N_OUTLIER_REASONS = 3

ISOLATION_TREES = 300

ARCHETYPE_ORDER = ["DF-0", "DF-1", "MF-0", "MF-1", "FW-0", "FW-1"]

# --------------------------------------------------------------------------------------
# Display names
# --------------------------------------------------------------------------------------

#: Short names for radar axes, kept short so the outer polar labels do not clip.
RADAR_LABELS = {
    "np_xg_p90": "npxG",
    "shots_p90": "Shots",
    "xa_p90": "xA",
    "key_passes_p90": "Key passes",
    "xg_chain_p90": "xGChain",
    "xg_buildup_p90": "xGBuildup",
    "crosses_p90": "Crosses",
    "tackles_won_p90": "Tackles won",
    "tackles_won_padj_p90": "Tackles won (adj)",
    "interceptions_p90": "Interceptions",
    "interceptions_padj_p90": "Interceptions (adj)",
    "fouls_drawn_p90": "Fouls drawn",
}

#: Compact names for the heatmap rows.
FEATURE_LABELS = {
    "np_xg_p90": "npxG/90",
    "xa_p90": "xA/90",
    "key_passes_p90": "Key passes/90",
    "xg_chain_p90": "xGChain/90",
    "xg_buildup_p90": "xGBuildup/90",
    "shots_p90": "Shots/90",
    "shots_on_target_p90": "Shots on target/90",
    "goals_non_penalty_p90": "Non-penalty goals/90",
    "assists_p90": "Assists/90",
    "crosses_p90": "Crosses/90",
    "interceptions_p90": "Interceptions/90",
    "interceptions_padj_p90": "Interceptions/90 (adj)",
    "tackles_won_p90": "Tackles won/90",
    "tackles_won_padj_p90": "Tackles won/90 (adj)",
    "fouls_committed_p90": "Fouls committed/90",
    "fouls_committed_padj_p90": "Fouls committed/90 (adj)",
    "fouls_drawn_p90": "Fouls drawn/90",
    "offsides_p90": "Offsides/90",
    "cards_yellow_p90": "Yellow cards/90",
    "shot_accuracy_pct": "Shot accuracy %",
    "goals_per_shot": "Goals per shot",
}

#: Spelled-out names for prose, so no justification or explanation contains a slash.
PROSE_LABELS = {
    "np_xg_p90": "non-penalty expected goals per 90",
    "xa_p90": "expected assists per 90",
    "key_passes_p90": "key passes per 90",
    "xg_chain_p90": "expected goals chain per 90",
    "xg_buildup_p90": "expected goals buildup per 90",
    "shots_p90": "shots per 90",
    "shots_on_target_p90": "shots on target per 90",
    "goals_non_penalty_p90": "non-penalty goals per 90",
    "assists_p90": "assists per 90",
    "crosses_p90": "crosses per 90",
    "interceptions_p90": "interceptions per 90",
    "interceptions_padj_p90": "interceptions per 90 (adj)",
    "tackles_won_p90": "tackles won per 90",
    "tackles_won_padj_p90": "tackles won per 90 (adj)",
    "fouls_committed_p90": "fouls committed per 90",
    "fouls_committed_padj_p90": "fouls committed per 90 (adj)",
    "fouls_drawn_p90": "fouls drawn per 90",
    "offsides_p90": "offsides per 90",
    "cards_yellow_p90": "yellow cards per 90",
    "shot_accuracy_pct": "shot accuracy",
    "goals_per_shot": "goals per shot",
}

GROUP_NOUN = {"DF": "defender", "MF": "midfield", "FW": "forward"}

# --------------------------------------------------------------------------------------
# Names, assigned after inspecting the centroid tables and the member lists
# --------------------------------------------------------------------------------------

ARCHETYPE_NAMES = {
    "DF-0": "Low attacking involvement defenders",
    "DF-1": "Crossing and chance creating defenders",
    "MF-0": "Ball winning, low attacking involvement midfielders",
    "MF-1": "High attacking involvement midfielders",
    "FW-0": "Shot taking central forwards",
    "FW-1": "Crossing and creating wide forwards",
}

ARCHETYPE_SUMMARIES = {
    "DF-0": (
        "Defenders below the defender average on every creation and shooting measure, "
        "with no compensating positive signature in the surviving data."
    ),
    "DF-1": (
        "Defenders whose crossing, key passing and assist output sits about one standard "
        "deviation above the defender average."
    ),
    "MF-0": (
        "Midfielders below the midfield average on every attacking measure, with modest "
        "positive deviations on interceptions and tackles won."
    ),
    "MF-1": (
        "Midfielders above the midfield average on shooting and creation together, and "
        "below it on defensive counting stats."
    ),
    "FW-0": ("Forwards who shoot more accurately and cross far less than the forward average."),
    "FW-1": (
        "Forwards who cross and create far more than the forward average and shoot much less."
    ),
}

#: Players named in each justification paragraph. Verified against the computed
#: representative list at run time; a mismatch raises rather than being papered over.
CITED_PLAYERS = {
    "DF-0": ["Calvin Bassey", "Emmanuel Agbadou", "Jan Paul van Hecke", "Max Kilman"],
    "DF-1": ["Malo Gusto", "Djed Spence", "Pervis Estupiñán", "Milos Kerkez", "Reece James"],
    "MF-0": ["Sam Morsy", "Kobbie Mainoo", "James Garner", "Wilfred Ndidi", "Sandro Tonali"],
    "MF-1": [
        "Morgan Gibbs-White",
        "Dejan Kulusevski",
        "Kaoru Mitoma",
        "Morgan Rogers",
        "Amad Diallo",
    ],
    "FW-0": [
        "Kai Havertz",
        "Nicolas Jackson",
        "Jean-Philippe Mateta",
        "Ollie Watkins",
        "Dominic Solanke",
    ],
    "FW-1": [
        "Leandro Trossard",
        "Gabriel Martinelli",
        "Anthony Gordon",
        "Harvey Barnes",
        "Son Heung-min",
    ],
}


# --------------------------------------------------------------------------------------
# Text helpers. Numbers and signs are read from the computed profile, never typed in.
# --------------------------------------------------------------------------------------


def _dev(centroid: pd.Series, feature: str) -> str:
    """Render one centroid coordinate as "0.44 standard deviations below"."""
    value = float(centroid[feature])
    direction = "above" if value >= 0 else "below"
    return f"{abs(value):.2f} standard deviations {direction}"


def _names(players: list[str]) -> str:
    """Join names as "A, B and C"."""
    if len(players) == 1:
        return players[0]
    return ", ".join(players[:-1]) + " and " + players[-1]


def _no_dashes(text: str, where: str) -> str:
    """Guard the ground rule that no em dash or en dash reaches the paper."""
    for bad, label in (("\u2014", "em dash"), ("\u2013", "en dash")):
        if bad in text:
            raise ValueError(f"{label} found in {where}: {text[:120]!r}")
    return text


# --------------------------------------------------------------------------------------
# Justifications. One paragraph per archetype, quoted verbatim by the paper.
# --------------------------------------------------------------------------------------


def _justify_df0(ctx: dict) -> str:
    c, cited = ctx["centroid"], ctx["cited"]
    return (
        f"This is the larger of the two defender clusters, holding {ctx['n']} of the "
        f"{ctx['n_group']} eligible defenders, and it is defined by what is absent rather than "
        "by any positive signature. All eight of its most distinguishing features are negative "
        f"deviations from the defender mean: key passes per 90 sit {_dev(c, 'key_passes_p90')} "
        f"it, assists per 90 {_dev(c, 'assists_p90')}, expected assists per 90 "
        f"{_dev(c, 'xa_p90')}, crosses per 90 {_dev(c, 'crosses_p90')} and shots per 90 "
        f"{_dev(c, 'shots_p90')}. Nothing in the eighteen features sits materially above the "
        f"defender average, the largest positive deviation being only "
        f"{ctx['max_positive']:.2f} standard deviations, so the honest description is low "
        "attacking involvement rather than a defensive speciality. That limitation is a "
        "property of the data and not of the players, because the withdrawal of the "
        "Opta-derived tables left no tackling detail, no pressing, no aerial duels and no ball "
        "recoveries with which a defender could be described positively. The members closest "
        f"to the centroid are {_names(cited)}, who are central defenders in the conventional "
        "sense. The split is provisional: bootstrap resampling of the defender scope gives a "
        f"mean adjusted Rand index of {ctx['ari']:.3f}, the lowest of the four scopes analysed "
        f"and far below the global figure of {ctx['ari_global']:.3f}, so the boundary between "
        "the two defender clusters moves substantially under resampling even though the "
        "direction of the contrast does not."
    )


def _justify_df1(ctx: dict) -> str:
    c, cited = ctx["centroid"], ctx["cited"]
    return (
        f"The smaller defender cluster, {ctx['n']} of {ctx['n_group']} eligible defenders, is "
        "defined by wide attacking output, and all eight of its distinguishing features are "
        f"positive. Key passes per 90 sit {_dev(c, 'key_passes_p90')} the defender mean, "
        f"assists per 90 {_dev(c, 'assists_p90')}, expected assists per 90 "
        f"{_dev(c, 'xa_p90')}, crosses per 90 {_dev(c, 'crosses_p90')} and shots per 90 "
        f"{_dev(c, 'shots_p90')}, with possession adjusted tackles won also {_dev(c, 'tackles_won_padj_p90')} "
        "the mean. Because the deviations are one-sided this is the high involvement end of a "
        "single dominant contrast rather than a separate tactical instruction, and the players "
        f"it selects are full backs and wing backs: {_names(cited)} are the members closest to "
        "the centroid. The label describes output and not intent, since the surviving features "
        "cannot separate a defender who is asked to overlap from one who plays in a team that "
        "crosses often. Bootstrap stability for the defender scope is a mean adjusted Rand "
        f"index of {ctx['ari']:.3f} with a standard deviation of {ctx['ari_sd']:.3f} and a "
        f"minimum of {ctx['ari_min']:.3f} across {ctx['n_boot']} resamples, so the membership "
        "of this cluster should be read as indicative rather than settled."
    )


def _justify_mf0(ctx: dict) -> str:
    c, cited = ctx["centroid"], ctx["cited"]
    return (
        "Unlike the defender split, the midfield split has real content on both sides. This "
        f"cluster of {ctx['n']} midfielders sits below the midfield "
        f"mean on every attacking measure, with shots on target per 90 "
        f"{_dev(c, 'shots_on_target_p90')} it, non-penalty expected goals per 90 "
        f"{_dev(c, 'np_xg_p90')}, shots per 90 {_dev(c, 'shots_p90')} and expected assists per "
        f"90 {_dev(c, 'xa_p90')}, and it is the only cluster in the study whose positive "
        "deviations are all defensive or disciplinary rather than attacking: interceptions per "
        f"90 sit {_dev(c, 'interceptions_padj_p90')} the midfield mean "
        f"and tackles won per 90 {_dev(c, 'tackles_won_padj_p90')} it. The negative attacking side "
        "of the contrast is the larger part of it, which is why the name records the ball "
        "winning tilt second rather than first. The members closest to the centroid are "
        f"{_names(cited)}. This is the best supported of the three within-group splits: "
        f"bootstrap resampling gives a mean adjusted Rand index of {ctx['ari']:.3f} for "
        f"midfielders with a standard deviation of {ctx['ari_sd']:.3f} and a minimum of "
        f"{ctx['ari_min']:.3f} over {ctx['n_boot']} resamples, so the partition survives "
        "resampling in a way the defender and forward partitions do not."
    )


def _justify_mf1(ctx: dict) -> str:
    c, cited = ctx["centroid"], ctx["cited"]
    return (
        f"The second midfield cluster, {ctx['n']} players, is the mirror of the first and is "
        "the high involvement side of the midfield contrast. Its centroid sits "
        f"{_dev(c, 'shots_on_target_p90')} the midfield mean on shots on target per 90, "
        f"{_dev(c, 'np_xg_p90')} on non-penalty expected goals per 90, {_dev(c, 'shots_p90')} "
        f"on shots per 90, {_dev(c, 'xa_p90')} on expected assists per 90 and "
        f"{_dev(c, 'key_passes_p90')} on key passes per 90, while interceptions per 90 are "
        f"{_dev(c, 'interceptions_padj_p90')} the mean and tackles won per 90 "
        f"{_dev(c, 'tackles_won_padj_p90')} it. Shooting and creation load on the same side of the "
        "split, which means the eighteen surviving features cannot separate a midfielder who "
        "scores from one who supplies, and no such distinction is claimed here. The members "
        f"closest to the centroid are {_names(cited)}, a set that mixes central attacking "
        "midfielders with wide midfielders and so cuts across the listed position strings. "
        f"Bootstrap stability for the midfield scope is a mean adjusted Rand index of "
        f"{ctx['ari']:.3f}, the strongest of the three position groups and close to the global "
        f"figure of {ctx['ari_global']:.3f}."
    )


def _justify_fw0(ctx: dict) -> str:
    c, cited = ctx["centroid"], ctx["cited"]
    return (
        f"The larger forward cluster, {ctx['n']} of {ctx['n_group']} eligible forwards, is "
        "separated from the other by an absence of wide play at least as much as by shooting. "
        f"Crosses per 90 is its single largest deviation at {_dev(c, 'crosses_p90')} the "
        f"forward mean, with key passes per 90 {_dev(c, 'key_passes_p90')}, expected assists "
        f"per 90 {_dev(c, 'xa_p90')} and expected goals buildup per 90 "
        f"{_dev(c, 'xg_buildup_p90')}, while non-penalty expected goals per 90 sit "
        f"{_dev(c, 'np_xg_p90')} the mean, shots on target per 90 "
        f"{_dev(c, 'shots_on_target_p90')} and shot accuracy {_dev(c, 'shot_accuracy_pct')}. "
        f"The members closest to the centroid are {_names(cited)}, who are centre forwards in "
        "ordinary football terms. The forward scope is the most volatile in the study. With "
        f"only {ctx['n_group']} eligible forwards the bootstrap adjusted Rand index has a mean "
        f"of {ctx['ari']:.3f} and a standard deviation of {ctx['ari_sd']:.3f} against "
        f"{ctx['ari_sd_mf']:.3f} for midfielders, and its minimum across {ctx['n_boot']} "
        f"resamples is {ctx['ari_min']:.3f}, which is below zero and so no better than a random "
        "relabelling in the worst case. The contrast described here is clear in the centroid, "
        "but the assignment of any individual borderline forward should not be relied on."
    )


def _justify_fw1(ctx: dict) -> str:
    c, cited = ctx["centroid"], ctx["cited"]
    return (
        f"The smaller forward cluster, {ctx['n']} players, inverts that profile. Crosses per 90 "
        f"sit {_dev(c, 'crosses_p90')} the forward mean, key passes per 90 "
        f"{_dev(c, 'key_passes_p90')}, expected assists per 90 {_dev(c, 'xa_p90')} and expected "
        f"goals buildup per 90 {_dev(c, 'xg_buildup_p90')}, while non-penalty expected goals "
        f"per 90 are {_dev(c, 'np_xg_p90')} the mean, shots on target per 90 "
        f"{_dev(c, 'shots_on_target_p90')} and goals per shot {_dev(c, 'goals_per_shot')}. The "
        f"members closest to the centroid are {_names(cited)}, which is a wide forward group in "
        f"ordinary football terms. Two cautions apply. The cluster holds only {ctx['n']} "
        "players, the smallest of the six archetypes, and the forward bootstrap mean adjusted "
        f"Rand index is {ctx['ari']:.3f} with a standard deviation of {ctx['ari_sd']:.3f} and a "
        f"minimum of {ctx['ari_min']:.3f}. It is reported as a description of the fitted "
        "centroid rather than as a stable partition of forwards, and no claim about an "
        "individual player's membership should rest on it."
    )


JUSTIFIERS = {
    "DF-0": _justify_df0,
    "DF-1": _justify_df1,
    "MF-0": _justify_mf0,
    "MF-1": _justify_mf1,
    "FW-0": _justify_fw0,
    "FW-1": _justify_fw1,
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_season(season: str) -> pd.DataFrame:
    """Within-group z-scored features joined to the Phase 5 within-group cluster labels."""
    z = pd.read_parquet(config.DATA_PROCESSED / f"outfield_z_bygroup_{season}.parquet")
    clusters = pd.read_parquet(config.DATA_PROCESSED / f"clusters_{season}.parquet")
    keep = ["player", "cluster_within_group", "cluster_within_group_label", "cluster_global"]
    frame = z.merge(clusters[keep], on="player", how="inner", validate="one_to_one")
    if len(frame) != len(z):
        raise ValueError(f"{season}: cluster labels do not cover every eligible player")
    if frame[F.OUTFIELD_CORE].isna().any().any():
        raise ValueError(f"{season}: missing values in the standardised feature matrix")
    return frame


def load_stability() -> dict:
    with open(config.METRICS / "cluster_stability.json") as fh:
        return json.load(fh)


def _write_json(payload: dict, path: Path) -> Path:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    _no_dashes(text, str(path.name))
    with open(path, "w") as fh:
        fh.write(text + "\n")
    print(f"  wrote {path.relative_to(config.ROOT)}")
    return path


# --------------------------------------------------------------------------------------
# Centroids, representatives and boundary cases
# --------------------------------------------------------------------------------------


def group_profiles(frame: pd.DataFrame, group: str) -> dict:
    """Centroids, per-player distances and boundary margins for one position group.

    The feature matrix is already z-scored within the position group, so a centroid
    coordinate is by construction the standardised deviation of that cluster from the
    position group mean. The boundary margin is the absolute difference between the
    distances to the two centroids, which is zero exactly on the KMeans decision boundary.
    """
    sub = frame[frame["position_group"] == group].reset_index(drop=True)
    labels = sorted(sub["cluster_within_group_label"].unique())
    if len(labels) != 2:
        raise ValueError(f"{group}: expected two within-group clusters, found {labels}")

    matrix = sub[F.OUTFIELD_CORE].to_numpy(float)
    centroids = {
        label: sub.loc[sub["cluster_within_group_label"] == label, F.OUTFIELD_CORE].mean()
        for label in labels
    }
    stack = np.vstack([centroids[label].to_numpy(float) for label in labels])
    distances = np.linalg.norm(matrix[:, None, :] - stack[None, :, :], axis=2)

    assigned = np.array([labels.index(label) for label in sub["cluster_within_group_label"]])
    sub = sub.assign(
        distance_to_centroid=distances[np.arange(len(sub)), assigned],
        boundary_margin=np.abs(distances[:, 0] - distances[:, 1]),
    )
    return {"group": group, "frame": sub, "labels": labels, "centroids": centroids}


def describe_cluster(profile: dict, label: str) -> dict:
    """Distinguishing features, representative members and boundary members."""
    sub = profile["frame"]
    members = sub[sub["cluster_within_group_label"] == label]
    centroid = profile["centroids"][label]

    ranked = centroid.reindex(centroid.abs().sort_values(ascending=False).index)
    distinguishing = [
        {"feature": name, "label": FEATURE_LABELS[name], "z_deviation": round(float(value), 4)}
        for name, value in ranked.head(N_DISTINGUISHING).items()
    ]
    representative = members.nsmallest(N_REPRESENTATIVE, "distance_to_centroid")
    boundary = members.nsmallest(N_BOUNDARY, "boundary_margin")
    return {
        "label": label,
        "position_group": profile["group"],
        "n_players": int(len(members)),
        "centroid": {name: round(float(value), 4) for name, value in centroid.items()},
        "distinguishing_features": distinguishing,
        "representative_players": [
            {
                "player": row.player,
                "team": row.team,
                "position_full": row.position_full,
                "minutes": int(row.minutes),
                "distance_to_centroid": round(float(row.distance_to_centroid), 3),
            }
            for row in representative.itertuples()
        ],
        "boundary_players": [
            {
                "player": row.player,
                "team": row.team,
                "position_full": row.position_full,
                "minutes": int(row.minutes),
                "boundary_margin": round(float(row.boundary_margin), 3),
            }
            for row in boundary.itertuples()
        ],
    }


def build_archetypes(frame: pd.DataFrame, stability: dict, season: str) -> list[dict]:
    """Assemble the named, justified archetype records for one season."""
    scopes = stability["seasons"][season]
    global_ari = scopes["global"]["ari_mean"]
    records = []
    for group in config.OUTFIELD_GROUPS:
        profile = group_profiles(frame, group)
        scope = scopes[group]
        for label in profile["labels"]:
            described = describe_cluster(profile, label)
            centroid = profile["centroids"][label]
            cited = CITED_PLAYERS[label]
            representative = [row["player"] for row in described["representative_players"]]
            missing = [name for name in cited if name not in representative]
            if missing:
                raise ValueError(
                    f"{label}: justification names {missing} but they are no longer among the "
                    f"{N_REPRESENTATIVE} representative players. Re-inspect the profile and "
                    "rewrite the paragraph rather than editing this check."
                )
            ctx = {
                "centroid": centroid,
                "cited": cited,
                "n": described["n_players"],
                "n_group": int((frame["position_group"] == group).sum()),
                "max_positive": float(centroid.max()),
                "ari": scope["ari_mean"],
                "ari_sd": scope["ari_sd"],
                "ari_min": scope["ari_min"],
                "n_boot": scope["n_bootstrap"],
                "ari_global": global_ari,
                "ari_sd_mf": scopes["MF"]["ari_sd"],
            }
            justification = _no_dashes(JUSTIFIERS[label](ctx), f"justification for {label}")
            records.append(
                {
                    "label": label,
                    "name": ARCHETYPE_NAMES[label],
                    "summary": ARCHETYPE_SUMMARIES[label],
                    "position_group": group,
                    "n_players": described["n_players"],
                    "n_position_group": ctx["n_group"],
                    "distinguishing_features": described["distinguishing_features"],
                    "representative_players": described["representative_players"],
                    "boundary_players": described["boundary_players"],
                    "bootstrap_ari": round(float(scope["ari_mean"]), 4),
                    "bootstrap_ari_sd": round(float(scope["ari_sd"]), 4),
                    "bootstrap_ari_min": round(float(scope["ari_min"]), 4),
                    "justification": justification,
                    "centroid": described["centroid"],
                }
            )
    return sorted(records, key=lambda record: ARCHETYPE_ORDER.index(record["label"]))


def archetype_table(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    """Player-level archetype assignment, the table every downstream module joins on."""
    pieces = []
    for group in config.OUTFIELD_GROUPS:
        profile = group_profiles(frame, group)
        sub = profile["frame"]
        pieces.append(
            sub.assign(
                season=season,
                archetype_name=sub["cluster_within_group_label"].map(ARCHETYPE_NAMES),
            )[
                [
                    "season",
                    "player",
                    "team",
                    "position_group",
                    "position_full",
                    "minutes",
                    "cluster_within_group_label",
                    "archetype_name",
                    "distance_to_centroid",
                    "boundary_margin",
                ]
            ]
        )
    table = pd.concat(pieces, ignore_index=True).sort_values("player").reset_index(drop=True)
    if table["archetype_name"].isna().any():
        raise ValueError(f"{season}: a cluster label has no archetype name")
    return table


def centroid_agreement(primary: pd.DataFrame, replication: pd.DataFrame) -> dict:
    """Check the replication centroids point the same way before reusing the names.

    Phase 5 matched the 2025-26 cluster ids to the 2024-25 centroids by optimal
    assignment, so a label should mean the same thing in both seasons. This verifies it
    instead of assuming it, because reusing a name across a flipped label would be a
    fabricated result.
    """
    agreement = {}
    for group in config.OUTFIELD_GROUPS:
        a = group_profiles(primary, group)["centroids"]
        b = group_profiles(replication, group)["centroids"]
        for label in sorted(a):
            correlation = float(np.corrcoef(a[label].to_numpy(), b[label].to_numpy())[0, 1])
            if correlation <= 0:
                raise ValueError(
                    f"{label}: replication centroid correlates {correlation:.3f} with the "
                    "primary centroid, so the archetype name cannot be carried over"
                )
            agreement[label] = round(correlation, 4)
    return agreement


# --------------------------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------------------------


def detect_outliers(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    """Isolation forest and Ledoit-Wolf Mahalanobis distance, fitted within each group."""
    pieces = []
    for group in config.OUTFIELD_GROUPS:
        sub = frame[frame["position_group"] == group].reset_index(drop=True)
        matrix = sub[F.OUTFIELD_CORE].to_numpy(float)

        forest = IsolationForest(
            n_estimators=ISOLATION_TREES,
            contamination="auto",
            random_state=config.RANDOM_STATE,
            bootstrap=False,
        ).fit(matrix)
        isolation = -forest.score_samples(matrix)  # higher is more anomalous

        covariance = LedoitWolf(store_precision=True, assume_centered=False).fit(matrix)
        mahalanobis = np.sqrt(np.maximum(covariance.mahalanobis(matrix), 0.0))

        pieces.append(
            sub.assign(
                isolation_score=isolation,
                isolation_pct=pd.Series(isolation).rank(pct=True).to_numpy(),
                mahalanobis=mahalanobis,
                mahalanobis_pct=pd.Series(mahalanobis).rank(pct=True).to_numpy(),
            )
        )
    table = pd.concat(pieces, ignore_index=True)
    table["outlier_rank_score"] = 0.5 * (table["isolation_pct"] + table["mahalanobis_pct"])
    table["season"] = season
    return table.sort_values("outlier_rank_score", ascending=False).reset_index(drop=True)


def explain_outlier(row: pd.Series) -> str:
    """Name the statistics that make one player unusual inside his position group."""
    deviations = row[F.OUTFIELD_CORE].astype(float)
    ranked = deviations.reindex(deviations.abs().sort_values(ascending=False).index)
    top = ranked.head(N_OUTLIER_REASONS)
    noun = GROUP_NOUN[row["position_group"]]
    parts = [
        f"{PROSE_LABELS[name]} {abs(value):.1f} standard deviations "
        f"{'above' if value >= 0 else 'below'} the {noun} mean"
        for name, value in top.items()
    ]
    return f"Unusual mainly through {_names(parts)}."


def outlier_report(table: pd.DataFrame, season: str) -> dict:
    rows = []
    for _, series in table.head(N_OUTLIERS).iterrows():
        rows.append(
            {
                "rank": len(rows) + 1,
                "player": series["player"],
                "team": series["team"],
                "position_group": series["position_group"],
                "position_full": series["position_full"],
                "minutes": int(series["minutes"]),
                "archetype_label": series["cluster_within_group_label"],
                "archetype_name": ARCHETYPE_NAMES[series["cluster_within_group_label"]],
                "isolation_score": round(float(series["isolation_score"]), 4),
                "isolation_percentile_in_group": round(float(series["isolation_pct"]), 4),
                "mahalanobis_distance": round(float(series["mahalanobis"]), 3),
                "mahalanobis_percentile_in_group": round(float(series["mahalanobis_pct"]), 4),
                "combined_percentile": round(float(series["outlier_rank_score"]), 4),
                "explanation": _no_dashes(explain_outlier(series), "outlier explanation"),
            }
        )
    return {
        "season": season,
        "method": {
            "space": "the 18 OUTFIELD_CORE features, z-scored within position group",
            "isolation_forest": (
                f"{ISOLATION_TREES} trees, contamination auto, fitted separately within DF, MF "
                f"and FW, seeded from RANDOM_STATE={config.RANDOM_STATE}"
            ),
            "mahalanobis": (
                "Ledoit-Wolf shrinkage covariance, also fitted within position group. Shrinkage "
                "is required because the forward group has fewer than three players per feature "
                "and the empirical covariance there is close to singular."
            ),
            "ranking": (
                "Both detectors are converted to a percentile within the position group so that "
                "groups of different size are comparable, and the table is ranked on the mean of "
                "the two percentiles."
            ),
            "caveat": (
                "An outlier here is a player whose per-90 profile is far from his position "
                "group in the surviving feature set. It is not a judgement of quality, and with "
                "only 18 features a single extreme rate such as goals per shot on a handful of "
                "shots can carry a player into this table on its own."
            ),
        },
        "detector_agreement": round(
            float(stats.spearmanr(table["isolation_score"], table["mahalanobis"]).statistic), 4
        ),
        "n_reported": len(rows),
        "players": rows,
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _radar_setup(
    ax: plt.Axes,
    limit: float,
    *,
    label_angle: float = 18.0,
    labelsize: float | None = None,
) -> np.ndarray:
    """Common polar furniture. Returns the closed angle array for RADAR_AXES.

    The radial tick labels are pushed onto the emptiest bisector between two spokes and
    given a surface-coloured box, because at the default position they sit underneath the
    plotted polygons and become unreadable.
    """
    n = len(F.RADAR_AXES)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(
        [RADAR_LABELS[name] for name in F.RADAR_AXES],
        fontsize=labelsize if labelsize is not None else plotting.BASE_FONT_PT - 2,
    )
    ax.tick_params(axis="x", pad=1)
    ax.set_rlim(-limit, limit)
    ticks = [-limit / 2.0, 0.0, limit / 2.0]
    ax.set_rgrids(
        ticks,
        labels=[f"{t:+.1f}" if t else "0" for t in ticks],
        fontsize=plotting.BASE_FONT_PT - 3,
        color=plotting.INK_SECONDARY,
    )
    ax.set_rlabel_position(label_angle)
    for text in ax.get_yticklabels():
        text.set_bbox({"facecolor": plotting.SURFACE, "edgecolor": "none", "pad": 0.8})
        text.set_zorder(7)
    ax.spines["polar"].set_color(plotting.GRID)
    ax.grid(color=plotting.GRID, linewidth=0.5)
    circle = np.linspace(0.0, 2.0 * np.pi, 181)
    ax.plot(circle, np.zeros_like(circle), color=plotting.INK_MUTED, lw=0.9, ls=(0, (3, 2)))
    return np.concatenate([angles, angles[:1]])


def _radar_values(centroid: pd.Series) -> np.ndarray:
    values = centroid[F.RADAR_AXES].to_numpy(float)
    return np.concatenate([values, values[:1]])


def _radar_limit(centroids: dict) -> float:
    peak = max(float(np.abs(c[F.RADAR_AXES]).max()) for c in centroids.values())
    return float(max(1.0, np.ceil(peak * 4.0) / 4.0))


def _quiet_angle(centroids: dict) -> float:
    """Bisector, in degrees, of the adjacent spoke pair carrying the least ink.

    The radial tick labels are parked there so that they do not land on a plotted line.
    """
    n = len(F.RADAR_AXES)
    loads = np.zeros(n)
    for centroid in centroids.values():
        loads += np.abs(centroid[F.RADAR_AXES].to_numpy(float))
    pairs = loads + np.roll(loads, -1)
    index = int(np.argmin(pairs))
    return float(360.0 * (index + 0.5) / n)


def figure_radar_panel(profile: dict, records: dict[str, dict]) -> None:
    """Small multiples, one panel per archetype, with the other drawn in grey for context."""
    group = profile["group"]
    labels = profile["labels"]
    limit = _radar_limit(profile["centroids"])
    colors = plotting.categorical(len(labels))
    marks = plotting.markers(len(labels))

    fig = plt.figure(figsize=(plotting.WIDTH_WIDE, 3.35))
    fig.set_layout_engine("none")
    width, height = 0.265, 0.575
    for index, (label, color, mark) in enumerate(zip(labels, colors, marks, strict=True)):
        centre = (index + 0.5) / len(labels)
        ax = fig.add_axes((centre - width / 2.0, 0.085, width, height), projection="polar")
        angles = _radar_setup(
            ax,
            limit,
            label_angle=_quiet_angle(profile["centroids"]),
            labelsize=plotting.BASE_FONT_PT - 3,
        )
        for other in labels:
            if other == label:
                continue
            ax.plot(
                angles,
                _radar_values(profile["centroids"][other]),
                color=plotting.CONTEXT_GREY,
                lw=1.0,
            )
        values = _radar_values(profile["centroids"][label])
        ax.plot(angles, values, color=color, lw=1.6, marker=mark, markersize=3.2)
        ax.fill(angles, values, color=color, alpha=0.14)
        record = records[label]
        fig.text(
            centre,
            0.865,
            f"{label} {textwrap.fill(record['name'], 26)} (n={record['n_players']})",
            ha="center",
            va="top",
            fontsize=plotting.BASE_FONT_PT - 1,
            color=plotting.INK_SECONDARY,
        )
    plotting.save_figure(fig, f"radar_panel_{group}")


def figure_archetype_profiles(records: list[dict]) -> None:
    """Every archetype centroid across all eighteen features, on the diverging ramp."""
    matrix = pd.DataFrame(
        {record["label"]: pd.Series(record["centroid"]) for record in records}
    ).reindex(F.OUTFIELD_CORE)
    values = matrix.to_numpy(float)
    peak = float(np.abs(values).max())
    norm = TwoSlopeNorm(vmin=-peak, vcenter=0.0, vmax=peak)

    fig, ax = plt.subplots(figsize=(plotting.WIDTH_FULL, 6.4))
    image = ax.imshow(values, cmap=plotting.DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        [f"{record['label']}\n{textwrap.fill(record['name'], 16)}" for record in records],
        fontsize=plotting.BASE_FONT_PT - 2,
    )
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(
        [FEATURE_LABELS[name] for name in matrix.index], fontsize=plotting.BASE_FONT_PT - 2
    )
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            ax.text(
                j,
                i,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=plotting.BASE_FONT_PT - 3,
                color=plotting.SURFACE if abs(value) > 0.62 * peak else plotting.INK_PRIMARY,
            )
    for index in range(1, matrix.shape[1]):
        if records[index]["position_group"] != records[index - 1]["position_group"]:
            ax.axvline(index - 0.5, color=plotting.SURFACE, lw=2.2)
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("standard deviations from the position group mean")
    ax.set_title(
        "Archetype centroid profiles\n"
        "columns are cluster centroids, rows are features, z-scored within position group",
        loc="left",
        pad=10,
    )
    plotting.save_figure(fig, "archetype_profiles")


def _stacked_offsets(
    ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, *, x_window: float, gap: float
) -> list[tuple[float, float]]:
    """Pick a vertical offset per label so that nearby labels do not overlap.

    Points are visited from the top down and each is given the first candidate offset
    that clears every label already placed within ``x_window`` display points.
    """
    ax.figure.canvas.draw()
    points = ax.transData.transform(np.column_stack([xs, ys]))
    candidates = [4.0, -16.0, 20.0, -32.0, 36.0, -48.0, 52.0, -64.0]
    offsets = [(4.0, 3.0)] * len(xs)
    placed: list[tuple[float, float]] = []
    for index in np.argsort(-points[:, 1]):
        px, py = points[index]
        chosen = candidates[-1]
        for candidate in candidates:
            clear = all(
                abs(px - qx) > x_window or abs(py + candidate - qy) > gap for qx, qy in placed
            )
            if clear:
                chosen = candidate
                break
        placed.append((px, py + chosen))
        offsets[index] = (4.0, chosen)
    return offsets


#: Points labelled directly on the scatter. Labelling all fifteen there produced
#: unreadable overlaps, so the remaining names are carried by the ranked panel instead.
N_SCATTER_LABELS = 5


def figure_outliers(table: pd.DataFrame, report: dict) -> None:
    """Left, the two detectors against each other. Right, the reported table, ranked."""
    reported = [row["player"] for row in report["players"]]
    flagged = table[table["player"].isin(reported)].set_index("player").loc[reported].reset_index()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(plotting.WIDTH_WIDE, 4.1),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )
    scatter, ranking = axes

    for group in config.OUTFIELD_GROUPS:
        sel = table["position_group"] == group
        scatter.scatter(
            table.loc[sel, "mahalanobis"],
            table.loc[sel, "isolation_score"],
            s=13,
            c=plotting.POSITION_COLORS[group],
            marker=plotting.POSITION_MARKERS[group],
            linewidths=0,
            alpha=0.8,
            label=f"{group} (n={int(sel.sum())})",
        )
    scatter.scatter(
        flagged["mahalanobis"],
        flagged["isolation_score"],
        s=46,
        facecolors="none",
        edgecolors=plotting.INK_PRIMARY,
        linewidths=0.7,
        zorder=5,
        label=f"top {len(flagged)} reported",
    )
    # Headroom on the right for the few direct labels, which all sit in the top corner.
    left, right = scatter.get_xlim()
    scatter.set_xlim(left, right + 0.22 * (right - left))
    head = flagged.head(N_SCATTER_LABELS)
    xs = head["mahalanobis"].to_numpy()
    ys = head["isolation_score"].to_numpy()
    for (dx, dy), x, y, name in zip(
        _stacked_offsets(scatter, xs, ys, x_window=150.0, gap=16.0),
        xs,
        ys,
        head["player"].tolist(),
        strict=True,
    ):
        plotting.annotate_points(scatter, [x], [y], [name], offset=(dx, dy))
    plotting.style_axis(
        scatter,
        xlabel="Mahalanobis distance (Ledoit-Wolf)",
        ylabel="isolation forest anomaly score",
        title=(
            f"detectors agree, Spearman {report['detector_agreement']:.2f}\n"
            f"top {N_SCATTER_LABELS} labelled, all {len(flagged)} circled"
        ),
    )
    scatter.legend(loc="lower right", fontsize=plotting.BASE_FONT_PT - 2)

    positions = np.arange(len(flagged))[::-1]
    ranking.barh(
        positions,
        flagged["mahalanobis"].to_numpy(),
        height=0.68,
        color=[plotting.POSITION_COLORS[g] for g in flagged["position_group"]],
        linewidth=0,
    )
    ranking.set_yticks(positions)
    ranking.set_yticklabels(
        [f"{i + 1}. {name}" for i, name in enumerate(flagged["player"])],
        fontsize=plotting.BASE_FONT_PT - 2,
    )
    ranking.set_ylim(-0.7, len(flagged) - 0.3)
    ranking.grid(True, axis="x")
    ranking.grid(False, axis="y")
    plotting.style_axis(
        ranking,
        xlabel="Mahalanobis distance",
        title="ranked on the mean of the two\nwithin-group percentiles",
    )
    plotting.save_figure(fig, "outliers_scatter")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    plotting.use_style()
    np.random.seed(config.RANDOM_STATE)

    stability = load_stability()
    primary = load_season(config.SEASON_PRIMARY)
    replication = load_season(config.SEASON_REPLICATION)

    print(f"=== archetypes, primary season {config.SEASON_PRIMARY} ===")
    records = build_archetypes(primary, stability, config.SEASON_PRIMARY)
    by_label = {record["label"]: record for record in records}
    for record in records:
        top = ", ".join(
            f"{item['feature']} {item['z_deviation']:+.2f}"
            for item in record["distinguishing_features"][:4]
        )
        print(f"  {record['label']:5s} n={record['n_players']:3d}  {record['name']}")
        print(f"        {top}")

    agreement = centroid_agreement(primary, replication)
    print("  replication centroid correlation:", agreement)

    # ---------------------------------------------------------------- figures
    for group in config.OUTFIELD_GROUPS:
        profile = group_profiles(primary, group)
        figure_radar_panel(profile, by_label)
    figure_archetype_profiles(records)

    outliers = detect_outliers(primary, config.SEASON_PRIMARY)
    report = outlier_report(outliers, config.SEASON_PRIMARY)
    figure_outliers(outliers, report)
    print(f"  top outlier: {report['players'][0]['player']}")

    # ---------------------------------------------------------------- parquet output
    for season, frame in (
        (config.SEASON_PRIMARY, primary),
        (config.SEASON_REPLICATION, replication),
    ):
        table = archetype_table(frame, season)
        path = config.DATA_PROCESSED / f"archetypes_{season}.parquet"
        table.to_parquet(path, index=False)
        print(f"  wrote {path.relative_to(config.ROOT)}  {table.shape}")

    # ---------------------------------------------------------------- json output
    _write_json(
        {
            "season": config.SEASON_PRIMARY,
            "k_per_position_group": 2,
            "space": (
                "the 18 OUTFIELD_CORE features z-scored within position group, so a centroid "
                "coordinate is the standardised deviation of the cluster from its position "
                "group mean"
            ),
            "naming_rule": (
                "Names were assigned after inspecting the centroid tables and the member lists, "
                "and are deliberately descriptive of the measured profile. Where a split is "
                "essentially high against low attacking involvement the name says so rather "
                "than implying a tactical role the data cannot support."
            ),
            "structural_note": (
                "Because the features are z-scored within position group, the group mean is "
                "the origin and the two centroids of a k=2 partition are exactly anti-parallel: "
                "the size-weighted sum of the two centroids is zero, so one is a negative "
                "multiple of the other. The two archetypes inside a position group are "
                "therefore the two ends of a single axis rather than two independently shaped "
                "profiles, and the larger cluster necessarily has the smaller deviations. This "
                "is a property of the partition and not a discovery about football, and it is "
                "stated here so that the paper does not read more structure into the result "
                "than the clustering can support."
            ),
            "n_distinguishing_features": N_DISTINGUISHING,
            "n_representative_players": N_REPRESENTATIVE,
            "n_boundary_players": N_BOUNDARY,
            "bootstrap_ari_global": round(
                float(stability["seasons"][config.SEASON_PRIMARY]["global"]["ari_mean"]), 4
            ),
            "replication_centroid_correlation": agreement,
            "archetypes": records,
        },
        config.RESULTS / "archetypes.json",
    )

    _write_json(
        {
            "features": F.OUTFIELD_CORE,
            "feature_labels": FEATURE_LABELS,
            "radar_axes": F.RADAR_AXES,
            "seasons": {
                season: {
                    label: {
                        "name": ARCHETYPE_NAMES[label],
                        "position_group": label.split("-")[0],
                        "n_players": int((frame["cluster_within_group_label"] == label).sum()),
                        "centroid": {
                            name: round(float(value), 4)
                            for name, value in group_profiles(frame, label.split("-")[0])[
                                "centroids"
                            ][label].items()
                        },
                    }
                    for label in ARCHETYPE_ORDER
                }
                for season, frame in (
                    (config.SEASON_PRIMARY, primary),
                    (config.SEASON_REPLICATION, replication),
                )
            },
            "replication_centroid_correlation": agreement,
        },
        config.METRICS / "archetype_profiles.json",
    )

    _write_json(report, config.METRICS / "outliers.json")


if __name__ == "__main__":
    main()
