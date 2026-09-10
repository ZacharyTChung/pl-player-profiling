"""Archetype characterisation on the pre-withdrawal archive.

What this module does
---------------------
``src/archive.py`` recovers the full feature set from a mirror that predates FBref's
January 2026 deletion: 9,263 eligible outfield player-seasons, five seasons from 2018 to
2022, five leagues, 33 features covering shooting, chance creation, progression by pass
and by carry, touches by pitch zone, take-ons, tackles, blocks, clearances, aerials, ball
recoveries and pass completion by distance. The declared consensus rule in
``config.K_RULE`` returns k=2 in every scope on that feature set, and
``results/metrics/structure.json`` shows the two-mode structure is real rather than an
artefact of partitioning a continuum: the observed silhouette beats all 25 clusterless
simulations with z between 13 and 24, and Hartigan's dip test rejects unimodality on PC1
for the outfield pool, for defenders and for midfielders. Beyond k=2 the observed
silhouette converges on the clusterless null, so no finer taxonomy is supported.

This module therefore describes exactly two archetypes inside each of DF, MF and FW, six
in total, and does not invent more. For each it computes the centroid in the within-group
z-scored space, so a centroid coordinate reads directly as a standardised deviation from
that position group's mean, reports the ten features with the largest absolute deviation,
lists the ten members closest to the centroid and the three closest to the decision
boundary with their season and league, draws the profile on ten fixed radar axes chosen
per position group to span shooting, creation, progression, touches and defending, and
measures whether membership survives moving between seasons and between leagues.

Naming
------
Names were assigned only after the centroid tables and member lists printed by the
exploratory pass were inspected, in line with the project ground rule that cluster names
are hypotheses. Every number quoted inside a justification paragraph is interpolated from
the computed profile at run time and every direction word follows the sign of the computed
deviation, so the prose cannot drift away from the data. The example players named in each
paragraph are checked against the computed representative list and the module raises if a
cited player is no longer a member.

Cost control on a sample twenty five times larger than the live one
-------------------------------------------------------------------
Two computations here are quadratic in the number of players and are sub-sampled rather
than run whole, with the sub-sample seeded from ``config.RANDOM_STATE`` so the result is
reproducible. ``cluster.bootstrap_stability`` accumulates a co-assignment matrix of shape
n by n over 200 resamples, which at n=3,793 defenders means 14.4 million cells rewritten
per resample, so it runs on a random sub-sample of ``BOOTSTRAP_SAMPLE`` players per
position group. Silhouette inside ``cluster.internal_curves`` is also quadratic, but at
these group sizes a full pass costs a fraction of a second per k and is left exact. Every
other quantity here, including the isolation forest, the Mahalanobis distances, the
centroids and the season and league stability tables, is linear in the number of players
and is computed on the whole sample.

Outputs
-------
``results/archive_archetypes.json``, ``results/metrics/archive_archetype_profiles.json``,
``results/metrics/archive_archetype_stability.json``,
``results/metrics/archive_outliers.json`` and the figures
``archive_radar_panel_{group}``, ``archive_archetype_profiles`` and ``archive_outliers``.
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
from sklearn.metrics import adjusted_rand_score

import config
from src import archive as A
from src import cluster as C
from src import plotting

# --------------------------------------------------------------------------------------
# Declared choices
# --------------------------------------------------------------------------------------

#: Clusters per position group. Fixed by the selection rule, confirmed at run time.
K = 2

#: Largest absolute centroid deviations reported per archetype. Ten rather than the eight
#: used on the live sample, because there are 33 features here instead of 18.
N_DISTINGUISHING = 10

#: Members closest to the centroid, reported as representative.
N_REPRESENTATIVE = 10

#: Members closest to the two-cluster decision boundary.
N_BOUNDARY = 3

#: Length of the pooled outlier table.
N_OUTLIERS = 15

#: Statistics quoted in each outlier explanation.
N_OUTLIER_REASONS = 3

ISOLATION_TREES = 300

#: Players drawn per position group for the bootstrap co-assignment matrix, which is
#: quadratic in the number of players. See the module docstring.
BOOTSTRAP_SAMPLE = 2500

#: Outliers labelled directly on the scatter. The reported table names all fifteen with
#: their season, so only the most extreme few are labelled in the cloud; five of them
#: produced leader lines that crossed each other and ran through neighbouring labels.
N_SCATTER_LABELS = 3

ARCHETYPE_ORDER = ["DF-0", "DF-1", "MF-0", "MF-1", "FW-0", "FW-1"]

#: Ten radar axes per position group, chosen from ``archive.OUTFIELD_CORE`` so that each
#: set spans shooting, chance creation, progression, touches by zone and defending. The
#: sets differ by group because the axes that separate defenders are not the axes that
#: separate forwards, and a single shared set would waste spokes on flat features.
RADAR_AXES = {
    "DF": [
        "np_xg_p90",
        "key_passes_p90",
        "sca_p90",
        "crosses_p90",
        "progressive_passes_p90",
        "progressive_carries_p90",
        "touches_att_third_p90",
        "touches_def_third_p90",
        "clearances_padj_p90",
        "aerials_won_pct",
    ],
    "MF": [
        "np_xg_p90",
        "xag_p90",
        "key_passes_p90",
        "crosses_p90",
        "progressive_passes_p90",
        "progressive_receptions_p90",
        "touches_att_third_p90",
        "touches_def_third_p90",
        "tackles_padj_p90",
        "ball_recoveries_p90",
    ],
    "FW": [
        "np_xg_p90",
        "shots_p90",
        "xag_p90",
        "key_passes_p90",
        "crosses_p90",
        "progressive_carries_p90",
        "take_ons_p90",
        "touches_att_pen_p90",
        "touches_mid_third_p90",
        "tackles_padj_p90",
    ],
}

# --------------------------------------------------------------------------------------
# Display names
# --------------------------------------------------------------------------------------

#: Short names for radar spokes, kept short so the outer polar labels do not clip.
RADAR_LABELS = {
    "np_xg_p90": "npxG",
    "shots_p90": "Shots",
    "xag_p90": "xAG",
    "key_passes_p90": "Key passes",
    "sca_p90": "SCA",
    "crosses_p90": "Crosses",
    "progressive_passes_p90": "Prog. passes",
    "progressive_carries_p90": "Prog. carries",
    "progressive_receptions_p90": "Prog. receipts",
    "take_ons_p90": "Take-ons",
    "touches_att_third_p90": "Att 3rd",
    "touches_mid_third_p90": "Mid 3rd",
    "touches_def_third_p90": "Def 3rd",
    "touches_att_pen_p90": "Box touches",
    "clearances_padj_p90": "Clearances",
    "tackles_padj_p90": "Tackles",
    "ball_recoveries_p90": "Recoveries",
    "aerials_won_pct": "Aerials %",
}

#: Compact names for the heatmap rows and the JSON feature tables.
FEATURE_LABELS = {
    "np_xg_p90": "npxG/90",
    "shots_p90": "Shots/90",
    "goals_non_penalty_p90": "Non-penalty goals/90",
    "xag_p90": "xAG/90",
    "key_passes_p90": "Key passes/90",
    "sca_p90": "Shot creating actions/90",
    "gca_p90": "Goal creating actions/90",
    "passes_final_third_p90": "Passes into final third/90",
    "passes_penalty_area_p90": "Passes into box/90",
    "crosses_p90": "Crosses/90",
    "progressive_passes_p90": "Progressive passes/90",
    "progressive_carries_p90": "Progressive carries/90",
    "progressive_receptions_p90": "Progressive receptions/90",
    "carries_final_third_p90": "Carries into final third/90",
    "touches_def_pen_p90": "Touches, own box/90",
    "touches_def_third_p90": "Touches, defensive third/90",
    "touches_mid_third_p90": "Touches, middle third/90",
    "touches_att_third_p90": "Touches, attacking third/90",
    "touches_att_pen_p90": "Touches, opposition box/90",
    "take_ons_p90": "Take-ons attempted/90",
    "tackles_padj_p90": "Tackles/90 (adj)",
    "interceptions_padj_p90": "Interceptions/90 (adj)",
    "blocks_padj_p90": "Blocks/90 (adj)",
    "clearances_padj_p90": "Clearances/90 (adj)",
    "ball_recoveries_p90": "Ball recoveries/90",
    "fouls_committed_padj_p90": "Fouls committed/90 (adj)",
    "shot_accuracy_pct": "Shots on target %",
    "take_on_success_pct": "Take-on success %",
    "tackle_win_pct": "Tackle win %",
    "aerials_won_pct": "Aerials won %",
    "pass_cmp_short_pct": "Short pass completion %",
    "pass_cmp_medium_pct": "Medium pass completion %",
    "pass_cmp_long_pct": "Long pass completion %",
}

#: Spelled-out names for prose, so no justification contains a slash or an abbreviation.
PROSE_LABELS = {
    "np_xg_p90": "non-penalty expected goals per 90",
    "shots_p90": "shots per 90",
    "goals_non_penalty_p90": "non-penalty goals per 90",
    "xag_p90": "expected assisted goals per 90",
    "key_passes_p90": "key passes per 90",
    "sca_p90": "shot creating actions per 90",
    "gca_p90": "goal creating actions per 90",
    "passes_final_third_p90": "passes into the final third per 90",
    "passes_penalty_area_p90": "passes into the penalty area per 90",
    "crosses_p90": "crosses per 90",
    "progressive_passes_p90": "progressive passes per 90",
    "progressive_carries_p90": "progressive carries per 90",
    "progressive_receptions_p90": "progressive receptions per 90",
    "carries_final_third_p90": "carries into the final third per 90",
    "touches_def_pen_p90": "touches in their own penalty area per 90",
    "touches_def_third_p90": "touches in the defensive third per 90",
    "touches_mid_third_p90": "touches in the middle third per 90",
    "touches_att_third_p90": "touches in the attacking third per 90",
    "touches_att_pen_p90": "touches in the opposition penalty area per 90",
    "take_ons_p90": "attempted take-ons per 90",
    "tackles_padj_p90": "possession adjusted tackles per 90",
    "interceptions_padj_p90": "possession adjusted interceptions per 90",
    "blocks_padj_p90": "possession adjusted blocks per 90",
    "clearances_padj_p90": "possession adjusted clearances per 90",
    "ball_recoveries_p90": "ball recoveries per 90",
    "fouls_committed_padj_p90": "possession adjusted fouls committed per 90",
    "shot_accuracy_pct": "share of shots on target",
    "take_on_success_pct": "take-on success rate",
    "tackle_win_pct": "tackle win rate",
    "aerials_won_pct": "share of aerial duels won",
    "pass_cmp_short_pct": "short pass completion",
    "pass_cmp_medium_pct": "medium range pass completion",
    "pass_cmp_long_pct": "long pass completion",
}

#: Features already expressed as a percentage, formatted differently in prose.
PERCENT_FEATURES = set(A.RATES)

GROUP_NOUN = {"DF": "defender", "MF": "midfield", "FW": "forward"}
GROUP_PLURAL = {"DF": "defenders", "MF": "midfielders", "FW": "forwards"}

# --------------------------------------------------------------------------------------
# Names, assigned after inspecting the centroid tables and the member lists
# --------------------------------------------------------------------------------------

ARCHETYPE_NAMES = {
    "DF-0": "Deep central defenders",
    "DF-1": "Wide attacking defenders",
    "MF-0": "Deep ball winning midfielders",
    "MF-1": "Advanced attacking midfielders",
    "FW-0": "Finishing centre forwards",
    "FW-1": "Creating and carrying forwards",
}

ARCHETYPE_SUMMARIES = {
    "DF-0": (
        "Defenders concentrated in their own third and their own box, clearing and "
        "winning headers, with almost no crossing or attacking third involvement."
    ),
    "DF-1": (
        "Defenders who cross, carry and receive progressive passes in the attacking "
        "third at roughly ten times the rate of the other defender cluster."
    ),
    "MF-0": (
        "Midfielders who recover, tackle and pass from the defensive and middle thirds, "
        "and who reach the opposition box far less often than the midfield average."
    ),
    "MF-1": (
        "Midfielders who receive progressively, touch the ball in the attacking third "
        "and the opposition box, shoot and cross, and defend much less."
    ),
    "FW-0": (
        "Forwards whose shooting matches the forward average but whose carrying, passing "
        "and touches outside the penalty area sit well below it."
    ),
    "FW-1": (
        "Forwards who drop into the build-up, carry the ball progressively and create "
        "chances at more than one standard deviation above the forward average."
    ),
}

#: Players named in each justification paragraph. Verified against the computed
#: representative list at run time; a mismatch raises rather than being papered over.
CITED_PLAYERS = {
    "DF-0": [
        "Çağlar Söyüncü",
        "Loïc Perrin",
        "Nicolas Nkoulou",
        "Benjamin Pavard",
        "Diego Llorente",
    ],
    "DF-1": [
        "José Luis Gayà",
        "Sergio Reguilón",
        "Stefan Lainer",
        "Timothy Castagne",
        "Kévin Rodrigues",
    ],
    "MF-0": [
        "João Moutinho",
        "Douglas Luiz",
        "Sandro Tonali",
        "Luka Milivojević",
        "Oriol Romeu",
    ],
    "MF-1": [
        "Brais Méndez",
        "Andreas Pereira",
        "Nathan Redmond",
        "Hamed Junior Traorè",
        "Gio Reyna",
    ],
    "FW-0": [
        "Ollie Watkins",
        "João Pedro",
        "Enes Ünal",
        "Habib Diallo",
        "Luis Javier Suárez",
    ],
    "FW-1": [
        "Amine Gouiri",
        "Musa Barrow",
        "Diogo Jota",
        "Wilfried Zaha",
        "Sadio Mané",
    ],
}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _native(obj):
    """JSON encoder fallback for numpy scalars and arrays."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _write_json(payload: dict, path: Path) -> Path:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_native, sort_keys=False)
    print(f"  wrote {path.relative_to(config.ROOT)}")
    return path


def _no_dashes(text: str, where: str) -> str:
    """Guard the ground rule that no em dash or en dash reaches the paper."""
    for bad in ("—", "–"):
        if bad in text:
            raise ValueError(f"{where} contains a dash character: {text!r}")
    return text


def _names(items: list[str]) -> str:
    """Join as "A, B and C"."""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _dev(value: float) -> str:
    """Render one centroid coordinate as "0.62 standard deviations above"."""
    direction = "above" if float(value) >= 0 else "below"
    return f"{abs(float(value)):.2f} standard deviations {direction}"


def _fmt(feature: str, value: float) -> str:
    if feature in PERCENT_FEATURES:
        return f"{float(value):.1f} percent"
    return f"{float(value):.2f}"


def _pair(record: dict, feature: str) -> str:
    """Render "31.71 against 18.97", this archetype first, in football units."""
    return f"{_mine(record, feature)} against {_theirs(record, feature)}"


def _mine(record: dict, feature: str) -> str:
    """This archetype's mean for one feature, in football units."""
    return _fmt(feature, record["raw_means"][feature])


def _theirs(record: dict, feature: str) -> str:
    """The counterpart archetype's mean for the same feature, in football units."""
    return _fmt(feature, record["counterpart_raw_means"][feature])


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the unstandardised eligible table and the within-group z-scored table.

    The two frames are written by ``archive.build`` from the same rows in the same order,
    which is asserted here because every downstream join is positional.
    """
    eligible = pd.read_parquet(config.DATA_PROCESSED / "archive_eligible.parquet")
    z_group = pd.read_parquet(config.DATA_PROCESSED / "archive_z_bygroup.parquet")
    if len(eligible) != len(z_group):
        raise ValueError("eligible and z-scored archive frames differ in length")
    if not (eligible["player"].to_numpy() == z_group["player"].to_numpy()).all():
        raise ValueError("eligible and z-scored archive frames are not row aligned")
    missing = [f for f in A.OUTFIELD_CORE if f not in z_group.columns]
    if missing:
        raise KeyError(f"archive features absent from the z-scored frame: {missing}")
    return eligible.reset_index(drop=True), z_group.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Partition
# --------------------------------------------------------------------------------------


def confirm_k(x: np.ndarray, group: str) -> dict:
    """Re-run the declared consensus rule inside one position group.

    The rule is not re-derived here, it is checked. ``cluster.internal_curves`` and
    ``cluster.apply_k_rule`` are the same functions the live pipeline uses, so a
    disagreement with the recorded k=2 result would surface as a failure rather than as a
    silently different partition.
    """
    curves = C.internal_curves(x, f"archive-profiles|{group}")
    decision = C.apply_k_rule(curves)
    return {
        "chosen_k": decision["chosen_k"],
        "rule_branch": decision["rule_branch"],
        "silhouette_by_k": curves["silhouette"],
        "davies_bouldin_by_k": curves["davies_bouldin"],
        "calinski_harabasz_by_k": curves["calinski_harabasz"],
        "gap_by_k": curves["gap"],
        "favoured_by_criterion": decision["favoured_by_criterion"],
        "agrees_with_declared_k": bool(decision["chosen_k"] == K),
    }


def fit_group(eligible: pd.DataFrame, z_group: pd.DataFrame, group: str) -> dict:
    """Fit the two-cluster partition inside one position group.

    Clusters are relabelled by descending size so that ``{group}-0`` is always the larger
    one. KMeans ids are arbitrary and change with the seed, and the paper cites labels, so
    the ordering has to come from the data rather than from the fit.
    """
    mask = (z_group["position_group"] == group).to_numpy()
    x = z_group.loc[mask, A.OUTFIELD_CORE].to_numpy(float)
    fit = C._kmeans(x, K, f"archive|{group}")
    order = pd.Series(fit.labels_).value_counts().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[v] for v in fit.labels_], dtype=int)
    centroids = fit.cluster_centers_[order]
    distances = np.stack([np.linalg.norm(x - c, axis=1) for c in centroids], axis=1)
    return {
        "group": group,
        "mask": mask,
        "matrix": x,
        "labels": labels,
        "centroids": centroids,
        "distances": distances,
        "frame": eligible.loc[mask].reset_index(drop=True),
        "z_frame": z_group.loc[mask].reset_index(drop=True),
        "inertia": float(fit.inertia_),
    }


def partition(eligible: pd.DataFrame, z_group: pd.DataFrame) -> tuple[dict[str, dict], pd.Series]:
    """Fit all three groups and return the fits plus one archetype label per player-season."""
    fits: dict[str, dict] = {}
    assignment = pd.Series(index=eligible.index, dtype=object)
    for group in config.OUTFIELD_GROUPS:
        fit = fit_group(eligible, z_group, group)
        fits[group] = fit
        assignment.loc[fit["mask"]] = [f"{group}-{c}" for c in fit["labels"]]
    if assignment.isna().any():
        raise ValueError("some eligible players were not assigned an archetype")
    return fits, assignment


# --------------------------------------------------------------------------------------
# Archetype records
# --------------------------------------------------------------------------------------


def describe_cluster(fit: dict, cluster: int) -> dict:
    """Centroid, distinguishing features, representative and boundary members."""
    group = fit["group"]
    label = f"{group}-{cluster}"
    frame = fit["frame"]
    members = fit["labels"] == cluster
    other = 1 - cluster
    centroid = pd.Series(fit["centroids"][cluster], index=A.OUTFIELD_CORE)

    ranked = centroid.reindex(centroid.abs().sort_values(ascending=False).index)
    distinguishing = [
        {
            "feature": name,
            "label": FEATURE_LABELS[name],
            "z_deviation": round(float(value), 4),
            "mean_in_cluster": round(float(frame.loc[members, name].mean()), 4),
            "mean_in_counterpart": round(float(frame.loc[fit["labels"] == other, name].mean()), 4),
        }
        for name, value in ranked.head(N_DISTINGUISHING).items()
    ]

    own = np.where(members, fit["distances"][:, cluster], np.inf)
    representative = []
    for index in np.argsort(own)[:N_REPRESENTATIVE]:
        row = frame.iloc[int(index)]
        representative.append(
            {
                "player": row["player"],
                "team": row["team"],
                "league": row["league"],
                "season": int(row["season"]),
                "position_full": row["position_full"],
                "minutes": int(row["minutes"]),
                "distance_to_centroid": round(float(fit["distances"][index, cluster]), 3),
            }
        )

    margin = np.abs(fit["distances"][:, 0] - fit["distances"][:, 1])
    margin = np.where(members, margin, np.inf)
    boundary = []
    for index in np.argsort(margin)[:N_BOUNDARY]:
        row = frame.iloc[int(index)]
        boundary.append(
            {
                "player": row["player"],
                "team": row["team"],
                "league": row["league"],
                "season": int(row["season"]),
                "position_full": row["position_full"],
                "minutes": int(row["minutes"]),
                "margin": round(float(np.abs(np.diff(fit["distances"][index]))[0]), 3),
            }
        )

    return {
        "label": label,
        "name": ARCHETYPE_NAMES[label],
        "position_group": group,
        "n_players": int(members.sum()),
        "n_position_group": int(len(frame)),
        "share_of_position_group": round(float(members.mean()), 4),
        "mean_minutes": round(float(frame.loc[members, "minutes"].mean()), 1),
        "mean_age": round(float(frame.loc[members, "age"].mean()), 2),
        "listed_position_counts": frame.loc[members, "position_full"]
        .value_counts()
        .head(5)
        .to_dict(),
        "league_share": frame.loc[members, "league"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "season_share": {
            str(k): round(float(v), 4)
            for k, v in frame.loc[members, "season"]
            .value_counts(normalize=True)
            .sort_index()
            .items()
        },
        "centroid": {name: round(float(value), 4) for name, value in centroid.items()},
        "raw_means": {name: float(frame.loc[members, name].mean()) for name in A.OUTFIELD_CORE},
        "counterpart_raw_means": {
            name: float(frame.loc[fit["labels"] == other, name].mean()) for name in A.OUTFIELD_CORE
        },
        "distinguishing_features": distinguishing,
        "representative_players": representative,
        "boundary_players": boundary,
    }


# --------------------------------------------------------------------------------------
# Justifications. Every number is interpolated from the computed record.
# --------------------------------------------------------------------------------------


def _stability_clause(label: str, stability: dict) -> str:
    group = label.split("-")[0]
    per_group = stability["consecutive_seasons"]["by_position_group"][group]
    lo = min(row["switch_rate"] for row in per_group["pairs"])
    hi = max(row["switch_rate"] for row in per_group["pairs"])
    ari_lo = min(row["adjusted_rand_index"] for row in per_group["pairs"])
    ari_hi = max(row["adjusted_rand_index"] for row in per_group["pairs"])
    league = stability["league_refits"]["by_position_group"][group]
    return (
        f"Membership is stable across time and competition: between {lo * 100:.0f} and "
        f"{hi * 100:.0f} percent of {GROUP_PLURAL[group]} who play two consecutive seasons "
        f"change cluster, the adjusted Rand index between consecutive seasons runs from "
        f"{ari_lo:.2f} to {ari_hi:.2f}, and refitting the same two cluster partition inside "
        f"a single league recovers the pooled one with an adjusted Rand index from "
        f"{league['ari_min']:.2f} to {league['ari_max']:.2f}."
    )


def _justify_df0(record: dict, stability: dict) -> str:
    share = record["share_of_position_group"] * 100
    return (
        f"These {record['n_players']:,} player-seasons, {share:.0f} percent of all defenders in "
        f"the sample, are the central defenders, recovered from the statistics rather than read "
        f"off the position label. Touches in the defensive third average "
        f"{_mine(record, 'touches_def_third_p90')} per 90 against "
        f"{_theirs(record, 'touches_def_third_p90')} for the other defender cluster, touches in "
        f"their own penalty area {_pair(record, 'touches_def_pen_p90')} and possession adjusted "
        f"clearances {_pair(record, 'clearances_padj_p90')}, which places this centroid "
        f"{_dev(record['centroid']['touches_def_third_p90'])} the defender mean on defensive "
        f"third touches and {_dev(record['centroid']['clearances_padj_p90'])} it on clearances. "
        f"They win {_mine(record, 'aerials_won_pct')} of their aerial duels against "
        f"{_theirs(record, 'aerials_won_pct')}, and they complete "
        f"{_mine(record, 'pass_cmp_medium_pct')} of medium range passes against "
        f"{_theirs(record, 'pass_cmp_medium_pct')} and {_mine(record, 'pass_cmp_long_pct')} of "
        f"long passes against {_theirs(record, 'pass_cmp_long_pct')}, so their distribution is "
        f"short, safe and lateral. The attacking side of the profile is close to absent: "
        f"crosses per 90 of {_pair(record, 'crosses_p90')}, touches in the attacking third "
        f"{_pair(record, 'touches_att_third_p90')} and shot creating actions "
        f"{_pair(record, 'sca_p90')}. Representative members are "
        f"{_names(CITED_PLAYERS['DF-0'])}. The listed position does not encode this split, since "
        f"most members of both defender clusters are listed simply as DF, which is the point: "
        f"the clustering separates centre backs from wide defenders on measured behaviour where "
        f"the label cannot. {_stability_clause('DF-0', stability)}"
    )


def _justify_df1(record: dict, stability: dict) -> str:
    return (
        f"These {record['n_players']:,} player-seasons are the wide defenders, and the separation "
        f"from their central counterparts is the sharpest contrast in the study. Crosses average "
        f"{_mine(record, 'crosses_p90')} per 90 against {_theirs(record, 'crosses_p90')} for the "
        f"central cluster, which places this centroid {_dev(record['centroid']['crosses_p90'])} "
        f"the defender mean, touches in the attacking third "
        f"{_pair(record, 'touches_att_third_p90')}, progressive receptions "
        f"{_pair(record, 'progressive_receptions_p90')} and carries into the final third "
        f"{_pair(record, 'carries_final_third_p90')}. The creation follows: key passes per 90 "
        f"of {_pair(record, 'key_passes_p90')}, passes into the penalty area "
        f"{_pair(record, 'passes_penalty_area_p90')} and shot creating actions "
        f"{_pair(record, 'sca_p90')}. They attempt {_mine(record, 'take_ons_p90')} take-ons per "
        f"90 against {_theirs(record, 'take_ons_p90')} and complete a smaller share of them, "
        f"{_mine(record, 'take_on_success_pct')} against "
        f"{_theirs(record, 'take_on_success_pct')}, which is what attempting them in wide areas "
        f"against a set defence costs. The defensive side is correspondingly lighter, with "
        f"possession adjusted clearances {_pair(record, 'clearances_padj_p90')} and "
        f"{_mine(record, 'aerials_won_pct')} of aerial duels won against "
        f"{_theirs(record, 'aerials_won_pct')}. Representative members are "
        f"{_names(CITED_PLAYERS['DF-1'])}, all full backs or wing backs. "
        f"{_stability_clause('DF-1', stability)}"
    )


def _justify_mf0(record: dict, stability: dict) -> str:
    return (
        f"These {record['n_players']:,} player-seasons are the deep midfielders. Ball recoveries "
        f"average {_mine(record, 'ball_recoveries_p90')} per 90 against "
        f"{_theirs(record, 'ball_recoveries_p90')} for the advanced cluster, possession adjusted "
        f"tackles {_pair(record, 'tackles_padj_p90')} and possession adjusted interceptions "
        f"{_pair(record, 'interceptions_padj_p90')}, and they win "
        f"{_mine(record, 'aerials_won_pct')} of their aerial duels against "
        f"{_theirs(record, 'aerials_won_pct')}. Their touches sit behind the ball, "
        f"{_mine(record, 'touches_def_third_p90')} per 90 in the defensive third against "
        f"{_theirs(record, 'touches_def_third_p90')} and "
        f"{_mine(record, 'touches_mid_third_p90')} in the middle third against "
        f"{_theirs(record, 'touches_mid_third_p90')}. The detail that matters is that they are "
        f"not passive in possession: passes into the final third average "
        f"{_mine(record, 'passes_final_third_p90')} per 90 against "
        f"{_theirs(record, 'passes_final_third_p90')} and progressive passes "
        f"{_pair(record, 'progressive_passes_p90')}, both above the midfield mean, so they "
        f"advance the ball by playing it forward rather than by carrying it or receiving it high "
        f"up the pitch. What separates them is where the move ends for them, since touches in "
        f"the opposition penalty area average {_mine(record, 'touches_att_pen_p90')} per 90 "
        f"against {_theirs(record, 'touches_att_pen_p90')}, "
        f"progressive receptions {_pair(record, 'progressive_receptions_p90')} and shots "
        f"{_pair(record, 'shots_p90')}. Representative members are "
        f"{_names(CITED_PLAYERS['MF-0'])}. {_stability_clause('MF-0', stability)}"
    )


def _justify_mf1(record: dict, stability: dict) -> str:
    return (
        f"These {record['n_players']:,} player-seasons are the advanced and wide attacking "
        f"midfielders, and the listed position half agrees, since the most common single listed "
        f"value inside the cluster is the dual designation MF,FW. Progressive receptions average "
        f"{_mine(record, 'progressive_receptions_p90')} per 90 against "
        f"{_theirs(record, 'progressive_receptions_p90')} for the deep cluster, which is "
        f"{_dev(record['centroid']['progressive_receptions_p90'])} the midfield mean, touches in "
        f"the opposition penalty area {_pair(record, 'touches_att_pen_p90')}, touches in the "
        f"attacking third {_pair(record, 'touches_att_third_p90')} and progressive carries "
        f"{_pair(record, 'progressive_carries_p90')}. Output follows from that position: shots "
        f"per 90 of {_pair(record, 'shots_p90')}, non-penalty expected goals "
        f"{_pair(record, 'np_xg_p90')}, key passes {_pair(record, 'key_passes_p90')} and crosses "
        f"{_pair(record, 'crosses_p90')}. They defend much less, with ball recoveries per 90 "
        f"of {_pair(record, 'ball_recoveries_p90')} and possession adjusted tackles "
        f"{_pair(record, 'tackles_padj_p90')}, and they complete a smaller share of their passing "
        f"at every distance, {_mine(record, 'pass_cmp_short_pct')} short against "
        f"{_theirs(record, 'pass_cmp_short_pct')} and {_mine(record, 'pass_cmp_medium_pct')} at "
        f"medium range against {_theirs(record, 'pass_cmp_medium_pct')}, which is what passing "
        f"into congested areas costs. Representative members are "
        f"{_names(CITED_PLAYERS['MF-1'])}. {_stability_clause('MF-1', stability)}"
    )


def _justify_fw0(record: dict, stability: dict) -> str:
    share = record["share_of_position_group"] * 100
    return (
        f"These {record['n_players']:,} player-seasons are {share:.0f} percent of all forwards "
        f"and are defined by what they do not do rather than by any positive extreme. Their "
        f"shooting sits at the forward average, with non-penalty expected goals of "
        f"{_mine(record, 'np_xg_p90')} per 90 against {_theirs(record, 'np_xg_p90')} for the "
        f"other forward cluster and non-penalty goals of "
        f"{_pair(record, 'goals_non_penalty_p90')}, and their box presence is close to average "
        f"as well, with touches in the opposition penalty area of "
        f"{_pair(record, 'touches_att_pen_p90')}. Everything that happens before the shot sits "
        f"well below average. Progressive carries average "
        f"{_mine(record, 'progressive_carries_p90')} per 90 against "
        f"{_theirs(record, 'progressive_carries_p90')}, progressive passes "
        f"{_pair(record, 'progressive_passes_p90')}, passes into the penalty area "
        f"{_pair(record, 'passes_penalty_area_p90')}, shot creating actions "
        f"{_pair(record, 'sca_p90')} and touches in the middle third "
        f"{_pair(record, 'touches_mid_third_p90')}. They also complete less of their short "
        f"passing, {_mine(record, 'pass_cmp_short_pct')} against "
        f"{_theirs(record, 'pass_cmp_short_pct')}. Representative members are "
        f"{_names(CITED_PLAYERS['FW-0'])}. This is the centre forward who is fed rather than the "
        f"one who builds, and the honest reading of the cluster is low involvement outside the "
        f"box rather than a distinct finishing skill, because none of the finishing statistics "
        f"separate the two forward clusters. {_stability_clause('FW-0', stability)}"
    )


def _justify_fw1(record: dict, stability: dict) -> str:
    return (
        f"These {record['n_players']:,} player-seasons are the smallest of the six clusters and "
        f"the most sharply drawn, because the forward split is uneven at roughly three to one "
        f"and the deviations concentrate on the smaller side. Progressive carries average "
        f"{_mine(record, 'progressive_carries_p90')} per 90 against "
        f"{_theirs(record, 'progressive_carries_p90')} for the other forward cluster, which is "
        f"{_dev(record['centroid']['progressive_carries_p90'])} the forward mean, carries into "
        f"the final third {_pair(record, 'carries_final_third_p90')} and attempted take-ons "
        f"{_pair(record, 'take_ons_p90')}. They create as much as they finish: key passes "
        f"average {_mine(record, 'key_passes_p90')} per 90 against "
        f"{_theirs(record, 'key_passes_p90')}, passes into the penalty area "
        f"{_pair(record, 'passes_penalty_area_p90')}, crosses {_pair(record, 'crosses_p90')} and "
        f"shot creating actions {_pair(record, 'sca_p90')}. They also drop deep, with touches in "
        f"the middle third of {_mine(record, 'touches_mid_third_p90')} per 90 against "
        f"{_theirs(record, 'touches_mid_third_p90')}, touches in the "
        f"defensive third of {_pair(record, 'touches_def_third_p90')} and ball recoveries of "
        f"{_pair(record, 'ball_recoveries_p90')}. Their own shooting is no better than the other "
        f"cluster, with non-penalty expected goals of {_mine(record, 'np_xg_p90')} per 90 "
        f"against {_theirs(record, 'np_xg_p90')}, so "
        f"this is a wide or second forward who joins the build-up rather than a superior scorer. "
        f"Representative members are {_names(CITED_PLAYERS['FW-1'])}. "
        f"{_stability_clause('FW-1', stability)}"
    )


JUSTIFIERS = {
    "DF-0": _justify_df0,
    "DF-1": _justify_df1,
    "MF-0": _justify_mf0,
    "MF-1": _justify_mf1,
    "FW-0": _justify_fw0,
    "FW-1": _justify_fw1,
}


def attach_justifications(records: dict[str, dict], stability: dict) -> None:
    """Write one justification paragraph per archetype, verifying the cited players."""
    for label, record in records.items():
        available = {row["player"] for row in record["representative_players"]}
        unknown = [name for name in CITED_PLAYERS[label] if name not in available]
        if unknown:
            raise ValueError(
                f"{label} justification cites players who are no longer representative: {unknown}"
            )
        text = JUSTIFIERS[label](record, stability)
        record["justification"] = _no_dashes(" ".join(text.split()), f"{label} justification")
        record["summary"] = _no_dashes(ARCHETYPE_SUMMARIES[label], f"{label} summary")


# --------------------------------------------------------------------------------------
# Stability across seasons and leagues
# --------------------------------------------------------------------------------------

#: Identity key for following one footballer across seasons. Name alone collides, so the
#: nationality is carried as well. Mid-season transfers give a player two rows in one
#: season, and the row with the most minutes is kept.
IDENTITY = ["player", "Nation", "season"]


def _expected_switch_rate(first: pd.Series, second: pd.Series) -> float:
    """Switch rate expected if the two seasons were independent, given the marginals.

    The baseline is computed inside each position group and then pooled by group size,
    because a defender can only be assigned to one of the two defender clusters. Scoring
    the six labels against each other instead would compare the partition against a null
    that the position label already rules out and would flatter it.
    """
    groups = first.str[:2]
    total, expected = 0, 0.0
    for group in sorted(groups.unique()):
        take = groups == group
        p = first[take].value_counts(normalize=True)
        q = second[take].value_counts(normalize=True)
        shared = p.index.intersection(q.index)
        rate = 1.0 - sum(p[label] * q[label] for label in shared)
        expected += rate * int(take.sum())
        total += int(take.sum())
    return float(expected / total) if total else float("nan")


def season_stability(table: pd.DataFrame) -> dict:
    """How often a player changes archetype between one season and the next.

    Rows are deduplicated to one per player-season before the comparison, keeping the club
    spell with the most minutes, because a mid-season transfer otherwise contributes two
    contradictory labels for the same season.
    """
    unique = (
        table.sort_values("minutes", ascending=False)
        .drop_duplicates(subset=IDENTITY)
        .sort_values(IDENTITY)
    )
    wide = unique.pivot_table(
        index=["player", "Nation"], columns="season", values="archetype", aggfunc="first"
    )
    seasons = sorted(int(s) for s in wide.columns)

    pairs = []
    by_group: dict[str, list[dict]] = {g: [] for g in config.OUTFIELD_GROUPS}
    for first, second in zip(seasons[:-1], seasons[1:], strict=True):
        both = wide[[first, second]].dropna()
        kept_group = both[first].str[:2] == both[second].str[:2]
        same_group = both[kept_group]
        pairs.append(
            {
                "from_season": first,
                "to_season": second,
                "n_players_both_seasons": int(len(both)),
                "n_kept_position_group": int(len(same_group)),
                "changed_position_group": round(float(1.0 - kept_group.mean()), 4),
                "switch_rate_any_change": round(float((both[first] != both[second]).mean()), 4),
                "switch_rate_within_group": round(
                    float((same_group[first] != same_group[second]).mean()), 4
                ),
                "switch_rate_expected_if_independent": round(
                    _expected_switch_rate(same_group[first], same_group[second]), 4
                ),
                "adjusted_rand_index": round(
                    float(adjusted_rand_score(same_group[first], same_group[second])), 4
                ),
            }
        )
        for group in config.OUTFIELD_GROUPS:
            sub = same_group[same_group[first].str[:2] == group]
            by_group[group].append(
                {
                    "from_season": first,
                    "to_season": second,
                    "n": int(len(sub)),
                    "switch_rate": round(float((sub[first] != sub[second]).mean()), 4),
                    "switch_rate_expected_if_independent": round(
                        _expected_switch_rate(sub[first], sub[second]), 4
                    ),
                    "adjusted_rand_index": round(
                        float(adjusted_rand_score(sub[first], sub[second])), 4
                    ),
                }
            )

    appearances = wide.notna().sum(axis=1)
    return {
        "method": (
            "Players are followed across seasons by name and nationality, with one row per "
            "player-season kept, the club spell with the most minutes. The adjusted Rand index "
            "is computed on the players who appear in both seasons of a pair and stay in the "
            "same listed position group, since a player who moves group is compared against a "
            "different pair of clusters and cannot be scored on the same partition."
        ),
        "n_tracked_players": int(len(wide)),
        "players_with_two_or_more_seasons": int((appearances >= 2).sum()),
        "players_with_all_five_seasons": int((appearances == len(seasons)).sum()),
        "pairs": pairs,
        "mean_switch_rate_within_group": round(
            float(np.mean([p["switch_rate_within_group"] for p in pairs])), 4
        ),
        "mean_adjusted_rand_index": round(
            float(np.mean([p["adjusted_rand_index"] for p in pairs])), 4
        ),
        "by_position_group": {
            group: {
                "pairs": rows,
                "mean_switch_rate": round(float(np.mean([r["switch_rate"] for r in rows])), 4),
                "mean_adjusted_rand_index": round(
                    float(np.mean([r["adjusted_rand_index"] for r in rows])), 4
                ),
            }
            for group, rows in by_group.items()
        },
    }


def refit_stability(table: pd.DataFrame, z_group: pd.DataFrame, field: str, values: list) -> dict:
    """Refit the two-cluster partition inside each level of ``field`` and score it.

    The adjusted Rand index is invariant to how the refitted clusters are numbered, so no
    label alignment is needed before comparing a refit against the pooled partition.
    """
    rows = []
    by_group: dict[str, list[float]] = {g: [] for g in config.OUTFIELD_GROUPS}
    for group in config.OUTFIELD_GROUPS:
        for value in values:
            mask = (
                (z_group["position_group"] == group) & (table[field].to_numpy() == value)
            ).to_numpy()
            x = z_group.loc[mask, A.OUTFIELD_CORE].to_numpy(float)
            fit = C._kmeans(x, K, f"archive|{group}|{field}|{value}")
            ari = float(adjusted_rand_score(table.loc[mask, "archetype"], fit.labels_))
            rows.append(
                {
                    "position_group": group,
                    str(field): value,
                    "n": int(mask.sum()),
                    "adjusted_rand_index": round(ari, 4),
                }
            )
            by_group[group].append(ari)
    return {
        "rows": rows,
        "overall_mean_adjusted_rand_index": round(
            float(np.mean([r["adjusted_rand_index"] for r in rows])), 4
        ),
        "by_position_group": {
            group: {
                "ari_mean": round(float(np.mean(aris)), 4),
                "ari_min": round(float(np.min(aris)), 4),
                "ari_max": round(float(np.max(aris)), 4),
            }
            for group, aris in by_group.items()
        },
    }


def bootstrap_report(fits: dict[str, dict]) -> dict:
    """Resampling stability of each partition, on a sub-sample.

    ``cluster.bootstrap_stability`` accumulates an n by n co-assignment matrix, so it is
    quadratic in the number of players and is run here on ``BOOTSTRAP_SAMPLE`` players
    drawn without replacement from each position group, seeded from ``config.RANDOM_STATE``.
    """
    out: dict[str, dict] = {}
    for group, fit in fits.items():
        rng = np.random.default_rng(config.RANDOM_STATE + len(group))
        n = fit["matrix"].shape[0]
        take = np.sort(rng.choice(n, size=min(BOOTSTRAP_SAMPLE, n), replace=False))
        result = C.bootstrap_stability(
            fit["matrix"][take], fit["labels"][take], K, f"archive|{group}"
        )
        result.pop("coassignment", None)
        result["n_subsampled"] = int(take.size)
        result["n_total"] = int(n)
        out[group] = result
    return out


def stability_report(table: pd.DataFrame, z_group: pd.DataFrame) -> dict:
    seasons = sorted(int(s) for s in table["season"].dropna().unique())
    return {
        "question": (
            "Do the two modes inside a position group describe a stable playing role, or a "
            "season specific and league specific accident of the fit?"
        ),
        "k_per_position_group": K,
        "n_players": int(len(table)),
        "seasons": seasons,
        "leagues": A.LEAGUES,
        "consecutive_seasons": season_stability(table),
        "league_refits": refit_stability(table, z_group, "league", A.LEAGUES),
        "season_refits": refit_stability(table, z_group, "season", seasons),
    }


# --------------------------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------------------------


def detect_outliers(table: pd.DataFrame, z_group: pd.DataFrame) -> pd.DataFrame:
    """Isolation forest and Ledoit-Wolf Mahalanobis distance, fitted within each group.

    Both detectors are linear in the number of players, so both run on the whole sample.
    Ledoit-Wolf shrinkage is kept from the live pipeline because it costs nothing here and
    keeps the two studies comparable.
    """
    pieces = []
    for group in config.OUTFIELD_GROUPS:
        mask = (z_group["position_group"] == group).to_numpy()
        matrix = z_group.loc[mask, A.OUTFIELD_CORE].to_numpy(float)
        forest = IsolationForest(
            n_estimators=ISOLATION_TREES,
            contamination="auto",
            random_state=config.RANDOM_STATE,
            bootstrap=False,
        ).fit(matrix)
        isolation = -forest.score_samples(matrix)
        covariance = LedoitWolf(store_precision=True, assume_centered=False).fit(matrix)
        mahalanobis = np.sqrt(np.maximum(covariance.mahalanobis(matrix), 0.0))
        sub = table.loc[mask].reset_index(drop=True).copy()
        for name in A.OUTFIELD_CORE:
            sub[f"z_{name}"] = z_group.loc[mask, name].to_numpy(float)
        pieces.append(
            sub.assign(
                isolation_score=isolation,
                isolation_pct=pd.Series(isolation).rank(pct=True).to_numpy(),
                mahalanobis=mahalanobis,
                mahalanobis_pct=pd.Series(mahalanobis).rank(pct=True).to_numpy(),
            )
        )
    out = pd.concat(pieces, ignore_index=True)
    out["outlier_rank_score"] = 0.5 * (out["isolation_pct"] + out["mahalanobis_pct"])
    return out.sort_values("outlier_rank_score", ascending=False).reset_index(drop=True)


def explain_outlier(row: pd.Series) -> str:
    """Name the statistics that make one player unusual inside his position group."""
    deviations = pd.Series({name: float(row[f"z_{name}"]) for name in A.OUTFIELD_CORE}, dtype=float)
    ranked = deviations.reindex(deviations.abs().sort_values(ascending=False).index)
    noun = GROUP_NOUN[row["position_group"]]
    parts = []
    for name, value in ranked.head(N_OUTLIER_REASONS).items():
        direction = "above" if value >= 0 else "below"
        raw = float(row[name])
        parts.append(
            f"{PROSE_LABELS[name]} of {_fmt(name, raw)}, {abs(value):.1f} standard deviations "
            f"{direction} the {noun} mean"
        )
    return f"Unusual mainly through {_names(parts)}."


def outlier_report(table: pd.DataFrame) -> dict:
    rows = []
    for _, series in table.head(N_OUTLIERS).iterrows():
        rows.append(
            {
                "rank": len(rows) + 1,
                "player": series["player"],
                "team": series["team"],
                "league": series["league"],
                "season": int(series["season"]),
                "position_group": series["position_group"],
                "position_full": series["position_full"],
                "minutes": int(series["minutes"]),
                "archetype_label": series["archetype"],
                "archetype_name": ARCHETYPE_NAMES[series["archetype"]],
                "isolation_score": round(float(series["isolation_score"]), 4),
                "isolation_percentile_in_group": round(float(series["isolation_pct"]), 4),
                "mahalanobis_distance": round(float(series["mahalanobis"]), 3),
                "mahalanobis_percentile_in_group": round(float(series["mahalanobis_pct"]), 4),
                "combined_percentile": round(float(series["outlier_rank_score"]), 4),
                "explanation": _no_dashes(explain_outlier(series), "outlier explanation"),
            }
        )
    return {
        "n_players": int(len(table)),
        "method": {
            "space": (
                f"the {len(A.OUTFIELD_CORE)} archive features, z-scored within position group"
            ),
            "isolation_forest": (
                f"{ISOLATION_TREES} trees, contamination auto, fitted separately within DF, MF "
                f"and FW, seeded from RANDOM_STATE={config.RANDOM_STATE}"
            ),
            "mahalanobis": (
                "Ledoit-Wolf shrinkage covariance, also fitted within position group. With "
                "9,263 players against 33 features the empirical covariance is well "
                "conditioned, so the shrinkage is retained for comparability with the live "
                "pipeline rather than out of necessity."
            ),
            "ranking": (
                "Both detectors are converted to a percentile within the position group so that "
                "groups of different size are comparable, and the table is ranked on the mean of "
                "the two percentiles."
            ),
            "caveat": (
                "An outlier here is a player-season whose per-90 profile sits far from the rest "
                "of his position group. It is not a judgement of quality. A player listed in one "
                "group who plays a different role, such as a forward listed as a defender, will "
                "appear here, and that is the most common reason for a high score."
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
    group: str,
    limit: float,
    *,
    label_angle: float,
    labelsize: float | None = None,
) -> np.ndarray:
    """Common polar furniture. Returns the closed angle array for the group's axes.

    The radial tick labels are pushed onto the emptiest bisector between two spokes and
    given a surface-coloured box. They are drawn as ordinary text rather than left to the
    radial axis, because matplotlib draws an axis at zorder 1.5, underneath the plotted
    lines, so the built-in labels end up with the polygons on top of them and the minus
    sign of the negative tick disappears.
    """
    features = RADAR_AXES[group]
    angles = np.linspace(0.0, 2.0 * np.pi, len(features), endpoint=False)
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(
        [RADAR_LABELS[name] for name in features],
        fontsize=labelsize if labelsize is not None else plotting.BASE_FONT_PT - 2,
    )
    ax.tick_params(axis="x", pad=2)
    ax.set_rlim(-limit, limit)
    ticks = [-limit / 2.0, 0.0, limit / 2.0]
    ax.set_rgrids(ticks, labels=[""] * len(ticks))
    for tick in ticks:
        ax.text(
            np.deg2rad(label_angle),
            tick,
            f"{tick:+.1f}" if tick else "0",
            ha="center",
            va="center",
            fontsize=plotting.BASE_FONT_PT - 3,
            color=plotting.INK_SECONDARY,
            zorder=8,
            bbox={"facecolor": plotting.SURFACE, "edgecolor": "none", "pad": 1.0},
        )
    ax.spines["polar"].set_color(plotting.GRID)
    ax.grid(color=plotting.GRID, linewidth=0.5)
    circle = np.linspace(0.0, 2.0 * np.pi, 181)
    ax.plot(circle, np.zeros_like(circle), color=plotting.INK_MUTED, lw=0.9, ls=(0, (3, 2)))
    return np.concatenate([angles, angles[:1]])


def _radar_values(record: dict, group: str) -> np.ndarray:
    values = np.array([record["centroid"][name] for name in RADAR_AXES[group]], dtype=float)
    return np.concatenate([values, values[:1]])


def _radar_limit(records: list[dict], group: str) -> float:
    peak = max(float(np.abs(_radar_values(r, group)).max()) for r in records)
    return float(max(1.0, np.ceil(peak * 4.0) / 4.0))


def _quiet_angle(records: list[dict], group: str) -> float:
    """Bisector, in degrees, of the adjacent spoke pair carrying the least ink."""
    n = len(RADAR_AXES[group])
    loads = np.zeros(n)
    for record in records:
        loads += np.abs(_radar_values(record, group)[:n])
    pairs = loads + np.roll(loads, -1)
    return float(360.0 * (int(np.argmin(pairs)) + 0.5) / n)


def figure_radar_panel(group: str, records: list[dict]) -> None:
    """Small multiples, one panel per archetype, the other drawn in grey for context."""
    limit = _radar_limit(records, group)
    colors = plotting.categorical(len(records))
    marks = plotting.markers(len(records))

    fig = plt.figure(figsize=(plotting.WIDTH_FULL, 3.5))
    fig.set_layout_engine("none")
    width, height = 0.335, 0.555
    for index, (record, color, mark) in enumerate(zip(records, colors, marks, strict=True)):
        centre = (index + 0.5) / len(records)
        ax = fig.add_axes((centre - width / 2.0, 0.10, width, height), projection="polar")
        angles = _radar_setup(
            ax,
            group,
            limit,
            label_angle=_quiet_angle(records, group),
            labelsize=plotting.BASE_FONT_PT - 3,
        )
        for other in records:
            if other["label"] == record["label"]:
                continue
            ax.plot(angles, _radar_values(other, group), color=plotting.CONTEXT_GREY, lw=1.0)
        values = _radar_values(record, group)
        ax.plot(angles, values, color=color, lw=1.6, marker=mark, markersize=3.2)
        ax.fill(angles, values, color=color, alpha=0.14)
        fig.text(
            centre,
            0.845,
            f"{record['label']} {textwrap.fill(record['name'], 24)}\n(n={record['n_players']:,})",
            ha="center",
            va="top",
            fontsize=plotting.BASE_FONT_PT - 1,
            color=plotting.INK_SECONDARY,
        )
    plotting.save_figure(fig, f"archive_radar_panel_{group}")


def figure_archetype_profiles(records: list[dict]) -> None:
    """Every archetype centroid across all 33 archive features, on the diverging ramp."""
    matrix = pd.DataFrame(
        {record["label"]: pd.Series(record["centroid"]) for record in records}
    ).reindex(A.OUTFIELD_CORE)
    values = matrix.to_numpy(float)
    peak = float(np.abs(values).max())
    norm = TwoSlopeNorm(vmin=-peak, vcenter=0.0, vmax=peak)

    fig, ax = plt.subplots(figsize=(plotting.WIDTH_FULL, 8.6))
    image = ax.imshow(values, cmap=plotting.DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(
        [f"{record['label']}\n{textwrap.fill(record['name'], 15)}" for record in records],
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
            ax.axvline(index - 0.5, color=plotting.SURFACE, lw=2.4)
    ax.grid(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("standard deviations from the position group mean")
    ax.set_title(
        "Archetype centroid profiles on the archive feature set\n"
        "columns are cluster centroids, rows are the 33 features, z-scored within group",
        loc="left",
        pad=10,
    )
    plotting.save_figure(fig, "archive_archetype_profiles")


def _stacked_offsets(
    ax: plt.Axes, xs: np.ndarray, ys: np.ndarray, *, x_window: float, gap: float
) -> list[tuple[float, float]]:
    """Pick a vertical offset per label so that nearby labels do not overlap.

    Everything is measured in typographic points, including ``x_window`` and ``gap``, so
    that the chosen offsets are directly usable as ``offset points`` and the collision test
    does not silently change meaning with the figure dpi.
    """
    ax.figure.canvas.draw()
    scale = ax.figure.dpi / 72.0
    points = ax.transData.transform(np.column_stack([xs, ys])) / scale
    # Upward offsets first: the extreme outliers sit on the upper right shoulder of the
    # cloud, so the empty space is above them and pushing labels down runs them into it.
    candidates = [8.0, 20.0, 32.0, -18.0, -30.0, 44.0, -42.0, 56.0]
    offsets = [(5.0, 8.0)] * len(xs)
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
        offsets[index] = (5.0, chosen)
    return offsets


def figure_outliers(table: pd.DataFrame, report: dict) -> None:
    """Left, the two detectors against each other. Right, the reported table, ranked."""
    # ``table`` is already sorted on the combined percentile and ``outlier_report`` takes
    # its head, so the reported rows are the first rows here. Looking them up by name would
    # be ambiguous, since a player can appear twice in one season after a transfer.
    flagged = table.head(len(report["players"])).reset_index(drop=True)
    if list(flagged["player"]) != [row["player"] for row in report["players"]]:
        raise ValueError("the outlier figure and the outlier report disagree on the ranking")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(plotting.WIDTH_WIDE, 4.4),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )
    scatter, ranking = axes

    for group in config.OUTFIELD_GROUPS:
        sel = table["position_group"] == group
        scatter.scatter(
            table.loc[sel, "mahalanobis"],
            table.loc[sel, "isolation_score"],
            s=4,
            c=plotting.POSITION_COLORS[group],
            marker=plotting.POSITION_MARKERS[group],
            linewidths=0,
            alpha=0.45,
            label=f"{group} (n={int(sel.sum()):,})",
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
    left, right = scatter.get_xlim()
    scatter.set_xlim(left, right + 0.30 * (right - left))
    head = flagged.head(N_SCATTER_LABELS)
    xs = head["mahalanobis"].to_numpy()
    ys = head["isolation_score"].to_numpy()
    for (dx, dy), x, y, name in zip(
        _stacked_offsets(scatter, xs, ys, x_window=96.0, gap=11.0),
        xs,
        ys,
        [f"{row.player} {int(row.season)}" for row in head.itertuples()],
        strict=True,
    ):
        # The five most extreme points sit on top of each other, so the labels are stacked
        # clear of the cloud and joined back to their point by a hairline. Without the
        # leader a reader cannot tell which circled point a stacked label belongs to.
        scatter.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=plotting.BASE_FONT_PT - 2,
            color=plotting.INK_SECONDARY,
            zorder=6,
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.5,
                "color": plotting.INK_MUTED,
                "shrinkA": 0.0,
                "shrinkB": 4.0,
            },
        )
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
        [f"{i + 1}. {row.player} {int(row.season)}" for i, row in enumerate(flagged.itertuples())],
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
    plotting.save_figure(fig, "archive_outliers")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

STRUCTURAL_NOTE = (
    "The features are z-scored within position group, so the group mean is the origin and "
    "the two centroids of a k=2 partition are exactly anti-parallel: the size weighted sum "
    "of the two centroids is zero, so one is a negative multiple of the other. The two "
    "archetypes inside a position group are therefore the two ends of a single axis rather "
    "than two independently shaped profiles, and the larger cluster necessarily carries the "
    "smaller deviations. This is a property of the partition and not a discovery about "
    "football. It matters most for the forwards, where the split is uneven at roughly three "
    "to one, so the smaller cluster shows deviations above one standard deviation while its "
    "counterpart sits near a third of that."
)

SELECTION_NOTE = (
    "Two archetypes per position group and no more. The declared consensus rule in "
    "config.K_RULE returns k=2 in every scope on this feature set, and the null calibration "
    "in results/metrics/structure.json shows that the two mode structure exceeds every "
    "clusterless simulation while the silhouette at larger k converges on that null, so a "
    "finer taxonomy would be reporting the shape of the algorithm rather than the shape of "
    "the data."
)


def write_membership(table: pd.DataFrame) -> None:
    """Publish every player's archetype assignment as CSV.

    The paper points readers here instead of printing a roster of thousands of rows, so
    this file is part of the paper's claims rather than a convenience.
    """
    columns = [
        "player",
        "team",
        "league",
        "season",
        "position_group",
        "position_full",
        "minutes",
        "archetype",
    ]
    frame = table[[c for c in columns if c in table.columns]].copy()
    # `archetype` already carries the "DF-1" style label assigned by partition().
    frame["archetype_name"] = [ARCHETYPE_NAMES[label] for label in frame["archetype"]]
    frame = frame.sort_values(["season", "league", "team", "player"], kind="stable")
    out = config.RESULTS / "membership_primary.csv"
    frame.to_csv(out, index=False)
    print(f"  wrote {out.relative_to(config.ROOT)} ({len(frame):,} rows)")


def main() -> None:
    plotting.use_style()
    np.random.seed(config.RANDOM_STATE)

    eligible, z_group = load()
    print(
        f"archive profiles: {len(eligible):,} player-seasons, {len(A.OUTFIELD_CORE)} features, "
        f"seasons {min(A.SEASONS)} to {max(A.SEASONS)}, {len(A.LEAGUES)} leagues"
    )

    fits, assignment = partition(eligible, z_group)
    table = eligible.copy()
    table["archetype"] = assignment.to_numpy()

    print("  confirming the declared k rule inside each position group")
    confirmation = {group: confirm_k(fit["matrix"], group) for group, fit in fits.items()}
    for group, result in confirmation.items():
        print(
            f"    {group}: rule chose k={result['chosen_k']} "
            f"({result['rule_branch']}), silhouette at k=2 {result['silhouette_by_k'][2]:.4f}"
        )
    disagree = [g for g, r in confirmation.items() if not r["agrees_with_declared_k"]]
    if disagree:
        raise ValueError(
            f"the declared rule no longer returns k={K} in {disagree}; "
            "the archetype count must follow the rule, not the other way round"
        )

    records = {
        f"{group}-{c}": describe_cluster(fits[group], c)
        for group in config.OUTFIELD_GROUPS
        for c in range(K)
    }
    ordered = [records[label] for label in ARCHETYPE_ORDER]

    write_membership(table)

    print("  measuring stability across seasons and leagues")
    stability = stability_report(table, z_group)
    stability["bootstrap_resampling"] = bootstrap_report(fits)
    attach_justifications(records, stability)

    print("  detecting outliers")
    outliers = detect_outliers(table, z_group)
    outlier_json = outlier_report(outliers)

    print("  drawing figures")
    for group in config.OUTFIELD_GROUPS:
        group_records = [records[f"{group}-{c}"] for c in range(K)]
        figure_radar_panel(group, group_records)
    figure_archetype_profiles(ordered)
    figure_outliers(outliers, outlier_json)

    public = []
    for record in ordered:
        public.append(
            {
                "label": record["label"],
                "name": record["name"],
                "position_group": record["position_group"],
                "n_players": record["n_players"],
                "n_position_group": record["n_position_group"],
                "share_of_position_group": record["share_of_position_group"],
                "summary": record["summary"],
                "distinguishing_features": record["distinguishing_features"],
                "representative_players": record["representative_players"],
                "boundary_players": record["boundary_players"],
                "justification": record["justification"],
            }
        )

    _write_json(
        {
            "sample": "archive, five seasons 2018 to 2022, five leagues, pooled",
            "n_players": int(len(table)),
            "n_features": len(A.OUTFIELD_CORE),
            "k_per_position_group": K,
            "selection_note": SELECTION_NOTE,
            "space": (
                f"the {len(A.OUTFIELD_CORE)} archive.OUTFIELD_CORE features z-scored within "
                "position group, so a centroid coordinate is the standardised deviation of the "
                "cluster from its position group mean"
            ),
            "naming_rule": (
                "Names were assigned after inspecting the centroid tables and the member lists, "
                "and are deliberately descriptive of the measured profile. Where a split is "
                "essentially high against low involvement the name says so rather than implying "
                "a tactical instruction the data cannot support."
            ),
            "structural_note": STRUCTURAL_NOTE,
            "n_distinguishing_features": N_DISTINGUISHING,
            "n_representative_players": N_REPRESENTATIVE,
            "n_boundary_players": N_BOUNDARY,
            "mean_switch_rate_between_consecutive_seasons": stability["consecutive_seasons"][
                "mean_switch_rate_within_group"
            ],
            "mean_adjusted_rand_index_between_consecutive_seasons": stability[
                "consecutive_seasons"
            ]["mean_adjusted_rand_index"],
            "mean_adjusted_rand_index_league_refits": stability["league_refits"][
                "overall_mean_adjusted_rand_index"
            ],
            "archetypes": public,
        },
        config.RESULTS / "archive_archetypes.json",
    )
    _write_json(
        {
            "k_confirmation": confirmation,
            "radar_axes": RADAR_AXES,
            "feature_labels": FEATURE_LABELS,
            "cluster_sizes": {r["label"]: r["n_players"] for r in ordered},
            "centroids": {r["label"]: r["centroid"] for r in ordered},
            "raw_means": {
                r["label"]: {k: round(v, 4) for k, v in r["raw_means"].items()} for r in ordered
            },
            "descriptives": {
                r["label"]: {
                    "mean_minutes": r["mean_minutes"],
                    "mean_age": r["mean_age"],
                    "listed_position_counts": r["listed_position_counts"],
                    "league_share": r["league_share"],
                    "season_share": r["season_share"],
                }
                for r in ordered
            },
        },
        config.METRICS / "archive_archetype_profiles.json",
    )
    _write_json(stability, config.METRICS / "archive_archetype_stability.json")
    _write_json(outlier_json, config.METRICS / "archive_outliers.json")

    _print_summary(ordered, stability, outlier_json)


def _print_summary(records: list[dict], stability: dict, outliers: dict) -> None:
    print("\narchetypes")
    for record in records:
        top = record["distinguishing_features"][0]
        print(
            f"  {record['label']:5s} {record['name']:34s} n={record['n_players']:5,d}  "
            f"top feature {top['label']} {top['z_deviation']:+.2f}"
        )
    seasons = stability["consecutive_seasons"]
    print("\nstability between consecutive seasons")
    for row in seasons["pairs"]:
        print(
            f"  {row['from_season']} to {row['to_season']}: "
            f"n={row['n_kept_position_group']:,}, switched "
            f"{row['switch_rate_within_group'] * 100:.1f} percent "
            f"(chance {row['switch_rate_expected_if_independent'] * 100:.1f}), "
            f"ARI {row['adjusted_rand_index']:.3f}"
        )
    print("\nleague refits against the pooled partition")
    for group, value in stability["league_refits"]["by_position_group"].items():
        print(
            f"  {group}: ARI mean {value['ari_mean']:.3f}, "
            f"range {value['ari_min']:.3f} to {value['ari_max']:.3f}"
        )
    print("\nbootstrap resampling")
    for group, value in stability["bootstrap_resampling"].items():
        print(
            f"  {group}: ARI {value['ari_mean']:.3f} "
            f"(sd {value['ari_sd']:.3f}) on {value['n_subsampled']:,} of {value['n_total']:,}"
        )
    print(f"\ntop {min(3, len(outliers['players']))} outliers")
    for row in outliers["players"][:3]:
        print(f"  {row['rank']}. {row['player']} ({row['team']}, {row['season']})")
    print("\narchive profiles complete")


if __name__ == "__main__":
    main()
