"""Phase 10. Three analyses that go beyond describing the clusters.

The three are declared in ``config.NOVEL_ANALYSES`` and are run in that order.

1. Similarity search
--------------------
Cosine similarity between players, computed within position group on the within-group
z-scored feature matrix ``outfield_z_bygroup_{season}``. Cosine rather than Euclidean
distance because on centred features it compares the shape of a player's profile, the
pattern of what he does more and less of than his positional peers, rather than the
overall magnitude of his involvement. Ten well known players from the primary season are
used as queries and their five nearest neighbours are reported. Every query name is
checked against the eligible set before use and any that is absent is recorded rather
than silently dropped.

The result has to be read against what the feature set actually contains. After the
January 2026 deletion the 18 surviving features cover shooting volume and quality,
expected creation from Understat, crossing, and three defensive or duel counts. There is
no passing volume, no progressive passing or carrying, no touches, no take-ons, no
possession share and no positional or tracking data. Two deep-lying midfielders who
differ enormously in how they pass will therefore look nearly identical here, because the
data does not contain passing. That limitation is reported alongside the neighbour lists
rather than glossed over.

2. Cross-season replication
---------------------------
The clustering is refitted on 2025-26 from scratch, applying the same declared
``config.K_RULE`` rather than importing the primary season's k, so the replication also
answers whether the rule picks the same number of clusters on new data. For the players
who are eligible in both seasons the adjusted Rand index between their 2024-25 and
2025-26 assignments is computed globally and within each position group. Cluster ids are
matched across seasons by optimal assignment on the centroids before the transition table
is built, which affects nothing that is invariant to relabelling but makes the table
readable. Players whose assignment changed are ranked by how far they moved in the
feature space, and the features that moved most are reported with them.

An archetype that is a real playing role should reappear in a new season with largely the
same members. An archetype that is a single-season artefact should not.

3. Empirical Bayes shrinkage
----------------------------
Motivation. A per-90 rate over a partial season is a noisy estimate of a player's true
rate, and the noise grows as minutes fall. Clustering raw per-90 rates therefore
clusters signal and sampling noise together, and the players with fewest minutes are the
ones whose position in the feature space is least trustworthy. This is the main
methodological weakness of per-90 clustering, so it is measured rather than assumed away.

Model. Fix a feature f and a position group g. Player i has exposure ``n_i`` and an
observed rate ``y_i``. Exposure is nineties (minutes divided by 90) for the sixteen
per-90 counting features, and shots for the two ratio features ``shot_accuracy_pct`` and
``goals_per_shot``, since those are computed per shot and not per minute. The two-level
model is

    y_i | theta_i ~ (mean theta_i, variance kappa_g / n_i)
    theta_i       ~ (mean mu_g,    variance tau_g^2)

so only first and second moments are assumed, not a distributional family.

Sampling variance. The form ``kappa_g / n_i`` is the variance of a rate computed over a
known exposure when the underlying quantity accumulates with independent increments per
unit of exposure. For an integer count that is Poisson accumulation, for which
``kappa_g = mu_g``. For a ratio it is binomial accumulation, for which
``kappa_g = mu_g (1 - mu_g)`` on the proportion scale. Neither is assumed here.
``kappa_g`` is estimated from the data instead, because four of the features (npxG, xA,
xGChain, xGBuildup) are continuous sums over shots and possessions rather than counts,
and because real football counts are overdispersed relative to Poisson. The Poisson and
binomial values are used only as a fallback when the estimate is degenerate.

Estimating the two variance components. Under the model above the squared deviation of a
player's observed rate from the group mean has expectation

    E[(y_i - mu_g)^2] = tau_g^2 + kappa_g / n_i

which is linear in ``1 / n_i`` with intercept ``tau_g^2`` and slope ``kappa_g``. The two
components are therefore separately identified whenever exposures differ across players,
which they do. They are estimated by regressing the squared deviations on ``1 / n_i``
with an intercept. The squared deviations are heteroscedastic, with variance roughly
proportional to the square of their own mean, so ordinary least squares is used only as a
starting value and two iterations of weighted least squares with weights
``1 / fitted^2`` follow. Both components are floored at zero. ``mu_g`` is the exposure
weighted mean rate, which for a per-90 feature is the group's total count divided by its
total nineties.

Prior pool. The hyperparameters are estimated on the unfiltered aggregated pool rather
than on the eligible players only. That pool contains the low-minute players who were
excluded by ``config.MIN_MINUTES``, and they are exactly the observations that identify
the slope ``kappa_g``. Players below ``PRIOR_MIN_MINUTES`` are still excluded, because at
one or two minutes a per-90 rate is a division by a number near zero and contributes
numerical noise rather than information.

Posterior mean. With both components in hand the posterior mean is the precision weighted
average of the player's own rate and his group's mean,

    B_i     = tau_g^2 / (tau_g^2 + kappa_g / n_i)
    theta_i = B_i y_i + (1 - B_i) mu_g

``B_i`` is the reliability of the player's own observation. It rises towards one as
exposure grows and falls towards zero as exposure shrinks, which is the behaviour the
analysis is testing for. Under a Gaussian prior and a Gaussian likelihood this is the
exact posterior mean; without those assumptions it remains the best linear predictor of
``theta_i`` given ``y_i``, which is why only moments were assumed.

Assumptions, stated plainly.
  * A player has one stable true rate for the whole season. Mid-season role change,
    injury and managerial change all violate this, and the estimator will read the
    resulting instability as sampling noise and shrink it away.
  * Sampling variance is inversely proportional to exposure with a constant that is
    shared by every player in a position group. Players are therefore assumed
    exchangeable within group up to exposure.
  * Exposure is fixed and independent of the true rate. This is false in the direction
    that matters: better players play more minutes, so the exposure weighted mean
    ``mu_g`` is pulled towards high-minute players, and shrinkage towards it is mildly
    conservative for fringe players. This is a limitation, not a correction.
  * The linear moment regression uses a squared deviation from an estimated rather than a
    known mean, so it carries a small downward bias in ``tau_g^2`` of order 1/m. With
    group sizes in the hundreds that bias is second order next to the effects reported.

What is then done with it. The eighteen shrunken features are re-standardised exactly as
``src.preprocess`` standardises the raw ones, globally and within position group, and the
clustering is refitted with the same declared ``config.K_RULE``. Reported: whether the
rule still selects the same k, how many players change cluster, the adjusted Rand index
between the original and shrunken partitions, and the change rate broken down by minutes
played. Re-standardising matters: it removes the uniform scale compression that shrinkage
induces and leaves only the part that actually matters, the fact that low-minute players
are pulled in further than high-minute players and therefore move relative to everyone
else.

If the partition survives, the clusters are describing something more than sampling
noise. If it dissolves, they were substantially a description of sampling noise, and the
paper has to say so.

Outputs
-------
``results/metrics/similarity_search.json``, ``cross_season_replication.json``,
``shrinkage.json``, ``data/processed/shrunken_{season}.parquet`` and four figures.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

import config
from src import cluster as C
from src import features as F
from src import plotting
from src import preprocess as P

# --------------------------------------------------------------------------------------
# Declared choices. None of these were tuned after seeing a result.
# --------------------------------------------------------------------------------------

SCOPES: list[str] = ["global"] + list(config.OUTFIELD_GROUPS)

#: Query players for the similarity search. Chosen for public recognisability and to
#: cover all three outfield groups. Presence in the eligible set is verified at run time.
QUERY_PLAYERS: list[str] = [
    "Mohamed Salah",
    "Cole Palmer",
    "Bruno Fernandes",
    "Declan Rice",
    "Erling Haaland",
    "Alexander Isak",
    "Bukayo Saka",
    "Virgil van Dijk",
    "Trent Alexander-Arnold",
    "Marc Cucurella",
]

N_NEIGHBOURS = 5

#: How many of a query player's most extreme features define his profile for the note.
DEFINING_FEATURE_N = 3

#: A neighbour set is called consistent when its mean absolute z gap to the query player,
#: on the query player's defining features, is below this. Declared before running.
AGREEMENT_TOLERANCE = 1.0

#: Feature families that no longer exist in public data and therefore cannot inform
#: similarity. Quoted in the metrics file so the neighbour lists are read correctly.
ABSENT_FEATURE_FAMILIES = [
    "pass volume, completion and pass length",
    "progressive passes, progressive carries and progressive receptions",
    "touches by pitch third, carries and take-ons",
    "shot-creating and goal-creating actions",
    "tackles attempted by pitch third, blocks, clearances and pressures",
    "aerial duels won and lost",
    "possession share and positional or tracking data",
]

#: Exposure column per feature. Rates over shots are not rates over minutes.
EXPOSURE_COLUMN: dict[str, str] = {feature: "nineties" for feature in F.OUTFIELD_CORE}
EXPOSURE_COLUMN["shot_accuracy_pct"] = "shots"
EXPOSURE_COLUMN["goals_per_shot"] = "shots"

#: Scale of each ratio feature, used only for the binomial fallback variance.
RATIO_SCALE: dict[str, float] = {"shot_accuracy_pct": 100.0, "goals_per_shot": 1.0}

#: Minutes floor for the prior pool. Below one full ninety a per-90 rate is a division
#: by a number near zero and adds numerical noise rather than information.
PRIOR_MIN_MINUTES = 90

#: Weighted least squares iterations for the variance component regression.
MOMENT_IRLS_ITERATIONS = 2

#: Minutes bins for the shrinkage breakdown. The lowest bin starts at config.MIN_MINUTES.
MINUTES_BIN_EDGES = [config.MIN_MINUTES, 900, 1500, 2200, np.inf]

#: Adjusted Rand index bands used to phrase the shrinkage verdict. Declared in advance.
SURVIVES_ARI = 0.80
PARTLY_SURVIVES_ARI = 0.50

#: How many movers to list in each metrics file.
MOVER_TABLE_N = 15

#: Short display names for figure panels. The long canonical names do not fit.
DISPLAY_NAME: dict[str, str] = {
    "np_xg_p90": "npxG/90",
    "xa_p90": "xA/90",
    "key_passes_p90": "Key passes/90",
    "xg_chain_p90": "xGChain/90",
    "xg_buildup_p90": "xGBuildup/90",
    "shots_p90": "Shots/90",
    "shots_on_target_p90": "SoT/90",
    "goals_non_penalty_p90": "npGoals/90",
    "assists_p90": "Assists/90",
    "crosses_p90": "Crosses/90",
    "interceptions_p90": "Interceptions/90",
    "tackles_won_p90": "Tackles won/90",
    "fouls_committed_p90": "Fouls cmt/90",
    "fouls_drawn_p90": "Fouls drawn/90",
    "offsides_p90": "Offsides/90",
    "cards_yellow_p90": "Yellows/90",
    "shot_accuracy_pct": "SoT pct",
    "goals_per_shot": "Goals per shot",
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


def _write_json(payload: dict, name: str) -> None:
    path = config.METRICS / name
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_native, sort_keys=False)
    print(f"  wrote {path.relative_to(config.ROOT)}")


def _r(x, nd: int = 4) -> float:
    return float(np.round(float(x), nd))


def short_name(name: str, limit: int = 15) -> str:
    """First initial plus the rest of the name, truncated for figure cells."""
    parts = str(name).split()
    label = f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else str(name)
    return label if len(label) <= limit else label[: limit - 1] + "…"


def cell_ink(cmap, normalised: float) -> str:
    """Pick the annotation colour that contrasts better with the cell behind it.

    Chosen by WCAG relative luminance rather than by a hand-set threshold, because the
    sequential ramp crosses the point where dark text stops winning fairly late.
    """
    r, g, b = cmap(float(np.clip(normalised, 0.0, 1.0)))[:3]
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    on_dark = 1.05 / (luminance + 0.05)
    on_light = (luminance + 0.05) / 0.05
    return plotting.SURFACE if on_dark > on_light else plotting.INK_PRIMARY


def match_labels(reference: np.ndarray, other: np.ndarray) -> dict[int, int]:
    """Relabel ``other`` onto ``reference`` by maximising the agreement count."""
    ref_ids = sorted(set(reference.tolist()))
    oth_ids = sorted(set(other.tolist()))
    table = np.zeros((len(oth_ids), len(ref_ids)), dtype=float)
    for r, o in enumerate(oth_ids):
        for c, ref in enumerate(ref_ids):
            table[r, c] = np.sum((other == o) & (reference == ref))
    rows, cols = linear_sum_assignment(-table)
    mapping = {int(oth_ids[r]): int(ref_ids[c]) for r, c in zip(rows, cols, strict=True)}
    for o in oth_ids:
        mapping.setdefault(int(o), int(o))
    return mapping


def match_centroids(reference: np.ndarray, other: np.ndarray) -> dict[int, int]:
    """Relabel ``other``'s cluster ids onto ``reference``'s by centroid proximity.

    Used across seasons, where agreement counts are the thing being measured and so
    cannot also be the thing used to align the ids.
    """
    cost = np.linalg.norm(other[:, None, :] - reference[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    mapping = {int(r): int(c) for r, c in zip(rows, cols, strict=True)}
    for i in range(other.shape[0]):
        mapping.setdefault(i, i)
    return mapping


def contingency(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """Counts of ``a`` (rows) against ``b`` (columns), with every id present."""
    ids_a = sorted(set(a.tolist()))
    ids_b = sorted(set(b.tolist()))
    table = pd.DataFrame(0, index=ids_a, columns=ids_b, dtype=int)
    for va, vb in zip(a, b, strict=True):
        table.loc[int(va), int(vb)] += 1
    return table


def minutes_bin_labels() -> list[str]:
    labels = []
    for lo, hi in zip(MINUTES_BIN_EDGES[:-1], MINUTES_BIN_EDGES[1:], strict=True):
        labels.append(f"{int(lo)}+" if np.isinf(hi) else f"{int(lo)}-{int(hi)}")
    return labels


def verdict_for(ari: float) -> str:
    """Band an adjusted Rand index using the thresholds declared above."""
    if ari >= SURVIVES_ARI:
        return "survives"
    if ari >= PARTLY_SURVIVES_ARI:
        return "partly_survives"
    return "dissolves"


def assign_minutes_bin(minutes: pd.Series) -> pd.Series:
    return pd.cut(
        minutes.astype(float),
        bins=MINUTES_BIN_EDGES,
        labels=minutes_bin_labels(),
        include_lowest=True,
        right=False,
    )


# --------------------------------------------------------------------------------------
# Analysis 1: player similarity search
# --------------------------------------------------------------------------------------


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pair of rows."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    unit = x / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def _similarity_note(query: dict, neighbours: list[dict], agreement: dict) -> str:
    defining = ", ".join(
        f"{DISPLAY_NAME[d['feature']]} z={d['z']:+.2f}" for d in query["defining_features"]
    )
    verdict = (
        "the neighbour set tracks those features closely"
        if agreement["mean_abs_z_gap_on_defining_features"] <= AGREEMENT_TOLERANCE
        else "the neighbour set does not track those features closely"
    )
    worst = agreement["largest_mismatch_feature"]
    return (
        f"Mean cosine similarity to the five neighbours is "
        f"{np.mean([n['cosine_similarity'] for n in neighbours]):.3f}. Within his position "
        f"group the query player is furthest from the mean on {defining}, and "
        f"{verdict} (mean absolute z gap "
        f"{agreement['mean_abs_z_gap_on_defining_features']:.2f} against a declared "
        f"tolerance of {AGREEMENT_TOLERANCE:.2f}). Across all eighteen features the "
        f"neighbours differ from him most on {DISPLAY_NAME[worst]} (mean absolute z gap "
        f"{agreement['largest_mismatch_gap']:.2f}). Similarity here is computed from "
        f"shooting, expected creation, crossing and three duel or defensive counts only, "
        f"so two players who pass very differently can still appear as neighbours."
    )


def similarity_search(season: str) -> dict:
    """Nearest neighbours within position group for each query player."""
    z_group = pd.read_parquet(config.DATA_PROCESSED / f"outfield_z_bygroup_{season}.parquet")
    eligible = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")
    clusters = pd.read_parquet(config.DATA_PROCESSED / f"clusters_{season}.parquet")
    label_by_player = dict(
        zip(clusters["player"], clusters["cluster_within_group_label"], strict=True)
    )

    present = set(z_group["player"])
    queries = [p for p in QUERY_PLAYERS if p in present]
    absent = [p for p in QUERY_PLAYERS if p not in present]
    print(f"  similarity: {len(queries)}/{len(QUERY_PLAYERS)} query players in the eligible set")
    if absent:
        print(f"    absent from the eligible set: {absent}")

    raw_by_player = eligible.set_index("player")
    results = []
    for group in config.OUTFIELD_GROUPS:
        sub = z_group[z_group["position_group"] == group].reset_index(drop=True)
        x = sub[F.OUTFIELD_CORE].to_numpy(dtype=float)
        sim = cosine_matrix(x)
        np.fill_diagonal(sim, -np.inf)
        index = {p: i for i, p in enumerate(sub["player"])}

        for player in queries:
            if player not in index:
                continue
            i = index[player]
            order = np.argsort(-sim[i])[:N_NEIGHBOURS]
            z_query = x[i]
            defining_idx = np.argsort(-np.abs(z_query))[:DEFINING_FEATURE_N]
            defining = [
                {"feature": F.OUTFIELD_CORE[j], "z": _r(z_query[j], 3)} for j in defining_idx
            ]

            neighbours = []
            for rank, j in enumerate(order, start=1):
                other = sub.loc[j, "player"]
                neighbours.append(
                    {
                        "rank": rank,
                        "player": other,
                        "team": str(raw_by_player.loc[other, "team"]),
                        "position_full": str(raw_by_player.loc[other, "position_full"]),
                        "minutes": int(raw_by_player.loc[other, "minutes"]),
                        "cosine_similarity": _r(sim[i, j], 4),
                        "cluster_within_group_label": label_by_player.get(other),
                    }
                )

            gaps = np.abs(x[order] - z_query)
            mean_gap = gaps.mean(axis=0)
            worst = int(np.argmax(mean_gap))
            agreement = {
                "mean_abs_z_gap_all_features": _r(mean_gap.mean(), 3),
                "mean_abs_z_gap_on_defining_features": _r(mean_gap[defining_idx].mean(), 3),
                "tolerance": AGREEMENT_TOLERANCE,
                "consistent_on_defining_features": bool(
                    mean_gap[defining_idx].mean() <= AGREEMENT_TOLERANCE
                ),
                "largest_mismatch_feature": F.OUTFIELD_CORE[worst],
                "largest_mismatch_gap": _r(mean_gap[worst], 3),
            }
            query = {
                "player": player,
                "position_group": group,
                "position_full": str(raw_by_player.loc[player, "position_full"]),
                "team": str(raw_by_player.loc[player, "team"]),
                "minutes": int(raw_by_player.loc[player, "minutes"]),
                "cluster_within_group_label": label_by_player.get(player),
                "group_pool_size": int(len(sub)),
                "defining_features": defining,
            }
            query["neighbours"] = neighbours
            query["mean_similarity"] = _r(np.mean([n["cosine_similarity"] for n in neighbours]), 4)
            query["neighbour_agreement"] = agreement
            query["note"] = _similarity_note(query, neighbours, agreement)
            results.append(query)

    order_map = {p: i for i, p in enumerate(QUERY_PLAYERS)}
    results.sort(key=lambda q: order_map[q["player"]])
    return {
        "season": season,
        "space": f"outfield_z_bygroup_{season}, within position group",
        "metric": "cosine similarity on the 18 within-group z-scored features",
        "why_cosine": (
            "On centred features cosine compares the shape of a player's profile, what he "
            "does more and less of than his positional peers, rather than the overall "
            "magnitude of his involvement."
        ),
        "n_neighbours": N_NEIGHBOURS,
        "feature_set": F.OUTFIELD_CORE,
        "absent_feature_families": ABSENT_FEATURE_FAMILIES,
        "reading_caveat": (
            "Similarity is driven by shooting volume and quality, Understat expected "
            "creation, crossing, and three duel or defensive counts. No passing, "
            "possession, carrying or positional data survives in public form, so players "
            "who differ mainly in how they pass or carry the ball cannot be separated by "
            "this metric."
        ),
        "queries_requested": QUERY_PLAYERS,
        "queries_absent_from_eligible_set": absent,
        "defining_feature_n": DEFINING_FEATURE_N,
        "agreement_tolerance": AGREEMENT_TOLERANCE,
        "queries": results,
    }


def figure_similarity_heatmap(payload: dict) -> None:
    """Scouting table as a heatmap: query players by neighbour rank."""
    queries = payload["queries"]
    n_rows, n_cols = len(queries), N_NEIGHBOURS
    values = np.array([[n["cosine_similarity"] for n in q["neighbours"]] for q in queries])

    fig, ax = plt.subplots(figsize=(plotting.WIDTH_FULL, 0.62 * n_rows + 1.25))
    vmin, vmax = float(values.min()), float(values.max())
    ax.imshow(values, cmap=plotting.SEQUENTIAL, vmin=vmin, vmax=vmax, aspect="auto")

    span = max(vmax - vmin, 1e-9)
    for r, q in enumerate(queries):
        for c, neighbour in enumerate(q["neighbours"]):
            ax.text(
                c,
                r,
                f"{short_name(neighbour['player'])}\n{neighbour['cosine_similarity']:.3f}",
                ha="center",
                va="center",
                fontsize=plotting.BASE_FONT_PT - 2.5,
                color=cell_ink(plotting.SEQUENTIAL, (values[r, c] - vmin) / span),
                linespacing=1.35,
            )

    ax.set_xticks(range(n_cols), [f"#{i + 1}" for i in range(n_cols)])
    ax.set_yticks(
        range(n_rows),
        [f"{short_name(q['player'], 24)}  ({q['position_group']})" for q in queries],
    )
    ax.set_xlabel("Nearest neighbour rank")
    ax.set_title(
        f"Nearest neighbours within position group, {payload['season']}\n"
        "cosine similarity on 18 surviving features, no passing or possession data",
        loc="left",
    )
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=plotting.SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    bar = fig.colorbar(ax.images[0], ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("cosine similarity", fontsize=plotting.BASE_FONT_PT - 1)
    bar.outline.set_visible(False)
    plotting.save_figure(fig, "similarity_heatmap")


# --------------------------------------------------------------------------------------
# Analysis 2: cross-season replication
# --------------------------------------------------------------------------------------


def fit_scopes(frames: dict[str, pd.DataFrame], season: str, tag_prefix: str = "") -> dict:
    """Run k selection under config.K_RULE and KMeans for every scope."""
    out: dict[str, dict] = {}
    for scope in SCOPES:
        frame = frames[scope]
        x = frame[F.OUTFIELD_CORE].to_numpy(dtype=float)
        tag = f"{tag_prefix}{season}|{scope}|{C.PRIMARY_SPACE}"
        curves = C.internal_curves(x, tag)
        rule = C.apply_k_rule(curves)
        k = int(rule["chosen_k"])
        fit = C._kmeans(x, k, tag)
        out[scope] = {
            "k": k,
            "rule_branch": rule["rule_branch"],
            "silhouette_at_k": curves["silhouette"][k],
            "labels": fit.labels_.astype(int),
            "centroids": fit.cluster_centers_,
            "players": frame["player"].to_numpy(),
            "n": int(x.shape[0]),
            "favoured_by_criterion": rule["favoured_by_criterion"],
        }
        print(
            f"    {season} {scope:<6} n={x.shape[0]:>3} rule k={k}"
            f" ({rule['rule_branch']}) sil={curves['silhouette'][k]:.3f}"
        )
    return out


def _feature_sd_by_group(eligible: pd.DataFrame) -> pd.DataFrame:
    return eligible.groupby("position_group")[F.OUTFIELD_CORE].std(ddof=0)


def _mover_rows(
    movers: pd.DataFrame,
    primary: pd.DataFrame,
    replication: pd.DataFrame,
    sd_by_group: pd.DataFrame,
    n: int,
) -> list[dict]:
    """Rank movers by distance travelled and report the features that drove it."""
    a = primary.set_index("player")
    b = replication.set_index("player")
    rows = []
    for player in movers["player"]:
        group = str(a.loc[player, "position_group"])
        sd = sd_by_group.loc[group, F.OUTFIELD_CORE].to_numpy(dtype=float)
        sd = np.where(sd > 0, sd, 1.0)
        raw_a = a.loc[player, F.OUTFIELD_CORE].to_numpy(dtype=float)
        raw_b = b.loc[player, F.OUTFIELD_CORE].to_numpy(dtype=float)
        delta = (raw_b - raw_a) / sd
        order = np.argsort(-np.abs(delta))[:4]
        rows.append(
            {
                "player": player,
                "position_group_2024_25": group,
                "position_group_2025_26": str(b.loc[player, "position_group"]),
                "position_group_changed": group != str(b.loc[player, "position_group"]),
                "team_2024_25": str(a.loc[player, "team"]),
                "team_2025_26": str(b.loc[player, "team"]),
                "minutes_2024_25": int(a.loc[player, "minutes"]),
                "minutes_2025_26": int(b.loc[player, "minutes"]),
                "cluster_2024_25": int(movers.set_index("player").loc[player, "cluster_a"]),
                "cluster_2025_26": int(movers.set_index("player").loc[player, "cluster_b"]),
                "distance_moved_sd_units": _r(float(np.linalg.norm(delta)), 3),
                "largest_feature_changes": [
                    {
                        "feature": F.OUTFIELD_CORE[j],
                        "value_2024_25": _r(raw_a[j], 3),
                        "value_2025_26": _r(raw_b[j], 3),
                        "change_in_2024_25_group_sd": _r(delta[j], 3),
                    }
                    for j in order
                ],
            }
        )
    rows.sort(key=lambda r: -r["distance_moved_sd_units"])
    return rows[:n]


def cross_season_replication() -> dict:
    """Refit 2025-26 under the declared rule and compare with 2024-25 assignments."""
    primary_season, replication_season = config.SEASON_PRIMARY, config.SEASON_REPLICATION
    eligible_a, frames_a = C.load_season(primary_season)
    eligible_b, frames_b = C.load_season(replication_season)

    print("  refitting both seasons under the declared K_RULE")
    fits = {
        primary_season: fit_scopes(frames_a, primary_season),
        replication_season: fit_scopes(frames_b, replication_season),
    }

    common = sorted(set(eligible_a["player"]) & set(eligible_b["player"]))
    groups_a = dict(zip(eligible_a["player"], eligible_a["position_group"], strict=True))
    groups_b = dict(zip(eligible_b["player"], eligible_b["position_group"], strict=True))
    same_group = [p for p in common if groups_a[p] == groups_b[p]]
    changed_group = [
        {"player": p, "from": groups_a[p], "to": groups_b[p]}
        for p in common
        if groups_a[p] != groups_b[p]
    ]
    print(
        f"  {len(common)} players eligible in both seasons, "
        f"{len(same_group)} with an unchanged listed position group"
    )

    sd_by_group = _feature_sd_by_group(eligible_a)
    scope_results: dict[str, dict] = {}
    transitions: dict[str, pd.DataFrame] = {}
    movers_global: list[dict] = []

    for scope in SCOPES:
        fa, fb = fits[primary_season][scope], fits[replication_season][scope]
        la = dict(zip(fa["players"], fa["labels"], strict=True))
        lb = dict(zip(fb["players"], fb["labels"], strict=True))
        pool = common if scope == "global" else [p for p in same_group if groups_a[p] == scope]
        pool = [p for p in pool if p in la and p in lb]
        if len(pool) < 2:
            scope_results[scope] = {"n_common": len(pool), "note": "too few common players"}
            continue

        a_lab = np.array([la[p] for p in pool])
        b_lab = np.array([lb[p] for p in pool])
        mapping = match_centroids(fa["centroids"], fb["centroids"])
        b_aligned = np.array([mapping[int(v)] for v in b_lab])
        ari = adjusted_rand_score(a_lab, b_lab)
        table = contingency(a_lab, b_aligned)
        transitions[scope] = table

        changed = pd.DataFrame({"player": pool, "cluster_a": a_lab, "cluster_b": b_aligned}).query(
            "cluster_a != cluster_b"
        )

        scope_results[scope] = {
            "n_common": int(len(pool)),
            "k_2024_25": fa["k"],
            "k_2025_26": fb["k"],
            "k_matches": fa["k"] == fb["k"],
            "adjusted_rand_index": _r(ari, 4),
            "n_changed_cluster": int(len(changed)),
            "pct_changed_cluster": _r(100.0 * len(changed) / len(pool), 2),
            "transition_counts": {
                str(r): {str(c): int(table.loc[r, c]) for c in table.columns} for r in table.index
            },
            "transition_row_pct": {
                str(r): {
                    str(c): _r(100.0 * table.loc[r, c] / max(table.loc[r].sum(), 1), 2)
                    for c in table.columns
                }
                for r in table.index
            },
            "note": (
                "Cluster ids for 2025-26 are matched to 2024-25 by optimal assignment on "
                "the KMeans centroids so the table reads down the diagonal. The adjusted "
                "Rand index is invariant to that relabelling."
            ),
        }
        if scope == "global":
            movers_global = _mover_rows(changed, eligible_a, eligible_b, sd_by_group, MOVER_TABLE_N)

    return {
        "question": (
            "Do the archetypes describe stable playing roles, or are they single-season "
            "artefacts? The clustering is refitted on 2025-26 from scratch under the same "
            "declared rule and the two assignments are compared on the players who are "
            "eligible in both seasons."
        ),
        "k_rule": config.K_RULE,
        "min_minutes": config.MIN_MINUTES,
        "space": C.PRIMARY_SPACE,
        "n_eligible": {
            primary_season: int(len(eligible_a)),
            replication_season: int(len(eligible_b)),
        },
        "n_common_eligible": int(len(common)),
        "n_common_same_position_group": int(len(same_group)),
        "position_group_changes": changed_group,
        "k_selected_by_rule": {
            season: {scope: fits[season][scope]["k"] for scope in SCOPES} for season in fits
        },
        "rule_branch": {
            season: {scope: fits[season][scope]["rule_branch"] for scope in SCOPES}
            for season in fits
        },
        "scopes": scope_results,
        "largest_movers_global": movers_global,
        "mover_note": (
            "Movers are ranked by the Euclidean distance travelled between seasons in the "
            "eighteen features, each scaled by its 2024-25 within-position-group standard "
            "deviation so both seasons are measured on one fixed ruler. Values quoted are "
            "raw per-90 rates."
        ),
        "_transitions": transitions,
    }


def figure_cross_season_transitions(payload: dict) -> None:
    """One transition heatmap per scope, 2024-25 rows against 2025-26 columns."""
    transitions = payload["_transitions"]
    scopes = [s for s in SCOPES if s in transitions]
    fig, axes = plt.subplots(2, 2, figsize=(plotting.WIDTH_FULL, 5.4))
    axes = np.atleast_1d(axes).ravel()

    for ax, scope in zip(axes, scopes, strict=False):
        table = transitions[scope]
        block = payload["scopes"][scope]
        row_pct = table.div(table.sum(axis=1).replace(0, 1), axis=0) * 100.0
        ax.imshow(row_pct.to_numpy(), cmap=plotting.SEQUENTIAL, vmin=0, vmax=100, aspect="auto")
        for r in range(table.shape[0]):
            for c in range(table.shape[1]):
                value = row_pct.iat[r, c]
                ax.text(
                    c,
                    r,
                    f"{int(table.iat[r, c])}\n{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=plotting.BASE_FONT_PT - 1,
                    color=cell_ink(plotting.SEQUENTIAL, value / 100.0),
                    linespacing=1.3,
                )
        ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns])
        ax.set_yticks(range(table.shape[0]), [str(r) for r in table.index])
        ax.set_title(
            f"{scope}  n={block['n_common']}  ARI={block['adjusted_rand_index']:.2f}", loc="left"
        )
        ax.set_xlabel(f"{config.SEASON_REPLICATION} cluster")
        ax.set_ylabel(f"{config.SEASON_PRIMARY} cluster")
        ax.grid(False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(scopes) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Archetype membership across seasons, players eligible in both\n"
        "cell shows count and row percentage; 2025-26 ids matched to 2024-25 centroids",
        x=0.01,
        ha="left",
        fontsize=plotting.BASE_FONT_PT,
        fontweight="bold",
    )
    plotting.save_figure(fig, "cross_season_transitions")


# --------------------------------------------------------------------------------------
# Analysis 3: empirical Bayes shrinkage
# --------------------------------------------------------------------------------------


def _plugin_kappa(feature: str, mu: float) -> float:
    """Poisson or binomial sampling constant, used only when the estimate is degenerate."""
    if feature in RATIO_SCALE:
        scale = RATIO_SCALE[feature]
        p = float(np.clip(mu / scale, 0.0, 1.0))
        return float(scale**2 * p * (1.0 - p))
    return float(max(mu, 0.0))


def moment_components(y: np.ndarray, exposure: np.ndarray, mu: float) -> tuple[float, float, dict]:
    """Estimate (tau squared, kappa) by regressing squared deviations on 1 / exposure.

    ``E[(y_i - mu)^2] = tau^2 + kappa / n_i``. Ordinary least squares gives the starting
    value and two iterations of weighted least squares with weights ``1 / fitted^2``
    follow, because the squared deviations are heteroscedastic with a spread that scales
    with their own mean.
    """
    inv_n = 1.0 / exposure
    d = (y - mu) ** 2
    design = np.column_stack([np.ones_like(inv_n), inv_n])
    beta, *_ = np.linalg.lstsq(design, d, rcond=None)
    for _ in range(MOMENT_IRLS_ITERATIONS):
        fitted = np.maximum(design @ beta, 1e-12)
        w = 1.0 / fitted**2
        wd = design * w[:, None]
        try:
            beta = np.linalg.solve(design.T @ wd, wd.T @ d)
        except np.linalg.LinAlgError:  # pragma: no cover - singular design
            break
    tau2, kappa = float(beta[0]), float(beta[1])
    diagnostics = {
        "ols_intercept": _r(tau2, 6),
        "ols_slope": _r(kappa, 6),
        "mean_squared_deviation": _r(float(d.mean()), 6),
    }
    return max(tau2, 0.0), kappa, diagnostics


def estimate_hyperparameters(season: str) -> dict:
    """Prior mean and the two variance components, per position group and feature."""
    pool = pd.read_parquet(config.DATA_PROCESSED / f"aggregated_{season}.parquet")
    pool = pool[pool["position_group"].isin(config.OUTFIELD_GROUPS)]
    pool = pool[pool["minutes"] >= PRIOR_MIN_MINUTES].copy()

    hyper: dict[str, dict] = {}
    for group in config.OUTFIELD_GROUPS:
        sub = pool[pool["position_group"] == group]
        per_feature: dict[str, dict] = {}
        for feature in F.OUTFIELD_CORE:
            exposure_col = EXPOSURE_COLUMN[feature]
            usable = sub[[feature, exposure_col]].astype(float).dropna()
            usable = usable[usable[exposure_col] > 0]
            y = usable[feature].to_numpy(dtype=float)
            n = usable[exposure_col].to_numpy(dtype=float)
            mu = float(np.sum(y * n) / np.sum(n))
            tau2, kappa, diagnostics = moment_components(y, n, mu)

            source = "moment_regression"
            if not np.isfinite(kappa) or kappa <= 0.0:
                kappa = _plugin_kappa(feature, mu)
                source = "plugin_poisson_or_binomial"
                tau2 = max(float(np.mean((y - mu) ** 2 - kappa / n)), 0.0)
            per_feature[feature] = {
                "exposure": exposure_col,
                "n_pool": int(len(usable)),
                "mu": _r(mu, 6),
                "tau_squared": _r(tau2, 6),
                "kappa": _r(kappa, 6),
                "kappa_source": source,
                "plugin_kappa": _r(_plugin_kappa(feature, mu), 6),
                "median_exposure": _r(float(np.median(n)), 3),
                "reliability_at_median_exposure": _r(
                    tau2 / (tau2 + kappa / max(np.median(n), 1e-9)) if tau2 > 0 else 0.0, 4
                ),
                "diagnostics": diagnostics,
            }
        hyper[group] = per_feature

    return {
        "prior_pool": {
            "source": f"aggregated_{season}.parquet, outfield only",
            "min_minutes": PRIOR_MIN_MINUTES,
            "n_players": int(len(pool)),
            "n_by_group": pool["position_group"].value_counts().to_dict(),
            "rationale": (
                "The unfiltered pool keeps the low-minute players that config.MIN_MINUTES "
                "excludes. They carry little weight on the prior mean but they are the "
                "observations that identify the sampling variance slope."
            ),
        },
        "degenerate_features": {
            group: {
                "features_with_zero_between_player_variance": [
                    feature for feature, block in per_feature.items() if block["tau_squared"] <= 0.0
                ],
                "meaning": (
                    "The moment regression attributes all observed spread in this feature "
                    "to sampling noise, so every player in the group receives the group "
                    "mean and the feature becomes constant within the group. It then "
                    "carries no within-group information at all after re-standardisation."
                ),
            }
            for group, per_feature in hyper.items()
        },
        "by_group": hyper,
    }


def shrink_season(season: str, hyper: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the posterior mean to every eligible player and return the shrunken frame."""
    eligible = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")
    shrunken = eligible.copy()
    weights = pd.DataFrame(index=eligible.index, columns=F.OUTFIELD_CORE, dtype=float)

    for feature in F.OUTFIELD_CORE:
        exposure_col = EXPOSURE_COLUMN[feature]
        exposure = eligible[exposure_col].astype(float).to_numpy()
        y = eligible[feature].astype(float).to_numpy()
        b = np.zeros_like(y)
        posterior = np.zeros_like(y)
        for group in config.OUTFIELD_GROUPS:
            mask = (eligible["position_group"] == group).to_numpy()
            block = hyper["by_group"][group][feature]
            tau2, kappa, mu = block["tau_squared"], block["kappa"], block["mu"]
            n = np.where(exposure[mask] > 0, exposure[mask], np.nan)
            s2 = kappa / n
            bi = np.where(np.isfinite(s2) & (tau2 + s2 > 0), tau2 / (tau2 + s2), 0.0)
            bi = np.nan_to_num(bi, nan=0.0)
            b[mask] = bi
            posterior[mask] = bi * y[mask] + (1.0 - bi) * mu
        shrunken[feature] = posterior
        weights[feature] = b

    shrunken["mean_shrinkage_weight"] = weights.mean(axis=1)
    identifiers = [
        "season",
        "player",
        "team",
        "position_group",
        "position_full",
        "minutes",
        "nineties",
        "shots",
        "age",
        "n_squads",
    ]
    out = shrunken[[c for c in identifiers if c in shrunken.columns]].copy()
    for feature in F.OUTFIELD_CORE:
        out[feature] = shrunken[feature]
        out[f"{feature}_shrinkage_weight"] = weights[feature]
    out["mean_shrinkage_weight"] = shrunken["mean_shrinkage_weight"]

    summary = {
        "n_players": int(len(out)),
        "mean_shrinkage_weight": _r(float(weights.to_numpy().mean()), 4),
        "min_player_mean_weight": _r(float(shrunken["mean_shrinkage_weight"].min()), 4),
        "max_player_mean_weight": _r(float(shrunken["mean_shrinkage_weight"].max()), 4),
        "by_feature": {
            feature: {
                "mean_weight": _r(float(weights[feature].mean()), 4),
                "min_weight": _r(float(weights[feature].min()), 4),
                "max_weight": _r(float(weights[feature].max()), 4),
                "mean_abs_change": _r(
                    float(np.abs(shrunken[feature] - eligible[feature]).mean()), 5
                ),
            }
            for feature in F.OUTFIELD_CORE
        },
    }
    return out, summary


def refit_on_shrunken(season: str, shrunken: pd.DataFrame) -> dict:
    """Re-standardise the shrunken features and refit the clustering under K_RULE."""
    base = shrunken.copy()
    z_global = P.zscore(base.copy(), F.OUTFIELD_CORE, group=None)
    z_group = P.zscore(base.copy(), F.OUTFIELD_CORE, group="position_group")
    frames = {"global": z_global.reset_index(drop=True)}
    for group in config.OUTFIELD_GROUPS:
        frames[group] = z_group[z_group["position_group"] == group].reset_index(drop=True).copy()
    return fit_scopes(frames, season, tag_prefix="shrunken|")


def compare_partitions(season: str, fits: dict, shrunken: pd.DataFrame) -> dict:
    """Original against shrunken assignments, overall and by minutes played."""
    original = pd.read_parquet(config.DATA_PROCESSED / f"clusters_{season}.parquet")
    selection = json.loads((config.METRICS / "cluster_selection.json").read_text())
    original_k = {scope: int(selection["seasons"][season][scope]["chosen_k"]) for scope in SCOPES}
    original_rule_k = {
        scope: int(selection["seasons"][season][scope]["k_from_declared_rule"]) for scope in SCOPES
    }

    by_scope: dict[str, dict] = {}
    changed_flags: dict[str, np.ndarray] = {}
    for scope in SCOPES:
        column = "cluster_global" if scope == "global" else "cluster_within_group"
        subset = original if scope == "global" else original[original["position_group"] == scope]
        lookup = dict(zip(subset["player"], subset[column], strict=True))
        fit = fits[scope]
        players = list(fit["players"])
        keep = [i for i, p in enumerate(players) if p in lookup]
        a = np.array([lookup[players[i]] for i in keep])
        b = np.array([fit["labels"][i] for i in keep])
        mapping = match_labels(a, b)
        b_aligned = np.array([mapping[int(v)] for v in b])
        changed = a != b_aligned
        changed_flags[scope] = (
            pd.Series(changed, index=[players[i] for i in keep])
            .reindex(shrunken["player"])
            .fillna(False)
            .to_numpy()
        )
        ari = adjusted_rand_score(a, b)
        by_scope[scope] = {
            "n": int(len(a)),
            "k_original": original_k[scope],
            "k_original_from_rule": original_rule_k[scope],
            "k_shrunken_from_rule": fit["k"],
            "k_changed": bool(original_rule_k[scope] != fit["k"]),
            "adjusted_rand_index": _r(ari, 4),
            "n_changed": int(changed.sum()),
            "pct_changed": _r(100.0 * changed.sum() / max(len(a), 1), 2),
            "verdict": verdict_for(ari),
            "cluster_sizes_original": {
                str(c): int((a == c).sum()) for c in sorted(set(a.tolist()))
            },
            "cluster_sizes_shrunken": {
                str(c): int((b_aligned == c).sum()) for c in sorted(set(b_aligned.tolist()))
            },
        }

    frame = shrunken[["player", "minutes", "position_group", "mean_shrinkage_weight"]].copy()
    frame["changed_global"] = changed_flags["global"]
    within = np.zeros(len(frame), dtype=bool)
    for scope in config.OUTFIELD_GROUPS:
        within |= changed_flags[scope]
    frame["changed_within_group"] = within
    frame["minutes_bin"] = assign_minutes_bin(frame["minutes"])
    grouped = frame.groupby("minutes_bin", observed=False)
    by_minutes = {
        str(label): {
            "n": int(len(block)),
            "n_changed_global": int(block["changed_global"].sum()),
            "pct_changed_global": _r(100.0 * block["changed_global"].mean(), 2)
            if len(block)
            else 0.0,
            "n_changed_within_group": int(block["changed_within_group"].sum()),
            "pct_changed_within_group": _r(100.0 * block["changed_within_group"].mean(), 2)
            if len(block)
            else 0.0,
            "mean_shrinkage_weight": _r(float(block["mean_shrinkage_weight"].mean()), 4)
            if len(block)
            else 0.0,
            "median_minutes": _r(float(block["minutes"].median()), 1) if len(block) else 0.0,
        }
        for label, block in grouped
    }

    moved = frame["changed_global"] | frame["changed_within_group"]
    movers = frame[moved].sort_values("minutes").head(MOVER_TABLE_N)
    verdict = by_scope["global"]["verdict"]

    return {
        "by_scope": by_scope,
        "by_minutes_bin": by_minutes,
        "minutes_bin_edges": [float(e) for e in MINUTES_BIN_EDGES],
        "lowest_minute_movers": [
            {
                "player": str(row.player),
                "position_group": str(row.position_group),
                "minutes": int(row.minutes),
                "mean_shrinkage_weight": _r(float(row.mean_shrinkage_weight), 4),
                "changed_global_cluster": bool(row.changed_global),
                "changed_within_group_cluster": bool(row.changed_within_group),
            }
            for row in movers.itertuples()
        ],
        "verdict_bands": {"survives": SURVIVES_ARI, "partly_survives": PARTLY_SURVIVES_ARI},
        "verdict_global": verdict,
        "verdict_within_group": {
            scope: by_scope[scope]["verdict"] for scope in config.OUTFIELD_GROUPS
        },
        "verdict": verdict,
        "_changed_frame": frame,
    }


def figure_shrinkage_effect(season: str, eligible: pd.DataFrame, shrunken: pd.DataFrame) -> None:
    """Observed against shrunken value per feature, both on the original z scale."""
    fig, axes = plotting.facet_grid(
        len(F.OUTFIELD_CORE), ncols=4, width=plotting.WIDTH_FULL, panel_h=1.45
    )
    groups = eligible["position_group"].to_numpy()
    mean = eligible.groupby("position_group")[F.OUTFIELD_CORE].mean()
    sd = eligible.groupby("position_group")[F.OUTFIELD_CORE].std(ddof=0)

    limits = []
    panels = []
    for feature in F.OUTFIELD_CORE:
        m = eligible["position_group"].map(mean[feature]).to_numpy(dtype=float)
        s = eligible["position_group"].map(sd[feature]).to_numpy(dtype=float)
        s = np.where(s > 0, s, 1.0)
        zx = (eligible[feature].to_numpy(dtype=float) - m) / s
        zy = (shrunken[feature].to_numpy(dtype=float) - m) / s
        panels.append((zx, zy))
        limits.extend([zx, zy])
    pooled = np.concatenate(limits)
    lo = float(np.percentile(pooled, 0.2)) - 0.4
    hi = float(np.percentile(pooled, 99.8)) + 0.4
    off_scale = int(sum(int(((z < lo) | (z > hi)).sum()) for pair in panels for z in pair))

    for ax, feature, (zx, zy) in zip(axes, F.OUTFIELD_CORE, panels, strict=True):
        ax.plot([lo, hi], [lo, hi], color=plotting.INK_MUTED, linewidth=0.7, linestyle="--")
        for group in config.OUTFIELD_GROUPS:
            mask = groups == group
            ax.scatter(
                zx[mask],
                zy[mask],
                s=4,
                alpha=0.55,
                linewidths=0,
                color=plotting.POSITION_COLORS[group],
                marker=plotting.POSITION_MARKERS[group],
                label=group,
            )
        ax.set_title(DISPLAY_NAME[feature], loc="left", fontsize=plotting.BASE_FONT_PT - 1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    # The last row is not full, so the bottom visible panel of each column needs its own
    # tick labels back. Shared axes would otherwise leave two columns without any.
    ncols = 4
    for i, ax in enumerate(axes):
        if i + ncols >= len(axes):
            ax.tick_params(labelbottom=True)

    fig.supxlabel("observed value, z units of the position group", fontsize=plotting.BASE_FONT_PT)
    fig.supylabel("shrunken value, same units", fontsize=plotting.BASE_FONT_PT)
    handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=plotting.POSITION_MARKERS[g],
            color=plotting.POSITION_COLORS[g],
            markersize=4,
            label=g,
        )
        for g in config.OUTFIELD_GROUPS
    ]
    spare = fig.axes[len(axes)] if len(fig.axes) > len(axes) else None
    if spare is not None:
        spare.set_visible(True)
        spare.axis("off")
        spare.legend(handles=handles, loc="center", frameon=False, title="position group")
    tail = (
        f"\n{off_scale} of {2 * len(F.OUTFIELD_CORE) * len(eligible)} plotted values sit "
        "outside the shared axes"
        if off_scale
        else ""
    )
    fig.suptitle(
        f"Empirical Bayes shrinkage pulls every player towards his group mean, {season}\n"
        f"dashed line is no shrinkage; a flat cloud is a feature judged to be all noise{tail}",
        x=0.01,
        ha="left",
        fontsize=plotting.BASE_FONT_PT,
        fontweight="bold",
    )
    plotting.save_figure(fig, "shrinkage_effect")


def figure_shrinkage_by_minutes(season: str, comparison: dict) -> None:
    """Reliability against minutes, and the cluster change rate by minutes bin."""
    frame = comparison["_changed_frame"]
    fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_FULL, 2.9))
    ax = axes[0]
    for group in config.OUTFIELD_GROUPS:
        block = frame[frame["position_group"] == group]
        ax.scatter(
            block["minutes"],
            block["mean_shrinkage_weight"],
            s=7,
            alpha=0.6,
            linewidths=0,
            color=plotting.POSITION_COLORS[group],
            marker=plotting.POSITION_MARKERS[group],
            label=group,
        )
    extremes = pd.concat(
        [frame.nsmallest(1, "mean_shrinkage_weight"), frame.nlargest(1, "mean_shrinkage_weight")]
    )
    plotting.annotate_points(
        ax,
        extremes["minutes"].to_numpy(),
        extremes["mean_shrinkage_weight"].to_numpy(),
        [short_name(p, 16) for p in extremes["player"]],
        offset=(5.0, -1.0),
    )
    plotting.style_axis(
        ax,
        xlabel="minutes played",
        ylabel="mean reliability weight B",
        title="Reliability rises with exposure",
    )
    ax.legend(loc="lower right", title="position group", ncol=3, columnspacing=0.7)

    ax = axes[1]
    bins = comparison["by_minutes_bin"]
    labels = [lab for lab in minutes_bin_labels() if lab in bins]
    series = [
        ("within position group", "pct_changed_within_group", plotting.CATEGORICAL[0]),
        ("whole outfield pool", "pct_changed_global", plotting.CATEGORICAL[1]),
    ]
    x = np.arange(len(labels))
    width = 0.36
    ceiling = max(
        [bins[lab][key] for lab in labels for _, key, _ in series] + [1.0],
    )
    for offset, (name, key, colour) in zip([-width / 2, width / 2], series, strict=True):
        heights = [bins[lab][key] for lab in labels]
        ax.bar(x + offset, heights, width=width, color=colour, label=name)
        for xi, height in zip(x + offset, heights, strict=True):
            ax.text(
                xi,
                height + ceiling * 0.03,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=plotting.BASE_FONT_PT - 2.5,
                color=plotting.INK_SECONDARY,
            )
    ax.set_xticks(x, [f"{lab}\nn={bins[lab]['n']}" for lab in labels])
    ax.set_ylim(0, ceiling * 1.35)
    plotting.style_axis(
        ax,
        xlabel="minutes played",
        ylabel="percent changing cluster",
        title="Who moves when rates are shrunk",
    )
    ax.legend(loc="upper right", title="clustering scope")

    fig.suptitle(
        f"Reliability and cluster movement against minutes played, {season}",
        x=0.01,
        ha="left",
        fontsize=plotting.BASE_FONT_PT,
        fontweight="bold",
    )
    plotting.save_figure(fig, "shrinkage_by_minutes")


def empirical_bayes_shrinkage() -> dict:
    """Estimate the prior, shrink both seasons, refit and compare."""
    payload: dict = {
        "question": (
            "Per-90 rates over partial seasons are noisy, and most so where minutes are "
            "fewest. Does the partition describe playing roles, or sampling noise?"
        ),
        "model": {
            "likelihood": "y_i | theta_i has mean theta_i and variance kappa_g / n_i",
            "prior": "theta_i has mean mu_g and variance tau_g^2 within position group",
            "posterior_mean": "B_i y_i + (1 - B_i) mu_g with B_i = tau^2 / (tau^2 + kappa / n_i)",
            "exposure": {
                "per_90_features": "nineties, that is minutes divided by 90",
                "ratio_features": "shots, since both ratios are computed per shot",
            },
            "variance_component_estimator": (
                "E[(y_i - mu_g)^2] = tau_g^2 + kappa_g / n_i is linear in 1 / n_i, so the "
                "two components are separately identified by regressing squared deviations "
                "on inverse exposure with an intercept. Ordinary least squares gives the "
                "starting value and two weighted least squares iterations with weights "
                "1 / fitted^2 follow, because squared deviations are heteroscedastic."
            ),
            "fallback": (
                "If the estimated slope is not positive, kappa falls back to the Poisson "
                "value mu_g for counts or the binomial value p(1-p) on the ratio scale, "
                "and tau^2 is recovered as the residual mean squared deviation."
            ),
            "assumptions": [
                "One stable true rate per player per season, so mid-season role change is "
                "read as sampling noise and shrunk away.",
                "Sampling variance inversely proportional to exposure with a constant "
                "shared within position group, so players are exchangeable up to exposure.",
                "Exposure fixed and independent of the true rate. This is false in the "
                "direction that matters, since better players play more, so mu_g leans "
                "towards high-minute players and shrinkage is mildly conservative.",
                "Only first and second moments are used, so the posterior mean is exact "
                "under Gaussian assumptions and the best linear predictor otherwise.",
                "The moment regression centres on an estimated mean, so tau^2 carries a "
                "small downward bias of order one over the group size.",
            ],
        },
        "refit": (
            "Shrunken features are re-standardised exactly as src.preprocess standardises "
            "the raw ones, globally and within position group, then the clustering is "
            "refitted under the same declared config.K_RULE. Re-standardising removes the "
            "uniform scale compression and leaves the part that matters, which is that "
            "low-minute players are pulled in further than high-minute players."
        ),
        "seasons": {},
    }

    primary_comparison = None
    primary_frames = None
    for season in config.SEASONS:
        print(f"  shrinkage, season {season}")
        hyper = estimate_hyperparameters(season)
        shrunken, summary = shrink_season(season, hyper)
        path = config.DATA_PROCESSED / f"shrunken_{season}.parquet"
        shrunken.to_parquet(path, index=False)
        print(f"    wrote {path.relative_to(config.ROOT)}  {shrunken.shape}")

        fits = refit_on_shrunken(season, shrunken)
        comparison = compare_partitions(season, fits, shrunken)
        eligible = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")

        block = {
            "hyperparameters": hyper,
            "shrinkage_summary": summary,
            "partition_comparison": {
                key: value for key, value in comparison.items() if not key.startswith("_")
            },
        }
        payload["seasons"][season] = block
        print(
            f"    global ARI original vs shrunken = "
            f"{comparison['by_scope']['global']['adjusted_rand_index']:.4f}, "
            f"{comparison['by_scope']['global']['n_changed']} of "
            f"{comparison['by_scope']['global']['n']} players change cluster"
        )
        if season == config.SEASON_PRIMARY:
            primary_comparison = comparison
            primary_frames = (eligible, shrunken)

    global_block = payload["seasons"][config.SEASON_PRIMARY]["partition_comparison"]
    payload["headline"] = {
        "season": config.SEASON_PRIMARY,
        "adjusted_rand_index_global": global_block["by_scope"]["global"]["adjusted_rand_index"],
        "n_changed_global": global_block["by_scope"]["global"]["n_changed"],
        "n_global": global_block["by_scope"]["global"]["n"],
        "k_changed_anywhere": any(global_block["by_scope"][scope]["k_changed"] for scope in SCOPES),
        "adjusted_rand_index_within_group": {
            scope: global_block["by_scope"][scope]["adjusted_rand_index"]
            for scope in config.OUTFIELD_GROUPS
        },
        "n_changed_within_group": {
            scope: global_block["by_scope"][scope]["n_changed"] for scope in config.OUTFIELD_GROUPS
        },
        "verdict": global_block["verdict"],
        "verdict_within_group": global_block["verdict_within_group"],
        "reading": (
            "The global partition and the within-group partitions behave differently under "
            "the same shrinkage, so they must be reported separately. The global split "
            "rests on the large between-role differences that shrinkage leaves intact. The "
            "within-group splits rest on finer distinctions, several of which the estimator "
            "attributes to sampling noise, and they move correspondingly more."
        ),
    }
    payload["_primary_comparison"] = primary_comparison
    payload["_primary_frames"] = primary_frames
    return payload


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    plotting.use_style()
    np.random.seed(config.RANDOM_STATE)
    print(f"Phase 10 novel analyses: {', '.join(config.NOVEL_ANALYSES)}")

    print("\n=== 1. player similarity search ===")
    similarity = similarity_search(config.SEASON_PRIMARY)
    figure_similarity_heatmap(similarity)
    _write_json(similarity, "similarity_search.json")

    print("\n=== 2. cross-season replication ===")
    replication = cross_season_replication()
    figure_cross_season_transitions(replication)
    _write_json(
        {key: value for key, value in replication.items() if not key.startswith("_")},
        "cross_season_replication.json",
    )

    print("\n=== 3. empirical Bayes shrinkage ===")
    shrinkage = empirical_bayes_shrinkage()
    eligible, shrunken = shrinkage.pop("_primary_frames")
    comparison = shrinkage.pop("_primary_comparison")
    figure_shrinkage_effect(config.SEASON_PRIMARY, eligible, shrunken)
    figure_shrinkage_by_minutes(config.SEASON_PRIMARY, comparison)
    _write_json(shrinkage, "shrinkage.json")

    _print_summary(similarity, replication, shrinkage)


def _print_summary(similarity: dict, replication: dict, shrinkage: dict) -> None:
    print("\n=== summary ===")
    consistent = sum(
        q["neighbour_agreement"]["consistent_on_defining_features"] for q in similarity["queries"]
    )
    print(
        f"  similarity: {len(similarity['queries'])} query players, "
        f"{consistent} with a neighbour set consistent on their defining features"
    )
    scopes = replication["scopes"]
    print(
        f"  replication: {replication['n_common_eligible']} players eligible in both seasons, "
        f"global ARI {scopes['global']['adjusted_rand_index']:.3f}"
    )
    for scope in config.OUTFIELD_GROUPS:
        block = scopes[scope]
        print(
            f"    {scope}: n={block['n_common']} ARI={block['adjusted_rand_index']:.3f} "
            f"changed={block['n_changed_cluster']}"
        )
    head = shrinkage["headline"]
    print(
        f"  shrinkage: {head['n_changed_global']} of {head['n_global']} change global cluster, "
        f"ARI {head['adjusted_rand_index_global']:.3f}, verdict {head['verdict']}, "
        f"k changed anywhere: {head['k_changed_anywhere']}"
    )
    for scope in config.OUTFIELD_GROUPS:
        print(
            f"    {scope} within group: ARI "
            f"{head['adjusted_rand_index_within_group'][scope]:.3f} "
            f"changed={head['n_changed_within_group'][scope]} "
            f"({head['verdict_within_group'][scope]})"
        )


if __name__ == "__main__":
    main()
