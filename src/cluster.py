"""Phase 5. Clustering, k selection, stability and the position comparison.

Scope
-----
Four scopes are analysed independently: the whole outfield pool ("global") and each of
the three listed position groups (DF, MF, FW). The global scope uses the pool-wide
z-scored table, ``outfield_z_global_{season}``. The three group scopes use the
within-group z-scored table, ``outfield_z_bygroup_{season}``, restricted to that group,
so a defender is compared against defenders rather than against the whole league.

Two feature spaces are fitted everywhere and both are reported:

``standardised``
    The 18 columns of ``features.OUTFIELD_CORE`` as z-scores.
``pca90``
    Principal components of that same matrix retaining ``config.PCA_VARIANCE_TARGET``
    of the variance.

Declared in advance, before any result was inspected: the reported clustering is KMeans
in the ``standardised`` space. That choice avoids conditioning the headline result on a
variance cutoff and keeps centroid distances interpretable in the original feature
basis. The ``pca90`` fit is carried through the whole pipeline as a sensitivity check
and the adjusted Rand index between the two solutions is recorded. Every result block in
``results/metrics/cluster_algorithms.json`` names the space it came from.

k selection
-----------
Silhouette, Davies-Bouldin, Calinski-Harabasz and the gap statistic are computed over
``config.K_RANGE``. The gap statistic follows Tibshirani, Walther and Hastie (2001) and
is implemented here because scikit-learn does not provide it: ``config.GAP_STATISTIC_B``
uniform reference sets are drawn over the axis-aligned bounding box of the data in the
space being fitted, and the gap is the mean log reference dispersion minus the log data
dispersion, with the dispersion taken as the KMeans within-cluster sum of squares.

The consensus rule is ``config.K_RULE``, fixed before the analysis was run, and is
implemented literally in :func:`apply_k_rule`. Which k each individual criterion favours
is logged alongside the rule outcome so a reader can see where they disagree.

Gaussian mixtures and HDBSCAN
-----------------------------
Gaussian mixture BIC is reported per k for two covariance types in both spaces. The
posterior used for the hybrid analysis comes from a full covariance mixture fitted in
the ``pca90`` space, because a full covariance model on 18 correlated features and a few
hundred players is not estimable, while the same model on nine decorrelated components
is. Only the global scope gets posteriors: on 49 forwards even the reduced model is
degenerate, and reporting a number from it would be dishonest. HDBSCAN is run at three
minimum cluster sizes and its noise fraction is reported for each.

Replication season
------------------
The 2025-26 labels exist so the Phase 10 replication can use them. Their k is taken from
the primary season so that a cross-season comparison is not confounded by a different
number of clusters, and the k the rule would have chosen on 2025-26 on its own is
recorded anyway. Cluster ids for the replication season are matched to the primary
season's centroids by optimal assignment, so that id 0 refers to a comparable group in
both seasons. Adjusted Rand index is invariant to that relabelling, so it changes no
downstream number.

Outputs
-------
``results/metrics/cluster_selection.json``, ``cluster_stability.json``,
``cluster_vs_position.json``, ``hybrid_players.json``, ``cluster_algorithms.json``,
``data/processed/clusters_{season}.parquet`` and four figures.
"""

from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

import config
from src import features as F
from src import plotting

# --------------------------------------------------------------------------------------
# Declared choices. None of these were tuned after seeing a result.
# --------------------------------------------------------------------------------------

SCOPES: list[str] = ["global"] + list(config.OUTFIELD_GROUPS)
SPACES: list[str] = ["standardised", "pca90"]

#: The space whose KMeans solution is the reported result. Sensitivity in the other.
PRIMARY_SPACE = "standardised"

KMEANS_N_INIT = 10
GAP_REFERENCE_N_INIT = 10
GMM_N_INIT = 5
GMM_COVARIANCES = ["full", "diag"]

#: Space and covariance used for the posteriors written to the parquet.
GMM_POSTERIOR_SPACE = "pca90"
GMM_POSTERIOR_COVARIANCE = "full"

#: Minimum cluster sizes swept for HDBSCAN.
HDBSCAN_MIN_CLUSTER_SIZES = [5, 10, 20]

#: A cluster is called low stability when its members co-assign below this on average.
LOW_STABILITY_THRESHOLD = 0.6

#: A Gaussian mixture posterior below this is called split.
POSTERIOR_SPLIT_THRESHOLD = 0.6

HYBRID_TABLE_N = 20

#: Reporting convention for calling the clusters a recovery of the listed positions.
RECOVERY_ARI_THRESHOLD = 0.5

#: Per-90 columns quoted in the hybrid table. Read from the unstandardised table so the
#: values in the paper are in football units, not z units.
HYBRID_STAT_COLS = [
    "np_xg_p90",
    "xa_p90",
    "key_passes_p90",
    "xg_buildup_p90",
    "crosses_p90",
    "tackles_won_padj_p90",
    "interceptions_padj_p90",
]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _seed(*parts: object) -> int:
    """A stable seed derived from ``config.RANDOM_STATE`` and a label.

    ``hash()`` is randomised per interpreter for strings, so a digest is used instead.
    Two runs of this module therefore draw the same reference sets and resamples.
    """
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return (config.RANDOM_STATE + int(digest[:8], 16)) % (2**31 - 1)


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


def _r(x: float, nd: int = 6) -> float:
    return float(np.round(float(x), nd))


def _kmeans(x: np.ndarray, k: int, tag: str, n_init: int = KMEANS_N_INIT) -> KMeans:
    return KMeans(n_clusters=k, n_init=n_init, random_state=_seed(tag, k)).fit(x)


# --------------------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------------------


def load_season(season: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return the unstandardised eligible table and one standardised frame per scope."""
    eligible = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")
    z_global = pd.read_parquet(config.DATA_PROCESSED / f"outfield_z_global_{season}.parquet")
    z_group = pd.read_parquet(config.DATA_PROCESSED / f"outfield_z_bygroup_{season}.parquet")
    if not eligible["player"].is_unique:
        raise ValueError(f"player is not a unique key for {season}")

    frames = {"global": z_global.reset_index(drop=True)}
    for group in config.OUTFIELD_GROUPS:
        frames[group] = z_group[z_group["position_group"] == group].reset_index(drop=True).copy()
    return eligible, frames


def build_spaces(x: np.ndarray, tag: str) -> tuple[dict[str, np.ndarray], PCA]:
    """Return the standardised matrix and its 90 percent variance PCA projection."""
    pca = PCA(n_components=config.PCA_VARIANCE_TARGET, random_state=_seed("pca", tag))
    projected = pca.fit_transform(x)
    return {"standardised": x, "pca90": projected}, pca


# --------------------------------------------------------------------------------------
# Gap statistic (Tibshirani, Walther and Hastie 2001)
# --------------------------------------------------------------------------------------


def gap_statistic(
    x: np.ndarray,
    k_range: list[int],
    tag: str,
    b: int = config.GAP_STATISTIC_B,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    """Return gap, its standard error and the log data dispersion for each k.

    The dispersion ``W_k`` is the pooled within-cluster sum of squares, which is exactly
    the KMeans objective, so ``inertia_`` is used directly. Reference sets are uniform
    over the axis-aligned bounding box of ``x`` in the space being fitted, which is the
    simpler of the two reference distributions in the original paper.
    """
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    n, d = x.shape
    gaps: dict[int, float] = {}
    sks: dict[int, float] = {}
    log_w: dict[int, float] = {}

    reference_logs = np.empty((len(k_range), b))
    rng = np.random.default_rng(_seed("gap", tag))
    references = [rng.uniform(low=lo, high=hi, size=(n, d)) for _ in range(b)]

    for i, k in enumerate(k_range):
        log_w[k] = float(np.log(_kmeans(x, k, f"{tag}|gap-data").inertia_))
        for j, ref in enumerate(references):
            fit = KMeans(
                n_clusters=k,
                n_init=GAP_REFERENCE_N_INIT,
                random_state=_seed(tag, "gap-ref", k, j),
            ).fit(ref)
            reference_logs[i, j] = np.log(fit.inertia_)
        mean_log = float(reference_logs[i].mean())
        sd_log = float(reference_logs[i].std(ddof=0))
        gaps[k] = mean_log - log_w[k]
        sks[k] = sd_log * float(np.sqrt(1.0 + 1.0 / b))
    return gaps, sks, log_w


def tibshirani_k(gaps: dict[int, float], sks: dict[int, float]) -> int | None:
    """Smallest k with ``gap(k) >= gap(k+1) - s(k+1)``, the original selection rule."""
    ks = sorted(gaps)
    for k, nxt in zip(ks[:-1], ks[1:], strict=True):
        if gaps[k] >= gaps[nxt] - sks[nxt]:
            return k
    return None


# --------------------------------------------------------------------------------------
# Internal validation curves and the declared consensus rule
# --------------------------------------------------------------------------------------


def internal_curves(x: np.ndarray, tag: str) -> dict[str, dict[int, float]]:
    """Silhouette, Davies-Bouldin, Calinski-Harabasz and the gap statistic over K_RANGE."""
    k_range = [k for k in config.K_RANGE if k < x.shape[0]]
    silhouette: dict[int, float] = {}
    davies: dict[int, float] = {}
    calinski: dict[int, float] = {}
    inertia: dict[int, float] = {}
    for k in k_range:
        labels = _kmeans(x, k, tag).labels_
        silhouette[k] = _r(silhouette_score(x, labels))
        davies[k] = _r(davies_bouldin_score(x, labels))
        calinski[k] = _r(calinski_harabasz_score(x, labels))
        inertia[k] = _r(_kmeans(x, k, tag).inertia_)
    gaps, sks, log_w = gap_statistic(x, k_range, tag)
    return {
        "silhouette": silhouette,
        "davies_bouldin": davies,
        "calinski_harabasz": calinski,
        "gap": {k: _r(v) for k, v in gaps.items()},
        "gap_se": {k: _r(v) for k, v in sks.items()},
        "log_dispersion": {k: _r(v) for k, v in log_w.items()},
        "inertia": inertia,
    }


def _ranks(values: dict[int, float], higher_is_better: bool) -> dict[int, int]:
    order = sorted(values, key=lambda k: (-values[k] if higher_is_better else values[k], k))
    return {k: i + 1 for i, k in enumerate(order)}


def apply_k_rule(curves: dict[str, dict[int, float]], rule: dict | None = None) -> dict:
    """Apply ``config.K_RULE`` literally.

    Choose the smallest k whose silhouette is within ``silhouette_tolerance`` of the best
    observed silhouette and which ranks in the top ``top_n_rank`` on at least
    ``min_criteria`` of silhouette, Calinski-Harabasz and gap. Davies-Bouldin, lower
    better, is the declared tiebreak and fires only when no k satisfies the rank
    condition.
    """
    rule = dict(config.K_RULE if rule is None else rule)
    silhouette = curves["silhouette"]
    calinski = curves["calinski_harabasz"]
    gap = curves["gap"]
    davies = curves["davies_bouldin"]
    ks = sorted(silhouette)

    best_silhouette = max(silhouette.values())
    tolerance = float(rule["silhouette_tolerance"])
    eligible = [k for k in ks if silhouette[k] >= best_silhouette - tolerance]

    ranks = {
        "silhouette": _ranks(silhouette, True),
        "calinski_harabasz": _ranks(calinski, True),
        "gap": _ranks(gap, True),
    }
    top_n = int(rule["top_n_rank"])
    passed = {k: [c for c, r in ranks.items() if r[k] <= top_n] for k in ks}
    candidates = [k for k in eligible if len(passed[k]) >= int(rule["min_criteria"])]

    if candidates:
        chosen = min(candidates)
        branch = "primary"
        tiebreak_used = False
    elif eligible:
        chosen = min(eligible, key=lambda k: (davies[k], k))
        branch = "tiebreak_davies_bouldin"
        tiebreak_used = True
    else:  # pragma: no cover - the best silhouette is always within tolerance of itself
        chosen = max(silhouette, key=lambda k: silhouette[k])
        branch = "fallback_max_silhouette"
        tiebreak_used = False

    return {
        "chosen_k": int(chosen),
        "rule_branch": branch,
        "tiebreak_applied": tiebreak_used,
        "best_silhouette": _r(best_silhouette),
        "silhouette_tolerance": tolerance,
        "eligible_k": eligible,
        "candidates": candidates,
        "criteria_passed_by_k": {int(k): passed[k] for k in ks},
        "ranks": {c: {int(k): int(v) for k, v in r.items()} for c, r in ranks.items()},
        "favoured_by_criterion": {
            "silhouette": int(max(silhouette, key=lambda k: silhouette[k])),
            "davies_bouldin": int(min(davies, key=lambda k: davies[k])),
            "calinski_harabasz": int(max(calinski, key=lambda k: calinski[k])),
            "gap_max": int(max(gap, key=lambda k: gap[k])),
            "gap_tibshirani_1se": tibshirani_k(curves["gap"], curves["gap_se"]),
        },
    }


# --------------------------------------------------------------------------------------
# Gaussian mixtures and HDBSCAN
# --------------------------------------------------------------------------------------


def _gmm_free_parameters(k: int, d: int, covariance: str) -> int:
    per_component = {"full": d * (d + 1) // 2, "diag": d}[covariance]
    return k * (d + per_component) + k - 1


def gmm_bic_curve(x: np.ndarray, tag: str) -> dict:
    """BIC per k for each covariance type, with the parameter count for context."""
    n, d = x.shape
    out: dict[str, dict] = {}
    for covariance in GMM_COVARIANCES:
        bics: dict[int, float] = {}
        params: dict[int, int] = {}
        for k in config.K_RANGE:
            if k >= n:
                continue
            params[k] = _gmm_free_parameters(k, d, covariance)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model = GaussianMixture(
                        n_components=k,
                        covariance_type=covariance,
                        n_init=GMM_N_INIT,
                        random_state=_seed(tag, "gmm", covariance, k),
                    ).fit(x)
                bics[k] = _r(model.bic(x), 2)
            except ValueError:
                bics[k] = None
        valid = {k: v for k, v in bics.items() if v is not None}
        out[covariance] = {
            "bic": bics,
            "free_parameters": params,
            "n_samples": int(n),
            "n_dimensions": int(d),
            "best_k_by_bic": int(min(valid, key=lambda k: valid[k])) if valid else None,
        }
    return out


def hdbscan_report(x: np.ndarray) -> dict:
    """Cluster count and noise fraction at each declared minimum cluster size."""
    out: dict[str, dict] = {}
    for min_size in HDBSCAN_MIN_CLUSTER_SIZES:
        if min_size >= x.shape[0]:
            continue
        labels = HDBSCAN(min_cluster_size=min_size, copy=True).fit(x).labels_
        found = sorted(set(labels) - {-1})
        noise = float((labels == -1).mean())
        sizes = {int(c): int((labels == c).sum()) for c in found}
        out[str(min_size)] = {
            "n_clusters": len(found),
            "noise_fraction": _r(noise, 4),
            "cluster_sizes": sizes,
        }
    return out


# --------------------------------------------------------------------------------------
# Bootstrap stability and the consensus co-assignment matrix
# --------------------------------------------------------------------------------------


def bootstrap_stability(
    x: np.ndarray,
    reference: np.ndarray,
    k: int,
    tag: str,
    n_boot: int = config.BOOTSTRAP_N,
) -> dict:
    """Resample with replacement, refit, and score against the reference clustering.

    The adjusted Rand index is computed on the players that appear in the resample,
    which is the overlap between the resampled clustering and the reference one. The
    co-assignment matrix counts, for every pair of players, the fraction of resamples in
    which both were drawn and both landed in the same cluster.
    """
    n = x.shape[0]
    rng = np.random.default_rng(_seed("bootstrap", tag, k))
    together = np.zeros((n, n), dtype=np.int32)
    seen = np.zeros((n, n), dtype=np.int32)
    aris: list[float] = []

    for b in range(n_boot):
        draw = rng.integers(0, n, size=n)
        unique, first = np.unique(draw, return_index=True)
        if unique.size <= k:
            continue
        fit = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=_seed(tag, "boot", k, b)).fit(
            x[draw]
        )
        labels_for_unique = fit.labels_[first]
        aris.append(float(adjusted_rand_score(reference[unique], labels_for_unique)))
        same = (labels_for_unique[:, None] == labels_for_unique[None, :]).astype(np.int32)
        grid = np.ix_(unique, unique)
        together[grid] += same
        seen[grid] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        coassign = np.where(seen > 0, together / np.maximum(seen, 1), np.nan)
    np.fill_diagonal(coassign, 1.0)

    per_cluster: dict[str, dict] = {}
    for c in sorted(set(reference.tolist())):
        members = np.flatnonzero(reference == c)
        if members.size < 2:
            per_cluster[str(int(c))] = {"n": int(members.size), "mean_coassignment": None}
            continue
        block = coassign[np.ix_(members, members)].copy()
        np.fill_diagonal(block, np.nan)
        mean_block = float(np.nanmean(block))
        per_cluster[str(int(c))] = {
            "n": int(members.size),
            "mean_coassignment": _r(mean_block, 4),
            "low_stability": bool(mean_block < LOW_STABILITY_THRESHOLD),
        }

    return {
        "n_bootstrap": int(len(aris)),
        "ari_mean": _r(float(np.mean(aris)), 4),
        "ari_sd": _r(float(np.std(aris, ddof=1)), 4),
        "ari_min": _r(float(np.min(aris)), 4),
        "ari_max": _r(float(np.max(aris)), 4),
        "per_cluster": per_cluster,
        "low_stability_clusters": [c for c, v in per_cluster.items() if v.get("low_stability")],
        "coassignment": coassign,
    }


# --------------------------------------------------------------------------------------
# Clusters against listed positions
# --------------------------------------------------------------------------------------


def cluster_vs_position(frame: pd.DataFrame, labels: np.ndarray) -> dict:
    """Contingency tables and agreement scores against the two listed position fields."""
    out: dict = {}
    for field in ("position_group", "position_full"):
        table = pd.crosstab(pd.Series(labels, name="cluster"), frame[field].to_numpy())
        out[field] = {
            "contingency": {str(int(r)): table.loc[r].to_dict() for r in table.index},
            "adjusted_rand_index": _r(adjusted_rand_score(frame[field], labels), 4),
            "normalized_mutual_info": _r(normalized_mutual_info_score(frame[field], labels), 4),
        }
        dominant = {}
        purity_num = 0
        for r in table.index:
            row = table.loc[r]
            dominant[str(int(r))] = {
                "label": str(row.idxmax()),
                "share": _r(float(row.max() / row.sum()), 4),
                "n": int(row.sum()),
            }
            purity_num += int(row.max())
        out[field]["dominant_label_per_cluster"] = dominant
        out[field]["cluster_purity"] = _r(purity_num / len(labels), 4)
    return out


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def figure_k_selection(selection: dict[str, dict], season: str) -> None:
    """Four validation curves, one panel each, all four scopes overlaid."""
    import matplotlib.pyplot as plt

    panels = [
        ("silhouette", "Silhouette (higher better)", False),
        ("davies_bouldin", "Davies-Bouldin (lower better)", False),
        ("calinski_harabasz", "Calinski-Harabasz, log scale (higher better)", True),
        ("gap", "Gap statistic (higher better)", False),
    ]
    colors = plotting.categorical(len(SCOPES))
    marks = plotting.markers(len(SCOPES))

    fig, axes = plt.subplots(2, 2, figsize=(plotting.WIDTH_FULL, 4.6))
    axes = axes.ravel()
    for index, (ax, (metric, label, log_y)) in enumerate(zip(axes, panels, strict=True)):
        for scope, color, mark in zip(SCOPES, colors, marks, strict=True):
            curves = selection[scope]["spaces"][PRIMARY_SPACE]["curves"]
            ks = sorted(int(k) for k in curves[metric])
            values = [curves[metric][k] for k in ks]
            if metric == "gap":
                errs = [curves["gap_se"][k] for k in ks]
                ax.errorbar(
                    ks,
                    values,
                    yerr=errs,
                    color=color,
                    marker=mark,
                    label=scope,
                    elinewidth=0.6,
                    capsize=1.5,
                    linewidth=1.4,
                    markersize=3.2,
                )
            else:
                ax.plot(
                    ks,
                    values,
                    color=color,
                    marker=mark,
                    label=scope,
                    linewidth=1.4,
                    markersize=3.2,
                )
            chosen = selection[scope]["chosen_k"]
            ax.plot(
                [chosen],
                [curves[metric][chosen]],
                marker="o",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.2,
                linestyle="none",
                zorder=5,
            )
        if log_y:
            ax.set_yscale("log")
        plotting.style_axis(ax, xlabel="number of clusters k" if index >= 2 else "", title=label)
        ax.set_xticks(config.K_RANGE[::2])
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="outside lower center", ncol=4)
    plotting.save_figure(fig, "k_selection_curves")


def figure_consensus(stability: dict[str, dict], labels: dict[str, np.ndarray]) -> None:
    """Co-assignment heat maps, one panel per scope, players ordered by cluster."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(plotting.WIDTH_FULL, 5.4))
    axes = axes.ravel()
    image = None
    for index, (ax, scope) in enumerate(zip(axes, SCOPES, strict=True)):
        matrix = stability[scope]["coassignment"]
        reference = labels[scope]
        strength = np.nanmean(matrix, axis=1)
        order = np.lexsort((-strength, reference))
        ordered = matrix[np.ix_(order, order)]
        image = ax.imshow(
            ordered,
            cmap=plotting.SEQUENTIAL,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
        )
        edges = np.cumsum(np.bincount(reference))[:-1]
        for edge in edges:
            ax.axhline(edge - 0.5, color=plotting.INK_PRIMARY, linewidth=0.7)
            ax.axvline(edge - 0.5, color=plotting.INK_PRIMARY, linewidth=0.7)
        mean_ari = stability[scope]["ari_mean"]
        ax.set_title(f"{scope}  (n={len(reference)}, mean ARI {mean_ari:.2f})", loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        if index >= 2:
            ax.set_xlabel("players ordered by cluster")
        ax.grid(False)
    bar = fig.colorbar(image, ax=axes.tolist(), fraction=0.03, pad=0.02)
    bar.set_label("fraction of bootstrap resamples in the same cluster")
    plotting.save_figure(fig, "consensus_matrix")


def _annotated_heatmap(ax, table: pd.DataFrame, title: str) -> None:
    import matplotlib.pyplot as plt  # noqa: F401

    shares = table.div(table.sum(axis=1), axis=0).to_numpy(float)
    ax.imshow(shares, cmap=plotting.SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns], rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]), [f"C{int(r)}" for r in table.index])
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            count = int(table.iat[i, j])
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                fontsize=plotting.BASE_FONT_PT - 2,
                color=plotting.SURFACE if shares[i, j] > 0.55 else plotting.INK_PRIMARY,
            )
    ax.set_title(title, loc="left")
    ax.grid(False)


def figure_cluster_vs_position(frame: pd.DataFrame, labels: np.ndarray, stats: dict) -> None:
    """Cluster by listed position contingency, counts printed, shading is the row share."""
    import matplotlib.pyplot as plt

    series = pd.Series(labels, name="cluster")
    group_table = pd.crosstab(series, frame["position_group"].to_numpy())
    full_table = pd.crosstab(series, frame["position_full"].to_numpy())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(plotting.WIDTH_FULL, 2.6),
        gridspec_kw={"width_ratios": [group_table.shape[1], full_table.shape[1]]},
    )
    g = stats["position_group"]
    f = stats["position_full"]
    _annotated_heatmap(
        axes[0],
        group_table,
        f"vs position group\nARI {g['adjusted_rand_index']:.2f}, "
        f"NMI {g['normalized_mutual_info']:.2f}",
    )
    _annotated_heatmap(
        axes[1],
        full_table,
        f"vs full FBref position string\nARI {f['adjusted_rand_index']:.2f}, "
        f"NMI {f['normalized_mutual_info']:.2f}",
    )
    axes[0].set_ylabel("data-driven cluster")
    for ax in axes:
        ax.set_xlabel("listed position")
    plotting.save_figure(fig, "cluster_vs_position")


def figure_cluster_scatter(
    projection: np.ndarray, pca: PCA, labels: np.ndarray, groups: pd.Series
) -> None:
    """Global clusters and listed groups in the same first two principal components."""
    import matplotlib.pyplot as plt

    var = pca.explained_variance_ratio_
    xlab = f"PC1 ({var[0] * 100:.0f}% of variance)"
    ylab = f"PC2 ({var[1] * 100:.0f}% of variance)"
    clusters = sorted(set(labels.tolist()))

    if len(clusters) <= plotting.MAX_CATEGORICAL:
        fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_FULL, 3.0), sharex=True, sharey=True)
        colors = plotting.categorical(len(clusters))
        marks = plotting.markers(len(clusters))
        for c, color, mark in zip(clusters, colors, marks, strict=True):
            sel = labels == c
            axes[0].scatter(
                projection[sel, 0],
                projection[sel, 1],
                s=11,
                c=color,
                marker=mark,
                linewidths=0,
                alpha=0.85,
                label=f"cluster {c} (n={int(sel.sum())})",
            )
        plotting.style_axis(axes[0], xlabel=xlab, ylabel=ylab, title="discovered clusters")
        axes[0].legend(loc="best", fontsize=plotting.BASE_FONT_PT - 2)
        for group in config.OUTFIELD_GROUPS:
            sel = (groups == group).to_numpy()
            axes[1].scatter(
                projection[sel, 0],
                projection[sel, 1],
                s=11,
                c=plotting.POSITION_COLORS[group],
                marker=plotting.POSITION_MARKERS[group],
                linewidths=0,
                alpha=0.85,
                label=f"{group} (n={int(sel.sum())})",
            )
        plotting.style_axis(axes[1], xlabel=xlab, title="listed position group")
        axes[1].legend(loc="best", fontsize=plotting.BASE_FONT_PT - 2)
    else:
        fig, axes = plotting.facet_grid(len(clusters), ncols=4, panel_h=1.7)
        for ax, c in zip(axes, clusters, strict=True):
            ax.scatter(
                projection[:, 0], projection[:, 1], s=6, c=plotting.CONTEXT_GREY, linewidths=0
            )
            sel = labels == c
            ax.scatter(
                projection[sel, 0], projection[sel, 1], s=8, c=plotting.CATEGORICAL[0], linewidths=0
            )
            ax.set_title(f"cluster {c} (n={int(sel.sum())})", loc="left")
        axes[0].set_ylabel(ylab)
        for ax in axes:
            ax.set_xlabel(xlab)
    plotting.save_figure(fig, "cluster_scatter_pca")


# --------------------------------------------------------------------------------------
# Per-season pipeline
# --------------------------------------------------------------------------------------


def analyse_season(
    season: str, fixed_k: dict[str, int] | None = None, reference_centroids: dict | None = None
) -> dict:
    """Run every scope for one season and return everything needed by the writers."""
    eligible, frames = load_season(season)
    print(f"\n=== season {season} ===")

    result: dict = {
        "selection": {},
        "algorithms": {},
        "stability": {},
        "position": {},
        "labels": {},
        "frames": frames,
        "eligible": eligible,
        "centroids": {},
        "projections": {},
        "pca": {},
    }

    for scope in SCOPES:
        frame = frames[scope]
        x = frame[F.OUTFIELD_CORE].to_numpy(dtype=float)
        tag = f"{season}|{scope}"
        spaces, pca = build_spaces(x, tag)
        result["pca"][scope] = pca
        result["projections"][scope] = spaces["pca90"]

        per_space: dict[str, dict] = {}
        for space in SPACES:
            curves = internal_curves(spaces[space], f"{tag}|{space}")
            decision = apply_k_rule(curves)
            per_space[space] = {"curves": curves, "rule": decision}

        rule_k = per_space[PRIMARY_SPACE]["rule"]["chosen_k"]
        k = int(fixed_k[scope]) if fixed_k else rule_k

        labels = {}
        centroids = {}
        for space in SPACES:
            fit = _kmeans(spaces[space], k, f"{tag}|{space}")
            labels[space] = fit.labels_
            centroids[space] = fit.cluster_centers_

        primary_labels = labels[PRIMARY_SPACE]
        if reference_centroids is not None and scope in reference_centroids:
            mapping = _align_labels(centroids[PRIMARY_SPACE], reference_centroids[scope])
            primary_labels = np.array([mapping[int(v)] for v in primary_labels])
            centroids[PRIMARY_SPACE] = centroids[PRIMARY_SPACE][
                np.argsort([mapping[i] for i in range(k)])
            ]
        else:
            mapping = None

        # Renumber so cluster 0 is the largest, which keeps figures readable.
        if mapping is None:
            order = np.argsort(-np.bincount(primary_labels, minlength=k))
            remap = {int(old): int(new) for new, old in enumerate(order)}
            primary_labels = np.array([remap[int(v)] for v in primary_labels])
            centroids[PRIMARY_SPACE] = centroids[PRIMARY_SPACE][order]

        result["labels"][scope] = primary_labels
        result["centroids"][scope] = centroids[PRIMARY_SPACE]

        selection_block = {
            "n_players": int(x.shape[0]),
            "standardisation": "z_global" if scope == "global" else "z_bygroup",
            "chosen_k": int(k),
            "k_from_declared_rule": int(rule_k),
            "k_source": "fixed_from_primary_season" if fixed_k else "declared_rule",
            "rule_branch": per_space[PRIMARY_SPACE]["rule"]["rule_branch"],
            "reported_space": PRIMARY_SPACE,
            "spaces": {
                space: {
                    "curves": {
                        name: {int(kk): vv for kk, vv in curve.items()}
                        for name, curve in per_space[space]["curves"].items()
                    },
                    "rule": per_space[space]["rule"],
                }
                for space in SPACES
            },
            "cluster_sizes": {
                str(c): int((primary_labels == c).sum()) for c in sorted(set(primary_labels))
            },
        }
        result["selection"][scope] = selection_block

        algo_block = {
            "reported_space": PRIMARY_SPACE,
            "pca90": {
                "n_components": int(pca.n_components_),
                "explained_variance": _r(float(pca.explained_variance_ratio_.sum()), 4),
            },
            "kmeans": {
                "k": int(k),
                "silhouette": {
                    space: per_space[space]["curves"]["silhouette"][k] for space in SPACES
                },
                "ari_standardised_vs_pca90": _r(
                    adjusted_rand_score(labels["standardised"], labels["pca90"]), 4
                ),
            },
            "gmm_bic": {space: gmm_bic_curve(spaces[space], f"{tag}|{space}") for space in SPACES},
            "hdbscan": {space: hdbscan_report(spaces[space]) for space in SPACES},
        }
        result["algorithms"][scope] = algo_block

        stability = bootstrap_stability(spaces[PRIMARY_SPACE], primary_labels, k, tag)
        result["stability"][scope] = stability

        result["position"][scope] = cluster_vs_position(frame, primary_labels)

        print(
            f"  {scope:<6} n={x.shape[0]:>3}  k={k} (rule would pick {rule_k})"
            f"  sil={per_space[PRIMARY_SPACE]['curves']['silhouette'][k]:.3f}"
            f"  bootstrap ARI={stability['ari_mean']:.3f}"
            f"  low-stability clusters={stability['low_stability_clusters']}"
        )

    # Global Gaussian mixture posteriors, in the declared space.
    global_frame = frames["global"]
    x_global = global_frame[F.OUTFIELD_CORE].to_numpy(dtype=float)
    projection = result["projections"]["global"]
    k_global = result["selection"]["global"]["chosen_k"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mixture = GaussianMixture(
            n_components=k_global,
            covariance_type=GMM_POSTERIOR_COVARIANCE,
            n_init=GMM_N_INIT,
            random_state=_seed(season, "gmm-posterior", k_global),
        ).fit(projection)
    posterior = mixture.predict_proba(projection)
    result["gmm"] = {
        "space": GMM_POSTERIOR_SPACE,
        "covariance_type": GMM_POSTERIOR_COVARIANCE,
        "k": int(k_global),
        "bic": _r(mixture.bic(projection), 2),
        "max_posterior": posterior.max(axis=1),
        "labels": posterior.argmax(axis=1),
        "ari_vs_kmeans": _r(
            adjusted_rand_score(result["labels"]["global"], posterior.argmax(axis=1)), 4
        ),
        "component_sizes": {
            str(c): int((posterior.argmax(axis=1) == c).sum()) for c in range(k_global)
        },
        "vs_position_group": {
            "adjusted_rand_index": _r(
                adjusted_rand_score(global_frame["position_group"], posterior.argmax(axis=1)), 4
            ),
            "normalized_mutual_info": _r(
                normalized_mutual_info_score(
                    global_frame["position_group"], posterior.argmax(axis=1)
                ),
                4,
            ),
        },
    }
    result["x_global"] = x_global
    return result


def _align_labels(centroids: np.ndarray, reference: np.ndarray) -> dict[int, int]:
    """Map this season's cluster ids onto the reference season's by centroid distance."""
    cost = np.linalg.norm(centroids[:, None, :] - reference[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols, strict=True)}


def build_cluster_frame(result: dict, season: str) -> pd.DataFrame:
    """Assemble the parquet that every downstream phase reads."""
    frame = result["frames"]["global"]
    out = pd.DataFrame(
        {
            "season": season,
            "player": frame["player"].to_numpy(),
            "team": frame["team"].to_numpy(),
            "position_group": frame["position_group"].to_numpy(),
            "position_full": frame["position_full"].to_numpy(),
            "minutes": frame["minutes"].to_numpy(),
            "cluster_global": result["labels"]["global"].astype("int32"),
            "gmm_max_posterior": result["gmm"]["max_posterior"].astype(float),
            "gmm_cluster_global": result["gmm"]["labels"].astype("int32"),
        }
    )
    within = pd.Series(index=out.index, dtype="float64")
    for group in config.OUTFIELD_GROUPS:
        sub = result["frames"][group]
        lookup = dict(zip(sub["player"].to_numpy(), result["labels"][group], strict=True))
        mask = out["position_group"] == group
        within.loc[mask] = out.loc[mask, "player"].map(lookup).astype(float)
    if within.isna().any():
        raise ValueError("a player received no within-group cluster")
    out["cluster_within_group"] = within.astype("int32")
    out["cluster_within_group_label"] = (
        out["position_group"].astype(str) + "-" + out["cluster_within_group"].astype(str)
    )
    return out[
        [
            "season",
            "player",
            "team",
            "position_group",
            "position_full",
            "minutes",
            "cluster_global",
            "cluster_within_group",
            "cluster_within_group_label",
            "gmm_max_posterior",
            "gmm_cluster_global",
        ]
    ]


def hybrid_players(result: dict, clusters: pd.DataFrame, season: str) -> dict:
    """Players the clustering and the team sheet disagree about.

    Two criteria are combined. A player disagrees when his listed position group is not
    the dominant listed group of the cluster he was assigned to. A player is split when
    his Gaussian mixture posterior never reaches ``POSTERIOR_SPLIT_THRESHOLD``. The table
    is ordered by tier, both criteria first, then split only, then disagreement only, and
    within a tier by ascending maximum posterior.
    """
    eligible = result["eligible"].set_index("player")
    dominant = result["position"]["global"]["position_group"]["dominant_label_per_cluster"]
    table = clusters.copy()
    table["cluster_dominant_group"] = (
        table["cluster_global"].astype(str).map(lambda c: dominant[c]["label"])
    )
    table["disagrees_with_listed_group"] = (
        table["cluster_dominant_group"] != table["position_group"]
    )
    table["posterior_split"] = table["gmm_max_posterior"] < POSTERIOR_SPLIT_THRESHOLD
    flagged = table[table["disagrees_with_listed_group"] | table["posterior_split"]].copy()
    flagged["tier"] = np.where(
        flagged["disagrees_with_listed_group"] & flagged["posterior_split"],
        0,
        np.where(flagged["posterior_split"], 1, 2),
    )
    flagged = flagged.sort_values(
        ["tier", "gmm_max_posterior", "player"], ascending=[True, True, True]
    )

    rows = []
    for _, row in flagged.head(HYBRID_TABLE_N).iterrows():
        stats = eligible.loc[row["player"]]
        rows.append(
            {
                "player": row["player"],
                "team": row["team"],
                "listed_position_group": row["position_group"],
                "listed_position_full": row["position_full"],
                "minutes": int(stats["minutes"]),
                "age": None if pd.isna(stats["age"]) else float(stats["age"]),
                "cluster_global": int(row["cluster_global"]),
                "cluster_dominant_group": row["cluster_dominant_group"],
                "cluster_within_group": row["cluster_within_group_label"],
                "gmm_max_posterior": _r(row["gmm_max_posterior"], 4),
                "disagrees_with_listed_group": bool(row["disagrees_with_listed_group"]),
                "posterior_split": bool(row["posterior_split"]),
                "stats_per90": {c: _r(float(stats[c]), 3) for c in HYBRID_STAT_COLS},
            }
        )

    return {
        "season": season,
        "definition": {
            "disagreement": (
                "listed position group differs from the dominant listed group of the "
                "assigned global cluster"
            ),
            "posterior_split": f"maximum GMM posterior below {POSTERIOR_SPLIT_THRESHOLD}",
            "posterior_source": (
                f"{GMM_POSTERIOR_COVARIANCE} covariance mixture in the "
                f"{GMM_POSTERIOR_SPACE} space at k={result['gmm']['k']}"
            ),
            "ordering": "tier (both, split only, disagreement only), then posterior ascending",
        },
        "disagreement_by_cluster_and_group": {
            str(int(cluster)): {
                "dominant_listed_group": dominant[str(int(cluster))]["label"],
                "listed_group_counts": (
                    table[table["cluster_global"] == cluster]["position_group"]
                    .value_counts()
                    .to_dict()
                ),
            }
            for cluster in sorted(table["cluster_global"].unique())
        },
        "counts": {
            "n_players": int(len(table)),
            "n_disagreeing": int(table["disagrees_with_listed_group"].sum()),
            "n_posterior_split": int(table["posterior_split"].sum()),
            "n_both": int((table["disagrees_with_listed_group"] & table["posterior_split"]).sum()),
            "n_flagged": int(len(flagged)),
        },
        "top_players": rows,
    }


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    plotting.use_style()

    primary = analyse_season(config.SEASON_PRIMARY)
    fixed_k = {scope: primary["selection"][scope]["chosen_k"] for scope in SCOPES}
    replication = analyse_season(
        config.SEASON_REPLICATION,
        fixed_k=fixed_k,
        reference_centroids=primary["centroids"],
    )
    seasons = {config.SEASON_PRIMARY: primary, config.SEASON_REPLICATION: replication}

    # ---------------------------------------------------------------- parquet output
    for season, result in seasons.items():
        frame = build_cluster_frame(result, season)
        path = config.DATA_PROCESSED / f"clusters_{season}.parquet"
        frame.to_parquet(path, index=False)
        print(f"  wrote {path.relative_to(config.ROOT)}  {frame.shape}")

    # ---------------------------------------------------------------- metrics output
    _write_json(
        {
            "rule": config.K_RULE,
            "k_range": config.K_RANGE,
            "reported_space": PRIMARY_SPACE,
            "space_definitions": {
                "standardised": "the 18 OUTFIELD_CORE z-scores, global or within group",
                "pca90": f"PCA retaining {config.PCA_VARIANCE_TARGET:.0%} of the variance",
            },
            "gap_statistic": {
                "reference": "uniform over the axis-aligned bounding box of the fitted space",
                "b": config.GAP_STATISTIC_B,
                "dispersion": "KMeans within-cluster sum of squares",
            },
            "seasons": {
                season: {scope: result["selection"][scope] for scope in SCOPES}
                for season, result in seasons.items()
            },
        },
        "cluster_selection.json",
    )

    _write_json(
        {
            "n_bootstrap_requested": config.BOOTSTRAP_N,
            "space": PRIMARY_SPACE,
            "low_stability_threshold": LOW_STABILITY_THRESHOLD,
            "note": (
                "ARI is computed against the reference clustering on the players drawn "
                "into each resample. Co-assignment is the fraction of resamples "
                "containing both players in which they share a cluster."
            ),
            "seasons": {
                season: {
                    scope: {
                        key: value
                        for key, value in result["stability"][scope].items()
                        if key != "coassignment"
                    }
                    for scope in SCOPES
                }
                for season, result in seasons.items()
            },
        },
        "cluster_stability.json",
    )

    _write_json(
        {
            "question": (
                "Do data-driven clusters recover listed positions, and where do they disagree?"
            ),
            "space": PRIMARY_SPACE,
            "seasons": {
                season: {scope: result["position"][scope] for scope in SCOPES}
                for season, result in seasons.items()
            },
            "answer": _position_answer(primary),
        },
        "cluster_vs_position.json",
    )

    _write_json(
        {
            "declared_in_advance": {
                "primary_space": PRIMARY_SPACE,
                "reason": (
                    "the reported clustering is fitted on the standardised features so "
                    "that it does not depend on a variance cutoff; the pca90 fit is a "
                    "sensitivity check and its agreement with the reported solution is "
                    "given as ari_standardised_vs_pca90"
                ),
                "gmm_posterior_space": GMM_POSTERIOR_SPACE,
                "gmm_posterior_covariance": GMM_POSTERIOR_COVARIANCE,
                "gmm_posterior_note": (
                    "a full covariance mixture on 18 correlated features is not "
                    "estimable at these sample sizes, so posteriors are taken in the "
                    "PCA space and only for the global scope"
                ),
                "hdbscan_min_cluster_sizes": HDBSCAN_MIN_CLUSTER_SIZES,
            },
            "seasons": {
                season: {
                    "scopes": {scope: result["algorithms"][scope] for scope in SCOPES},
                    "gmm_posterior_fit": {
                        key: value
                        for key, value in result["gmm"].items()
                        if key not in {"max_posterior", "labels"}
                    },
                }
                for season, result in seasons.items()
            },
        },
        "cluster_algorithms.json",
    )

    hybrids = {
        season: hybrid_players(result, build_cluster_frame(result, season), season)
        for season, result in seasons.items()
    }
    _write_json(
        {"primary_season": config.SEASON_PRIMARY, "seasons": hybrids},
        "hybrid_players.json",
    )

    # ---------------------------------------------------------------- figures
    figure_k_selection(primary["selection"], config.SEASON_PRIMARY)
    figure_consensus(primary["stability"], primary["labels"])
    figure_cluster_vs_position(
        primary["frames"]["global"],
        primary["labels"]["global"],
        primary["position"]["global"],
    )
    figure_cluster_scatter(
        primary["projections"]["global"],
        primary["pca"]["global"],
        primary["labels"]["global"],
        primary["frames"]["global"]["position_group"],
    )
    print("  wrote 4 figures to figures/")

    _print_summary(primary, hybrids[config.SEASON_PRIMARY])


def _position_answer(primary: dict) -> dict:
    """A plain answer to the recovery question, stored so the paper can quote it."""
    global_stats = primary["position"]["global"]["position_group"]
    ari = global_stats["adjusted_rand_index"]
    purity = global_stats["cluster_purity"]
    k = primary["selection"]["global"]["chosen_k"]
    recovers = bool(ari >= RECOVERY_ARI_THRESHOLD)
    return {
        "global_k": k,
        "adjusted_rand_index_vs_position_group": ari,
        "cluster_purity_vs_position_group": purity,
        "recovery_threshold_ari": RECOVERY_ARI_THRESHOLD,
        "recovery_threshold_note": (
            "a declared reporting convention, not a test: an adjusted Rand index of at "
            "least this value is called substantial recovery"
        ),
        "clusters_recover_listed_positions": recovers,
        "statement": (
            f"At k={k} the global clustering scores an adjusted Rand index of {ari:.3f} "
            f"against the listed position group, with a cluster purity of {purity:.3f}. "
            + (
                "That is a substantial recovery of the listed positions."
                if recovers
                else "That is weak agreement: the partition the data supports is not the "
                "listed position partition, it is a coarser attacking against defending "
                "split that cuts across the midfield."
            )
        ),
    }


def _print_summary(primary: dict, hybrids: dict) -> None:
    print("\n--- summary, primary season ---")
    for scope in SCOPES:
        sel = primary["selection"][scope]
        stab = primary["stability"][scope]
        fav = sel["spaces"][PRIMARY_SPACE]["rule"]["favoured_by_criterion"]
        print(
            f"{scope:<6} k={sel['chosen_k']} via {sel['rule_branch']:<22}"
            f" favoured: sil={fav['silhouette']} db={fav['davies_bouldin']}"
            f" ch={fav['calinski_harabasz']} gap={fav['gap_max']}"
            f" gap1se={fav['gap_tibshirani_1se']}"
            f" | ARI {stab['ari_mean']:.3f} +/- {stab['ari_sd']:.3f}"
            f" | low stability {stab['low_stability_clusters']}"
        )
    pos = primary["position"]["global"]
    print(
        f"global vs position_group: ARI {pos['position_group']['adjusted_rand_index']:.3f},"
        f" NMI {pos['position_group']['normalized_mutual_info']:.3f}"
    )
    print(
        f"global vs position_full:  ARI {pos['position_full']['adjusted_rand_index']:.3f},"
        f" NMI {pos['position_full']['normalized_mutual_info']:.3f}"
    )
    print(f"hybrid counts: {hybrids['counts']}")


if __name__ == "__main__":
    main()
