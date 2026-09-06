"""Phase 4: dimensionality reduction, PCA, UMAP and t-SNE with a numeric comparison.

The eighteen surviving per-90 features are reduced to two dimensions three different
ways so that the clustering phase can be run on a representation whose distortion is
measured rather than assumed.

* PCA is linear, deterministic and interpretable through its loadings, so it is the
  reference embedding and the one whose axes are given a reading in the paper.
* UMAP is swept over the twelve hyperparameter combinations in ``config.UMAP_GRID`` and
  scored by trustworthiness, so the configuration used downstream is selected by a
  measurement rather than by which picture looked best.
* t-SNE is computed at perplexity 30 for visual contrast only. Its inter-cluster
  distances and cluster sizes carry no meaning, so it is never used for clustering,
  never used to select k and never used to assign an archetype. The saved metrics repeat
  that statement in the ``used_for_clustering`` field.

A global embedding of the whole outfield pool is dominated by the defender to forward
axis, which is the one thing already known from the listed positions. PCA is therefore
also fit separately inside DF, MF and FW on the within-group standardised features, so
that the variation that survives after position is held constant becomes visible. Both
the global and the position-stratified embeddings are written to
``data/processed/embeddings_{season}.parquet`` for the clustering and profiling phases.

Outputs
-------
results/metrics/pca_loadings.json
results/metrics/umap_grid.json
results/metrics/embedding_comparison.json
results/metrics/pca_by_position.json
data/processed/embeddings_{season}.parquet
figures/pca_scree.{pdf,png}
figures/pca_biplot.{pdf,png}
figures/embedding_comparison.{pdf,png}
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness

import config
from src import features as F
from src import plotting
from src.descriptive import label

#: Neighbourhood size shared by trustworthiness, continuity and the position-purity
#: score, so that the three numbers in the comparison table describe the same scale of
#: local structure.
NEIGHBOURS = config.KNN_PURITY_K

#: Recorded in every metrics file that mentions t-SNE.
TSNE_ROLE = "visual contrast only, not used for clustering, k selection or archetype assignment"


def load_z(season: str, scope: str) -> pd.DataFrame:
    """Load a standardised outfield table. ``scope`` is 'global' or 'bygroup'."""
    return pd.read_parquet(config.DATA_PROCESSED / f"outfield_z_{scope}_{season}.parquet")


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[F.OUTFIELD_CORE].to_numpy(dtype=float)


# --------------------------------------------------------------------------------------
# Embedding quality
# --------------------------------------------------------------------------------------


def _neighbour_ranks(x: np.ndarray) -> np.ndarray:
    """Rank of every point as a neighbour of every other point. Self has rank 0."""
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    order = np.argsort(dist, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(x.shape[0])[:, None]
    ranks[rows, order] = np.arange(x.shape[0])[None, :]
    return ranks


def continuity(x_high: np.ndarray, x_low: np.ndarray, n_neighbors: int = NEIGHBOURS) -> float:
    """Continuity of an embedding, the mirror image of trustworthiness.

    Trustworthiness penalises points that the embedding places close together although
    they are far apart in the original space. Continuity penalises the opposite failure:
    points that were genuine neighbours in the original space but which the embedding
    has pushed apart. Formally it is trustworthiness with the roles of the two spaces
    swapped, so ``continuity(high, low, k) == trustworthiness(low, high, k)``.

    With ``U_i`` the set of points among the ``k`` nearest neighbours of ``i`` in the
    original space but not among its ``k`` nearest in the embedding, and ``r_low(i, j)``
    the rank of ``j`` around ``i`` in the embedding,

        C(k) = 1 - 2 / (n k (2n - 3k - 1)) * sum_i sum_{j in U_i} (r_low(i, j) - k)

    It is computed directly here rather than by calling the trustworthiness routine
    backwards, so the definition used is visible in the source. The equivalence is
    asserted once in ``_selfcheck_continuity``.
    """
    n = x_high.shape[0]
    k = int(n_neighbors)
    if k >= n / 2:
        raise ValueError(f"n_neighbors={k} must be less than n/2={n / 2}")
    rank_high = _neighbour_ranks(x_high)
    rank_low = _neighbour_ranks(x_low)
    in_high_knn = (rank_high >= 1) & (rank_high <= k)
    outside_low_knn = rank_low > k
    penalty = np.where(in_high_knn & outside_low_knn, rank_low - k, 0).sum()
    return float(1.0 - (2.0 / (n * k * (2 * n - 3 * k - 1))) * penalty)


def _selfcheck_continuity(x_high: np.ndarray, x_low: np.ndarray) -> None:
    """Confirm the direct implementation matches the swapped-space definition."""
    mine = continuity(x_high, x_low, NEIGHBOURS)
    swapped = float(trustworthiness(x_low, x_high, n_neighbors=NEIGHBOURS))
    if not np.isclose(mine, swapped, atol=1e-10):
        raise AssertionError(f"continuity {mine} does not match swapped trustworthiness {swapped}")


def knn_position_purity(emb: np.ndarray, groups: np.ndarray, k: int = NEIGHBOURS) -> float:
    """Share of each point's k nearest embedding neighbours sharing its position group."""
    ranks = _neighbour_ranks(emb)
    neighbour_mask = (ranks >= 1) & (ranks <= k)
    same = (groups[None, :] == groups[:, None]) & neighbour_mask
    return float(same.sum(axis=1).mean() / k)


def purity_baseline(groups: np.ndarray) -> float:
    """Purity a random embedding would reach, the sum of squared group shares."""
    _, counts = np.unique(groups, return_counts=True)
    shares = counts / counts.sum()
    return float((shares**2).sum())


def score_embedding(x: np.ndarray, emb: np.ndarray, groups: np.ndarray) -> dict:
    return {
        "trustworthiness": float(trustworthiness(x, emb, n_neighbors=NEIGHBOURS)),
        "continuity": continuity(x, emb, NEIGHBOURS),
        "knn_position_purity": knn_position_purity(emb, groups, NEIGHBOURS),
        "k": NEIGHBOURS,
    }


# --------------------------------------------------------------------------------------
# PCA
# --------------------------------------------------------------------------------------


def fit_pca(x: np.ndarray, n_components: int | None = None) -> PCA:
    model = PCA(n_components=n_components, svd_solver="full", random_state=config.RANDOM_STATE)
    model.fit(x)
    return model


def pca_report(model: PCA, feats: list[str], n_pcs: int = 5) -> dict:
    evr = model.explained_variance_ratio_
    cum = np.cumsum(evr)
    n_pcs = min(n_pcs, model.n_components_)
    loadings = {}
    for i in range(n_pcs):
        vec = model.components_[i]
        order = np.argsort(-np.abs(vec))
        loadings[f"PC{i + 1}"] = {
            "explained_variance_ratio": float(evr[i]),
            "cumulative": float(cum[i]),
            "loadings": {feats[j]: float(vec[j]) for j in range(len(feats))},
            "top_positive": [
                {"feature": feats[j], "label": label(feats[j]), "loading": float(vec[j])}
                for j in order
                if vec[j] > 0
            ][:5],
            "top_negative": [
                {"feature": feats[j], "label": label(feats[j]), "loading": float(vec[j])}
                for j in order
                if vec[j] < 0
            ][:5],
        }
    return {
        "n_features": len(feats),
        "explained_variance_ratio": [float(v) for v in evr],
        "cumulative_explained_variance": [float(v) for v in cum],
        "n_components_for_90pct": int(np.searchsorted(cum, config.PCA_VARIANCE_TARGET) + 1),
        "variance_target": config.PCA_VARIANCE_TARGET,
        "components": loadings,
    }


# --------------------------------------------------------------------------------------
# UMAP
# --------------------------------------------------------------------------------------


def fit_umap(x: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_neighbors=int(n_neighbors),
            min_dist=float(min_dist),
            n_components=2,
            metric="euclidean",
            random_state=config.RANDOM_STATE,
        )
        return np.asarray(reducer.fit_transform(x), dtype=float)


def umap_sweep(x: np.ndarray, groups: np.ndarray) -> tuple[dict, dict]:
    """Score every ``config.UMAP_GRID`` combination and pick the best trustworthiness.

    Returns the report and a map from ``(n_neighbors, min_dist)`` to the embedding, so
    the selected configuration is plotted from the very array that was scored rather
    than from a second fit.
    """
    results = []
    embeddings: dict[tuple[int, float], np.ndarray] = {}
    for n_neighbors in config.UMAP_GRID["n_neighbors"]:
        for min_dist in config.UMAP_GRID["min_dist"]:
            emb = fit_umap(x, n_neighbors, min_dist)
            embeddings[(int(n_neighbors), float(min_dist))] = emb
            scores = score_embedding(x, emb, groups)
            results.append({"n_neighbors": int(n_neighbors), "min_dist": float(min_dist), **scores})
            print(
                f"  umap n_neighbors={n_neighbors:>2} min_dist={min_dist:.1f} "
                f"trust={scores['trustworthiness']:.4f} cont={scores['continuity']:.4f} "
                f"purity={scores['knn_position_purity']:.4f}",
                flush=True,
            )
    best = max(results, key=lambda r: r["trustworthiness"])
    trusts = np.array([r["trustworthiness"] for r in results])
    report = {
        "grid": config.UMAP_GRID,
        "n_combinations": len(results),
        "selection_criterion": (
            f"highest trustworthiness at k={NEIGHBOURS} against the standardised 18-feature space"
        ),
        "random_state": config.RANDOM_STATE,
        "results": results,
        "best": {"n_neighbors": best["n_neighbors"], "min_dist": best["min_dist"]},
        "best_trustworthiness": best["trustworthiness"],
        "trustworthiness_range": [float(trusts.min()), float(trusts.max())],
        "trustworthiness_spread": float(trusts.max() - trusts.min()),
    }
    return report, embeddings


def fit_tsne(x: np.ndarray) -> np.ndarray:
    model = TSNE(
        n_components=2,
        perplexity=config.TSNE_PERPLEXITY,
        init="pca",
        learning_rate="auto",
        random_state=config.RANDOM_STATE,
    )
    return np.asarray(model.fit_transform(x), dtype=float)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def figure_scree(report: dict, season: str) -> None:
    evr = np.array(report["explained_variance_ratio"])
    cum = np.array(report["cumulative_explained_variance"])
    idx = np.arange(1, len(evr) + 1)

    fig, ax = plotting.plt.subplots(figsize=(plotting.WIDTH_COLUMN, 3.2))
    ax.bar(idx, evr * 100, color=plotting.CATEGORICAL[0], width=0.7, label="per component")
    ax.set_ylim(0, max(evr * 100) * 1.18)
    plotting.style_axis(
        ax,
        xlabel="Principal component",
        ylabel="Variance explained (%)",
        title=f"PCA scree, 18 features, {season}",
    )
    ax.set_xticks(idx[::2])

    ax2 = ax.twinx()
    ax2.plot(
        idx,
        cum * 100,
        color=plotting.CATEGORICAL[1],
        marker="s",
        markersize=3,
        lw=1.4,
        label="cumulative",
    )
    ax2.axhline(
        config.PCA_VARIANCE_TARGET * 100,
        color=plotting.INK_MUTED,
        lw=0.8,
        ls=(0, (3, 3)),
    )
    n90 = report["n_components_for_90pct"]
    ax2.annotate(
        f"{config.PCA_VARIANCE_TARGET:.0%} at {n90} components",
        (len(evr), config.PCA_VARIANCE_TARGET * 100),
        textcoords="offset points",
        xytext=(-4, 5),
        ha="right",
        fontsize=plotting.BASE_FONT_PT - 2,
        color=plotting.INK_SECONDARY,
    )
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative (%)", color=plotting.CATEGORICAL[1])
    ax2.tick_params(axis="y", colors=plotting.CATEGORICAL[1])
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(plotting.CATEGORICAL[1])
    ax2.grid(False)
    plotting.save_figure(fig, "pca_scree")


def figure_biplot(scores: np.ndarray, model: PCA, groups: np.ndarray, season: str) -> None:
    evr = model.explained_variance_ratio_
    fig, ax = plotting.plt.subplots(figsize=(plotting.WIDTH_FULL, 4.4))

    for group in config.OUTFIELD_GROUPS:
        m = groups == group
        ax.scatter(
            scores[m, 0],
            scores[m, 1],
            s=16,
            alpha=0.7,
            color=plotting.POSITION_COLORS[group],
            marker=plotting.POSITION_MARKERS[group],
            linewidth=0,
            label=f"{group} (n={int(m.sum())})",
            zorder=2,
        )

    # Loading arrows, scaled to the score cloud. Only the eight strongest are drawn,
    # because eighteen labelled arrows on one panel is unreadable. One scale factor is
    # used for both axes so the angles between arrows stay meaningful, and it is chosen
    # so the longest arrow fits inside whichever axis is tighter.
    load = model.components_[:2].T
    norms = np.linalg.norm(load, axis=1)
    keep = np.argsort(-norms)[:8]
    span_x = np.abs(scores[:, 0]).max() / np.abs(load[keep, 0]).max()
    span_y = np.abs(scores[:, 1]).max() / np.abs(load[keep, 1]).max()
    scale = 0.62 * min(span_x, span_y)

    plotting.style_axis(
        ax,
        xlabel=f"PC1 ({evr[0]:.1%} of variance)",
        ylabel=f"PC2 ({evr[1]:.1%})",
        title=f"PCA biplot of the surviving feature set, {season}",
    )
    ax.axhline(0, color=plotting.GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=plotting.GRID, lw=0.6, zorder=0)

    # Limits are fixed before the labels are placed, because the label positions are
    # resolved in axes fractions.
    xs = np.concatenate([scores[:, 0], load[keep, 0] * scale * 1.35])
    ys = np.concatenate([scores[:, 1], load[keep, 1] * scale * 1.35])
    ax.set_xlim(xs.min() - 0.05 * np.ptp(xs), xs.max() + 0.05 * np.ptp(xs))
    ax.set_ylim(ys.min() - 0.05 * np.ptp(ys), ys.max() + 0.08 * np.ptp(ys))

    for j in keep:
        ax.annotate(
            "",
            xy=(load[j, 0] * scale, load[j, 1] * scale),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "-|>", "color": plotting.INK_PRIMARY, "lw": 0.9},
            zorder=4,
        )
    _place_loading_labels(ax, load, keep, scale)

    ax.legend(loc="upper left", ncol=1)
    plotting.save_figure(fig, "pca_biplot")


def _place_loading_labels(ax, load: np.ndarray, keep: np.ndarray, scale: float) -> None:
    """Label loading arrows without letting near-parallel ones stack their text.

    xA, key passes and assists load almost identically on the first two components, so
    their arrow tips sit within a few points of each other and naive labelling produces
    overlapping words. The labels are therefore placed just beyond each tip and then
    pushed apart vertically, in axes fractions, with a thin leader line back to the tip
    whenever a label had to move. Arrows pointing left and arrows pointing right are
    separated independently so the two fans never interfere.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    min_sep = 0.055
    for sign in (1.0, -1.0):
        side = [j for j in keep if np.sign(load[j, 0]) in (sign, 0.0)]
        if not side:
            continue
        tips = [
            ((load[j, 0] * scale - x0) / (x1 - x0), (load[j, 1] * scale - y0) / (y1 - y0))
            for j in side
        ]
        order = np.argsort([-t[1] for t in tips])
        placed: dict[int, float] = {}
        previous = None
        for pos in order:
            y = tips[pos][1]
            if previous is not None and previous - y < min_sep:
                y = previous - min_sep
            placed[int(pos)] = float(np.clip(y, 0.02, 0.97))
            previous = placed[int(pos)]
        for pos, j in enumerate(side):
            tx, ty = tips[pos]
            ly = placed[pos]
            lx = tx + 0.012 * sign
            moved = abs(ly - ty) > 0.01
            ax.annotate(
                label(F.OUTFIELD_CORE[j]),
                xy=(tx, ty),
                xytext=(lx, ly),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=plotting.BASE_FONT_PT - 2,
                color=plotting.INK_PRIMARY,
                ha="left" if sign > 0 else "right",
                va="center",
                zorder=6,
                bbox={
                    "facecolor": plotting.SURFACE,
                    "edgecolor": "none",
                    "pad": 1.0,
                    "alpha": 0.85,
                },
                arrowprops=(
                    {
                        "arrowstyle": "-",
                        "color": plotting.INK_MUTED,
                        "lw": 0.5,
                        "shrinkA": 1,
                        "shrinkB": 1,
                    }
                    if moved
                    else None
                ),
            )


def figure_embedding_comparison(panels: list[dict], groups: np.ndarray, season: str) -> None:
    fig, axes = plotting.plt.subplots(
        1, 3, figsize=(plotting.WIDTH_FULL, 3.0), constrained_layout=True
    )
    for ax, panel in zip(axes, panels, strict=True):
        emb = panel["embedding"]
        for group in config.OUTFIELD_GROUPS:
            m = groups == group
            ax.scatter(
                emb[m, 0],
                emb[m, 1],
                s=9,
                alpha=0.75,
                color=plotting.POSITION_COLORS[group],
                marker=plotting.POSITION_MARKERS[group],
                linewidth=0,
                label=group,
            )
        s = panel["scores"]
        ax.set_title(panel["title"], loc="left")
        ax.set_xlabel(
            f"T {s['trustworthiness']:.2f}  C {s['continuity']:.2f}  "
            f"purity {s['knn_position_purity']:.2f}",
            fontsize=plotting.BASE_FONT_PT - 1,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0.07)
        ax.grid(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="outside lower center", ncol=3, markerscale=1.8)
    fig.suptitle(
        f"Two-dimensional embeddings of the same 18 features, {season}\n"
        f"T trustworthiness, C continuity, purity at k={NEIGHBOURS}",
        fontsize=plotting.BASE_FONT_PT,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    plotting.save_figure(fig, "embedding_comparison")


# --------------------------------------------------------------------------------------
# Position-stratified reduction
# --------------------------------------------------------------------------------------


def stratified_reduction(season: str, umap_params: dict) -> tuple[dict, pd.DataFrame]:
    """Fit PCA and UMAP inside each position group on within-group standardised data.

    The global embedding is dominated by the defender to forward contrast, which merely
    reproduces the listed positions. Refitting inside each group removes that contrast by
    construction and exposes the variation that remains once position is held constant.
    """
    zb = load_z(season, "bygroup")
    report: dict = {"umap_params_requested": dict(umap_params), "groups": {}}
    coords = np.full((len(zb), 4), np.nan)

    for group in config.OUTFIELD_GROUPS:
        mask = (zb["position_group"] == group).to_numpy()
        sub = zb.loc[mask]
        x = feature_matrix(sub)
        model = fit_pca(x)
        scores = model.transform(x)

        # n_neighbors cannot exceed the group size. FW is the smallest group and is the
        # only one where this bites, so the applied value is recorded per group.
        n_nb = int(min(umap_params["n_neighbors"], len(sub) - 1))
        emb = fit_umap(x, n_nb, umap_params["min_dist"])

        rep = pca_report(model, F.OUTFIELD_CORE)
        rep["n_players"] = int(len(sub))
        rep["umap_n_neighbors_applied"] = n_nb
        rep["umap_n_neighbors_clamped"] = bool(n_nb != umap_params["n_neighbors"])
        rep["umap"] = score_embedding(x, emb, sub["position_group"].to_numpy())
        rep["umap"].pop("knn_position_purity")  # single group, purity is 1 by definition
        rep["pca2d"] = {
            "trustworthiness": float(trustworthiness(x, scores[:, :2], n_neighbors=NEIGHBOURS)),
            "continuity": continuity(x, scores[:, :2], NEIGHBOURS),
        }
        report["groups"][group] = rep
        coords[mask, 0:2] = scores[:, :2]
        coords[mask, 2:4] = emb

    frame = pd.DataFrame(
        {
            "player": zb["player"].to_numpy(),
            "team": zb["team"].to_numpy(),
            "position_group": zb["position_group"].to_numpy(),
            "pca1": coords[:, 0],
            "pca2": coords[:, 1],
            "umap1": coords[:, 2],
            "umap2": coords[:, 3],
        }
    )
    return report, frame


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def run_primary(season: str) -> tuple[dict, dict, dict, np.ndarray]:
    zg = load_z(season, "global")
    x = feature_matrix(zg)
    groups = zg["position_group"].to_numpy()

    model = fit_pca(x)
    scores = model.transform(x)
    loadings = pca_report(model, F.OUTFIELD_CORE)
    loadings["season"] = season

    figure_scree(loadings, season)
    figure_biplot(scores, model, groups, season)

    _selfcheck_continuity(x, scores[:, :2])

    print(f"{season}: UMAP sweep over {len(config.UMAP_GRID['n_neighbors']) * 3} combinations")
    sweep, sweep_embeddings = umap_sweep(x, groups)
    sweep["season"] = season

    best_emb = sweep_embeddings[(sweep["best"]["n_neighbors"], sweep["best"]["min_dist"])]
    tsne_emb = fit_tsne(x)

    comparison = {
        "season": season,
        "n_players": int(len(zg)),
        "k": NEIGHBOURS,
        "purity_baseline_random": purity_baseline(groups),
        "purity_note": (
            "Position purity is the share of a player's k nearest embedding neighbours "
            "sharing his listed position group. The baseline is what a random embedding "
            "reaches given the group sizes."
        ),
        "embeddings": {
            "pca": {**score_embedding(x, scores[:, :2], groups), "used_for_clustering": True},
            "umap": {
                **score_embedding(x, best_emb, groups),
                "n_neighbors": sweep["best"]["n_neighbors"],
                "min_dist": sweep["best"]["min_dist"],
                "used_for_clustering": True,
            },
            "tsne": {
                **score_embedding(x, tsne_emb, groups),
                "perplexity": config.TSNE_PERPLEXITY,
                "used_for_clustering": False,
                "role": TSNE_ROLE,
            },
        },
    }

    figure_embedding_comparison(
        [
            {
                "embedding": scores[:, :2],
                "scores": comparison["embeddings"]["pca"],
                "title": "PCA",
            },
            {
                "embedding": best_emb,
                "scores": comparison["embeddings"]["umap"],
                "title": (
                    f"UMAP (n={sweep['best']['n_neighbors']}, d={sweep['best']['min_dist']:g})"
                ),
            },
            {
                "embedding": tsne_emb,
                "scores": comparison["embeddings"]["tsne"],
                "title": f"t-SNE (perp {config.TSNE_PERPLEXITY}, not clustered)",
            },
        ],
        groups,
        season,
    )
    return loadings, sweep, comparison, best_emb


def run_replication(season: str, umap_params: dict) -> tuple[dict, dict, np.ndarray]:
    """Same pipeline on the replication season, reusing the primary season's UMAP choice."""
    zg = load_z(season, "global")
    x = feature_matrix(zg)
    groups = zg["position_group"].to_numpy()

    model = fit_pca(x)
    scores = model.transform(x)
    loadings = pca_report(model, F.OUTFIELD_CORE)
    loadings["season"] = season

    best_emb = fit_umap(x, umap_params["n_neighbors"], umap_params["min_dist"])
    tsne_emb = fit_tsne(x)
    comparison = {
        "season": season,
        "n_players": int(len(zg)),
        "k": NEIGHBOURS,
        "purity_baseline_random": purity_baseline(groups),
        "umap_params_source": "selected on the primary season, not re-tuned here",
        "embeddings": {
            "pca": {**score_embedding(x, scores[:, :2], groups), "used_for_clustering": True},
            "umap": {
                **score_embedding(x, best_emb, groups),
                **umap_params,
                "used_for_clustering": True,
            },
            "tsne": {
                **score_embedding(x, tsne_emb, groups),
                "perplexity": config.TSNE_PERPLEXITY,
                "used_for_clustering": False,
                "role": TSNE_ROLE,
            },
        },
    }
    return loadings, comparison, best_emb


def main() -> None:
    np.random.seed(config.RANDOM_STATE)
    plotting.use_style()

    loadings: dict = {}
    comparisons: dict = {}
    by_position: dict = {}

    primary = config.SEASON_PRIMARY
    load_primary, sweep, comp_primary, umap_primary = run_primary(primary)
    loadings[primary] = load_primary
    comparisons[primary] = comp_primary
    best = sweep["best"]

    evr = load_primary["explained_variance_ratio"]
    print(
        f"{primary}: PC1 {evr[0]:.1%} PC2 {evr[1]:.1%} PC3 {evr[2]:.1%} "
        f"cumulative3 {sum(evr[:3]):.1%} "
        f"90% at {load_primary['n_components_for_90pct']} components",
        flush=True,
    )
    print(
        f"{primary}: best UMAP n_neighbors={best['n_neighbors']} "
        f"min_dist={best['min_dist']:g} trustworthiness={sweep['best_trustworthiness']:.4f}",
        flush=True,
    )

    global_umap = {primary: umap_primary}
    for season in config.SEASONS:
        if season != primary:
            load_s, comp_s, umap_s = run_replication(season, best)
            loadings[season] = load_s
            comparisons[season] = comp_s
            global_umap[season] = umap_s
            e = load_s["explained_variance_ratio"]
            print(
                f"{season}: PC1 {e[0]:.1%} PC2 {e[1]:.1%} PC3 {e[2]:.1%} "
                f"cumulative3 {sum(e[:3]):.1%}",
                flush=True,
            )

    for season in config.SEASONS:
        strat, frame = stratified_reduction(season, best)
        strat["season"] = season
        by_position[season] = strat

        # The global and by-group standardised tables are written from the same eligible
        # frame in the same order, so the two embeddings can be joined positionally. The
        # check below fails loudly if that ever stops being true.
        zg = load_z(season, "global")
        if not np.array_equal(zg["player"].to_numpy(), frame["player"].to_numpy()):
            raise AssertionError(f"{season}: global and by-group tables are not row aligned")
        out = frame.copy()
        out["umap_global1"] = global_umap[season][:, 0]
        out["umap_global2"] = global_umap[season][:, 1]
        out.to_parquet(config.DATA_PROCESSED / f"embeddings_{season}.parquet", index=False)
        print(
            f"{season}: stratified PCA PC1 "
            + " ".join(
                f"{g}={strat['groups'][g]['explained_variance_ratio'][0]:.1%}"
                for g in config.OUTFIELD_GROUPS
            )
            + f" | embeddings rows={len(out)} "
            f"nulls={int(out.isna().sum().sum())}",
            flush=True,
        )

    for name, payload in (
        ("pca_loadings", loadings),
        ("umap_grid", {primary: sweep}),
        ("embedding_comparison", comparisons),
        ("pca_by_position", by_position),
    ):
        with open(config.METRICS / f"{name}.json", "w") as fh:
            json.dump(payload, fh, indent=2)

    print("reduce complete")


if __name__ == "__main__":
    main()
