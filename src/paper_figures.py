"""Composite figures for the main text.

The rest of the pipeline draws one figure per analysis, which is the right unit while the
analysis is being done and the wrong one for a reader. This module assembles the two main
text figures that no single analysis owns, each carrying one argument across its panels:

``main_geometry``
    The shape of the space. How many directions it has, what the leading two are made of,
    where the players sit along the axis the partition cuts, and how that axis relates to
    the positions the data already records, from the clustering and from a classifier that
    shares none of its machinery.

``main_modes``
    What the two modes are, one panel per position group, as the signed deviation of each
    archetype from its group mean on the features that separate them most.

Everything here reads from artefacts the analysis modules already wrote. Nothing is
recomputed except the pooled two-cluster fit, which is refitted from the same standardised
matrix under the same seed tag as :mod:`src.archive_validation`, so the axis drawn here is
the axis reported there rather than a second one that resembles it.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import archive as A
from src import cluster as C
from src import plotting as P

#: The number of clusters the paper reports, matching src.archive_validation.
K = 2

#: Features shown per loading panel and per archetype panel. Ten is what fits at nine
#: point type across a third of the text width without the labels colliding.
TOP_N = 10

CLASS_ORDER = ["DF", "MF", "FW"]
GROUP_LABELS = {"DF": "defenders", "MF": "midfielders", "FW": "forwards"}


def _metrics(name: str) -> dict:
    with open(config.METRICS / f"{name}.json") as fh:
        return json.load(fh)


def _label(feature: str) -> str:
    """Short display name for a feature, borrowed from the validation module's map."""
    from src.archive_validation import label

    return label(feature)


# --------------------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------------------


def _panel_scree(ax, pca: dict) -> None:
    ratio = np.array(pca["explained_variance_ratio"], dtype=float)
    cumulative = np.array(pca["cumulative_variance"], dtype=float)
    idx = np.arange(1, len(ratio) + 1)
    ax.bar(idx, ratio * 100, color=P.CONTEXT_GREY, width=0.8, linewidth=0)
    twin = ax.twinx()
    twin.plot(idx, cumulative * 100, color=P.CATEGORICAL[0], linewidth=1.6)
    twin.axhline(90, color=P.INK_MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    twin.set_ylim(0, 104)
    twin.set_ylabel("cumulative, %", color=P.CATEGORICAL[0])
    twin.tick_params(axis="y", labelcolor=P.CATEGORICAL[0])
    twin.grid(False)
    twin.spines["right"].set_visible(True)
    twin.spines["right"].set_color(P.CATEGORICAL[0])
    P.style_axis(ax, "component", "variance explained, %", "one dominant axis")
    ax.set_xlim(0.2, len(ratio) + 0.8)


def _panel_loadings(ax, pca: dict, component: str, header: str) -> None:
    loadings = pca["components"][component]["loadings"]
    items = sorted(loadings.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_N]
    items = sorted(items, key=lambda kv: kv[1])
    y = np.arange(len(items))
    values = [v for _, v in items]
    colours = [P.CATEGORICAL[0] if v >= 0 else P.CATEGORICAL[1] for v in values]
    ax.barh(y, values, color=colours, height=0.72, linewidth=0)
    ax.axvline(0, color=P.INK_SECONDARY, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([_label(f) for f, _ in items], fontsize=P.BASE_FONT_PT - 2)
    P.style_axis(ax, "loading", "", header)
    ax.grid(False, axis="y")


def _pooled_axis() -> tuple[pd.DataFrame, np.ndarray]:
    """Refit the pooled two-cluster solution and project every player onto its axis.

    The projection is signed so that the attacking mode sits on the right, and it is
    centred on the midpoint between the two centroids, so zero is the cut the partition
    makes and the units are standard deviations of the feature space along that direction.
    """
    z_global = pd.read_parquet(config.DATA_PROCESSED / "archive_z_global.parquet")
    z_global = z_global.reset_index(drop=True)
    features = [c for c in A.OUTFIELD_CORE if c in z_global.columns]
    x = z_global[features].to_numpy(float)
    fit = C._kmeans(x, K, "archive|All outfield")
    centres = fit.cluster_centers_

    # Orient the axis toward whichever centroid holds more forwards, so the sign of the
    # projection means the same thing on every run regardless of the label KMeans hands out.
    groups = z_global["position_group"].to_numpy()
    forward_share = [float((groups[fit.labels_ == c] == "FW").mean()) for c in range(K)]
    attacking = int(np.argmax(forward_share))
    direction = centres[attacking] - centres[1 - attacking]
    direction = direction / np.linalg.norm(direction)
    midpoint = 0.5 * (centres[0] + centres[1])
    projection = (x - midpoint) @ direction
    return z_global, projection


def _panel_axis_density(ax, z_global: pd.DataFrame, projection: np.ndarray) -> None:
    edges = np.linspace(projection.min(), projection.max(), 60)
    groups = z_global["position_group"].to_numpy()
    bottom = np.zeros(len(edges) - 1)
    for group in CLASS_ORDER:
        counts, _ = np.histogram(projection[groups == group], bins=edges)
        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            bottom=bottom,
            align="edge",
            color=P.POSITION_COLORS[group],
            linewidth=0,
            label=GROUP_LABELS[group],
        )
        bottom = bottom + counts
    # The stack shows which recorded position sits where; the outline is the quantity the
    # paper's claim is about, since bimodality is a property of the whole distribution and
    # not of any one group within it.
    ax.step(edges[:-1], bottom, where="post", color=P.INK_PRIMARY, linewidth=1.2)
    ax.axvline(0.0, color=P.INK_PRIMARY, linewidth=0.9, linestyle=(0, (4, 3)))
    # Headroom for the legend, so it never sits on the taller of the two peaks.
    ax.set_ylim(0, bottom.max() * 1.3)
    P.style_axis(ax, "position along the two-mode axis", "players", "two poles, a full middle")
    ax.legend(loc="upper right", fontsize=P.BASE_FONT_PT - 2, handlelength=1.0)


def _heat(ax, matrix, xlabels, ylabels, header, *, xlabel="", ylabel="") -> None:
    ax.imshow(matrix, cmap=P.SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value * 100:.0f}",
                ha="center",
                va="center",
                fontsize=P.BASE_FONT_PT - 1,
                color=P.SURFACE if value > 0.55 else P.INK_PRIMARY,
            )
    P.style_axis(ax, xlabel, ylabel, header)
    ax.grid(False)


def _panel_contingency(ax, validation: dict) -> None:
    table = validation["scopes"]["All outfield"]["vs_position"]["position_group"]["contingency"]
    counts = np.array(
        [[float(table[str(c)][g]) for c in range(K)] for g in CLASS_ORDER], dtype=float
    )
    shares = counts / counts.sum(axis=1, keepdims=True)
    _heat(
        ax,
        shares,
        ["defensive", "attacking"],
        [GROUP_LABELS[g] for g in CLASS_ORDER],
        "modes against positions",
        xlabel="percent of each position group",
    )


def _panel_confusion(ax, supervised: dict) -> None:
    block = supervised["models"]["lightgbm"]["confusion_matrix"]
    order = block["labels"]
    shares = np.array(block["row_normalised"], dtype=float)
    _heat(
        ax,
        shares,
        order,
        [GROUP_LABELS[g] for g in order],
        "classifier errors",
        xlabel="percent predicted",
        ylabel="listed",
    )


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def figure_geometry() -> None:
    """Six panels on the shape of the space and what the two modes are not."""
    pca = _metrics("archive_pca")
    validation = _metrics("archive_cluster_validation")
    supervised = _metrics("archive_supervised")
    z_global, projection = _pooled_axis()

    fig, axes = plt.subplots(2, 3, figsize=(P.WIDTH_FULL, 5.4))
    _panel_scree(axes[0, 0], pca)
    _panel_loadings(axes[0, 1], pca, "PC1", "PC1: territory")
    _panel_loadings(axes[0, 2], pca, "PC2", "PC2: progression")
    _panel_axis_density(axes[1, 0], z_global, projection)
    _panel_contingency(axes[1, 1], validation)
    _panel_confusion(axes[1, 2], supervised)
    P.panel_labels(axes)
    P.save_figure(fig, "main_geometry")


def figure_modes() -> None:
    """One panel per position group: what separates its two archetypes."""
    profiles = _metrics("archive_archetype_profiles")
    with open(config.RESULTS / "archive_archetypes.json") as fh:
        names = {entry["label"]: entry["name"] for entry in json.load(fh)["archetypes"]}

    fig, axes = plt.subplots(1, len(CLASS_ORDER), figsize=(P.WIDTH_FULL, 4.4), sharex=True)
    for ax, group in zip(np.atleast_1d(axes), CLASS_ORDER, strict=True):
        keys = [f"{group}-{c}" for c in range(K)]
        centroids = [profiles["centroids"][k] for k in keys]
        features = list(centroids[0])
        separation = {f: abs(centroids[0][f] - centroids[1][f]) for f in features}
        chosen = sorted(separation, key=separation.get, reverse=True)[:TOP_N]
        chosen = sorted(chosen, key=lambda f: centroids[1][f])
        y = np.arange(len(chosen))
        for i, key in enumerate(keys):
            ax.barh(
                y + (0.2 if i == 1 else -0.2),
                [centroids[i][f] for f in chosen],
                height=0.36,
                color=P.CATEGORICAL[i],
                linewidth=0,
                label=names[key],
            )
        ax.axvline(0, color=P.INK_SECONDARY, linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([_label(f) for f in chosen], fontsize=P.BASE_FONT_PT - 2)
        P.style_axis(ax, "", "", GROUP_LABELS[group])
        ax.grid(False, axis="y")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.06),
            ncol=1,
            fontsize=P.BASE_FONT_PT - 2,
            handlelength=1.0,
        )
    fig.supxlabel(
        "deviation from the position group mean, standard deviations",
        fontsize=P.BASE_FONT_PT,
    )
    P.panel_labels(np.atleast_1d(axes))
    P.save_figure(fig, "main_modes")


def main() -> None:
    P.use_style()
    figure_geometry()
    figure_modes()
    print("paper_figures: main_geometry, main_modes")


if __name__ == "__main__":
    main()
