"""Three-dimensional views of the player and club spaces.

Two dimensions are not enough here, and the reason is specific rather than aesthetic. The
first two principal components carry shooting volume and wide creation, and together they
account for roughly half the variance. The third carries duels and discipline, fouls
committed, cards and tackles won, which is a behavioural dimension orthogonal to both of
the others and one that a two-dimensional projection collapses entirely. Adding it raises
captured variance by around eleven percentage points.

A static three-dimensional scatter is easy to do badly. Three habits are used throughout to
keep these readable in print:

* every scene is drawn from two viewpoints, so a reader can tell genuine structure from an
  artefact of one projection,
* points are given a floor projection in light grey, which supplies the depth cue that
  stereo vision would otherwise provide, and
* marker size decreases with distance from the camera, reinforcing the same cue.

Colour still obeys the validated palette: at most four categorical hues, marker shape as a
secondary channel, and a sequential ramp wherever the quantity is a magnitude.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import config
from src import features as F
from src import plotting as P

#: Two azimuths far enough apart to expose occlusion, at a shared elevation.
VIEWS = ((22, -60), (22, 25))

#: Pure white. Matplotlib fills a 3D pane with light grey by default, which prints as
#: the grey box behind the plot that the rest of the style exists to avoid.
PANE = (1.0, 1.0, 1.0, 1.0)

#: How many clubs at each end of the table get a direct label.
LABEL_EXTREMES = 5


def _load(season: str) -> tuple[pd.DataFrame, np.ndarray, PCA, list[str]]:
    """Standardised outfield matrix and its three-component projection."""
    frame = pd.read_parquet(config.DATA_PROCESSED / f"outfield_z_global_{season}.parquet")
    usable = [c for c in F.OUTFIELD_CORE if c in frame.columns]
    matrix = frame[usable].to_numpy(dtype=float)
    pca = PCA(n_components=3, random_state=config.RANDOM_STATE).fit(matrix)
    return frame, pca.transform(matrix), pca, usable


def _prepare_axis(ax, pca: PCA, title: str) -> None:
    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  {evr[0] * 100:.0f}%", labelpad=-4, fontsize=P.BASE_FONT_PT - 2)
    ax.set_ylabel(f"PC2  {evr[1] * 100:.0f}%", labelpad=-4, fontsize=P.BASE_FONT_PT - 2)
    # The z label rotates to follow the axis by default, which overlaps its own ticks at
    # this figure size, so it is pinned horizontal and given real padding.
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(f"PC3  {evr[2] * 100:.0f}%", labelpad=4, fontsize=P.BASE_FONT_PT - 2, rotation=90)
    # The viewpoint caption sits inside the panel, because a per-axes title at this size
    # collides with the figure title above it.
    ax.text2D(
        0.03,
        0.88,
        title,
        transform=ax.transAxes,
        ha="left",
        fontsize=P.BASE_FONT_PT - 2,
        color=P.INK_SECONDARY,
    )
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color(PANE)
        pane._axinfo["grid"]["color"] = P.GRID
        pane._axinfo["grid"]["linewidth"] = 0.4
    ax.tick_params(labelsize=P.BASE_FONT_PT - 3, pad=-2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])


def _depth_sizes(coords: np.ndarray, elev: float, azim: float, base: float = 13.0) -> np.ndarray:
    """Shrink markers with distance from the camera, as a monocular depth cue."""
    e, a = np.radians(elev), np.radians(azim)
    direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    depth = coords @ direction
    span = depth.max() - depth.min()
    scaled = (depth - depth.min()) / span if span > 0 else np.full(len(depth), 0.5)
    return base * (0.55 + 0.75 * scaled)


def _floor(ax, coords: np.ndarray, zfloor: float) -> None:
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        np.full(len(coords), zfloor),
        s=3.0,
        c=P.CONTEXT_GREY,
        alpha=0.45,
        linewidths=0,
        zorder=1,
    )


def _new_scene(width: float = P.WIDTH_FULL, height: float = 3.1):
    fig = plt.figure(figsize=(width, height), layout="none")
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    return fig, axes


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def figure_positions(season: str) -> dict:
    frame, coords, pca, _ = _load(season)
    zfloor = coords[:, 2].min() - 0.35 * np.ptp(coords[:, 2])

    fig, axes = _new_scene()
    for ax, (elev, azim) in zip(axes, VIEWS, strict=True):
        _floor(ax, coords, zfloor)
        for group, colour in P.POSITION_COLORS.items():
            if group == "GK":
                continue
            mask = (frame["position_group"] == group).to_numpy()
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                s=_depth_sizes(coords[mask], elev, azim),
                c=colour,
                marker=P.POSITION_MARKERS[group],
                label=group,
                edgecolors=P.SURFACE,
                linewidths=0.3,
                depthshade=False,
                zorder=3,
            )
        ax.view_init(elev=elev, azim=azim)
        _prepare_axis(ax, pca, f"view {azim:+.0f} degrees")
        ax.set_zlim(zfloor, coords[:, 2].max())

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=P.BASE_FONT_PT - 1,
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.09, wspace=0.02)
    P.save_figure(fig, "pca3d_positions")

    return {
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "cumulative_two_components": round(float(pca.explained_variance_ratio_[:2].sum()), 4),
        "cumulative_three_components": round(float(pca.explained_variance_ratio_.sum()), 4),
    }


def figure_clusters(season: str) -> dict:
    frame, coords, pca, _ = _load(season)
    clusters = pd.read_parquet(config.DATA_PROCESSED / f"clusters_{season}.parquet")
    merged = frame[["player"]].merge(
        clusters[["player", "cluster_global"]], on="player", how="left"
    )
    labels = merged["cluster_global"].to_numpy()
    zfloor = coords[:, 2].min() - 0.35 * np.ptp(coords[:, 2])
    palette = P.categorical(2)

    fig, axes = _new_scene()
    centroids = {}
    for ax, (elev, azim) in zip(axes, VIEWS, strict=True):
        _floor(ax, coords, zfloor)
        for value in sorted(pd.unique(labels[~pd.isna(labels)])):
            mask = labels == value
            idx = int(value)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                coords[mask, 2],
                s=_depth_sizes(coords[mask], elev, azim),
                c=palette[idx % len(palette)],
                marker=P.MARKERS[idx % len(P.MARKERS)],
                label=f"cluster {idx}",
                edgecolors=P.SURFACE,
                linewidths=0.3,
                depthshade=False,
                zorder=3,
            )
            centre = coords[mask].mean(axis=0)
            centroids[str(idx)] = [round(float(v), 4) for v in centre]
            ax.scatter(
                *centre,
                s=140,
                marker="X",
                c=palette[idx % len(palette)],
                edgecolors=P.INK_PRIMARY,
                linewidths=1.0,
                depthshade=False,
                zorder=6,
            )
        ax.view_init(elev=elev, azim=azim)
        _prepare_axis(ax, pca, f"view {azim:+.0f} degrees")
        ax.set_zlim(zfloor, coords[:, 2].max())

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=P.BASE_FONT_PT - 1,
    )
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.09, wspace=0.02)
    P.save_figure(fig, "pca3d_clusters")
    return {"cluster_centroids_pc123": centroids}


def figure_teams(season: str) -> dict:
    """Clubs as minutes-weighted centroids, shaded by final league position."""
    frame, coords, pca, _ = _load(season)
    eligible = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")
    table = pd.read_parquet(config.DATA_PROCESSED / f"league_table_{season}.parquet")

    work = frame[["player", "team"]].copy()
    work[["pc1", "pc2", "pc3"]] = coords
    work = work.merge(eligible[["player", "minutes"]], on="player", how="left")
    work["minutes"] = pd.to_numeric(work["minutes"], errors="coerce").fillna(0.0)
    work["team"] = work["team"].astype(str).str.split(" / ").str[0].str.strip()

    rows = []
    for team, grp in work.groupby("team"):
        w = grp["minutes"].to_numpy(dtype=float)
        if w.sum() <= 0:
            continue
        rows.append(
            {
                "team": team,
                "pc1": float(np.average(grp["pc1"], weights=w)),
                "pc2": float(np.average(grp["pc2"], weights=w)),
                "pc3": float(np.average(grp["pc3"], weights=w)),
            }
        )
    teams = (
        pd.DataFrame(rows)
        .merge(table[["team", "league_position"]], on="team", how="left")
        .dropna(subset=["league_position"])
    )

    pos = teams["league_position"].to_numpy(dtype=float)
    colours = P.SEQUENTIAL(1.0 - (pos - pos.min()) / max(pos.max() - pos.min(), 1))
    pts = teams[["pc1", "pc2", "pc3"]].to_numpy(dtype=float)
    zfloor = pts[:, 2].min() - 0.4 * np.ptp(pts[:, 2])

    fig, axes = _new_scene(height=3.3)
    for ax, (elev, azim) in zip(axes, VIEWS, strict=True):
        for x, y, z in pts:
            ax.plot([x, x], [y, y], [zfloor, z], color=P.GRID, linewidth=0.6, zorder=1)
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=52,
            c=colours,
            edgecolors=P.INK_SECONDARY,
            linewidths=0.5,
            depthshade=False,
            zorder=4,
        )
        # Labelling all twenty points collides badly in a projected scene and breaks the
        # rule against labelling every mark. Only the extremes of the league are named;
        # the shading carries the ordering for everyone else, which is what the figure is
        # actually about.
        span = np.ptp(pts[:, 2])
        order = np.argsort(pos)
        named = set(order[:LABEL_EXTREMES]) | set(order[-LABEL_EXTREMES:])
        rank_of = {int(i): r for r, i in enumerate(order)}
        for i, ((x, y, z), name) in enumerate(zip(pts, teams["team"], strict=True)):
            if i not in named:
                continue
            stagger = (0.10 if rank_of[i] % 2 == 0 else -0.14) * span
            ax.text(
                x,
                y,
                z + stagger,
                name,
                fontsize=P.BASE_FONT_PT - 3,
                color=P.INK_SECONDARY,
                ha="center",
                va="bottom" if stagger > 0 else "top",
                zorder=7,
            )
        ax.view_init(elev=elev, azim=azim)
        _prepare_axis(ax, pca, f"view {azim:+.0f} degrees")
        ax.set_zlim(zfloor, pts[:, 2].max() + 0.15 * np.ptp(pts[:, 2]))

    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.01, wspace=0.02)
    P.save_figure(fig, "pca3d_teams")

    corr = {f"pc{i + 1}": round(float(np.corrcoef(pts[:, i], pos)[0, 1]), 4) for i in range(3)}
    return {"n_teams": int(len(teams)), "corr_with_league_position": corr}


def figure_archetype_axes(season: str) -> dict:
    """The six archetype centroids, which lie on three axes rather than six directions.

    Because features are standardised within position group before a two-way split, the
    two centroids in a group are anti-parallel by construction. Drawing them as vectors
    from the origin makes that visible: each pair is one line through the origin, not two
    independent directions.
    """
    arche = pd.read_parquet(config.DATA_PROCESSED / f"archetypes_{season}.parquet")
    zgroup = pd.read_parquet(config.DATA_PROCESSED / f"outfield_z_bygroup_{season}.parquet")
    usable = [c for c in F.OUTFIELD_CORE if c in zgroup.columns]

    merged = zgroup.merge(
        arche[["player", "cluster_within_group_label", "archetype_name"]], on="player", how="inner"
    )
    pca = PCA(n_components=3, random_state=config.RANDOM_STATE).fit(
        merged[usable].to_numpy(dtype=float)
    )

    groups = sorted(merged["position_group"].unique())
    palette = P.categorical(min(len(groups), P.MAX_CATEGORICAL))
    colour_of = dict(zip(groups, palette, strict=False))

    vectors, cosines, magnitudes = {}, {}, {}
    for label, grp in merged.groupby("cluster_within_group_label"):
        raw = pca.transform(grp[usable].to_numpy(dtype=float).mean(axis=0)[None, :])[0]
        norm = float(np.linalg.norm(raw))
        magnitudes[label] = round(norm, 4)
        # Drawn as unit directions. The claim being illustrated is that the two centroids
        # in a group point in opposite directions, which is a statement about direction
        # and not magnitude, and unit length also keeps the short vectors from piling up
        # on the origin where their labels would overlap.
        vectors[label] = raw / norm if norm else raw
    for group in groups:
        pair = sorted(k for k in vectors if k.startswith(f"{group}-"))
        if len(pair) == 2:
            a, b = vectors[pair[0]], vectors[pair[1]]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cosines[group] = round(float(a @ b / denom), 4) if denom else None

    allv = np.vstack(list(vectors.values()))
    lim = 1.45 * np.abs(allv).max()
    fig, axes = _new_scene(height=3.3)
    for ax, (elev, azim) in zip(axes, VIEWS, strict=True):
        # Endpoints can project close together from any single viewpoint even when the
        # vectors are well separated in three dimensions, so labels are pushed out and
        # nudged alternately along PC3 to break ties.
        for nudge, (label, vec) in enumerate(sorted(vectors.items())):
            group = label.split("-")[0]
            colour = colour_of.get(group, P.CATEGORICAL[0])
            ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]], color=colour, linewidth=2.0, zorder=3)
            ax.scatter(
                *vec,
                s=45,
                c=colour,
                marker=P.POSITION_MARKERS.get(group, "o"),
                edgecolors=P.SURFACE,
                linewidths=0.5,
                depthshade=False,
                zorder=5,
            )
            offset = 0.10 * lim * (1 if nudge % 2 == 0 else -1)
            ax.text(
                vec[0] * 1.24,
                vec[1] * 1.24,
                vec[2] * 1.24 + offset,
                label,
                fontsize=P.BASE_FONT_PT - 3,
                color=P.INK_SECONDARY,
                ha="center",
                zorder=7,
            )
        ax.scatter(0, 0, 0, s=20, c=P.INK_MUTED, marker="o", depthshade=False, zorder=4)
        ax.view_init(elev=elev, azim=azim)
        _prepare_axis(ax, pca, f"view {azim:+.0f} degrees")
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
            setter(-lim, lim)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.01, wspace=0.02)
    P.save_figure(fig, "pca3d_archetype_axes")

    return {
        "within_group_centroid_cosine": cosines,
        "centroid_magnitudes": magnitudes,
        "drawn_as": "unit vectors, so the figure shows direction rather than magnitude",
        "note": (
            "A cosine of minus one means the two centroids in a position group point in "
            "exactly opposite directions, which is forced by standardising within group "
            "before a two-way split. The six archetypes therefore describe three axes."
        ),
    }


def main() -> None:
    P.use_style()
    season = config.SEASON_PRIMARY
    payload = {"season": season, "views": [{"elev": e, "azim": a} for e, a in VIEWS]}
    payload["positions"] = figure_positions(season)
    payload["clusters"] = figure_clusters(season)
    payload["teams"] = figure_teams(season)
    payload["archetype_axes"] = figure_archetype_axes(season)

    with open(config.METRICS / "threed.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    pos = payload["positions"]
    print(
        f"3D: two components capture {pos['cumulative_two_components']:.3f} of variance, "
        f"three capture {pos['cumulative_three_components']:.3f}",
        flush=True,
    )
    print(
        f"    club centroid correlations with league position: "
        f"{payload['teams']['corr_with_league_position']}",
        flush=True,
    )
    print(
        f"    within-group centroid cosines: "
        f"{payload['archetype_axes']['within_group_centroid_cosine']}",
        flush=True,
    )
    print("threed complete")


if __name__ == "__main__":
    main()
