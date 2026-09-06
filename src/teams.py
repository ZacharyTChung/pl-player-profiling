"""Team stylistic signatures.

Each club is summarised twice: as a minutes-weighted centroid of its outfield players in
the shared principal component space, and as the share of its outfield minutes played by
each archetype. Clubs are then clustered on archetype composition and compared against
the final league table, which is computed from match scores rather than taken on trust.

Both summaries are deliberately minutes-weighted. A club's style is what its players
actually did on the pitch, so a fringe player with two hundred minutes should not count
as much as an ever-present.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

import config
from src import plotting as P

SCORE_SEPARATORS = ("–", "—", "-")


# --------------------------------------------------------------------------------------
# League table
# --------------------------------------------------------------------------------------


def _parse_score(text: object) -> tuple[int, int] | None:
    """Split an FBref score string into goals for and against.

    FBref separates the two figures with an en dash rather than a hyphen, so the
    separator is tried in order rather than assumed.
    """
    if not isinstance(text, str):
        return None
    for sep in SCORE_SEPARATORS:
        if sep in text:
            left, _, right = text.partition(sep)
            try:
                return int(left.strip()), int(right.strip())
            except ValueError:
                continue
    return None


def build_league_table(season: str) -> pd.DataFrame:
    """Compute the final table from match results, caching the outcome."""
    cached = config.DATA_PROCESSED / f"league_table_{season}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    from src.ingest import FBrefFull

    schedule = FBrefFull(leagues=config.LEAGUE, seasons=season).read_schedule().reset_index()
    rows: dict[str, dict[str, int]] = {}

    def bump(team: str, gf: int, ga: int) -> None:
        rec = rows.setdefault(
            team, {"played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0}
        )
        rec["played"] += 1
        rec["gf"] += gf
        rec["ga"] += ga
        if gf > ga:
            rec["won"] += 1
        elif gf == ga:
            rec["drawn"] += 1
        else:
            rec["lost"] += 1

    for _, game in schedule.iterrows():
        parsed = _parse_score(game.get("score"))
        if parsed is None:
            continue
        home_goals, away_goals = parsed
        bump(str(game["home_team"]), home_goals, away_goals)
        bump(str(game["away_team"]), away_goals, home_goals)

    table = pd.DataFrame.from_dict(rows, orient="index").reset_index(names="team")
    table["points"] = table["won"] * 3 + table["drawn"]
    table["goal_difference"] = table["gf"] - table["ga"]
    table = table.sort_values(["points", "goal_difference", "gf"], ascending=False).reset_index(
        drop=True
    )
    table["league_position"] = np.arange(1, len(table) + 1)
    table["season"] = season
    table.to_parquet(cached, index=False)
    return table


# --------------------------------------------------------------------------------------
# Team profiles
# --------------------------------------------------------------------------------------


def archetype_composition(season: str) -> pd.DataFrame:
    """Share of each club's outfield minutes played by each archetype."""
    merged = pd.read_parquet(config.DATA_PROCESSED / f"archetypes_{season}.parquet")
    if "minutes" not in merged.columns:
        eligible = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")
        merged = merged.merge(eligible[["player", "minutes"]], on="player", how="left")
    # Cast out of any nullable extension dtype: pandas cannot pivot on those here.
    merged["minutes"] = (
        pd.to_numeric(merged["minutes"], errors="coerce").astype("float64").fillna(0.0)
    )

    # Aggregation names a mover's clubs as "A / B", ordered by minutes played, so the
    # first entry is the club he played most for. Without this split those composite
    # names would be counted as additional clubs and the league would appear larger
    # than twenty.
    merged["team"] = merged["team"].astype(str).str.split(" / ").str[0].str.strip()
    pivot = merged.pivot_table(
        index="team", columns="archetype_name", values="minutes", aggfunc="sum", fill_value=0.0
    )
    shares = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    shares.columns = [str(c) for c in shares.columns]
    return shares


def pca_centroids(season: str) -> pd.DataFrame:
    """Minutes-weighted club centroid in the shared principal component space."""
    emb = pd.read_parquet(config.DATA_PROCESSED / f"embeddings_{season}.parquet")
    eligible = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")
    merged = emb.merge(eligible[["player", "minutes"]], on="player", how="left")
    merged["minutes"] = (
        pd.to_numeric(merged["minutes"], errors="coerce").astype("float64").fillna(0.0)
    )
    merged["team"] = merged["team"].astype(str).str.split(" / ").str[0].str.strip()

    out = []
    for team, grp in merged.groupby("team"):
        weight = grp["minutes"].to_numpy(dtype=float)
        if weight.sum() <= 0:
            continue
        out.append(
            {
                "team": team,
                "pca1": float(np.average(grp["pca1"], weights=weight)),
                "pca2": float(np.average(grp["pca2"], weights=weight)),
                "minutes": float(weight.sum()),
            }
        )
    return pd.DataFrame(out).set_index("team")


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def figure_dendrogram(shares: pd.DataFrame, link: np.ndarray, season: str) -> None:
    fig, ax = plt.subplots(figsize=(P.WIDTH_FULL, 3.6))
    dendrogram(
        link,
        labels=list(shares.index),
        ax=ax,
        color_threshold=0,
        link_color_func=lambda _: P.INK_SECONDARY,
        leaf_rotation=90,
    )
    P.style_axis(ax, "", "Distance", f"Clubs clustered on archetype composition, {season}")
    ax.grid(False, axis="x")
    ax.tick_params(axis="x", labelsize=P.BASE_FONT_PT - 2)
    P.save_figure(fig, "team_dendrogram")


def figure_composition(shares: pd.DataFrame, table: pd.DataFrame, season: str) -> None:
    """Stacked composition ordered by final league position."""
    order = table.set_index("team")["league_position"]
    ordered = shares.loc[[t for t in order.sort_values().index if t in shares.index]]

    n_arch = ordered.shape[1]
    if n_arch > P.MAX_CATEGORICAL:
        # Six archetypes exceed the validated palette, so encode the position group by
        # hue and separate the two archetypes within a group by shade.
        groups = sorted({c.split()[-1] for c in ordered.columns})
        colors = {}
        base = P.categorical(min(len(groups), P.MAX_CATEGORICAL))
        for i, col in enumerate(ordered.columns):
            grp = col.split()[-1]
            hue = base[groups.index(grp) % len(base)]
            colors[col] = hue if i % 2 == 0 else _lighten(hue, 0.45)
    else:
        colors = dict(zip(ordered.columns, P.categorical(n_arch), strict=False))

    fig, ax = plt.subplots(figsize=(P.WIDTH_FULL, 3.8))
    bottom = np.zeros(len(ordered))
    for col in ordered.columns:
        vals = ordered[col].to_numpy()
        ax.bar(
            np.arange(len(ordered)),
            vals,
            bottom=bottom,
            color=colors[col],
            edgecolor=P.SURFACE,
            linewidth=0.8,
            label=col,
        )
        bottom += vals
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels(ordered.index, rotation=90, fontsize=P.BASE_FONT_PT - 2)
    ax.set_ylim(0, 1)
    P.style_axis(ax, "", "Share of outfield minutes", f"Archetype composition, {season}")
    ax.grid(False, axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=P.BASE_FONT_PT - 2)
    P.save_figure(fig, "team_composition")


def _lighten(hex_color: str, amount: float) -> str:
    rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    rgb = rgb + (255.0 - rgb) * amount
    return "#" + "".join(f"{int(round(v)):02X}" for v in rgb)


def figure_team_panels(season: str) -> None:
    """One panel per club, that club's players highlighted against the whole league."""
    emb = pd.read_parquet(config.DATA_PROCESSED / f"embeddings_{season}.parquet")
    emb["team"] = emb["team"].astype(str).str.split(" / ").str[0].str.strip()
    teams = sorted(emb["team"].unique())
    ncols = 5
    nrows = int(np.ceil(len(teams) / ncols))

    # Twenty clubs fill a four by five grid exactly, so there is no spare cell for the
    # key. Constrained layout reserves room for a figure legend only when the location is
    # an "outside" one, which is why that form is used below rather than manual margins.
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(P.WIDTH_WIDE, 1.55 * nrows + 0.35), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, team in zip(axes, teams, strict=False):
        ax.scatter(emb["umap1"], emb["umap2"], s=3.0, c=P.CONTEXT_GREY, linewidths=0, zorder=1)
        sub = emb[emb["team"] == team]
        for grp, colour in P.POSITION_COLORS.items():
            g = sub[sub["position_group"] == grp]
            if g.empty:
                continue
            ax.scatter(
                g["umap1"],
                g["umap2"],
                s=11.0,
                c=colour,
                marker=P.POSITION_MARKERS[grp],
                linewidths=0.4,
                edgecolors=P.SURFACE,
                zorder=3,
            )
        ax.set_title(team, fontsize=P.BASE_FONT_PT - 2, loc="center", pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(P.GRID)
    for ax in axes[len(teams) :]:
        ax.set_visible(False)

    handles = [
        plt.Line2D(
            [],
            [],
            marker=P.POSITION_MARKERS[g],
            color="none",
            markerfacecolor=c,
            markeredgecolor=P.SURFACE,
            markersize=5,
            label=g,
        )
        for g, c in P.POSITION_COLORS.items()
        if g != "GK"
    ]
    # The key sits directly under the title, where no panel competes with it. A key
    # placed at the foot would overlap the bottom row, since twenty clubs fill the grid.
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Each club's outfield players in the shared embedding, {season}",
        fontsize=P.BASE_FONT_PT,
        fontweight="bold",
    )
    P.save_figure(fig, "team_panels")


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def analyse_season(season: str) -> dict:
    shares = archetype_composition(season)
    centroids = pca_centroids(season)
    table = build_league_table(season)

    common = [t for t in shares.index if t in set(table["team"])]
    unmatched = sorted(set(shares.index) - set(table["team"]))

    distances = pdist(shares.to_numpy(), metric="euclidean")
    link = linkage(distances, method="ward")
    labels = fcluster(link, t=3, criterion="maxclust")
    style_groups = dict(zip(shares.index, (int(x) for x in labels), strict=True))

    merged = shares.loc[common].join(table.set_index("team")[["league_position", "points"]])
    correlations = {}
    for col in shares.columns:
        rho, p = spearmanr(merged[col], merged["league_position"])
        correlations[col] = {"spearman_rho": round(float(rho), 4), "p_value": round(float(p), 4)}

    cent = centroids.join(table.set_index("team")[["league_position", "points"]], how="inner")
    rho_pc1, p_pc1 = spearmanr(cent["pca1"], cent["league_position"])

    figure_dendrogram(shares, link, season)
    figure_composition(shares, table, season)
    if season == config.SEASON_PRIMARY:
        figure_team_panels(season)

    most_distinctive = (
        shares.sub(shares.mean(axis=0), axis=1).abs().sum(axis=1).sort_values(ascending=False)
    )

    return {
        "season": season,
        "n_teams": int(len(shares)),
        "archetypes": list(shares.columns),
        "unmatched_teams": unmatched,
        "style_groups": style_groups,
        "n_style_groups": int(len(set(labels))),
        "archetype_share_vs_league_position": correlations,
        "pca1_centroid_vs_league_position": {
            "spearman_rho": round(float(rho_pc1), 4),
            "p_value": round(float(p_pc1), 4),
        },
        "most_distinctive_teams": {
            k: round(float(v), 4) for k, v in most_distinctive.head(5).items()
        },
        "least_distinctive_teams": {
            k: round(float(v), 4) for k, v in most_distinctive.tail(5).items()
        },
        "composition": {
            t: {c: round(float(shares.loc[t, c]), 4) for c in shares.columns} for t in shares.index
        },
        "league_table": table.to_dict(orient="records"),
    }


def main() -> None:
    P.use_style()
    report = {}
    for season in config.SEASONS:
        report[season] = analyse_season(season)
        r = report[season]
        print(
            f"{season}: {r['n_teams']} clubs, {r['n_style_groups']} style groups, "
            f"PC1 centroid against league position rho "
            f"{r['pca1_centroid_vs_league_position']['spearman_rho']}",
            flush=True,
        )
    with open(config.METRICS / "team_signatures.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("teams complete")


if __name__ == "__main__":
    main()
