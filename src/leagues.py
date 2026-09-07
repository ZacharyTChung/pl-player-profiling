"""Does the two cluster result belong to the feature set or to the Premier League?

Every result in the main analysis rests on one league, which leaves an obvious
alternative explanation: perhaps English football happens to divide in two and a different
competition would divide differently. The question is answerable, because the withdrawal
that motivated this study applied to the whole site, so the other four of the big five
leagues are in exactly the same reduced state and Understat covers all of them.

This module repeats the pipeline on each of them for the primary season and asks three
questions. Does the declared selection rule still return two clusters? Does the partition
still fail to recover listed positions? And do the possession elasticities that justify the
one sided adjustment reproduce outside England?

Team name reconciliation is done by fitting an alias map per league rather than by the
hand written table the Premier League uses, because four more competitions would need four
more hand written tables and the failure mode of getting one wrong is a silently reduced
match rate rather than an error.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

import config
from src import cluster as C
from src import features as F
from src import plotting as P
from src import preprocess as PP

#: The other four of the big five. All are covered by both sources.
LEAGUES = ["ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]

SHORT = {
    "ENG-Premier League": "England",
    "ESP-La Liga": "Spain",
    "GER-Bundesliga": "Germany",
    "ITA-Serie A": "Italy",
    "FRA-Ligue 1": "France",
}

ROOT = config.DATA_RAW / "leagues"


def slug(league: str) -> str:
    return league.replace(" ", "_").replace("-", "_")


def league_root(league: str):
    return ROOT / slug(league)


# --------------------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------------------


def ingest_league(league: str, season: str) -> dict:
    """Pull the surviving FBref tables and the Understat block for one league."""
    os.environ.setdefault("SOCCERDATA_DIR", str(config.HTML_CACHE))
    import soccerdata as sd

    from src.ingest import FBrefFull, flatten_columns

    out_dir = league_root(league) / season
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only the tables that still carry values are worth the requests.
    needed = ["standard", "shooting", "misc", "playing_time"]
    fb = FBrefFull(leagues=league, seasons=season)
    report = {"league": league, "season": season, "tables": {}}

    for scope, reader in (
        ("players", fb.read_player_season_stats),
        ("teams", fb.read_team_season_stats),
    ):
        for stat in needed:
            path = out_dir / f"{scope}_{stat}.parquet"
            if path.exists():
                report["tables"][f"{scope}_{stat}"] = "cached"
                continue
            flat, _ = flatten_columns(reader(stat_type=stat).reset_index())
            flat.to_parquet(path, index=False)
            report["tables"][f"{scope}_{stat}"] = list(flat.shape)
            print(f"  {league} {scope}/{stat}: {flat.shape}", flush=True)

    us_path = out_dir / "understat_players.parquet"
    if not us_path.exists():
        us = sd.Understat(leagues=league, seasons=season).read_player_season_stats().reset_index()
        us.to_parquet(us_path, index=False)
        report["understat"] = list(us.shape)
        print(f"  {league} understat: {us.shape}", flush=True)
    else:
        report["understat"] = "cached"
    return report


# --------------------------------------------------------------------------------------
# Team name reconciliation
# --------------------------------------------------------------------------------------


def fit_alias_map(fbref_teams, understat_teams) -> dict[str, str]:
    """Pair Understat club names to FBref ones without a hand written table.

    Exact folded match first, then prefix containment, then a single best token overlap.
    A pairing is only accepted when it is unambiguous, so an unmatched club stays
    unmatched and is counted rather than being attached to the wrong side.
    """
    fb_keys = {PP.normalize_name(t): t for t in fbref_teams}
    alias: dict[str, str] = {}
    for raw in understat_teams:
        key = PP.normalize_name(raw)
        if key in fb_keys:
            alias[key] = fb_keys[key]
            continue
        hits = [k for k in fb_keys if k.startswith(key) or key.startswith(k)]
        if len(hits) == 1:
            alias[key] = fb_keys[hits[0]]
            continue
        tokens = set(key.split())
        scored = [(len(tokens & set(k.split())), k) for k in fb_keys]
        best = max(scored)[0]
        if best > 0 and sum(1 for s, _ in scored if s == best) == 1:
            alias[key] = fb_keys[max(scored)[1]]
    return alias


def prepare_league(league: str, season: str) -> tuple[pd.DataFrame, dict]:
    """Build the eligible outfield frame for one league, reusing the main loaders."""
    root = league_root(league)
    fbref = PP.load_fbref_canonical(season, root=root)
    understat = PP.load_understat(season, root=root)

    alias = fit_alias_map(fbref["team"].unique(), understat["_team_key"].unique())
    understat = understat.copy()
    understat["_team_key"] = understat["_team_key"].map(
        lambda k: PP.normalize_name(alias.get(k, k))
    )
    merged, match_report = PP.match_understat(fbref, understat)
    merged = PP.parse_positions(merged)

    possession = PP.load_team_possession(season, root=root)
    merged = merged.merge(possession, on="team", how="left")
    merged["team_possession"] = merged["team_possession"].fillna(
        possession["team_possession"].mean()
    )

    aggregated = PP.aggregate_per_player(merged)
    aggregated = PP.to_per90(aggregated, F.OUTFIELD_COUNTS)
    elasticity = elasticities(league, season)
    aggregated = PP.possession_adjust(aggregated, elasticity["alphas"])

    outfield = aggregated[aggregated["position_group"].isin(config.OUTFIELD_GROUPS)].copy()
    eligible = outfield[outfield["minutes"] >= config.MIN_MINUTES].copy()
    usable = [c for c in F.OUTFIELD_CORE if c in eligible.columns]
    eligible, _ = PP.impute_within_group(eligible, usable, "position_group")

    info = {
        "n_aliased_clubs": len(alias),
        "understat_match_rate": match_report["match_rate"],
        "n_outfield": int(len(outfield)),
        "n_eligible": int(len(eligible)),
        "elasticities": elasticity["report"]["elasticities"],
    }
    return eligible, info


def elasticities(league: str, season: str) -> dict:
    """Fit the defensive possession elasticities inside this league."""
    root = league_root(league)
    misc = pd.read_parquet(root / season / "teams_misc.parquet")
    std = pd.read_parquet(root / season / "teams_standard.parquet")
    merged = misc.merge(std[["team", "Poss"]], on="team", how="inner")
    merged["nineties"] = pd.to_numeric(merged["90s"], errors="coerce")
    merged["opponent"] = 100.0 - pd.to_numeric(merged["Poss"], errors="coerce")

    alphas, report = {}, {}
    for count, source in config.PADJ_TEAM_SOURCES.items():
        merged[count] = pd.to_numeric(merged[source], errors="coerce") / merged["nineties"]
        sub = merged[[count, "opponent"]].dropna()
        sub = sub[(sub[count] > 0) & (sub["opponent"] > 0)]
        alpha = (
            float(np.polyfit(np.log(sub["opponent"]), np.log(sub[count]), 1)[0])
            if len(sub) > 2
            else 1.0
        )
        alphas[count] = alpha
        report[count] = round(alpha, 4)
    return {"alphas": alphas, "report": {"elasticities": report, "n_teams": int(len(merged))}}


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def analyse_league(league: str, eligible: pd.DataFrame) -> dict:
    usable = [c for c in F.OUTFIELD_CORE if c in eligible.columns]
    standardised = PP.zscore(eligible.copy(), usable, group=None)
    matrix = standardised[usable].to_numpy(dtype=float)

    tag = f"league-{slug(league)}"
    curves = C.internal_curves(matrix, tag)
    rule = C.apply_k_rule(curves)
    k = int(rule["chosen_k"])
    labels = C._kmeans(matrix, k, tag).labels_
    stability = C.bootstrap_stability(matrix, labels, k, tag)

    groups = eligible["position_group"].to_numpy()
    return {
        "league": league,
        "n_eligible": int(len(eligible)),
        "chosen_k": k,
        "rule_branch": rule["rule_branch"],
        "silhouette": round(float(silhouette_score(matrix, labels)), 4),
        "bootstrap_ari_mean": round(float(stability["ari_mean"]), 4),
        "bootstrap_ari_sd": round(float(stability["ari_sd"]), 4),
        "ari_vs_position_group": round(float(adjusted_rand_score(groups, labels)), 4),
        "position_group_counts": eligible["position_group"].value_counts().to_dict(),
    }


def figure_leagues(rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    frame["label"] = frame["league"].map(lambda x: SHORT.get(x, x))
    frame = frame.sort_values("silhouette", ascending=True)

    fig, axes = plt.subplots(1, 3, figsize=(P.WIDTH_FULL, 2.6))
    panels = [
        ("silhouette", "Silhouette"),
        ("bootstrap_ari_mean", "Bootstrap ARI"),
        ("ari_vs_position_group", "ARI against listed position"),
    ]
    y = np.arange(len(frame))
    for ax, (key, title) in zip(axes, panels, strict=True):
        colours = [
            P.CATEGORICAL[1] if lg == config.LEAGUE else P.CATEGORICAL[0] for lg in frame["league"]
        ]
        ax.barh(y, frame[key], color=colours, height=0.62, edgecolor=P.SURFACE, linewidth=0.8)
        for yi, v in zip(y, frame[key], strict=True):
            ax.text(
                v + max(frame[key]) * 0.03,
                yi,
                f"{v:.3f}",
                va="center",
                fontsize=P.BASE_FONT_PT - 3,
                color=P.INK_SECONDARY,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(frame["label"], fontsize=P.BASE_FONT_PT - 2)
        P.style_axis(ax, "", "", title)
        ax.grid(False, axis="y")
        ax.set_xlim(0, max(frame[key]) * 1.28)

    ks = ", ".join(f"{SHORT.get(r['league'], r['league'])} k={r['chosen_k']}" for r in rows)
    fig.suptitle(
        f"The same pipeline across the big five, {config.SEASON_PRIMARY}.  {ks}",
        fontsize=P.BASE_FONT_PT,
        fontweight="bold",
    )
    P.save_figure(fig, "league_comparison")


def main() -> None:
    P.use_style()
    season = config.SEASON_PRIMARY

    rows, details = [], {}
    # The Premier League is re-analysed from its own processed table so the comparison is
    # like for like rather than quoting a number computed by a different code path.
    epl = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")
    rows.append(analyse_league(config.LEAGUE, epl))

    for league in LEAGUES:
        ingest_league(league, season)
        eligible, info = prepare_league(league, season)
        details[league] = info
        rows.append(analyse_league(league, eligible))

    figure_leagues(rows)

    ks = {r["league"]: r["chosen_k"] for r in rows}
    payload = {
        "season": season,
        "question": (
            "Is the two cluster result a property of the reduced feature set or of the "
            "Premier League?"
        ),
        "leagues": rows,
        "preparation": details,
        "k_by_league": ks,
        "k_is_unanimous": len(set(ks.values())) == 1,
        "verdict": (
            "The selection rule returns the same number of clusters in every big five "
            "league, so the result follows from what the data can still measure rather "
            "than from English football."
            if len(set(ks.values())) == 1
            else "The number of clusters differs by league, so the result is not purely a "
            "property of the feature set."
        ),
    }
    with open(config.METRICS / "league_comparison.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    for r in rows:
        print(
            f"{SHORT.get(r['league'], r['league']):9s} n={r['n_eligible']:>4} k={r['chosen_k']} "
            f"silhouette={r['silhouette']:.3f} bootstrapARI={r['bootstrap_ari_mean']:.3f} "
            f"ARIvsPos={r['ari_vs_position_group']:.3f}",
            flush=True,
        )
    print(f"k unanimous: {payload['k_is_unanimous']}")
    print("leagues complete")


if __name__ == "__main__":
    main()
