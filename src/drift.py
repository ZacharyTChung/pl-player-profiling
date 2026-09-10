"""Role drift within a season.

The main analysis treats a season as one observation per player, which assumes a player's
profile is homogeneous across it. That assumption is testable with match level data, and
the test matters: if profiles move substantially within a season, a season aggregate
describes an average of several roles rather than one role.

Match reports survived the January 2026 withdrawal better than the season tables did. The
season tables kept their full headers and lost their values, while the match summary table
simply dropped the withdrawn columns from its schema. What remains is populated: shots,
shots on target, goals, assists, crosses, interceptions, tackles won, fouls committed and
drawn, offsides and cards. The expected goals block is absent entirely, so this analysis
runs in a reduced feature space and says so.

Because a full season is 380 match reports at seven seconds each, the scrape is restricted
to one club, which is what the study design allowed for. Every match report covers both
sides, so a single club's fixtures also yield partial coverage of the other nineteen, but
only the chosen club has a complete set and only its players are analysed.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import config
from src import cluster as C
from src import plotting as P
from src import preprocess as PP

#: The club whose fixtures are scraped: the most eligible outfield players spanning all
#: six archetypes, so drift has the widest range of roles to move between.
DRIFT_TEAM = "Tottenham"

#: Match summary column -> canonical name. These are exactly the season features that
#: survive at match level; the Understat expected-goals block has no match equivalent.
MATCH_FEATURES = {
    "Performance__Sh": "shots",
    "Performance__SoT": "shots_on_target",
    "Performance__Ast": "assists",
    "Performance__Crs": "crosses",
    "Performance__Int": "interceptions",
    "Performance__TklW": "tackles_won",
    "Performance__Fls": "fouls_committed",
    "Performance__Fld": "fouls_drawn",
    "Performance__Off": "offsides",
    "Performance__CrdY": "cards_yellow",
}

#: Rolling window in matches. Six is a compromise: long enough that a per-90 rate is not
#: dominated by one fixture, short enough that a change within a season can still show.
WINDOW = 6
MIN_WINDOW_MINUTES = 180


def _flat(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        c if isinstance(c, str) else "__".join(str(x) for x in c).rstrip("_") for c in out.columns
    ]
    return out


def scrape_team(season: str, team: str = DRIFT_TEAM) -> pd.DataFrame:
    """Pull every match report for one club, caching the assembled frame."""
    cached = config.DATA_PROCESSED / f"match_stats_{team.replace(' ', '_')}_{season}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    from src.ingest import FBrefFull

    fb = FBrefFull(leagues=config.LEAGUE, seasons=season)
    schedule = fb.read_schedule(force_cache=True).reset_index()
    schedule = schedule[schedule["game_id"].notna()]
    mask = (schedule["home_team"] == team) | (schedule["away_team"] == team)
    ids = schedule.loc[mask, "game_id"].tolist()
    if not ids:
        raise ValueError(f"no fixtures found for {team} in {season}")

    frames = []
    for i, gid in enumerate(ids, start=1):
        stats = _flat(fb.read_player_match_stats("summary", match_id=gid, force_cache=True))
        stats["_order"] = i
        frames.append(stats.reset_index())
        print(f"  match {i}/{len(ids)}", flush=True)

    out = _flat(pd.concat(frames, ignore_index=True))
    out.to_parquet(cached, index=False)
    return out


def resolve_team(matches: pd.DataFrame, team: str = DRIFT_TEAM) -> str:
    """Match reports name clubs in full, the rest of the pipeline uses short forms.

    The schedule and season tables call the club Tottenham while a match report calls it
    Tottenham Hotspur, so an exact comparison silently selects no rows. Names are folded
    and compared by prefix, and the resolution is asserted rather than assumed.
    """
    target = PP.normalize_team(team)
    candidates = {
        raw: PP.normalize_team(raw) for raw in matches["team"].dropna().astype(str).unique()
    }
    hits = [
        raw
        for raw, key in candidates.items()
        if key == target or key.startswith(target) or target.startswith(key)
    ]
    if len(hits) != 1:
        raise ValueError(f"{team!r} resolved to {hits} among {sorted(candidates.values())}")
    return hits[0]


def rolling_profiles(matches: pd.DataFrame, team: str = DRIFT_TEAM) -> pd.DataFrame:
    """Rolling per-90 profile for each of the club's players across the season."""
    df = matches[matches["team"] == resolve_team(matches, team)].copy()
    if df.empty:
        raise ValueError(f"no match rows for {team!r} after name resolution")
    df["minutes"] = pd.to_numeric(df.get("min", df.get("min__")), errors="coerce").fillna(0.0)
    for raw, canon in MATCH_FEATURES.items():
        df[canon] = pd.to_numeric(df[raw], errors="coerce").fillna(0.0)
    df = df.sort_values(["player", "_order"])

    rows = []
    counts = list(MATCH_FEATURES.values())
    for player, grp in df.groupby("player"):
        if len(grp) < WINDOW:
            continue
        rolled = grp[counts].rolling(WINDOW, min_periods=WINDOW).sum()
        minutes = grp["minutes"].rolling(WINDOW, min_periods=WINDOW).sum()
        for idx in range(len(grp)):
            total_minutes = minutes.iloc[idx]
            if not np.isfinite(total_minutes) or total_minutes < MIN_WINDOW_MINUTES:
                continue
            nineties = total_minutes / 90.0
            record = {
                "player": player,
                "window_end_match": int(grp["_order"].iloc[idx]),
                "window_minutes": float(total_minutes),
            }
            for c in counts:
                record[f"{c}_p90"] = float(rolled[c].iloc[idx]) / nineties
            rows.append(record)
    return pd.DataFrame(rows)


def reduced_reference(season: str) -> dict:
    """Cluster the season pool on only the features that exist at match level.

    Reported alongside the adjusted Rand index against the main partition, so a reader
    knows how much of the main result this coarser space reproduces before any drift
    claim is made on top of it.
    """
    eligible = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")
    usable = [f"{c}_p90" for c in MATCH_FEATURES.values()]
    usable = [c for c in usable if c in eligible.columns]

    standardised = PP.zscore(eligible.copy(), usable, group=None)
    matrix = standardised[usable].to_numpy(dtype=float)
    curves = C.internal_curves(matrix, "drift-reference")
    k = int(C.apply_k_rule(curves)["chosen_k"])
    labels = C._kmeans(matrix, k, "drift-reference").labels_

    main = pd.read_parquet(config.DATA_PROCESSED / f"clusters_{season}.parquet")
    merged = (
        eligible[["player"]]
        .assign(reduced=labels)
        .merge(main[["player", "cluster_global"]], on="player", how="inner")
    )
    ari = float(adjusted_rand_score(merged["cluster_global"], merged["reduced"]))

    centroids = np.vstack([matrix[labels == c].mean(axis=0) for c in range(k)])
    return {
        "features": usable,
        "k": k,
        "labels": pd.Series(labels, index=eligible["player"].to_numpy()),
        "centroids": centroids,
        "mean": standardised.attrs.get("mean"),
        "raw_mean": eligible[usable].mean().to_numpy(dtype=float),
        "raw_std": eligible[usable].std(ddof=0).to_numpy(dtype=float),
        "ari_vs_main": round(ari, 4),
        "n_compared": int(len(merged)),
    }


def assign_windows(profiles: pd.DataFrame, reference: dict) -> pd.DataFrame:
    """Place each rolling window at its nearest reduced-space centroid."""
    usable = reference["features"]
    values = profiles[usable].to_numpy(dtype=float)
    std = np.where(reference["raw_std"] > 0, reference["raw_std"], 1.0)
    z = (values - reference["raw_mean"]) / std
    distances = np.linalg.norm(z[:, None, :] - reference["centroids"][None, :, :], axis=2)
    out = profiles.copy()
    out["assigned"] = distances.argmin(axis=1)
    out["margin"] = np.abs(distances[:, 0] - distances[:, 1]) if distances.shape[1] == 2 else np.nan
    return out


def figure_drift(assigned: pd.DataFrame, movers: pd.DataFrame) -> None:
    names = movers["player"].tolist()[:6]
    if not names:
        return
    fig, axes = plt.subplots(
        len(names), 1, figsize=(P.WIDTH_FULL, 0.72 * len(names) + 1.0), sharex=True
    )
    axes = np.atleast_1d(axes)
    palette = P.categorical(2)

    for ax, name in zip(axes, names, strict=True):
        sub = assigned[assigned["player"] == name].sort_values("window_end_match")
        ax.plot(
            sub["window_end_match"], sub["assigned"], color=P.INK_MUTED, linewidth=1.0, zorder=2
        )
        for value in (0, 1):
            m = sub["assigned"] == value
            ax.scatter(
                sub.loc[m, "window_end_match"],
                sub.loc[m, "assigned"],
                s=26,
                c=palette[value],
                marker=P.MARKERS[value],
                edgecolors=P.SURFACE,
                linewidths=0.5,
                zorder=4,
            )
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["cluster 0", "cluster 1"], fontsize=P.BASE_FONT_PT - 3)
        ax.set_ylim(-0.45, 1.45)
        ax.set_ylabel("")
        ax.text(
            0.005,
            0.5,
            name,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=P.BASE_FONT_PT - 2,
            color=P.INK_PRIMARY,
        )
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    axes[-1].set_xlabel(f"Match number, rolling window of {WINDOW}")
    P.save_figure(fig, "role_drift")


def main() -> None:
    P.use_style()
    season = config.SEASON_PRIMARY

    matches = scrape_team(season)
    resolved = resolve_team(matches)
    profiles = rolling_profiles(matches)
    if profiles.empty:
        raise ValueError("no rolling windows met the minutes requirement")
    reference = reduced_reference(season)
    assigned = assign_windows(profiles, reference)

    summary = []
    for player, grp in assigned.groupby("player"):
        seq = grp.sort_values("window_end_match")["assigned"].to_numpy()
        switches = int((np.diff(seq) != 0).sum())
        summary.append(
            {
                "player": player,
                "n_windows": int(len(seq)),
                "switches": switches,
                "share_cluster_one": round(float(seq.mean()), 3),
            }
        )
    table = pd.DataFrame(summary).sort_values(["switches", "n_windows"], ascending=False)
    movers = table[table["switches"] > 0]

    payload = {
        "season": season,
        "team": DRIFT_TEAM,
        "team_as_named_in_match_reports": resolved,
        "window_matches": WINDOW,
        "min_window_minutes": MIN_WINDOW_MINUTES,
        "match_features": list(MATCH_FEATURES.values()),
        "n_matches_scraped": int(matches["_order"].nunique()),
        "n_players_with_windows": int(table.shape[0]),
        "reduced_space": {
            "k": reference["k"],
            "adjusted_rand_index_vs_main_partition": reference["ari_vs_main"],
            "n_compared": reference["n_compared"],
            "caveat": (
                "The match level feature set omits the Understat expected goals block, "
                "which has no match equivalent, so drift is measured in a coarser space "
                "than the main analysis. The adjusted Rand index above says how much of "
                "the main partition that coarser space reproduces."
            ),
        },
        "n_players_switching": int(len(movers)),
        "share_of_players_switching": round(float(len(movers) / max(len(table), 1)), 3),
        "per_player": table.to_dict(orient="records"),
    }
    with open(config.METRICS / "role_drift.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    figure_drift(assigned, movers)

    print(
        f"{DRIFT_TEAM}: {payload['n_matches_scraped']} matches, "
        f"{payload['n_players_with_windows']} players with a full window",
        flush=True,
    )
    print(
        f"reduced space k={reference['k']} agrees with the main partition at "
        f"ARI {reference['ari_vs_main']}",
        flush=True,
    )
    print(
        f"{payload['n_players_switching']} players change cluster at least once "
        f"({payload['share_of_players_switching']:.0%})",
        flush=True,
    )
    print("drift complete")


if __name__ == "__main__":
    main()
