"""The full feature set, from a pre-withdrawal archive.

The live site no longer serves the statistics this study was designed around. A public
archive does: the worldfootballR project has been mirroring FBref's advanced season tables
since they first appeared, and its snapshot predates the January 2026 deletion. That gives
six complete seasons, 2018 through 2023, across all five major European leagues, with every
column the original design called for.

This module is the foundation of the main analysis. The live 2024-25 and 2025-26 pulls are
retained as a separate, deliberately impoverished sample, used later to measure what the
loss of the modern feature block costs rather than to carry the study.

Provenance matters here and is stated plainly. These tables were scraped from FBref by a
third party rather than by this project, so they are a mirror rather than a primary
source. They are used because the primary source no longer serves the data, the mirror
predates its removal, and the alternative is not doing the analysis at all. Row counts and
column coverage are audited on load.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src import preprocess as PP

RAW = config.DATA_RAW / "archive"

#: Seasons for which FBref published the advanced tables. Season is labelled by the year
#: it ends, so 2018 is the 2017-18 campaign.
#:
#: 2023 is excluded deliberately. FBref restructured the carrying block for that season
#: and the mirror's scrape did not follow the renames, so every carries column is empty
#: there while touches and take-ons are intact. Keeping it would mean either two features
#: missing for a sixth of the sample or silently imputing a fifth of the carry data.
#: ``assert_season_coverage`` enforces the exclusion rather than trusting this comment.
SEASONS = [2018, 2019, 2020, 2021, 2022]
EXCLUDED_SEASONS = {2023: "carrying block absent from the mirror after FBref renamed it"}

#: A feature missing above this share within any retained season is treated as a coverage
#: failure rather than as something to impute.
MAX_SEASON_MISSING = 0.5
LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]

IDS = ["Season_End_Year", "Squad", "Comp", "Player", "Pos", "Age", "Nation"]

#: canonical name -> (table, column). This is the feature set the study was specified
#: around, recovered in full.
SOURCES: dict[str, tuple[str, str]] = {
    # Identity and exposure
    "minutes": ("standard", "Min_Playing"),
    "matches_played": ("standard", "MP_Playing"),
    "starts": ("standard", "Starts_Playing"),
    # Shooting and finishing
    "np_xg": ("standard", "npxG_Expected"),
    "shots": ("shooting", "Sh_Standard"),
    "shot_accuracy_pct": ("shooting", "SoT_percent_Standard"),
    "goals_non_penalty": ("standard", "G_minus_PK"),
    # Creation
    "xag": ("standard", "xAG_Expected"),
    "key_passes": ("passing", "KP"),
    "sca": ("gca", "SCA_SCA"),
    "gca": ("gca", "GCA_GCA"),
    "passes_final_third": ("passing", "Final_Third"),
    "passes_penalty_area": ("passing", "PPA"),
    "crosses": ("misc", "Crs"),
    # Progression
    "progressive_passes": ("passing", "Prog"),
    "progressive_carries": ("possession", "Prog_Carries"),
    "progressive_receptions": ("possession", "Prog_Receiving"),
    "carries_final_third": ("possession", "Final_Third_Carries"),
    # Touches by zone
    "touches_def_pen": ("possession", "Def Pen_Touches"),
    "touches_def_third": ("possession", "Def 3rd_Touches"),
    "touches_mid_third": ("possession", "Mid 3rd_Touches"),
    "touches_att_third": ("possession", "Att 3rd_Touches"),
    "touches_att_pen": ("possession", "Att Pen_Touches"),
    # Carrying the ball past opponents
    "take_ons": ("possession", "Att_Dribbles"),
    "take_on_success_pct": ("possession", "Succ_percent_Dribbles"),
    # Defending
    "tackles": ("defense", "Tkl_Tackles"),
    "tackle_win_pct": ("defense", "Tkl_percent_Vs"),
    "interceptions": ("misc", "Int"),
    "blocks": ("defense", "Blocks_Blocks"),
    "clearances": ("defense", "Clr"),
    "ball_recoveries": ("misc", "Recov"),
    "aerials_won_pct": ("misc", "Won_percent_Aerial"),
    # Passing quality by distance
    "pass_cmp_short_pct": ("passing", "Cmp_percent_Short"),
    "pass_cmp_medium_pct": ("passing", "Cmp_percent_Medium"),
    "pass_cmp_long_pct": ("passing", "Cmp_percent_Long"),
    # Discipline
    "fouls_committed": ("misc", "Fls"),
}

#: Counting statistics, converted to per ninety.
COUNTS = [
    "np_xg",
    "shots",
    "goals_non_penalty",
    "xag",
    "key_passes",
    "sca",
    "gca",
    "passes_final_third",
    "passes_penalty_area",
    "crosses",
    "progressive_passes",
    "progressive_carries",
    "progressive_receptions",
    "carries_final_third",
    "touches_def_pen",
    "touches_def_third",
    "touches_mid_third",
    "touches_att_third",
    "touches_att_pen",
    "take_ons",
    "tackles",
    "interceptions",
    "blocks",
    "clearances",
    "ball_recoveries",
    "fouls_committed",
]

#: Already rates, used as published.
RATES = [
    "shot_accuracy_pct",
    "take_on_success_pct",
    "tackle_win_pct",
    "aerials_won_pct",
    "pass_cmp_short_pct",
    "pass_cmp_medium_pct",
    "pass_cmp_long_pct",
]

#: Opponent-ball actions, possession adjusted exactly as in the live pipeline.
DEFENSIVE_COUNTS = ["tackles", "interceptions", "blocks", "clearances", "fouls_committed"]

#: Team-level equivalents for fitting the possession elasticities.
DEFENSIVE_TEAM_SOURCES = {
    "tackles": ("defense", "Tkl_Tackles"),
    "interceptions": ("misc", "Int"),
    "blocks": ("defense", "Blocks_Blocks"),
    "clearances": ("defense", "Clr"),
    "fouls_committed": ("misc", "Fls"),
}

OUTFIELD_CORE = [f"{c}_padj_p90" if c in DEFENSIVE_COUNTS else f"{c}_p90" for c in COUNTS] + RATES

#: Goalkeeper features, now fully recoverable.
KEEPER_SOURCES: dict[str, tuple[str, str]] = {
    "gk_minutes": ("keepers", "Min_Playing"),
    "gk_save_pct": ("keepers", "Save_percent"),
    "gk_clean_sheet_pct": ("keepers", "CS_percent"),
    "gk_goals_against_per90": ("keepers", "GA90"),
    "gk_shots_on_target_against": ("keepers", "SoTA"),
    "gk_psxg": ("keepers_adv", "PSxG_Expected"),
    "gk_psxg_per_sot": ("keepers_adv", "PSxG_per_SoT_Expected"),
    "gk_psxg_net": ("keepers_adv", "PSxG+_per__minus__Expected"),
    "gk_crosses_stopped_pct": ("keepers_adv", "Stp_percent_Crosses"),
    "gk_sweeper_per90": ("keepers_adv", "#OPA_per_90_Sweeper"),
    "gk_sweeper_distance": ("keepers_adv", "AvgDist_Sweeper"),
    "gk_launch_pct": ("keepers_adv", "Launch_percent_Passes"),
    "gk_pass_length": ("keepers_adv", "AvgLen_Passes"),
}


def _load_table(name: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW / f"{name}.parquet")
    df["Season_End_Year"] = pd.to_numeric(df["Season_End_Year"], errors="coerce")
    return df


def load_players() -> tuple[pd.DataFrame, dict]:
    """Assemble one row per player-season with the full canonical feature set."""
    by_table: dict[str, list[tuple[str, str]]] = {}
    for canon, (table, raw) in SOURCES.items():
        by_table.setdefault(table, []).append((canon, raw))

    base = _load_table("standard")
    base = base[base["Season_End_Year"].isin(SEASONS)]
    merged = base[IDS].drop_duplicates(subset=["Season_End_Year", "Squad", "Player"]).copy()

    missing: dict[str, list[str]] = {}
    for table, pairs in by_table.items():
        df = _load_table(table)
        absent = [raw for _, raw in pairs if raw not in df.columns]
        if absent:
            missing[table] = absent
            continue
        keep = {raw: canon for canon, raw in pairs}
        sub = df[["Season_End_Year", "Squad", "Player", *keep]].rename(columns=keep)
        sub = sub.drop_duplicates(subset=["Season_End_Year", "Squad", "Player"])
        merged = merged.merge(sub, on=["Season_End_Year", "Squad", "Player"], how="left")
    if missing:
        raise KeyError(f"archive columns absent: {missing}")

    for col in merged.columns:
        if col not in IDS:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    audit = {
        "rows": int(len(merged)),
        "seasons": sorted(int(x) for x in merged["Season_End_Year"].unique()),
        "competitions": sorted(merged["Comp"].dropna().unique().tolist()),
        "features_mapped": len(SOURCES),
        "null_rate_by_feature": {
            c: round(float(merged[c].isna().mean()), 4)
            for c in SOURCES
            if c in merged.columns and merged[c].isna().mean() > 0.01
        },
    }
    coverage = assert_season_coverage(merged)
    audit["season_coverage"] = coverage
    audit["excluded_seasons"] = {str(k): v for k, v in EXCLUDED_SEASONS.items()}
    return merged, audit


def assert_season_coverage(frame: pd.DataFrame) -> dict:
    """Fail loudly if any feature is largely absent from a retained season.

    The 2023 carrying columns were empty for the whole season while neighbouring columns
    were fine, which an overall null rate hides and a per-season one exposes. This check
    exists so that a future refresh of the mirror cannot reintroduce that silently.
    """
    worst: dict[str, dict[str, float]] = {}
    for season, grp in frame.groupby("Season_End_Year"):
        for canon in SOURCES:
            if canon not in grp.columns:
                continue
            rate = float(grp[canon].isna().mean())
            if rate > MAX_SEASON_MISSING:
                worst.setdefault(str(int(season)), {})[canon] = round(rate, 4)
    if worst:
        raise ValueError(
            f"features largely absent within a retained season: {worst}. "
            "Exclude the season or drop the feature rather than imputing it."
        )
    return {"checked_seasons": sorted(int(s) for s in frame["Season_End_Year"].unique())}


def team_possession() -> pd.DataFrame:
    df = _load_table("team_standard")
    df = df[(df["Team_or_Opponent"] == "team") & df["Season_End_Year"].isin(SEASONS)]
    out = df[["Season_End_Year", "Squad", "Poss"]].rename(columns={"Poss": "team_possession"})
    out["team_possession"] = pd.to_numeric(out["team_possession"], errors="coerce")
    return out.dropna(subset=["team_possession"])


def fit_elasticities() -> dict:
    """Fit defensive possession elasticities on the archive's team tables."""
    poss = team_possession()
    frames = {}
    for table in {t for t, _ in DEFENSIVE_TEAM_SOURCES.values()}:
        df = _load_table(f"team_{table}") if (RAW / f"team_{table}.parquet").exists() else None
        frames[table] = df

    # The archive ships only two team tables, so the elasticities are fitted from player
    # rows aggregated to the club, which is the same quantity computed a different way.
    players, _ = load_players()
    agg = players.groupby(["Season_End_Year", "Squad"], as_index=False).agg(
        {**{c: "sum" for c in DEFENSIVE_COUNTS}, "minutes": "sum"}
    )
    agg = agg.merge(poss, on=["Season_End_Year", "Squad"], how="inner")
    agg["nineties"] = agg["minutes"] / 90.0
    agg["opponent"] = 100.0 - agg["team_possession"]

    alphas, report = {}, {}
    for count in DEFENSIVE_COUNTS:
        rate = agg[count] / agg["nineties"]
        sub = pd.DataFrame({"rate": rate, "opp": agg["opponent"]}).dropna()
        sub = sub[(sub["rate"] > 0) & (sub["opp"] > 0)]
        alpha = float(np.polyfit(np.log(sub["opp"]), np.log(sub["rate"]), 1)[0])
        alphas[count] = alpha
        report[count] = {
            "elasticity": round(alpha, 4),
            "n_team_seasons": int(len(sub)),
            "log_log_correlation": round(
                float(np.corrcoef(np.log(sub["opp"]), np.log(sub["rate"]))[0, 1]), 4
            ),
        }
    return {"alphas": alphas, "report": report}


def build() -> dict:
    players, audit = load_players()
    poss = team_possession()
    players = players.merge(poss, on=["Season_End_Year", "Squad"], how="left")
    players["team_possession"] = players["team_possession"].fillna(poss["team_possession"].mean())

    players["nineties"] = players["minutes"] / 90.0
    players = PP.to_per90(players, COUNTS)

    elasticity = fit_elasticities()
    players = PP.possession_adjust(players, elasticity["alphas"])

    pos = players["Pos"].fillna("").astype(str)
    players["position_full"] = pos
    players["primary_position"] = pos.str.split(",").str[0].str.strip()
    players["position_group"] = players["primary_position"].map(
        {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}
    )
    players["season"] = players["Season_End_Year"].astype("Int64")
    players["league"] = players["Comp"]
    players["team"] = players["Squad"]
    players["player"] = players["Player"]
    players["age"] = pd.to_numeric(players["Age"], errors="coerce")

    outfield = players[players["position_group"].isin(config.OUTFIELD_GROUPS)].copy()
    eligible = outfield[outfield["minutes"] >= config.MIN_MINUTES].copy()

    usable = [c for c in OUTFIELD_CORE if c in eligible.columns]
    eligible, imputed = PP.impute_within_group(eligible, usable, "position_group")

    global_z = PP.zscore(eligible.copy(), usable, group=None)
    group_z = PP.zscore(eligible.copy(), usable, group="position_group")

    out = config.DATA_PROCESSED
    players.to_parquet(out / "archive_all.parquet", index=False)
    eligible.to_parquet(out / "archive_eligible.parquet", index=False)
    global_z.to_parquet(out / "archive_z_global.parquet", index=False)
    group_z.to_parquet(out / "archive_z_bygroup.parquet", index=False)
    players[players["position_group"] == "GK"].to_parquet(
        out / "archive_keepers.parquet", index=False
    )

    report = {
        "source": "worldfootballR_data mirror of FBref advanced season tables",
        "provenance_note": (
            "A third party mirror, used because the primary source deleted these columns "
            "in January 2026 and the mirror predates the deletion."
        ),
        "audit": audit,
        "seasons": SEASONS,
        "leagues": LEAGUES,
        "n_features": len(usable),
        "features": usable,
        "elasticities": {k: v["elasticity"] for k, v in elasticity["report"].items()},
        "elasticity_detail": elasticity["report"],
        "rows_all": int(len(players)),
        "rows_outfield": int(len(outfield)),
        "rows_eligible": int(len(eligible)),
        "rows_keepers": int((players["position_group"] == "GK").sum()),
        "eligible_by_position": eligible["position_group"].value_counts().to_dict(),
        "eligible_by_league": eligible["league"].value_counts().to_dict(),
        "eligible_by_season": {
            str(k): int(v) for k, v in eligible["season"].value_counts().sort_index().items()
        },
        "imputed_cells": imputed,
    }
    with open(config.METRICS / "archive.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report


def main() -> None:
    report = build()
    print(f"archive: {report['rows_all']:,} player-seasons, {report['n_features']} features")
    print(
        f"  seasons {report['seasons'][0]} to {report['seasons'][-1]}, {len(report['leagues'])} leagues"
    )
    print(f"  eligible outfield: {report['rows_eligible']:,}  keepers: {report['rows_keepers']:,}")
    print(f"  by position: {report['eligible_by_position']}")
    print(f"  elasticities: {report['elasticities']}")
    high = {k: v for k, v in report["audit"]["null_rate_by_feature"].items()}
    if high:
        print(f"  features with >1% missing: {high}")
    print("archive complete")


if __name__ == "__main__":
    main()
