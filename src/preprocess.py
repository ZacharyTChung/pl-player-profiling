"""Merge, filter, normalise and standardise the raw pulls.

Produces the analysis tables under ``data/processed`` and a set of provenance metrics
under ``results/metrics``. Nothing here invents data: where a value is missing it is
either imputed with an explicit flag or the row is dropped, and both are counted.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import get_close_matches

import numpy as np
import pandas as pd

import config
from src import features as F

# Characters that NFKD does not decompose but that differ between the two sources.
_TRANSLIT = str.maketrans(
    {
        "ø": "o",
        "Ø": "o",
        "đ": "d",
        "Đ": "d",
        "ß": "ss",
        "ł": "l",
        "Ł": "l",
        "æ": "ae",
        "Æ": "ae",
        "þ": "th",
        "ð": "d",
        "ı": "i",
    }
)

#: Understat team name -> FBref team name. The two sites disagree on seven clubs.
TEAM_ALIASES = {
    "ipswich": "ipswich town",
    "leicester": "leicester city",
    "manchester united": "manchester utd",
    "newcastle united": "newcastle",
    "nottingham forest": "nottingham",
    "wolverhampton wanderers": "wolves",
    "leeds": "leeds united",
}


def normalize_name(value: str) -> str:
    """Fold a player or team name to a comparable ASCII key."""
    text = str(value).translate(_TRANSLIT)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("-", " ").replace("'", "").replace(".", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_team(value: str) -> str:
    key = normalize_name(value)
    return TEAM_ALIASES.get(key, key)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_fbref_canonical(season: str) -> pd.DataFrame:
    """Assemble one row per (team, player) with the canonical FBref columns."""
    by_table: dict[str, list[tuple[str, str]]] = {}
    for canon, (table, raw) in F.FBREF_SOURCES.items():
        by_table.setdefault(table, []).append((canon, raw))

    # The standard table defines the population. Every other table is joined onto it, so
    # that rows unique to playing_time (squad members who never took the field) do not
    # enter the analysis or distort the cross-source match rate.
    ordered = ["standard", *[t for t in by_table if t != "standard"]]

    merged: pd.DataFrame | None = None
    for table in ordered:
        pairs = by_table[table]
        path = config.DATA_RAW / season / f"players_{table}.parquet"
        df = pd.read_parquet(path)
        keep = {raw: canon for canon, raw in pairs if raw in df.columns}
        missing = [raw for canon, raw in pairs if raw not in df.columns]
        if missing:
            raise KeyError(f"{season}/{table}: expected columns absent: {missing}")

        cols = ["team", "player", *keep]
        sub = df[cols].rename(columns=keep)
        # Carry the identity columns once, from the standard table.
        if table == "standard":
            sub = sub.join(df[["pos", "age", "nation"]])
        sub = sub.drop_duplicates(subset=["team", "player"])
        merged = sub if merged is None else merged.merge(sub, on=["team", "player"], how="left")

    assert merged is not None
    merged.insert(0, "season", season)
    for col in merged.columns:
        if col not in {"season", "team", "player", "pos", "nation"}:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


def load_understat(season: str) -> pd.DataFrame:
    df = pd.read_parquet(config.DATA_RAW / season / "understat_players.parquet")
    keep = {raw: canon for canon, raw in F.UNDERSTAT_SOURCES.items()}
    out = df[["team", "player", *keep]].rename(columns=keep)
    out["_team_key"] = out["team"].map(normalize_team)
    out["_name_key"] = out["player"].map(normalize_name)
    return out.drop(columns=["team", "player"])


def load_team_possession(season: str) -> pd.DataFrame:
    """Team possession share, needed to adjust defensive counts for exposure.

    Possession survived the January 2026 withdrawal even though the touch counts it is
    derived from did not, which is what makes the adjustment possible at all.
    """
    df = pd.read_parquet(config.DATA_RAW / season / "teams_standard.parquet")
    if "Poss" not in df.columns:
        raise KeyError(f"{season}: teams_standard has no Poss column")
    out = df[["team", "Poss"]].rename(columns={"Poss": "team_possession"})
    out["team_possession"] = pd.to_numeric(out["team_possession"], errors="coerce")
    if out["team_possession"].isna().any():
        raise ValueError(f"{season}: team possession is not fully populated")
    return out


# --------------------------------------------------------------------------------------
# Cross-source matching
# --------------------------------------------------------------------------------------


def match_understat(fbref: pd.DataFrame, understat: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach Understat columns to FBref rows.

    Three passes, each more permissive than the last, and every pass requires the two
    records to belong to the same club:

    1. exact normalised name,
    2. token subset, which resolves short forms such as "Kepa" against
       "Kepa Arrizabalaga", accepted only when exactly one candidate matches,
    3. close string match at a 0.84 cutoff.

    A player who cannot be matched keeps NaN Understat values and is counted.
    """
    fb = fbref.copy()
    fb["_team_key"] = fb["team"].map(normalize_team)
    fb["_name_key"] = fb["player"].map(normalize_name)

    us = understat.copy()
    us_by_team: dict[str, pd.DataFrame] = dict(tuple(us.groupby("_team_key")))

    us_cols = [c for c in us.columns if not c.startswith("_")]
    assignments: dict[int, int] = {}
    method_counts = {"exact": 0, "token_subset": 0, "fuzzy": 0, "unmatched": 0}
    ambiguous: list[str] = []

    for idx, row in fb.iterrows():
        pool = us_by_team.get(row["_team_key"])
        if pool is None or pool.empty:
            method_counts["unmatched"] += 1
            continue

        hit = pool.index[pool["_name_key"] == row["_name_key"]]
        if len(hit) == 1:
            assignments[idx] = hit[0]
            method_counts["exact"] += 1
            continue

        tokens = set(row["_name_key"].split())
        subset = [
            j
            for j, cand in pool["_name_key"].items()
            if tokens and (tokens <= set(cand.split()) or set(cand.split()) <= tokens)
        ]
        if len(subset) == 1:
            assignments[idx] = subset[0]
            method_counts["token_subset"] += 1
            continue
        if len(subset) > 1:
            ambiguous.append(row["player"])

        close = get_close_matches(row["_name_key"], pool["_name_key"].tolist(), n=1, cutoff=0.84)
        if close:
            j = pool.index[pool["_name_key"] == close[0]][0]
            assignments[idx] = j
            method_counts["fuzzy"] += 1
            continue

        method_counts["unmatched"] += 1

    for col in us_cols:
        fb[col] = np.nan
    if assignments:
        fb_idx = list(assignments.keys())
        us_idx = list(assignments.values())
        fb.loc[fb_idx, us_cols] = us.loc[us_idx, us_cols].to_numpy()

    matched = len(assignments)
    report = {
        "fbref_rows": int(len(fb)),
        "understat_rows": int(len(understat)),
        "matched": int(matched),
        "match_rate": round(matched / len(fb), 4) if len(fb) else 0.0,
        "by_method": method_counts,
        "ambiguous_token_matches": sorted(set(ambiguous)),
    }
    return fb.drop(columns=["_team_key", "_name_key"]), report


# --------------------------------------------------------------------------------------
# Shaping
# --------------------------------------------------------------------------------------

_POSITION_GROUP = {"GK": "GK", "DF": "DF", "MF": "MF", "FW": "FW"}


def parse_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Split FBref's comma separated position string."""
    out = df.copy()
    pos = out["pos"].fillna("").astype(str)
    out["position_full"] = pos
    out["primary_position"] = pos.str.split(",").str[0].str.strip()
    out["position_group"] = out["primary_position"].map(_POSITION_GROUP)
    out["is_hybrid_listed"] = pos.str.contains(",")
    return out


#: Counting columns that must be summed when a player's spells are combined.
_SUMMABLE = [
    "minutes",
    "matches_played",
    "starts",
    "goals",
    "assists",
    "goals_non_penalty",
    "penalties_scored",
    "penalties_attempted",
    "shots",
    "shots_on_target",
    "cards_yellow",
    "cards_red",
    "fouls_committed",
    "fouls_drawn",
    "offsides",
    "crosses",
    "interceptions",
    "tackles_won",
    "subs_on",
    "starts_completed",
    "np_xg",
    "xa",
    "key_passes",
    "xg_chain",
    "xg_buildup",
    "us_shots",
    "us_goals_non_penalty",
    "us_assists",
    "us_minutes",
]


def aggregate_per_player(df: pd.DataFrame) -> pd.DataFrame:
    """Combine a player's rows across clubs into one season row.

    Counting statistics are summed. Rates are recomputed from their components rather
    than averaged, because averaging rates across unequal spells is wrong. Team context
    columns are minutes-weighted since they describe the environment, not the player.
    """
    out = df.copy()
    out["_squads"] = 1

    weighted = [
        "team_points_per_match",
        "team_plus_minus_per90",
        "on_off",
        "pct_squad_minutes",
        "team_possession",
    ]
    for col in weighted:
        out[f"_w_{col}"] = out[col] * out["minutes"]

    agg: dict[str, str] = {c: "sum" for c in _SUMMABLE if c in out.columns}
    agg["_squads"] = "sum"
    for col in weighted:
        agg[f"_w_{col}"] = "sum"

    first = [
        "pos",
        "age",
        "nation",
        "position_full",
        "primary_position",
        "position_group",
        "is_hybrid_listed",
    ]
    for col in first:
        agg[col] = "first"

    grouped = out.sort_values("minutes", ascending=False).groupby(
        ["season", "player"], as_index=False
    )
    res = grouped.agg(agg)

    teams = (
        out.sort_values("minutes", ascending=False)
        .groupby(["season", "player"])["team"]
        .agg(lambda s: " / ".join(dict.fromkeys(s)))
        .reset_index()
    )
    res = res.merge(teams, on=["season", "player"], how="left")

    for col in weighted:
        res[col] = np.where(res["minutes"] > 0, res[f"_w_{col}"] / res["minutes"], np.nan)
        res = res.drop(columns=[f"_w_{col}"])

    res["nineties"] = res["minutes"] / 90.0
    res["shot_accuracy_pct"] = np.where(
        res["shots"] > 0, 100.0 * res["shots_on_target"] / res["shots"], np.nan
    )
    res["goals_per_shot"] = np.where(res["shots"] > 0, res["goals"] / res["shots"], np.nan)
    res = res.rename(columns={"_squads": "n_squads"})
    return res


def estimate_possession_elasticity(seasons: list[str]) -> dict:
    """Fit how strongly team defensive volume actually responds to opponent possession.

    The textbook possession adjustment multiplies a defensive rate by
    ``reference / opponent_possession``, which assumes the rate is directly proportional
    to time spent out of possession. That is an empirical claim, not a definition, and it
    can be checked. Regressing log team defensive volume per ninety on log opponent
    possession across the available team-seasons gives the elasticity: the percentage
    change in defensive volume for a one percent change in opponent possession.

    Estimating from team totals rather than player rates is deliberate. The quantity being
    corrected is a team-level exposure effect, and player rates carry position and role
    variation that would swamp it.

    Pooling the seasons is also deliberate. Twenty clubs give a noisy slope, and the
    exponent is a nuisance parameter rather than a target: it uses no player-level or
    cluster-level information, so a shared value cannot manufacture the cross-season
    agreement that the replication analysis measures.
    """
    frames = []
    for season in seasons:
        misc = pd.read_parquet(config.DATA_RAW / season / f"teams_{'misc'}.parquet")
        std = pd.read_parquet(config.DATA_RAW / season / "teams_standard.parquet")
        merged = misc.merge(std[["team", "Poss"]], on="team", how="inner")
        merged["nineties"] = pd.to_numeric(merged["90s"], errors="coerce")
        merged["opponent_possession"] = 100.0 - pd.to_numeric(merged["Poss"], errors="coerce")
        for count, source in config.PADJ_TEAM_SOURCES.items():
            merged[count] = pd.to_numeric(merged[source], errors="coerce") / merged["nineties"]
        merged["season"] = season
        frames.append(merged)
    pooled = pd.concat(frames, ignore_index=True)

    report: dict = {
        "method": (
            "ordinary least squares of log team defensive volume per ninety on log "
            "opponent possession, pooled across team-seasons"
        ),
        "n_team_seasons": int(len(pooled)),
        "opponent_possession_min": round(float(pooled["opponent_possession"].min()), 2),
        "opponent_possession_max": round(float(pooled["opponent_possession"].max()), 2),
        "per_feature": {},
    }
    alphas: dict[str, float] = {}
    for count in config.PADJ_COUNTS:
        sub = pooled[[count, "opponent_possession", "season"]].dropna()
        sub = sub[(sub[count] > 0) & (sub["opponent_possession"] > 0)]
        x = np.log(sub["opponent_possession"].to_numpy(dtype=float))
        y = np.log(sub[count].to_numpy(dtype=float))
        alpha, intercept = np.polyfit(x, y, 1)
        by_season = {}
        for season, grp in sub.groupby("season"):
            xs = np.log(grp["opponent_possession"].to_numpy(dtype=float))
            ys = np.log(grp[count].to_numpy(dtype=float))
            by_season[str(season)] = round(float(np.polyfit(xs, ys, 1)[0]), 4)
        alphas[count] = float(alpha)
        report["per_feature"][count] = {
            "elasticity": round(float(alpha), 4),
            "elasticity_by_season": by_season,
            "log_log_correlation": round(float(np.corrcoef(x, y)[0, 1]), 4),
            "n": int(len(sub)),
            "unit_elasticity_overcorrection_factor": round(1.0 / float(alpha), 2),
        }
    report["elasticities"] = {k: round(v, 4) for k, v in alphas.items()}
    return {"alphas": alphas, "report": report}


def possession_adjust(df: pd.DataFrame, alphas: dict[str, float]) -> pd.DataFrame:
    """Rescale defensive per-90 counts to a common opponent-possession baseline.

    Defensive actions require the opponent to have the ball, so their raw rate confounds
    what a player does with how often his team is out of possession. Each count is
    multiplied by ``(reference / opponent_possession) ** alpha``, where alpha is the
    measured elasticity from :func:`estimate_possession_elasticity` rather than the
    conventional 1.0. A player at a club with league-average possession is unchanged; the
    rest are corrected toward what they would have recorded facing an average share of
    the ball, by the amount the data says exposure actually matters.

    The adjustment is applied to the per-90 rate, so the result is still a rate per ninety
    minutes and remains directly comparable across players.
    """
    out = df.copy()
    opponent = (100.0 - out["team_possession"]).clip(lower=config.PADJ_MIN_OPPONENT_POSSESSION)
    ratio = config.PADJ_REFERENCE_POSSESSION / opponent
    out["possession_exposure_ratio"] = ratio
    for col, alpha in alphas.items():
        raw = f"{col}_p90"
        if raw not in out.columns:
            continue
        factor = ratio**alpha
        out[f"{col}_padj_factor"] = factor
        out[f"{col}_padj_p90"] = out[raw] * factor
    # A single representative factor for reporting, using the mean fitted elasticity.
    mean_alpha = float(np.mean(list(alphas.values()))) if alphas else 1.0
    out["possession_adjustment_factor"] = ratio**mean_alpha
    return out


def to_per90(df: pd.DataFrame, counts: list[str]) -> pd.DataFrame:
    """Add a ``*_p90`` column for each counting stat. Rates are left alone."""
    out = df.copy()
    nineties = out["nineties"].replace(0, np.nan)
    for col in counts:
        if col in out.columns:
            out[f"{col}_p90"] = out[col] / nineties
    return out


# --------------------------------------------------------------------------------------
# Missingness, imputation, standardisation
# --------------------------------------------------------------------------------------


def missingness_report(df: pd.DataFrame, cols: list[str]) -> dict:
    present = [c for c in cols if c in df.columns]
    frac = df[present].isna().mean().sort_values(ascending=False)
    return {
        "per_column": {c: round(float(v), 4) for c, v in frac.items()},
        "dropped_over_threshold": [c for c, v in frac.items() if v > config.MAX_MISSING_FRACTION],
        "threshold": config.MAX_MISSING_FRACTION,
    }


def impute_within_group(df: pd.DataFrame, cols: list[str], group: str) -> tuple[pd.DataFrame, dict]:
    """Median-impute within position group and flag every imputed cell."""
    out = df.copy()
    counts: dict[str, int] = {}
    for col in cols:
        if col not in out.columns:
            continue
        missing = out[col].isna()
        if not missing.any():
            continue
        counts[col] = int(missing.sum())
        out[col] = out.groupby(group)[col].transform(lambda s: s.fillna(s.median()))
        out[col] = out[col].fillna(out[col].median())
    out["n_imputed_features"] = 0
    if counts:
        out["n_imputed_features"] = df[list(counts)].isna().sum(axis=1).to_numpy()
    return out, counts


def zscore(df: pd.DataFrame, cols: list[str], group: str | None = None) -> pd.DataFrame:
    """Standardise features, either across the whole pool or within position group."""
    out = df.copy()
    if group is None:
        for col in cols:
            std = out[col].std(ddof=0)
            out[col] = (out[col] - out[col].mean()) / std if std and std > 0 else 0.0
        return out
    for col in cols:
        grouped = out.groupby(group)[col]
        std = grouped.transform(lambda s: s.std(ddof=0))
        mean = grouped.transform("mean")
        out[col] = np.where(std > 0, (out[col] - mean) / std, 0.0)
    return out


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def _padj_confound_report(eligible: pd.DataFrame, alphas: dict[str, float]) -> dict:
    """Measure how much team dependence the adjustment actually removes.

    Reports the correlation between each defensive rate and team possession before and
    after adjustment, alongside the counterfactual under the conventional exponent of
    one. The counterfactual is the evidence that direct proportionality overcorrects.
    """
    ratio = eligible["possession_exposure_ratio"]
    possession = eligible["team_possession"]
    out = {}
    for count, alpha in alphas.items():
        raw_col, adj_col = f"{count}_p90", f"{count}_padj_p90"
        if raw_col not in eligible.columns:
            continue
        unit = eligible[raw_col] * ratio**config.PADJ_UNIT_ELASTICITY
        out[count] = {
            "elasticity_used": round(float(alpha), 4),
            "corr_raw_with_possession": round(float(eligible[raw_col].corr(possession)), 4),
            "corr_adjusted_with_possession": round(float(eligible[adj_col].corr(possession)), 4),
            "corr_unit_elasticity_with_possession": round(float(unit.corr(possession)), 4),
        }
    return out


def build_season(season: str) -> dict:
    fb = load_fbref_canonical(season)
    us = load_understat(season)
    merged, match_report = match_understat(fb, us)
    merged = parse_positions(merged)

    possession = load_team_possession(season)
    merged = merged.merge(possession, on="team", how="left")
    unmatched_teams = sorted(merged.loc[merged["team_possession"].isna(), "team"].unique())
    if unmatched_teams:
        raise KeyError(f"{season}: no team possession for {unmatched_teams}")

    per_squad = merged.copy()
    per_squad["nineties"] = per_squad["minutes"] / 90.0
    per_squad = to_per90(per_squad, F.OUTFIELD_COUNTS)
    elasticity = estimate_possession_elasticity(config.SEASONS)
    alphas = (
        elasticity["alphas"]
        if config.PADJ_ELASTICITY_MODE == "estimated"
        else dict.fromkeys(config.PADJ_COUNTS, config.PADJ_UNIT_ELASTICITY)
    )
    per_squad = possession_adjust(per_squad, alphas)

    # Possession is minutes-weighted across clubs for movers, so the adjustment reflects
    # the share of the ball a player actually played in front of.
    aggregated = aggregate_per_player(merged)
    aggregated = to_per90(aggregated, F.OUTFIELD_COUNTS)
    aggregated = possession_adjust(aggregated, alphas)

    outfield = aggregated[aggregated["position_group"].isin(config.OUTFIELD_GROUPS)].copy()
    miss = missingness_report(outfield, F.OUTFIELD_CORE)
    usable = [c for c in F.OUTFIELD_CORE if c not in miss["dropped_over_threshold"]]

    eligible = outfield[outfield["minutes"] >= config.MIN_MINUTES].copy()
    eligible, imputed = impute_within_group(eligible, usable, "position_group")

    global_z = zscore(eligible.copy(), usable, group=None)
    group_z = zscore(eligible.copy(), usable, group="position_group")

    out_dir = config.DATA_PROCESSED
    per_squad.to_parquet(out_dir / f"per_squad_{season}.parquet", index=False)
    aggregated.to_parquet(out_dir / f"aggregated_{season}.parquet", index=False)
    eligible.to_parquet(out_dir / f"outfield_eligible_{season}.parquet", index=False)
    global_z.to_parquet(out_dir / f"outfield_z_global_{season}.parquet", index=False)
    group_z.to_parquet(out_dir / f"outfield_z_bygroup_{season}.parquet", index=False)

    keepers = aggregated[aggregated["position_group"] == "GK"].copy()
    keepers.to_parquet(out_dir / f"keepers_all_{season}.parquet", index=False)

    sensitivity = {
        str(m): int((outfield["minutes"] >= m).sum()) for m in config.MIN_MINUTES_SENSITIVITY
    }

    return {
        "season": season,
        "understat_match": match_report,
        "rows_per_squad": int(len(per_squad)),
        "rows_aggregated": int(len(aggregated)),
        "rows_outfield": int(len(outfield)),
        "rows_outfield_eligible": int(len(eligible)),
        "rows_keepers": int(len(keepers)),
        "players_multi_club": int((aggregated["n_squads"] > 1).sum()),
        "position_group_counts": aggregated["position_group"].value_counts().to_dict(),
        "features_used": usable,
        "features_dropped_missing": miss["dropped_over_threshold"],
        "missingness": miss["per_column"],
        "imputed_cells": imputed,
        "minutes_threshold": config.MIN_MINUTES,
        "minutes_sensitivity_counts": sensitivity,
        "possession_adjustment": {
            "reference_possession": config.PADJ_REFERENCE_POSSESSION,
            "adjusted_counts": list(config.PADJ_COUNTS),
            "elasticity_mode": config.PADJ_ELASTICITY_MODE,
            "elasticity": elasticity["report"],
            "team_possession_min": round(float(possession["team_possession"].min()), 2),
            "team_possession_max": round(float(possession["team_possession"].max()), 2),
            "team_possession_mean": round(float(possession["team_possession"].mean()), 2),
            "factor_min": round(float(eligible["possession_adjustment_factor"].min()), 4),
            "factor_max": round(float(eligible["possession_adjustment_factor"].max()), 4),
            "factor_mean": round(float(eligible["possession_adjustment_factor"].mean()), 4),
            "confound_removal": _padj_confound_report(eligible, alphas),
            "note": (
                "Factor is the exposure ratio, the reference opponent possession divided "
                "by the actual one, raised to the fitted elasticity. A value above one "
                "belongs to a player whose team had more of the ball than average and "
                "whose raw defensive counts therefore understate his rate of action per "
                "opportunity."
            ),
        },
    }


def main() -> None:
    report = {}
    for season in config.SEASONS:
        report[season] = build_season(season)
        r = report[season]
        print(
            f"{season}: outfield eligible={r['rows_outfield_eligible']} "
            f"keepers={r['rows_keepers']} "
            f"understat match={r['understat_match']['match_rate']:.1%} "
            f"features={len(r['features_used'])}",
            flush=True,
        )
    with open(config.METRICS / "preprocess.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("preprocess complete")


if __name__ == "__main__":
    main()
