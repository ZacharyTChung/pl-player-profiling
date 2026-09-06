"""Should the attacking statistics be possession adjusted too?

The main pipeline corrects only the three statistics that require the opponent to have
the ball. That leaves an obvious objection: attacking output also depends on possession,
so the feature set is still team dependent and the correction is half a correction.

This module tests the objection rather than waving at it, and finds that the two sides of
the ball behave so differently that treating them alike would be a mistake.

Defensive volume responds to opponent possession sub-proportionally, with elasticities
well below one, because a side pinned back defends deeper and concedes territory without
contesting it. Opponent possession there is an exposure denominator in the plain sense:
a tackle cannot happen while your own team has the ball.

Attacking volume responds to a team's own possession super-proportionally, with
elasticities above one for shots, goals and assists. Ten percent more of the ball buys
appreciably more than ten percent more output. That is not an exposure effect. It is a
statement that keeping the ball is productive, which is a football fact rather than a
measurement artefact, and dividing it out would remove signal instead of bias.

The counterfactual is still computed and reported, because the argument is only worth
making if the alternative is quantified.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

import config
from src import cluster as C
from src import features as F
from src import plotting as P
from src import preprocess as PP

#: Team-level source columns for the FBref attacking counts, by the table that carries them.
ATTACK_TEAM_SOURCES = {
    "shots": ("shooting", "Standard__Sh"),
    "shots_on_target": ("shooting", "Standard__SoT"),
    "goals_non_penalty": ("standard", "Performance__G-PK"),
    "assists": ("standard", "Performance__Ast"),
    "crosses": ("misc", "Performance__Crs"),
}

#: The Understat attacking block has no team-level table, so team totals are summed from
#: the player rows. Omitting these would leave the counterfactual adjusting only half the
#: attacking features, which would not be a fair test of the objection it exists to answer.
ATTACK_UNDERSTAT_SOURCES = ["np_xg", "xa", "key_passes", "xg_chain", "xg_buildup"]


def _team_panel(seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        tables = {
            name: pd.read_parquet(config.DATA_RAW / season / f"teams_{name}.parquet")
            for name in ("standard", "shooting", "misc")
        }
        base = tables["standard"][["team", "Poss", "Playing Time__90s"]].copy()
        base["nineties"] = pd.to_numeric(base["Playing Time__90s"], errors="coerce")
        base["own_possession"] = pd.to_numeric(base["Poss"], errors="coerce")
        base["opponent_possession"] = 100.0 - base["own_possession"]
        for canon, (table, column) in ATTACK_TEAM_SOURCES.items():
            merged = tables[table][["team", column]]
            base = base.merge(merged, on="team", how="left")
            base[canon] = pd.to_numeric(base[column], errors="coerce") / base["nineties"]
            base = base.drop(columns=[column])
        understat = pd.read_parquet(config.DATA_RAW / season / "understat_players.parquet")
        totals = understat.groupby("team")[ATTACK_UNDERSTAT_SOURCES].sum().reset_index()
        totals["team"] = totals["team"].map(PP.normalize_team)
        base["_key"] = base["team"].map(PP.normalize_team)
        base = base.merge(totals.rename(columns={"team": "_key"}), on="_key", how="left")
        for canon in ATTACK_UNDERSTAT_SOURCES:
            base[canon] = pd.to_numeric(base[canon], errors="coerce") / base["nineties"]
        base = base.drop(columns=["_key"])
        base["season"] = season
        frames.append(base)
    return pd.concat(frames, ignore_index=True)


def estimate_attacking_elasticity(seasons: list[str]) -> dict:
    """Fit the elasticity of attacking volume with respect to a team's own possession."""
    panel = _team_panel(seasons)
    out = {}
    for canon in [*ATTACK_TEAM_SOURCES, *ATTACK_UNDERSTAT_SOURCES]:
        sub = panel[[canon, "own_possession"]].dropna()
        sub = sub[(sub[canon] > 0) & (sub["own_possession"] > 0)]
        x = np.log(sub["own_possession"].to_numpy(dtype=float))
        y = np.log(sub[canon].to_numpy(dtype=float))
        beta = float(np.polyfit(x, y, 1)[0])
        out[canon] = {
            "elasticity": round(beta, 4),
            "log_log_correlation": round(float(np.corrcoef(x, y)[0, 1]), 4),
            "n": int(len(sub)),
            "super_proportional": bool(beta > 1.0),
        }
    return {"n_team_seasons": int(len(panel)), "per_feature": out}


def build_symmetric(season: str, betas: dict[str, float]) -> tuple[pd.DataFrame, list[str]]:
    """Apply the counterfactual correction to the attacking counts as well."""
    frame = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet").copy()
    own = frame["team_possession"].clip(lower=config.PADJ_MIN_OPPONENT_POSSESSION)
    ratio = config.PADJ_REFERENCE_POSSESSION / own

    usable = [c for c in F.OUTFIELD_CORE if c in frame.columns]
    for canon, beta in betas.items():
        raw = f"{canon}_p90"
        if raw in frame.columns:
            frame[raw] = frame[raw] * ratio**beta
    return frame, usable


def partition(frame: pd.DataFrame, usable: list[str], tag: str) -> tuple[np.ndarray, int, float]:
    standardised = PP.zscore(frame.copy(), usable, group=None)
    matrix = standardised[usable].to_numpy(dtype=float)
    curves = C.internal_curves(matrix, tag)
    k = int(C.apply_k_rule(curves)["chosen_k"])
    labels = C._kmeans(matrix, k, tag).labels_
    return labels, k, float(silhouette_score(matrix, labels))


def team_correlation(frame: pd.DataFrame, usable: list[str], season: str) -> float:
    """Correlation between the club centroid on PC1 and final league position."""
    standardised = PP.zscore(frame.copy(), usable, group=None)
    coords = PCA(n_components=3, random_state=config.RANDOM_STATE).fit_transform(
        standardised[usable].to_numpy(dtype=float)
    )
    work = frame[["team", "minutes"]].copy()
    work["pc1"] = coords[:, 0]
    work["team"] = work["team"].astype(str).str.split(" / ").str[0].str.strip()
    work["minutes"] = pd.to_numeric(work["minutes"], errors="coerce").fillna(0.0)

    table = pd.read_parquet(config.DATA_PROCESSED / f"league_table_{season}.parquet")
    rows = []
    for team, grp in work.groupby("team"):
        w = grp["minutes"].to_numpy(dtype=float)
        if w.sum() > 0:
            rows.append({"team": team, "pc1": float(np.average(grp["pc1"], weights=w))})
    centroids = pd.DataFrame(rows).merge(table[["team", "league_position"]], on="team", how="inner")
    return float(np.corrcoef(centroids["pc1"], centroids["league_position"])[0, 1])


def figure_elasticities(defensive: dict, attacking: dict) -> None:
    labels, values, colours = [], [], []
    for name, block in defensive.items():
        labels.append(name.replace("_", " "))
        values.append(block["elasticity"])
        colours.append(P.CATEGORICAL[0])
    for name, block in attacking["per_feature"].items():
        labels.append(name.replace("_", " "))
        values.append(block["elasticity"])
        colours.append(P.CATEGORICAL[1])

    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colours = [colours[i] for i in order]

    fig, ax = plt.subplots(figsize=(P.WIDTH_FULL, 3.6))
    ypos = np.arange(len(values))
    ax.barh(ypos, values, color=colours, height=0.66, edgecolor=P.SURFACE, linewidth=0.8)
    ax.axvline(1.0, color=P.INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(
        1.02,
        len(values) - 0.4,
        "proportional",
        fontsize=P.BASE_FONT_PT - 2,
        color=P.INK_SECONDARY,
        va="center",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=P.BASE_FONT_PT - 1)
    for y, v in zip(ypos, values, strict=True):
        ax.text(
            v + 0.04, y, f"{v:.2f}", va="center", fontsize=P.BASE_FONT_PT - 2, color=P.INK_SECONDARY
        )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=P.CATEGORICAL[0], label="defensive, vs opponent"),
        plt.Rectangle((0, 0), 1, 1, color=P.CATEGORICAL[1], label="attacking, vs own"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=P.BASE_FONT_PT - 2)
    P.style_axis(ax, "Elasticity with respect to possession", "", "Possession elasticities")
    ax.grid(False, axis="y")
    ax.set_xlim(0, max(values) * 1.22)
    P.save_figure(fig, "possession_elasticities")


def main() -> None:
    P.use_style()
    season = config.SEASON_PRIMARY

    pre = json.loads((config.METRICS / "preprocess.json").read_text())
    defensive = pre[season]["possession_adjustment"]["elasticity"]["per_feature"]
    attacking = estimate_attacking_elasticity(config.SEASONS)
    figure_elasticities(defensive, attacking)

    betas = {k: v["elasticity"] for k, v in attacking["per_feature"].items()}

    baseline = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")
    usable = [c for c in F.OUTFIELD_CORE if c in baseline.columns]
    base_labels, base_k, base_sil = partition(baseline.copy(), usable, "symmetric-baseline")
    base_corr = team_correlation(baseline.copy(), usable, season)

    sym_frame, _ = build_symmetric(season, betas)
    sym_labels, sym_k, sym_sil = partition(sym_frame, usable, "symmetric-counterfactual")
    sym_corr = team_correlation(sym_frame, usable, season)

    payload = {
        "season": season,
        "question": (
            "Should the attacking statistics be possession adjusted in the same way as the "
            "defensive ones?"
        ),
        "defensive_elasticities": {k: v["elasticity"] for k, v in defensive.items()},
        "attacking_elasticities": attacking,
        "asymmetry": (
            "Defensive elasticities are below one and attacking elasticities above it. "
            "Opponent possession is an exposure denominator for defending, because a "
            "tackle cannot occur while your own team holds the ball. Own possession is "
            "not merely exposure for attacking: returns to it are increasing, which is a "
            "property of football rather than of the measurement, so dividing it out "
            "would remove signal rather than bias."
        ),
        "baseline": {
            "chosen_k": base_k,
            "silhouette": round(base_sil, 4),
            "team_pc1_vs_league_position": round(base_corr, 4),
        },
        "symmetric_counterfactual": {
            "chosen_k": sym_k,
            "silhouette": round(sym_sil, 4),
            "team_pc1_vs_league_position": round(sym_corr, 4),
            "adjusted_rand_index_vs_baseline": round(
                float(adjusted_rand_score(base_labels, sym_labels)), 4
            ),
            "n_changed": int((base_labels != sym_labels).sum()) if base_k == sym_k else None,
        },
        "verdict": (
            "Adopted: one sided adjustment. The counterfactual is reported so the choice "
            "can be checked, not because it is preferred."
        ),
    }
    with open(config.METRICS / "symmetric_adjustment.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    print(
        "attacking elasticities: " + ", ".join(f"{k} {v:.2f}" for k, v in betas.items()),
        flush=True,
    )
    print(
        f"baseline      k={base_k} silhouette={base_sil:.3f} "
        f"team PC1 vs league position {base_corr:+.3f}",
        flush=True,
    )
    print(
        f"symmetric     k={sym_k} silhouette={sym_sil:.3f} "
        f"team PC1 vs league position {sym_corr:+.3f} "
        f"ARI vs baseline {payload['symmetric_counterfactual']['adjusted_rand_index_vs_baseline']}",
        flush=True,
    )
    print("symmetric complete")


if __name__ == "__main__":
    main()
