"""Is a player's archetype related to his age?

The question is worth asking because the two archetypes inside each position group differ
mainly in attacking involvement, and involvement is plausibly something a career moves
through rather than a fixed trait. A full back who overlaps might do so at twenty three and
not at thirty three.

The test is run within position group rather than pooled, because archetypes are nested
inside positions and a pooled table would mostly recover the fact that forwards and
defenders have different age profiles. Association is measured with a chi-square test of
independence and reported alongside Cramer's V, since a p-value on a few hundred players
says whether an association exists and nothing about whether it matters.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, spearmanr

import config
from src import plotting as P

#: Bands chosen before looking at any archetype assignment, on football rather than
#: statistical grounds: academy graduate, establishing, peak, late peak, veteran.
AGE_BINS = [15, 21, 24, 27, 30, 45]
AGE_LABELS = ["21 and under", "22 to 24", "25 to 27", "28 to 30", "31 and over"]

#: The chi-square approximation is unreliable when expected counts fall below this.
MIN_EXPECTED = 5.0


def load(season: str) -> pd.DataFrame:
    arche = pd.read_parquet(config.DATA_PROCESSED / f"archetypes_{season}.parquet")
    eligible = pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")
    clusters = pd.read_parquet(config.DATA_PROCESSED / f"clusters_{season}.parquet")

    frame = arche.merge(eligible[["player", "age", "minutes"]], on="player", how="left")
    frame = frame.merge(clusters[["player", "cluster_global"]], on="player", how="left")
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    frame = frame.dropna(subset=["age"])
    frame["age_band"] = pd.cut(frame["age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    return frame


def association(table: pd.DataFrame) -> dict:
    """Chi-square test with Cramer's V, and an honest note on the approximation."""
    counts = table.to_numpy(dtype=float)
    if counts.shape[0] < 2 or counts.shape[1] < 2 or counts.sum() == 0:
        return {"testable": False, "reason": "table too small"}
    chi2, p, dof, expected = chi2_contingency(counts)
    n = counts.sum()
    denom = n * (min(counts.shape) - 1)
    cramers_v = float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")
    low = int((expected < MIN_EXPECTED).sum())
    return {
        "testable": True,
        "chi2": round(float(chi2), 4),
        "dof": int(dof),
        "p_value": round(float(p), 4),
        "cramers_v": round(cramers_v, 4),
        "n": int(n),
        "cells_with_expected_below_five": low,
        "approximation_reliable": bool(low == 0),
    }


def analyse(frame: pd.DataFrame) -> dict:
    out: dict = {"by_position_group": {}}

    for group, grp in frame.groupby("position_group", observed=True):
        table = pd.crosstab(grp["archetype_name"], grp["age_band"])
        ages = [sub["age"].to_numpy(dtype=float) for _, sub in grp.groupby("archetype_name")]
        block = {
            "contingency": table.to_dict(),
            "association": association(table),
            "mean_age_by_archetype": {
                str(k): round(float(v), 2)
                for k, v in grp.groupby("archetype_name")["age"].mean().items()
            },
            "median_age_by_archetype": {
                str(k): round(float(v), 1)
                for k, v in grp.groupby("archetype_name")["age"].median().items()
            },
            "n_by_archetype": {
                str(k): int(v) for k, v in grp["archetype_name"].value_counts().items()
            },
        }
        # A chi-square tests general association and is weak against a monotonic
        # alternative, which is exactly the shape the defender panel shows. Spearman
        # between age and the archetype indicator is the powerful test here, so it is
        # reported alongside rather than instead, and the gap between them is the point.
        names = sorted(grp["archetype_name"].unique())
        if len(names) == 2:
            indicator = (grp["archetype_name"] == names[1]).astype(float)
            rho, p_trend = spearmanr(grp["age"].to_numpy(dtype=float), indicator.to_numpy())
            block["age_trend"] = {
                "indicator_is": names[1],
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(p_trend), 4),
                "note": (
                    "Positive means the indicated archetype becomes more common with age. "
                    "This is a test against a monotonic alternative, unlike the chi-square "
                    "above, which spends its degrees of freedom on any departure from "
                    "independence."
                ),
            }

        if len(ages) == 2 and all(len(a) > 1 for a in ages):
            stat, p = kruskal(*ages)
            block["kruskal_age_difference"] = {
                "statistic": round(float(stat), 4),
                "p_value": round(float(p), 4),
                "mean_difference_years": round(float(ages[0].mean() - ages[1].mean()), 2),
            }
        out["by_position_group"][str(group)] = block

    global_table = pd.crosstab(frame["cluster_global"], frame["age_band"])
    out["global_cluster"] = {
        "contingency": global_table.to_dict(),
        "association": association(global_table),
        "mean_age_by_cluster": {
            str(int(k)): round(float(v), 2)
            for k, v in frame.groupby("cluster_global")["age"].mean().items()
        },
    }
    out["age_bands"] = AGE_LABELS
    out["n_players"] = int(len(frame))
    out["multiple_testing_note"] = (
        "Four tests are reported, one per position group and one global. No correction is "
        "applied, so a single p-value near the conventional threshold should be read as "
        "suggestive rather than decisive."
    )
    return out


def _wrap(name: str, width: int = 22) -> str:
    """Wrap an archetype name onto two lines instead of truncating it mid-word."""
    words, lines, current = name.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines[:3])


def figure_age(frame: pd.DataFrame, results: dict) -> None:
    groups = sorted(frame["position_group"].unique())
    fig, axes = plt.subplots(1, len(groups), figsize=(P.WIDTH_FULL, 2.9), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, group in zip(axes, groups, strict=True):
        grp = frame[frame["position_group"] == group]
        archetypes = sorted(grp["archetype_name"].unique())
        colours = P.categorical(min(len(archetypes), P.MAX_CATEGORICAL))
        table = pd.crosstab(grp["age_band"], grp["archetype_name"], normalize="index")

        x = np.arange(len(AGE_LABELS))
        width = 0.38
        for i, name in enumerate(archetypes):
            share = [
                table.loc[b, name] if b in table.index and name in table.columns else 0.0
                for b in AGE_LABELS
            ]
            ax.bar(
                x + (i - (len(archetypes) - 1) / 2) * width,
                share,
                width * 0.92,
                color=colours[i % len(colours)],
                edgecolor=P.SURFACE,
                linewidth=0.8,
                label=_wrap(name),
            )
        stats = results["by_position_group"][group]["association"]
        subtitle = (
            f"V {stats['cramers_v']:.2f}, p {stats['p_value']:.2f}"
            if stats.get("testable")
            else "not testable"
        )
        P.style_axis(ax, "", "Share of band" if group == groups[0] else "", f"{group}   {subtitle}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [b.replace(" and ", "\n and ").replace(" to ", "\nto ") for b in AGE_LABELS],
            fontsize=P.BASE_FONT_PT - 3,
        )
        ax.set_ylim(0, 1)
        ax.grid(False, axis="x")
        ax.legend(
            loc="upper center",
            fontsize=P.BASE_FONT_PT - 3,
            ncol=1,
            bbox_to_anchor=(0.5, -0.20),
            labelspacing=0.35,
        )

    fig.suptitle("Archetype share within each age band", fontsize=P.BASE_FONT_PT, fontweight="bold")
    P.save_figure(fig, "age_archetype")


def main() -> None:
    P.use_style()
    season = config.SEASON_PRIMARY
    frame = load(season)
    results = analyse(frame)
    results["season"] = season
    figure_age(frame, results)

    with open(config.METRICS / "age_archetype.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    for group, block in results["by_position_group"].items():
        a = block["association"]
        line = f"{group}: "
        if a.get("testable"):
            line += (
                f"chi2={a['chi2']:.2f} dof={a['dof']} p={a['p_value']:.3f} V={a['cramers_v']:.3f}"
            )
            if not a["approximation_reliable"]:
                line += f" (warning: {a['cells_with_expected_below_five']} cells expected below 5)"
        else:
            line += "not testable"
        tr = block.get("age_trend")
        if tr:
            line += f" | trend rho={tr['spearman_rho']:+.3f} p={tr['p_value']:.3f}"
        kr = block.get("kruskal_age_difference")
        if kr:
            line += (
                f" | mean age gap {kr['mean_difference_years']:+.2f} years, p={kr['p_value']:.3f}"
            )
        print(line, flush=True)
    print("age analysis complete")


if __name__ == "__main__":
    main()
