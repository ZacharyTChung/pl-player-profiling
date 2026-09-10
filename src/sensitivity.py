"""Minutes-threshold sensitivity.

The main analysis excludes outfield players below a minutes threshold, and that threshold
is a judgement call rather than a fact. A low threshold admits players whose per-90 rates
are dominated by sampling noise; a high one discards genuine squad roles such as the
impact substitute. This module repeats the whole selection and stability procedure at each
candidate threshold so the paper can report what the choice actually costs.

Everything reuses the functions in :mod:`src.cluster` rather than reimplementing them, so
the selection rule applied here is literally the declared rule and cannot drift from it.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

import config
from src import cluster as C
from src import features as F
from src import plotting as P
from src import preprocess as PP


def build_pool(season: str, threshold: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Rebuild the eligible pool and its standardised matrix at one threshold.

    Standardisation is redone inside each pool rather than carried over from the main
    analysis, because a z-score is defined relative to the population it is computed in
    and reusing the 450-minute scaling would quietly compare different quantities.
    """
    aggregated = pd.read_parquet(config.DATA_PROCESSED / f"aggregated_{season}.parquet")
    outfield = aggregated[aggregated["position_group"].isin(config.OUTFIELD_GROUPS)].copy()
    eligible = outfield[outfield["minutes"] >= threshold].copy()

    usable = [c for c in F.OUTFIELD_CORE if c in eligible.columns]
    eligible, _ = PP.impute_within_group(eligible, usable, "position_group")
    standardised = PP.zscore(eligible.copy(), usable, group=None)
    return eligible, standardised[usable].to_numpy(dtype=float), usable


def analyse_threshold(season: str, threshold: int) -> dict:
    tag = f"sensitivity-{season}-{threshold}"
    eligible, matrix, usable = build_pool(season, threshold)

    curves = C.internal_curves(matrix, tag)
    rule = C.apply_k_rule(curves)
    k = int(rule["chosen_k"])

    labels = C._kmeans(matrix, k, tag).labels_
    stability = C.bootstrap_stability(matrix, labels, k, tag)

    return {
        "minutes_threshold": threshold,
        "n_eligible": int(len(eligible)),
        "n_features": len(usable),
        "chosen_k": k,
        "rule_branch": rule["rule_branch"],
        "silhouette": round(float(silhouette_score(matrix, labels)), 4),
        "bootstrap_ari_mean": round(float(stability["ari_mean"]), 4),
        "bootstrap_ari_sd": round(float(stability["ari_sd"]), 4),
        "position_group_counts": eligible["position_group"].value_counts().to_dict(),
        "_labels": pd.Series(labels, index=eligible["player"].to_numpy()),
    }


def agreement_with_baseline(results: dict[int, dict]) -> dict:
    """How far the partition itself moves when the threshold moves.

    A stable count of clusters is weak evidence on its own, because the same k can
    describe a different split. Comparing assignments on the players common to both pools
    asks the stronger question.
    """
    baseline = results.get(config.MIN_MINUTES)
    if baseline is None:
        return {}
    base_labels = baseline["_labels"]
    out = {}
    for threshold, res in results.items():
        if threshold == config.MIN_MINUTES:
            continue
        other = res["_labels"]
        common = base_labels.index.intersection(other.index)
        if len(common) < 2:
            continue
        out[str(threshold)] = {
            "n_common_players": int(len(common)),
            "adjusted_rand_index_vs_baseline": round(
                float(adjusted_rand_score(base_labels.loc[common], other.loc[common])), 4
            ),
        }
    return out


def figure_sensitivity(results: dict[int, dict], season: str) -> None:
    thresholds = sorted(results)
    fig, axes = plt.subplots(1, 3, figsize=(P.WIDTH_FULL, 2.5))
    colour = P.CATEGORICAL[0]

    panels = [
        ("n_eligible", "Eligible players"),
        ("silhouette", "Silhouette"),
        ("bootstrap_ari_mean", "Bootstrap ARI"),
    ]
    for ax, (key, label) in zip(axes, panels, strict=True):
        values = [results[t][key] for t in thresholds]
        ax.plot(thresholds, values, marker="o", color=colour, linewidth=2.0, markersize=5)
        for t, v in zip(thresholds, values, strict=True):
            marker = " (used)" if t == config.MIN_MINUTES else ""
            ax.annotate(
                f"{v:g}{marker}" if key == "n_eligible" else f"{v:.3f}{marker}",
                (t, v),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=P.BASE_FONT_PT - 2,
                color=P.INK_SECONDARY,
            )
        ax.axvline(config.MIN_MINUTES, color=P.INK_MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        P.style_axis(ax, "Minutes threshold", label, label)
        ax.set_xticks(thresholds)
        ax.margins(y=0.25)

    P.save_figure(fig, "minutes_sensitivity")


def main() -> None:
    P.use_style()
    season = config.SEASON_PRIMARY
    results = {t: analyse_threshold(season, t) for t in config.MIN_MINUTES_SENSITIVITY}
    agreement = agreement_with_baseline(results)

    figure_sensitivity(results, season)

    payload = {
        "season": season,
        "baseline_threshold": config.MIN_MINUTES,
        "note": (
            "Each threshold is analysed as an independent pool: imputation and "
            "standardisation are recomputed inside it, and the declared selection rule is "
            "reapplied from scratch. Agreement is the adjusted Rand index against the "
            "baseline partition on the players common to both pools."
        ),
        "agreement_with_baseline": agreement,
    }
    for threshold, res in results.items():
        payload[str(threshold)] = {k: v for k, v in res.items() if not k.startswith("_")}

    with open(config.METRICS / "minutes_sensitivity.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    for threshold in sorted(results):
        r = results[threshold]
        extra = agreement.get(str(threshold), {})
        tail = (
            f" | ARI vs baseline {extra['adjusted_rand_index_vs_baseline']}"
            f" on {extra['n_common_players']} common"
            if extra
            else " | baseline"
        )
        print(
            f"{threshold:>4} min: n={r['n_eligible']:>3} k={r['chosen_k']} "
            f"silhouette={r['silhouette']:.3f} bootstrapARI={r['bootstrap_ari_mean']:.3f}{tail}",
            flush=True,
        )
    print("sensitivity complete")


if __name__ == "__main__":
    main()
