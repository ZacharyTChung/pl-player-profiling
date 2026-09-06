"""Phase 3: descriptive statistics, correlation structure and distributional shape.

This module answers three questions before any modelling happens.

1. What does a typical defender, midfielder and forward look like on the twelve headline
   per-90 metrics that survived the January 2026 removal of the Opta-derived FBref
   columns.
2. How are the eighteen surviving features related to one another, and does that
   relationship hold across position groups. The position-stratified correlation
   comparison is a substantive result rather than a diagnostic: a statistic whose
   correlation with another statistic flips sign between midfielders and forwards means
   something different depending on who is being measured, which constrains how a single
   global embedding can be interpreted.
3. How far the features are from normal. PCA itself assumes nothing about normality, but
   heavy right skew inflates the influence of a small number of extreme players on the
   principal axes, so the actual skewness, kurtosis, univariate normality tests and
   Mardia multivariate tests are reported rather than assumed away.

Outputs
-------
results/metrics/descriptive_summary.json
results/metrics/correlation_divergence.json
results/metrics/normality.json
figures/corr_heatmap_all.{pdf,png}
figures/corr_divergence_top10.{pdf,png}
figures/distributions_by_position.{pdf,png}
"""

from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

import config
from src import features as F
from src import plotting

#: Short display names. The raw canonical names carry a "_p90" suffix that wastes axis
#: space when eighteen of them have to fit along one edge of a heatmap.
FEATURE_LABELS = {
    "np_xg_p90": "npxG",
    "xa_p90": "xA",
    "key_passes_p90": "Key passes",
    "xg_chain_p90": "xGChain",
    "xg_buildup_p90": "xGBuildup",
    "shots_p90": "Shots",
    "shots_on_target_p90": "Shots on target",
    "goals_non_penalty_p90": "NP goals",
    "assists_p90": "Assists",
    "crosses_p90": "Crosses",
    "interceptions_p90": "Interceptions",
    "tackles_won_p90": "Tackles won",
    "fouls_committed_p90": "Fouls committed",
    "fouls_drawn_p90": "Fouls drawn",
    "offsides_p90": "Offsides",
    "cards_yellow_p90": "Yellow cards",
    "shot_accuracy_pct": "Shot accuracy %",
    "goals_per_shot": "Goals per shot",
}

#: Metrics shown in the distribution figure, chosen to span the creation, finishing and
#: defensive ends of the surviving feature set.
DISTRIBUTION_METRICS = ["np_xg_p90", "xa_p90", "tackles_won_p90", "key_passes_p90"]


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def load_eligible(season: str) -> pd.DataFrame:
    """Raw per-90 values, one row per eligible outfield player."""
    return pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")


# --------------------------------------------------------------------------------------
# 1. Summary table
# --------------------------------------------------------------------------------------


def _describe(values: pd.Series) -> dict[str, float]:
    clean = values.dropna()
    q25, q50, q75 = np.percentile(clean, [25, 50, 75])
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def summary_by_position(df: pd.DataFrame) -> dict:
    """Mean, standard deviation, median and IQR of the headline metrics per group."""
    out: dict = {
        "n_by_group": {g: int((df["position_group"] == g).sum()) for g in config.OUTFIELD_GROUPS},
        "groups": {},
        "all_outfield": {m: _describe(df[m]) for m in F.HEADLINE_METRICS},
    }
    for group in config.OUTFIELD_GROUPS:
        sub = df[df["position_group"] == group]
        out["groups"][group] = {m: _describe(sub[m]) for m in F.HEADLINE_METRICS}

    # Kruskal-Wallis across the three groups, so the table can say which headline metrics
    # actually separate positions and which do not.
    tests = {}
    for metric in F.HEADLINE_METRICS:
        samples = [
            df.loc[df["position_group"] == g, metric].dropna().to_numpy()
            for g in config.OUTFIELD_GROUPS
        ]
        stat, pval = stats.kruskal(*samples)
        tests[metric] = {"kruskal_h": float(stat), "p": float(pval)}
    out["kruskal_wallis_by_metric"] = tests
    return out


# --------------------------------------------------------------------------------------
# 2 and 3. Correlation structure
# --------------------------------------------------------------------------------------


def spearman_matrix(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Spearman rank correlation. Rank based, so the heavy right skew does not matter."""
    return df[feats].corr(method="spearman")


def cluster_order(corr: pd.DataFrame) -> list[str]:
    """Order features by average-linkage clustering on the correlation distance.

    Distance is 1 minus the correlation, so features that move together end up adjacent
    and the block structure of the matrix becomes visible. Optimal leaf ordering is on,
    which makes the ordering deterministic for a given matrix.
    """
    dist = 1.0 - corr.to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 2.0)
    link = linkage(squareform(dist, checks=False), method="average", optimal_ordering=True)
    return [corr.columns[i] for i in leaves_list(link)]


def correlation_divergence(df: pd.DataFrame, feats: list[str]) -> dict:
    """Per-pair spread of the Spearman correlation across DF, MF and FW."""
    mats = {
        g: spearman_matrix(df[df["position_group"] == g], feats) for g in config.OUTFIELD_GROUPS
    }
    global_mat = spearman_matrix(df, feats)
    rows = []
    for a, b in combinations(feats, 2):
        vals = {g: float(mats[g].loc[a, b]) for g in config.OUTFIELD_GROUPS}
        hi_group = max(vals, key=vals.get)
        lo_group = min(vals, key=vals.get)
        rows.append(
            {
                "feature_a": a,
                "feature_b": b,
                "label_a": label(a),
                "label_b": label(b),
                "by_group": vals,
                "global": float(global_mat.loc[a, b]),
                "spread": float(vals[hi_group] - vals[lo_group]),
                "max_group": hi_group,
                "min_group": lo_group,
                "sign_flip": bool(vals[hi_group] > 0 > vals[lo_group]),
            }
        )
    rows.sort(key=lambda r: r["spread"], reverse=True)
    spreads = np.array([r["spread"] for r in rows])
    return {
        "method": "spearman",
        "n_pairs": len(rows),
        "n_by_group": {g: int((df["position_group"] == g).sum()) for g in config.OUTFIELD_GROUPS},
        "spread_mean": float(spreads.mean()),
        "spread_median": float(np.median(spreads)),
        "spread_max": float(spreads.max()),
        "n_pairs_sign_flip": int(sum(r["sign_flip"] for r in rows)),
        "top10": rows[:10],
        "matrices": {g: mats[g].round(6).to_dict() for g in config.OUTFIELD_GROUPS},
    }


# --------------------------------------------------------------------------------------
# 5. Normality and shape
# --------------------------------------------------------------------------------------


def mardia_test(x: np.ndarray) -> dict:
    """Mardia multivariate skewness and kurtosis.

    With ``n`` rows and ``p`` columns, let ``S`` be the maximum likelihood covariance and
    ``D = Xc S^-1 Xc'`` the matrix of Mahalanobis-type products of the centred data. Then
    ``b1p`` is the mean cube of the off-diagonal entries and ``b2p`` the mean square of the
    diagonal entries. Under multivariate normality ``n*b1p/6`` follows a chi-square with
    ``p(p+1)(p+2)/6`` degrees of freedom, and ``b2p`` is asymptotically normal with mean
    ``p(p+2)`` and variance ``8p(p+2)/n``.

    The statistic is affine invariant, so the columns are standardised first. That makes
    ``S`` the correlation matrix and its condition number a scale-free measure of how
    collinear the surviving feature set is, which the raw covariance would confound with
    the fact that goals per shot and crosses per 90 live on very different scales.
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    x = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
    xc = x - x.mean(axis=0)
    cov = np.cov(xc, rowvar=False, bias=True)
    cond = float(np.linalg.cond(cov))
    inv = np.linalg.pinv(cov)
    d = xc @ inv @ xc.T

    b1p = float((d**3).sum() / (n * n))
    b2p = float((np.diag(d) ** 2).mean())

    skew_stat = n * b1p / 6.0
    skew_df = p * (p + 1) * (p + 2) / 6.0
    skew_p = float(stats.chi2.sf(skew_stat, skew_df))

    kurt_expected = p * (p + 2)
    kurt_z = (b2p - kurt_expected) / np.sqrt(8.0 * p * (p + 2) / n)
    kurt_p = float(2 * stats.norm.sf(abs(kurt_z)))

    return {
        "n": int(n),
        "p": int(p),
        "correlation_condition_number": cond,
        "skewness_b1p": b1p,
        "skewness_stat": float(skew_stat),
        "skewness_df": float(skew_df),
        "skewness_p": skew_p,
        "kurtosis_b2p": b2p,
        "kurtosis_expected": float(kurt_expected),
        "kurtosis_z": float(kurt_z),
        "kurtosis_p": kurt_p,
        "multivariate_normal_rejected": bool(skew_p < 0.05 or kurt_p < 0.05),
    }


def normality_report(df: pd.DataFrame, feats: list[str]) -> dict:
    per_feature = {}
    for feat in feats:
        v = df[feat].dropna().to_numpy()
        shapiro_w, shapiro_p = stats.shapiro(v)
        k2, k2_p = stats.normaltest(v)
        per_feature[feat] = {
            "n": int(v.size),
            "skew": float(stats.skew(v, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(v, fisher=True, bias=False)),
            "shapiro_w": float(shapiro_w),
            "shapiro_p": float(shapiro_p),
            "dagostino_k2": float(k2),
            "dagostino_p": float(k2_p),
            "normal_at_005": bool(shapiro_p >= 0.05),
        }
    skews = np.array([v["skew"] for v in per_feature.values()])
    n_reject = sum(not v["normal_at_005"] for v in per_feature.values())
    worst = max(per_feature.items(), key=lambda kv: abs(kv[1]["skew"]))
    return {
        "n_features": len(feats),
        "per_feature": per_feature,
        "n_features_rejecting_normality_shapiro_005": int(n_reject),
        "skew_mean_abs": float(np.abs(skews).mean()),
        "skew_max_feature": worst[0],
        "skew_max": float(worst[1]["skew"]),
        "n_features_skew_gt_1": int((skews > 1.0).sum()),
        "mardia": mardia_test(df[feats].to_numpy()),
        "interpretation": (
            "PCA does not require normality, but every principal axis is a variance-weighted "
            "combination, so features with heavy right skew let a handful of extreme players "
            "pull an axis toward themselves. The numbers here quantify how much of that risk "
            "the surviving feature set carries."
        ),
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def figure_corr_heatmap(corr: pd.DataFrame, season: str) -> None:
    order = cluster_order(corr)
    m = corr.loc[order, order]
    labels = [label(c) for c in order]

    fig, ax = plotting.plt.subplots(figsize=(plotting.WIDTH_FULL, plotting.WIDTH_FULL * 0.86))
    im = ax.imshow(m.to_numpy(), cmap=plotting.DIVERGING, vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=plotting.BASE_FONT_PT - 2)
    ax.set_yticklabels(labels, fontsize=plotting.BASE_FONT_PT - 2)
    ax.grid(False)
    ax.set_title(
        f"Spearman correlation, {len(order)} surviving features, {season}\n"
        "average-linkage ordering",
        loc="left",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cbar.set_label("Spearman rho", fontsize=plotting.BASE_FONT_PT - 1)
    cbar.outline.set_visible(False)
    plotting.save_figure(fig, "corr_heatmap_all")


def figure_corr_divergence(div: dict, season: str) -> None:
    rows = div["top10"][::-1]  # largest spread at the top of the axis
    y = np.arange(len(rows))

    fig, ax = plotting.plt.subplots(figsize=(plotting.WIDTH_FULL, 4.3))
    for i, row in enumerate(rows):
        vals = [row["by_group"][g] for g in config.OUTFIELD_GROUPS]
        ax.plot([min(vals), max(vals)], [i, i], color=plotting.CONTEXT_GREY, lw=1.6, zorder=1)
    for group in config.OUTFIELD_GROUPS:
        ax.scatter(
            [r["by_group"][group] for r in rows],
            y,
            color=plotting.POSITION_COLORS[group],
            marker=plotting.POSITION_MARKERS[group],
            s=34,
            label=group,
            zorder=3,
            edgecolor=plotting.SURFACE,
            linewidth=0.5,
        )
    ax.axvline(0.0, color=plotting.INK_MUTED, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['label_a']} vs {r['label_b']}" for r in rows],
        fontsize=plotting.BASE_FONT_PT - 1,
    )
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-1.02, 1.10)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    plotting.style_axis(
        ax,
        xlabel="Spearman rho within position group",
        title=f"Ten feature pairs whose correlation depends most on position, {season}",
    )
    for i, row in enumerate(rows):
        right = max(row["by_group"][g] for g in config.OUTFIELD_GROUPS)
        ax.annotate(
            f"spread {row['spread']:.2f}",
            (right, i),
            textcoords="offset points",
            xytext=(7, -2.5),
            fontsize=plotting.BASE_FONT_PT - 3,
            color=plotting.INK_MUTED,
        )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    plotting.save_figure(fig, "corr_divergence_top10")


def figure_distributions(df: pd.DataFrame, season: str) -> None:
    """Ridge-style KDE per position group for four representative metrics."""
    fig, axes = plotting.plt.subplots(
        2, 2, figsize=(plotting.WIDTH_FULL, 4.4), constrained_layout=True
    )
    axes = axes.ravel()
    groups = config.OUTFIELD_GROUPS

    for ax, metric in zip(axes, DISTRIBUTION_METRICS, strict=True):
        upper = float(np.percentile(df[metric].dropna(), 99.5)) * 1.10
        grid = np.linspace(0.0, upper, 256)
        for j, group in enumerate(groups):
            v = df.loc[df["position_group"] == group, metric].dropna().to_numpy()
            dens = stats.gaussian_kde(v)(grid)
            dens = dens / dens.max()
            base = float(len(groups) - 1 - j)
            ax.fill_between(
                grid,
                base,
                base + dens * 0.92,
                color=plotting.POSITION_COLORS[group],
                alpha=0.55,
                lw=0,
                zorder=2 + j,
            )
            ax.plot(
                grid,
                base + dens * 0.92,
                color=plotting.POSITION_COLORS[group],
                lw=1.0,
                zorder=2 + j,
            )
            ax.plot(
                [np.median(v), np.median(v)],
                [base, base + 0.5],
                color=plotting.SURFACE,
                lw=1.2,
                zorder=6 + j,
            )
        ax.set_yticks([float(len(groups) - 1 - j) for j in range(len(groups))])
        ax.set_yticklabels(groups)
        ax.set_ylim(-0.1, len(groups))
        ax.set_xlim(0.0, upper)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.set_title(label(metric), loc="left")
        ax.set_xlabel("per 90 minutes")

    fig.suptitle(
        f"Distribution by position group, {season} (white tick marks the median)",
        fontsize=plotting.BASE_FONT_PT,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    plotting.save_figure(fig, "distributions_by_position")


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def run_season(season: str) -> tuple[dict, dict, dict, pd.DataFrame]:
    df = load_eligible(season)
    feats = F.OUTFIELD_CORE
    summary = summary_by_position(df)
    corr = spearman_matrix(df, feats)
    div = correlation_divergence(df, feats)
    div["global_matrix"] = corr.round(6).to_dict()
    norm = normality_report(df, feats)
    return summary, div, norm, corr


def main() -> None:
    np.random.seed(config.RANDOM_STATE)
    plotting.use_style()

    summaries: dict = {}
    divergences: dict = {}
    normalities: dict = {}

    for season in config.SEASONS:
        summary, div, norm, corr = run_season(season)
        summaries[season] = summary
        divergences[season] = div
        normalities[season] = norm
        if season == config.SEASON_PRIMARY:
            figure_corr_heatmap(corr, season)
            figure_corr_divergence(div, season)
            figure_distributions(load_eligible(season), season)
        top = div["top10"][0]
        print(
            f"{season}: n={sum(summary['n_by_group'].values())} "
            f"largest correlation spread {top['spread']:.3f} "
            f"({top['label_a']} vs {top['label_b']}) "
            f"| sign flips {div['n_pairs_sign_flip']}/{div['n_pairs']} "
            f"| features rejecting normality "
            f"{norm['n_features_rejecting_normality_shapiro_005']}/{norm['n_features']}",
            flush=True,
        )

    for name, payload in (
        ("descriptive_summary", summaries),
        ("correlation_divergence", divergences),
        ("normality", normalities),
    ):
        with open(config.METRICS / f"{name}.json", "w") as fh:
            json.dump(payload, fh, indent=2)

    print("descriptive complete")


if __name__ == "__main__":
    main()
