"""Is role space clustered, or is it a continuum?

Every partitioning method returns a partition. Run k-means on a single multivariate
Gaussian and it will hand back two tidy clusters with a respectable silhouette, because
that is what it is built to do. So the finding that the selection rule chooses two
clusters is not by itself evidence that two groups exist, and the published habit of
reporting player archetypes from season aggregate data rests on exactly that inference.

This module tests the inference instead of making it. Four instruments, each sensitive to
a different way the claim could fail:

* a **null calibration**, which is the decisive one. Simulate data with the same size,
  dimension and covariance as the real feature matrix but built to contain no clusters
  at all, run the identical k-selection on it, and ask whether the real data's silhouette
  is any better. If it is not, the observed partition is what a continuum looks like when
  it is partitioned. Three nulls are run rather than one, because a single Gaussian is
  a soft reference: per-90 rates are skewed and bounded below, and skew alone lifts
  silhouette against a Gaussian. A Gaussian copula keeps every real marginal and the
  real rank correlations, and a uniform box over the principal axes assumes no
  distribution at all. The three are described in :func:`simulate`, and which of them
  is hardest to beat in each scope is recorded in the summary as a result rather than
  assumed.
* **Hartigan's dip test** on the leading principal components, which asks directly whether
  a distribution is unimodal. At nine thousand players the p-value rejects trivially, so
  the dip statistic is treated as the effect size and placed against the dip of each
  null's own leading component.
* **Gaussian mixture BIC**, which unlike silhouette carries no built-in preference for
  small numbers of components.
* **HDBSCAN**, which is permitted to answer that there are no clusters, and does so by
  labelling points as noise.

One further check is aimed at the stability evidence rather than the existence evidence.
A bootstrap adjusted Rand index is the usual argument that a partition is real. It is
computed here on the clusterless simulations as well as on the data, by the same routine
with the same seeding, so the paper can say with a number whether a resampling stability
check distinguishes real structure from a partitioned continuum.

A continuum is not a negative result. It is a different and more useful description of
the same data, and it explains why archetype taxonomies published from this kind of data
tend not to reproduce.
"""

from __future__ import annotations

import json

import diptest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

import config
from src import archive as A
from src import cluster as C
from src import plotting as P

#: Simulated datasets per scope and per null. Enough to place the observed value in a
#: distribution rather than against a single draw.
N_SIMULATIONS = 25

#: Sub-sample used for silhouette, which is quadratic in the number of points.
SILHOUETTE_SAMPLE = 4000

K_RANGE = list(range(2, 11))

#: The number of clusters the paper reports. Every null is scored against it.
K_REPORTED = 2

#: The clusterless nulls, keyed by name, with the metrics key each is written under. The
#: Gaussian null keeps the original ``null_calibration`` key so that ``src/tables.py`` and
#: the paper macros continue to find it; the two added nulls get keys of their own.
NULL_KEYS = {
    "gaussian": "null_calibration",
    "gaussian_copula": "null_calibration_copula",
    "uniform_principal_box": "null_calibration_uniform",
}

#: What each null keeps of the data and what it destroys, written into the metrics file
#: beside the numbers so the JSON is honest on its own.
NULL_DESCRIPTIONS = {
    "gaussian": (
        "A single multivariate normal with the empirical mean and covariance. Matches the "
        "data in size, dimension and covariance and in nothing else: every marginal is "
        "symmetric and unbounded where the real per-90 rates are skewed and bounded below, "
        "and skew alone lifts silhouette against this reference, which is why the copula "
        "null stands beside it."
    ),
    "gaussian_copula": (
        "A Gaussian copula fitted to the normal scores of the observed ranks and mapped back "
        "through the empirical quantile function of each observed column. Preserves every "
        "marginal distribution exactly, including skew, bounds, ties and any bimodality a "
        "single feature carries on its own, and preserves the pairwise rank correlation up "
        "to sampling noise. Destroys joint structure beyond pairwise rank correlation, which "
        "is where two groups that differ in several features at once would live. It tests "
        "whether the joint structure carries anything beyond the marginals, not whether the "
        "marginals are unimodal."
    ),
    "uniform_principal_box": (
        "Uniform within the bounding box of the data after rotation to its principal axes, "
        "rotated back. The reference distribution of the gap statistic in Tibshirani, "
        "Walther and Hastie (2001). Assumes no distributional form. Every principal axis "
        "is filled evenly over its full observed range, and a bounded or heavy-tailed axis "
        "ranges over several times its standard deviation, so the minor axes carry more "
        "spread than in the data and the reference is more isotropic than the data it is "
        "matched to. That lowers its silhouette at k=2, so for silhouette it is a soft "
        "reference. Its leading component is flat rather than peaked, which gives it the "
        "largest dip of the three, so for the dip it is the hard one."
    ),
}

#: Figure encoding per null. Identity rests on the dash pattern rather than on hue, since
#: the palette validates four categorical colours and the observed curve takes one.
NULL_LABELS = {
    "gaussian": "Gaussian null",
    "gaussian_copula": "copula null",
    "uniform_principal_box": "uniform null",
}
NULL_LINESTYLES = {
    "gaussian": (0, (4, 3)),
    "gaussian_copula": (0, (1, 1.5)),
    "uniform_principal_box": (0, (5, 2, 1, 2)),
}

#: Simulated datasets per null on which the bootstrap is repeated. Each one costs a full
#: bootstrap, and five is enough to give the null index a spread as well as a level.
BOOTSTRAP_DATASETS = 5

#: Resamples per bootstrap. ``config.BOOTSTRAP_N`` is 200 on the live sample and
#: ``archive_validation.N_BOOT`` reduces it to 50 on this one, because every resample
#: refits KMeans on up to 9,263 players and accumulates a co-assignment matrix of that
#: dimension squared. The same count and the same tag are used here, so the observed index
#: reproduces the one in ``archive_cluster_validation.json`` to the last digit.
N_BOOTSTRAP = 50

#: A bootstrap index at or above this level is what a published taxonomy cites as evidence
#: that its partition is stable. A clusterless dataset that reaches it passes the same
#: screen, and the summary says so in those terms.
STABLE_ARI = 0.8


def _silhouette(x: np.ndarray, labels: np.ndarray, seed: int) -> float:
    n = x.shape[0]
    if n <= SILHOUETTE_SAMPLE:
        return float(silhouette_score(x, labels))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, SILHOUETTE_SAMPLE, replace=False)
    return float(silhouette_score(x[idx], labels[idx]))


def _z(observed: float, sims: np.ndarray) -> float | None:
    sd = float(sims.std(ddof=1))
    return round(float((observed - sims.mean()) / sd), 2) if sd > 0 else None


def kmeans_profile(x: np.ndarray, seed: int) -> dict[int, float]:
    """Silhouette across the candidate range for one dataset."""
    out = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(x)
        out[k] = _silhouette(x, labels, seed)
    return out


def simulate(x: np.ndarray, null: str) -> list[np.ndarray]:
    """``N_SIMULATIONS`` clusterless datasets with the shape of ``x``, under one null.

    Every null is seeded from ``config.RANDOM_STATE`` and draws its datasets in sequence
    from one generator, so the Gaussian datasets are the ones the original single-null
    calibration used and its numbers do not move.

    ``gaussian``
        A single multivariate normal with the empirical mean and covariance.
    ``gaussian_copula``
        Each observed column is transformed to normal scores through its ranks, the
        correlation of those scores is estimated, a multivariate normal with that
        correlation is drawn, and each simulated column is mapped back through the
        empirical quantile function of the observed column. The map is a permutation:
        the sorted observed column is handed out in the rank order of the simulated one,
        so every simulated column is exactly the observed column reordered. That keeps
        every marginal exactly, skew, bounds, ties and any single-feature bimodality
        included, and keeps the pairwise rank correlation up to sampling noise. It does
        not keep anything joint beyond that, which is what a real division into groups
        is. So this null asks whether the joint structure carries something the marginals
        do not; it does not ask whether the marginals are unimodal.
    ``uniform_principal_box``
        Uniform within the bounding box of the data after rotation to its principal axes,
        then rotated back. This is the reference distribution of the gap statistic,
        Tibshirani, Walther and Hastie (2001), and it assumes no distribution at all.
    """
    n, d = x.shape
    rng = np.random.default_rng(config.RANDOM_STATE)
    if null == "gaussian":
        mean = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        return [
            rng.multivariate_normal(mean, cov, size=n, method="cholesky")
            for _ in range(N_SIMULATIONS)
        ]
    if null == "gaussian_copula":
        # Normal scores of the ranks, ties at their average rank, and the correlation of
        # those scores. That correlation is the copula parameter.
        scores = norm.ppf(rankdata(x, axis=0) / (n + 1))
        corr = np.corrcoef(scores, rowvar=False)
        sorted_columns = np.sort(x, axis=0)
        out = []
        for _ in range(N_SIMULATIONS):
            gauss = rng.multivariate_normal(np.zeros(d), corr, size=n, method="cholesky")
            order = np.argsort(np.argsort(gauss, axis=0), axis=0)
            out.append(np.take_along_axis(sorted_columns, order, axis=0))
        return out
    if null == "uniform_principal_box":
        # The gap statistic's reference in Tibshirani, Walther and Hastie (2001): uniform
        # over the box aligned with the principal axes of the data, rotated back so the
        # simulated matrix sits in the original feature basis.
        mean = x.mean(axis=0)
        _, _, axes = np.linalg.svd(x - mean, full_matrices=False)
        rotated = (x - mean) @ axes.T
        low, high = rotated.min(axis=0), rotated.max(axis=0)
        return [rng.uniform(low, high, size=(n, d)) @ axes + mean for _ in range(N_SIMULATIONS)]
    raise ValueError(f"unknown null {null!r}")


def rank_correlation_check(x: np.ndarray, simulated: list[np.ndarray]) -> dict:
    """How closely the copula datasets reproduce the observed Spearman correlations."""
    target = spearmanr(x).correlation
    upper = np.triu_indices_from(target, 1)
    deviations = [np.abs(spearmanr(sim).correlation - target)[upper] for sim in simulated]
    return {
        "observed_mean_abs_spearman": round(float(np.abs(target[upper]).mean()), 4),
        "mean_abs_deviation_mean_over_simulations": round(
            float(np.mean([d.mean() for d in deviations])), 4
        ),
        "max_abs_deviation_mean_over_simulations": round(
            float(np.mean([d.max() for d in deviations])), 4
        ),
        "max_abs_deviation_worst_simulation": round(float(max(d.max() for d in deviations)), 4),
    }


def null_calibration(
    x: np.ndarray,
    tag: str,
    observed: dict[int, float],
    simulated: list[np.ndarray],
    null: str,
) -> dict:
    """Compare the observed silhouette profile against clusterless data of the same shape.

    Every simulated matrix goes through the identical :func:`kmeans_profile`, seeded as
    the observed one is, so the only thing that differs between the two sides is the data.
    """
    profiles = [kmeans_profile(sim, config.RANDOM_STATE + i) for i, sim in enumerate(simulated)]

    rows = {}
    for k in K_RANGE:
        sims = np.array([s[k] for s in profiles])
        obs = observed[k]
        rows[str(k)] = {
            "observed": round(obs, 4),
            "null_mean": round(float(sims.mean()), 4),
            "null_sd": round(float(sims.std(ddof=1)), 4),
            "null_max": round(float(sims.max()), 4),
            "z_against_null": _z(obs, sims),
            "exceeds_every_simulation": bool(obs > sims.max()),
        }
    best_k = max(observed, key=lambda k: observed[k])
    null_best = float(np.mean([max(s.values()) for s in profiles]))
    out = {
        "scope": tag,
        "n": int(x.shape[0]),
        "d": int(x.shape[1]),
        "n_simulations": N_SIMULATIONS,
        "by_k": rows,
        "observed_best_k": int(best_k),
        "observed_best_silhouette": round(observed[best_k], 4),
        "null_best_silhouette_mean": round(null_best, 4),
        "separation_ratio": round(observed[best_k] / null_best, 3),
        "null": null,
        "description": NULL_DESCRIPTIONS[null],
    }
    if null == "gaussian_copula":
        out["rank_correlation_check"] = rank_correlation_check(x, simulated)
    return out


def bootstrap_ari(x: np.ndarray, tag: str) -> dict:
    """Resampling stability of the ``K_REPORTED`` partition, by ``cluster.bootstrap_stability``.

    The reference partition is the KMeans fit on the full matrix, each resample is drawn
    with replacement and refitted, and the adjusted Rand index is scored on the players
    present in the resample. Nothing is reimplemented: the routine and its seeding are
    the ones the paper's own stability numbers come from, so the index on a simulated
    matrix is strictly comparable with the index on the real one.
    """
    labels = C._kmeans(x, K_REPORTED, tag).labels_
    boot = C.bootstrap_stability(x, labels, K_REPORTED, tag, n_boot=N_BOOTSTRAP)
    return {key: boot[key] for key in ("n_bootstrap", "ari_mean", "ari_sd", "ari_min", "ari_max")}


def bootstrap_calibration(x: np.ndarray, tag: str, simulated: dict[str, list[np.ndarray]]) -> dict:
    """The bootstrap index on the data beside the same index on clusterless data.

    A high index is routinely offered as evidence that a partition is real. Whether it
    can be that is an empirical question: if k-means partitions a continuum the same way
    on every resample, the index is high there too and the check has no power against
    the null it is meant to exclude.
    """
    # The tag ``archive_validation`` uses, so the observed index here is the paper's own.
    observed = bootstrap_ari(x, f"archive|{tag}")
    by_null = {}
    for null, sims in simulated.items():
        runs = [
            bootstrap_ari(sim, f"structure|{tag}|{null}|{i}")
            for i, sim in enumerate(sims[:BOOTSTRAP_DATASETS])
        ]
        means = np.array([r["ari_mean"] for r in runs])
        by_null[null] = {
            "n_datasets": len(runs),
            "ari_mean": round(float(means.mean()), 4),
            "ari_sd": round(float(means.std(ddof=1)), 4),
            "ari_min": round(float(means.min()), 4),
            "ari_max": round(float(means.max()), 4),
            "within_dataset_sd_mean": round(float(np.mean([r["ari_sd"] for r in runs])), 4),
            "observed_minus_null": round(observed["ari_mean"] - float(means.mean()), 4),
            "observed_exceeds_every_dataset": bool(observed["ari_mean"] > means.max()),
            "per_dataset": runs,
        }
    return {
        "k": K_REPORTED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_datasets_per_null": BOOTSTRAP_DATASETS,
        "procedure": (
            "cluster.bootstrap_stability, unchanged: KMeans reference fit on the full "
            "matrix, resamples drawn with replacement and refitted, adjusted Rand index on "
            "the players present in each resample. ari_mean and ari_sd under each null are "
            "taken over the per-dataset means; within_dataset_sd_mean is the mean of the "
            "between-resample standard deviations."
        ),
        "observed": observed,
        "by_null": by_null,
    }


def pc1_dip(x: np.ndarray) -> float:
    """The dip statistic on the leading principal component, the dip test's effect size."""
    scores = PCA(n_components=1, random_state=config.RANDOM_STATE).fit_transform(x)
    return float(diptest.dipstat(np.ascontiguousarray(scores[:, 0])))


def dip_calibration(x: np.ndarray, simulated: dict[str, list[np.ndarray]]) -> dict:
    """Place the observed dip on PC1 against the dip on PC1 of each clusterless null.

    With thousands of players the dip test rejects unimodality for departures far too
    small to matter, so the p-value in :func:`dip_tests` says little. The statistic says
    more. Each simulated dataset gets its own principal axis and its own dip, exactly as
    the observed data does, and the observed dip is placed against that distribution in
    the same form as the silhouette calibration.
    """
    observed = pc1_dip(x)
    by_null = {}
    for null, sims in simulated.items():
        dips = np.array([pc1_dip(sim) for sim in sims])
        by_null[null] = {
            "n_simulations": int(dips.size),
            "observed": round(observed, 5),
            "null_mean": round(float(dips.mean()), 5),
            "null_sd": round(float(dips.std(ddof=1)), 5),
            "null_max": round(float(dips.max()), 5),
            "z_against_null": _z(observed, dips),
            "exceeds_every_simulation": bool(observed > dips.max()),
        }
    scored = {n: v["z_against_null"] for n, v in by_null.items() if v["z_against_null"] is not None}
    hardest = min(scored, key=lambda n: scored[n])
    return {
        "statistic": "Hartigan dip on PC1, each dataset projected on its own leading component",
        "observed_dip": round(observed, 5),
        "by_null": by_null,
        "hardest_null": hardest,
        "min_z": scored[hardest],
        "exceeds_every_simulation_under_every_null": all(
            v["exceeds_every_simulation"] for v in by_null.values()
        ),
        "note": (
            "The p-value reported in dip_tests rejects trivially at this sample size and is "
            "not informative. The dip statistic is the effect size, and the z against each "
            "null's own dip distribution is the calibrated quantity."
        ),
    }


def dip_tests(x: np.ndarray, n_components: int = 5) -> dict:
    """Hartigan's dip test for unimodality on the leading principal components."""
    pca = PCA(n_components=min(n_components, x.shape[1]), random_state=config.RANDOM_STATE)
    scores = pca.fit_transform(x)
    out = {}
    for i in range(scores.shape[1]):
        stat, p = diptest.diptest(np.ascontiguousarray(scores[:, i]))
        out[f"PC{i + 1}"] = {
            "dip": round(float(stat), 5),
            "p_value": round(float(p), 4),
            "multimodal_at_005": bool(p < 0.05),
            "explained_variance": round(float(pca.explained_variance_ratio_[i]), 4),
        }
    return out


def mixture_bic(x: np.ndarray) -> dict:
    """Gaussian mixture BIC, which does not share silhouette's preference for small k."""
    out = {}
    for k in K_RANGE:
        gm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=config.RANDOM_STATE,
            reg_covar=1e-4,
            max_iter=200,
        ).fit(x)
        out[str(k)] = round(float(gm.bic(x)), 1)
    best = min(out, key=lambda k: out[k])
    return {
        "bic_by_k": out,
        "best_k": int(best),
        "monotone": out[str(K_RANGE[-1])] == min(out.values()),
    }


def density_report(x: np.ndarray) -> dict:
    """HDBSCAN, which is allowed to conclude that nothing is clustered."""
    out = {}
    for size in (25, 50, 100):
        model = HDBSCAN(min_cluster_size=size).fit(x)
        labels = model.labels_
        n_clusters = int(len({int(v) for v in labels if v >= 0}))
        out[str(size)] = {
            "n_clusters": n_clusters,
            "noise_fraction": round(float((labels < 0).mean()), 4),
        }
    return out


def figure_structure(results: dict) -> None:
    """Observed silhouette against all three null bands, one panel per scope.

    The bands share one grey, since three ribbons a few thousandths tall overlap and a
    fourth hue is not available; each null's mean carries its own dash pattern and the
    legend keys those, so the nulls are told apart by line style.
    """
    scopes = [s for s in results["scopes"]]
    fig, axes = plt.subplots(1, len(scopes), figsize=(P.WIDTH_FULL, 2.7), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, scope in zip(axes, scopes, strict=True):
        entry = results["scopes"][scope]
        ks = [int(k) for k in entry["null_calibration"]["by_k"]]
        for j, (null, key) in enumerate(NULL_KEYS.items()):
            cal = entry[key]
            nul = np.array([cal["by_k"][str(k)]["null_mean"] for k in ks])
            sd = np.array([cal["by_k"][str(k)]["null_sd"] for k in ks])
            ax.fill_between(
                ks,
                nul - 2 * sd,
                nul + 2 * sd,
                color=P.CONTEXT_GREY,
                alpha=0.4,
                linewidth=0,
                label="clusterless nulls, 2 sd" if j == 0 else "_nolegend_",
            )
            ax.plot(
                ks,
                nul,
                color=P.INK_SECONDARY,
                linewidth=1.4,
                linestyle=NULL_LINESTYLES[null],
                label=NULL_LABELS[null],
            )
        obs = [entry["null_calibration"]["by_k"][str(k)]["observed"] for k in ks]
        ax.plot(
            ks,
            obs,
            color=P.CATEGORICAL[1],
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="observed",
        )
        P.style_axis(ax, "Number of clusters", "Silhouette" if scope == scopes[0] else "", scope)
        ax.set_xticks(ks[::2])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle(
        "Observed cluster quality against data built to contain no clusters",
        fontsize=P.BASE_FONT_PT,
        fontweight="bold",
    )
    P.save_figure(fig, "structure_null")


def summarise(results: dict) -> dict:
    """The cross-null summary: the hardest null per scope for silhouette, bootstrap and dip."""
    scopes = results["scopes"]

    k2 = {}
    for name, entry in scopes.items():
        zs = {
            null: entry[key]["by_k"][str(K_REPORTED)]["z_against_null"]
            for null, key in NULL_KEYS.items()
        }
        hardest = min(zs, key=lambda n: zs[n])
        k2[name] = {
            "z_by_null": zs,
            "min_z": zs[hardest],
            "hardest_null": hardest,
            "exceeds_every_simulation_under_every_null": all(
                entry[key]["by_k"][str(K_REPORTED)]["exceeds_every_simulation"]
                for key in NULL_KEYS.values()
            ),
        }

    null_aris = [
        v["ari_mean"]
        for entry in scopes.values()
        for v in entry["null_bootstrap"]["by_null"].values()
    ]
    null_ari_datasets = [
        r["ari_mean"]
        for entry in scopes.values()
        for v in entry["null_bootstrap"]["by_null"].values()
        for r in v["per_dataset"]
    ]
    observed_aris = [entry["null_bootstrap"]["observed"]["ari_mean"] for entry in scopes.values()]
    gaps = [
        v["observed_minus_null"]
        for entry in scopes.values()
        for v in entry["null_bootstrap"]["by_null"].values()
    ]
    null_min, null_max = min(null_aris), max(null_aris)
    observed_min = min(observed_aris)
    exceeds = all(
        v["observed_exceeds_every_dataset"]
        for entry in scopes.values()
        for v in entry["null_bootstrap"]["by_null"].values()
    )
    passes_screen = min(null_ari_datasets) >= STABLE_ARI
    if passes_screen:
        reading = (
            f"Every clusterless dataset in every scope returns a bootstrap ARI of at least "
            f"{min(null_ari_datasets):.3f}, above the {STABLE_ARI} that a stability screen "
            f"accepts, against an observed minimum of {observed_min:.3f}. A resampling "
            "stability check therefore does not distinguish real structure from a "
            "partitioned continuum."
        )
    elif exceeds:
        reading = (
            f"The observed bootstrap ARI exceeds every clusterless dataset in every scope, "
            f"with the null falling as low as {min(null_ari_datasets):.3f}, so on this "
            "sample the resampling check does separate real structure from a partitioned "
            "continuum."
        )
    else:
        reading = (
            "The clusterless bootstrap ARI is mixed: it falls below the stability screen on "
            "some datasets and matches or exceeds the observed index on others."
        )
    bootstrap = {
        "k": K_REPORTED,
        "stability_screen_ari": STABLE_ARI,
        "observed_ari_min_over_scopes": round(observed_min, 4),
        "observed_ari_max_over_scopes": round(max(observed_aris), 4),
        "null_ari_min_over_scopes_and_nulls": round(null_min, 4),
        "null_ari_max_over_scopes_and_nulls": round(null_max, 4),
        "null_ari_min_single_dataset": round(min(null_ari_datasets), 4),
        "observed_minus_null_min": round(min(gaps), 4),
        "observed_minus_null_max": round(max(gaps), 4),
        "observed_exceeds_every_null_dataset_everywhere": bool(exceeds),
        "null_passes_stability_screen_everywhere": bool(passes_screen),
        "reading": reading,
    }

    dip = {
        name: {
            "observed_dip": entry["dip_calibration"]["observed_dip"],
            "z_by_null": {
                null: v["z_against_null"] for null, v in entry["dip_calibration"]["by_null"].items()
            },
            "min_z": entry["dip_calibration"]["min_z"],
            "hardest_null": entry["dip_calibration"]["hardest_null"],
            "exceeds_every_simulation_under_every_null": entry["dip_calibration"][
                "exceeds_every_simulation_under_every_null"
            ],
        }
        for name, entry in scopes.items()
    }

    return {
        "k2_against_three_nulls": k2,
        "k2_min_z_over_scopes_and_nulls": min(v["min_z"] for v in k2.values()),
        "k2_hardest_null_by_scope": {name: v["hardest_null"] for name, v in k2.items()},
        "k2_exceeds_every_simulation_under_every_null_everywhere": all(
            v["exceeds_every_simulation_under_every_null"] for v in k2.values()
        ),
        "bootstrap": bootstrap,
        "dip_pc1_against_three_nulls": dip,
        "dip_scopes_exceeding_every_null": [
            name for name, v in dip.items() if v["exceeds_every_simulation_under_every_null"]
        ],
    }


def main() -> None:
    P.use_style()
    z = pd.read_parquet(config.DATA_PROCESSED / "archive_z_global.parquet")
    zg = pd.read_parquet(config.DATA_PROCESSED / "archive_z_bygroup.parquet")
    usable = [c for c in A.OUTFIELD_CORE if c in z.columns]

    scopes = {"All outfield": z[usable].to_numpy(float)}
    for grp in config.OUTFIELD_GROUPS:
        sub = zg[zg["position_group"] == grp]
        scopes[grp] = sub[usable].to_numpy(float)

    results = {
        "question": "Is the role space clustered, or continuous?",
        "n_features": len(usable),
        "nulls": NULL_DESCRIPTIONS,
        "scopes": {},
    }
    for name, x in scopes.items():
        observed = kmeans_profile(x, config.RANDOM_STATE)
        simulated = {null: simulate(x, null) for null in NULL_KEYS}
        calibrations = {
            null: null_calibration(x, name, observed, sims, null)
            for null, sims in simulated.items()
        }
        cal = calibrations["gaussian"]
        print(
            f"{name:14s} n={x.shape[0]:>5,}  observed best silhouette "
            f"{cal['observed_best_silhouette']:.3f} at k={cal['observed_best_k']}  "
            f"null {cal['null_best_silhouette_mean']:.3f}  ratio {cal['separation_ratio']}",
            flush=True,
        )
        for null, c in calibrations.items():
            row = c["by_k"][str(K_REPORTED)]
            print(
                f"  {null:22s} k={K_REPORTED} null {row['null_mean']:.4f} sd {row['null_sd']:.4f} "
                f"max {row['null_max']:.4f}  z {row['z_against_null']}  ratio {c['separation_ratio']}",
                flush=True,
            )
        results["scopes"][name] = {
            "null_calibration": cal,
            "dip_tests": dip_tests(x),
            "mixture_bic": mixture_bic(x),
            "density": density_report(x),
            "null_calibration_copula": calibrations["gaussian_copula"],
            "null_calibration_uniform": calibrations["uniform_principal_box"],
            "dip_calibration": dip_calibration(x, simulated),
            "null_bootstrap": bootstrap_calibration(x, name, simulated),
        }
        boot = results["scopes"][name]["null_bootstrap"]
        print(
            f"  bootstrap ARI observed {boot['observed']['ari_mean']:.4f}  "
            + "  ".join(f"{n} {v['ari_mean']:.4f}" for n, v in boot["by_null"].items()),
            flush=True,
        )

    figure_structure(results)

    ratios = [v["null_calibration"]["separation_ratio"] for v in results["scopes"].values()]
    dips = [v["dip_tests"]["PC1"]["multimodal_at_005"] for v in results["scopes"].values()]
    results["summary"] = {
        "separation_ratio_min": min(ratios),
        "separation_ratio_max": max(ratios),
        "pc1_multimodal_anywhere": any(dips),
        "verdict": (
            "The observed partition is not meaningfully better separated than one imposed "
            "on data built to contain no clusters, and the leading component is unimodal, "
            "so the role space is better described as a continuum than as a set of types."
            if max(ratios) < 1.25 and not any(dips)
            else "At least one scope shows separation beyond the clusterless null."
        ),
        **summarise(results),
    }
    with open(config.METRICS / "structure.json", "w") as fh:
        json.dump(results, fh, indent=2)

    summary = results["summary"]
    print()
    print("silhouette at k=2, z against each null (Gaussian / copula / uniform):")
    for name, v in summary["k2_against_three_nulls"].items():
        zs = v["z_by_null"]
        print(
            f"  {name:14s} {zs['gaussian']:>6} / {zs['gaussian_copula']:>6} / "
            f"{zs['uniform_principal_box']:>6}   hardest {v['hardest_null']}"
        )
    print()
    print(f"bootstrap ARI at k={K_REPORTED}, observed against clusterless data:")
    for name, v in results["scopes"].items():
        boot = v["null_bootstrap"]
        print(
            f"  {name:14s} observed {boot['observed']['ari_mean']:.3f}  "
            + "  ".join(
                f"{n} {b['ari_mean']:.3f} (sd {b['ari_sd']:.3f})"
                for n, b in boot["by_null"].items()
            )
        )
    print(f"  {summary['bootstrap']['reading']}")
    print()
    print("dip on PC1, observed against each null's dip distribution (z):")
    for name, v in summary["dip_pc1_against_three_nulls"].items():
        zs = v["z_by_null"]
        print(
            f"  {name:14s} dip={v['observed_dip']:.5f}  {zs['gaussian']:>6} / "
            f"{zs['gaussian_copula']:>6} / {zs['uniform_principal_box']:>6}   hardest {v['hardest_null']}"
        )
    print()
    print("verdict:", summary["verdict"])
    print("structure complete")


if __name__ == "__main__":
    main()
