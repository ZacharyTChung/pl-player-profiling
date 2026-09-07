"""Is role space clustered, or is it a continuum?

Every partitioning method returns a partition. Run k-means on a single multivariate
Gaussian and it will hand back two tidy clusters with a respectable silhouette, because
that is what it is built to do. So the finding that the selection rule chooses two
clusters is not by itself evidence that two groups exist, and the published habit of
reporting player archetypes from season aggregate data rests on exactly that inference.

This module tests the inference instead of making it. Four instruments, each sensitive to
a different way the claim could fail:

* a **null calibration**, which is the decisive one. Simulate data with the same size,
  dimension and covariance as the real feature matrix but drawn from a single Gaussian,
  so that by construction it contains no clusters at all. Run the identical k-selection on
  it. If the real data's silhouette is no better than the simulated one, then the observed
  partition is what a continuum looks like when it is partitioned.
* **Hartigan's dip test** on the leading principal components, which asks directly whether
  a distribution is unimodal.
* **Gaussian mixture BIC**, which unlike silhouette carries no built-in preference for
  small numbers of components.
* **HDBSCAN**, which is permitted to answer that there are no clusters, and does so by
  labelling points as noise.

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
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

import config
from src import archive as A
from src import plotting as P

#: Simulated datasets per scope. Enough to place the observed value in a distribution
#: rather than against a single draw.
N_SIMULATIONS = 25

#: Sub-sample used for silhouette, which is quadratic in the number of points.
SILHOUETTE_SAMPLE = 4000

K_RANGE = list(range(2, 11))


def _silhouette(x: np.ndarray, labels: np.ndarray, seed: int) -> float:
    n = x.shape[0]
    if n <= SILHOUETTE_SAMPLE:
        return float(silhouette_score(x, labels))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, SILHOUETTE_SAMPLE, replace=False)
    return float(silhouette_score(x[idx], labels[idx]))


def kmeans_profile(x: np.ndarray, seed: int) -> dict[int, float]:
    """Silhouette across the candidate range for one dataset."""
    out = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(x)
        out[k] = _silhouette(x, labels, seed)
    return out


def null_calibration(x: np.ndarray, tag: str) -> dict:
    """Compare the observed silhouette profile against clusterless data of the same shape.

    The simulated matrices are drawn from a multivariate normal with the empirical mean
    and covariance, so they match the real data in size, dimension and correlation
    structure and differ only in containing no group structure whatsoever.
    """
    observed = kmeans_profile(x, config.RANDOM_STATE)
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    rng = np.random.default_rng(config.RANDOM_STATE)

    simulated: list[dict[int, float]] = []
    for i in range(N_SIMULATIONS):
        sim = rng.multivariate_normal(mean, cov, size=x.shape[0], method="cholesky")
        simulated.append(kmeans_profile(sim, config.RANDOM_STATE + i))

    rows = {}
    for k in K_RANGE:
        sims = np.array([s[k] for s in simulated])
        obs = observed[k]
        rows[str(k)] = {
            "observed": round(obs, 4),
            "null_mean": round(float(sims.mean()), 4),
            "null_sd": round(float(sims.std(ddof=1)), 4),
            "null_max": round(float(sims.max()), 4),
            "z_against_null": round(float((obs - sims.mean()) / sims.std(ddof=1)), 2)
            if sims.std(ddof=1) > 0
            else None,
            "exceeds_every_simulation": bool(obs > sims.max()),
        }
    best_k = max(observed, key=lambda k: observed[k])
    return {
        "scope": tag,
        "n": int(x.shape[0]),
        "d": int(x.shape[1]),
        "n_simulations": N_SIMULATIONS,
        "by_k": rows,
        "observed_best_k": int(best_k),
        "observed_best_silhouette": round(observed[best_k], 4),
        "null_best_silhouette_mean": round(float(np.mean([max(s.values()) for s in simulated])), 4),
        "separation_ratio": round(
            observed[best_k] / float(np.mean([max(s.values()) for s in simulated])), 3
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
    scopes = [s for s in results["scopes"]]
    fig, axes = plt.subplots(1, len(scopes), figsize=(P.WIDTH_FULL, 2.7), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, scope in zip(axes, scopes, strict=True):
        cal = results["scopes"][scope]["null_calibration"]
        ks = [int(k) for k in cal["by_k"]]
        obs = [cal["by_k"][str(k)]["observed"] for k in ks]
        nul = [cal["by_k"][str(k)]["null_mean"] for k in ks]
        sd = [cal["by_k"][str(k)]["null_sd"] for k in ks]

        ax.fill_between(
            ks,
            np.array(nul) - 2 * np.array(sd),
            np.array(nul) + 2 * np.array(sd),
            color=P.CONTEXT_GREY,
            alpha=0.55,
            linewidth=0,
            label="clusterless null, 2 sd",
        )
        ax.plot(
            ks, nul, color=P.INK_SECONDARY, linewidth=1.4, linestyle=(0, (4, 3)), label="null mean"
        )
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
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Observed cluster quality against data built to contain no clusters",
        fontsize=P.BASE_FONT_PT,
        fontweight="bold",
    )
    P.save_figure(fig, "structure_null")


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
        "scopes": {},
    }
    for name, x in scopes.items():
        cal = null_calibration(x, name)
        results["scopes"][name] = {
            "null_calibration": cal,
            "dip_tests": dip_tests(x),
            "mixture_bic": mixture_bic(x),
            "density": density_report(x),
        }
        print(
            f"{name:14s} n={x.shape[0]:>5,}  observed best silhouette "
            f"{cal['observed_best_silhouette']:.3f} at k={cal['observed_best_k']}  "
            f"null {cal['null_best_silhouette_mean']:.3f}  ratio {cal['separation_ratio']}",
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
    }
    with open(config.METRICS / "structure.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print()
    print("dip test on PC1 (is the dominant axis unimodal?):")
    for name, v in results["scopes"].items():
        pc1 = v["dip_tests"]["PC1"]
        print(
            f"  {name:14s} dip={pc1['dip']:.5f} p={pc1['p_value']:.4f} "
            f"multimodal={pc1['multimodal_at_005']}"
        )
    print()
    print("verdict:", results["summary"]["verdict"])
    print("structure complete")


if __name__ == "__main__":
    main()
