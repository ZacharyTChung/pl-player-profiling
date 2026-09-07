"""Does the two-mode result depend on any one block of features?

``src/structure.py`` established that the archive's 33 features support exactly two modes:
the declared consensus rule returns k=2 in every scope and the observed k=2 silhouette
exceeds every one of the clusterless Gaussian simulations. A reviewer will ask whether that
rests on one block of statistics. Touches by zone alone could split the pool into a
defensive half and an attacking half, and if the two modes were nothing more than that
block restated, dropping it would dissolve them.

This module answers by ablation. The 33 features are partitioned into six families that
follow the archive's own grouping of its sources, and each family is treated twice:

* **leave one family out** asks whether the two modes survive the removal of any single
  block. Survival means that the declared rule still returns k=2 and that the observed
  k=2 silhouette still exceeds every clusterless simulation. Whether the survivors are
  the same two modes is a separate question, answered by the adjusted Rand index between
  the k=2 partition on the remaining features and the k=2 partition on the full set,
  scored on the same players.
* **keep one family only** asks the converse: which blocks carry the two-mode structure
  on their own. A family that shows no separation from the null on its own is not a
  failure of the analysis, it is the finding that this family does not describe the
  division.

Both run in the same four scopes as the calibration, with the identical rule from
``src/cluster.py`` and the identical Gaussian null from ``src/structure.py``, so nothing
here can drift from the headline result. The full set is run alongside as the reference
row, under the same seeds, so its numbers reproduce ``results/metrics/structure.json``
and ``results/metrics/archive_cluster_validation.json`` rather than approximating them.

Cost
----
Every cell reruns the gap statistic with its fifty reference sets over eleven values of
k, and one feature set costs about six minutes across the four scopes. The fifty two
cells are therefore run as independent processes, each restricted to a single thread.
Every seed in the two imported modules is derived from ``config.RANDOM_STATE`` or from a
digest of the cell's tag, never from execution order, so the schedule cannot change a
number, and one thread per cell removes the thread scheduling that could otherwise
reorder floating point sums inside a KMeans fit. At these sizes a single-threaded cell
is also faster than a threaded one.

Outputs
-------
``results/metrics/ablation.json`` and the figure ``ablation``.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

import config
from src import archive as A
from src import cluster as C
from src import plotting as P
from src import structure as S

#: The six feature families, in the archive's canonical names. They follow the source
#: groups annotated in ``archive.SOURCES``, folding two of its small groups into
#: neighbours: take-ons sit with progression because a take-on moves the ball past an
#: opponent, and fouls committed sit with defending because a foul is overwhelmingly a
#: defensive event, the same reading ``config.PADJ_COUNTS`` makes.
FAMILIES: dict[str, list[str]] = {
    "shooting": ["np_xg", "shots", "goals_non_penalty", "shot_accuracy_pct"],
    "creation": [
        "xag",
        "key_passes",
        "sca",
        "gca",
        "passes_final_third",
        "passes_penalty_area",
        "crosses",
    ],
    "progression": [
        "progressive_passes",
        "progressive_carries",
        "progressive_receptions",
        "carries_final_third",
        "take_ons",
        "take_on_success_pct",
    ],
    "territory": [
        "touches_def_pen",
        "touches_def_third",
        "touches_mid_third",
        "touches_att_third",
        "touches_att_pen",
    ],
    "defending": [
        "tackles",
        "interceptions",
        "blocks",
        "clearances",
        "ball_recoveries",
        "fouls_committed",
        "tackle_win_pct",
        "aerials_won_pct",
    ],
    "passing": ["pass_cmp_short_pct", "pass_cmp_medium_pct", "pass_cmp_long_pct"],
}

#: The number of clusters the paper reports, taken from the calibration so the ablation
#: cannot ask about a different k than the headline result.
K = S.K_REPORTED

#: The null every cell is scored against: the Gaussian one, which is the paper's headline
#: calibration and the one the reviewer's question is about.
NULL = "gaussian"

#: Simulated datasets per cell. The same count as the calibration, so the reference row
#: reproduces ``structure.json`` rather than approximating it.
N_SIMULATIONS = S.N_SIMULATIONS

#: A family carries the two modes alone when, on its own, the rule still returns ``K``
#: and the observed silhouette at ``K`` sits this many null standard deviations above the
#: null mean. Three is deliberately far below the headline z of 13 to 24, so the threshold
#: asks whether a family separates at all, not whether it matches the full set.
CARRIES_ALONE_MIN_Z = 3.0

#: Worker processes for the cells, in joblib's convention: every core.
N_JOBS = -1


# --------------------------------------------------------------------------------------
# The families, mapped to the z-scored columns and audited against OUTFIELD_CORE
# --------------------------------------------------------------------------------------


def column_for(feature: str) -> str:
    """The z-scored column that carries one canonical archive feature.

    Mirrors the construction of ``archive.OUTFIELD_CORE``: possession adjusted counts end
    in ``_padj_p90``, other counts in ``_p90``, and rates are used under their own name.
    """
    if feature in A.RATES:
        return feature
    if feature in A.DEFENSIVE_COUNTS:
        return f"{feature}_padj_p90"
    return f"{feature}_p90"


def audit_families() -> dict[str, list[str]]:
    """Map every family to its columns, refusing to run unless the union is OUTFIELD_CORE.

    The check is what stops the mapping drifting: a feature added to the archive without
    a family, or listed in two families, fails here rather than silently shrinking or
    double counting an ablation.
    """
    columns = {name: [column_for(f) for f in feats] for name, feats in FAMILIES.items()}
    flat = [c for cols in columns.values() for c in cols]
    duplicated = sorted({c for c in flat if flat.count(c) > 1})
    missing = sorted(set(A.OUTFIELD_CORE) - set(flat))
    unknown = sorted(set(flat) - set(A.OUTFIELD_CORE))
    if duplicated or missing or unknown:
        raise ValueError(
            "feature families do not partition archive.OUTFIELD_CORE: "
            f"duplicated {duplicated}, missing {missing}, unknown {unknown}"
        )
    return columns


FAMILY_COLUMNS = audit_families()


def ablations() -> list[tuple[str, str, str | None, list[str]]]:
    """Every feature set to run, as (name, kind, family, columns), in a fixed order.

    Columns keep the order of ``archive.OUTFIELD_CORE`` in every set, so a cell differs
    from the reference row in which columns it holds and in nothing else.
    """
    full = list(A.OUTFIELD_CORE)
    out: list[tuple[str, str, str | None, list[str]]] = [("full", "full", None, full)]
    for family, cols in FAMILY_COLUMNS.items():
        kept = [c for c in full if c not in cols]
        out.append((f"without {family}", "leave_one_out", family, kept))
    for family, cols in FAMILY_COLUMNS.items():
        kept = [c for c in full if c in cols]
        out.append((f"{family} only", "keep_one_only", family, kept))
    return out


# --------------------------------------------------------------------------------------
# One cell: a feature set in a scope
# --------------------------------------------------------------------------------------


def load_scopes() -> dict[str, pd.DataFrame]:
    """The calibration's four scopes: the pool on global z-scores, each group on its own."""
    z = pd.read_parquet(config.DATA_PROCESSED / "archive_z_global.parquet")
    zg = pd.read_parquet(config.DATA_PROCESSED / "archive_z_bygroup.parquet")
    scopes = {"All outfield": z}
    for grp in config.OUTFIELD_GROUPS:
        scopes[grp] = zg[zg["position_group"] == grp].reset_index(drop=True)
    return scopes


def _tag(scope: str, name: str) -> str:
    """The seed tag for one cell.

    The full set reuses the tag ``archive_validation`` uses, so the reference row is the
    KMeans fit that module reports and the membership file records, not a re-seeded twin
    of it. Every ablation gets a tag of its own, so no two cells share a seed.
    """
    return f"archive|{scope}" if name == "full" else f"ablation|{scope}|{name}"


def measure_cell(x: np.ndarray, tag: str) -> tuple[dict, np.ndarray]:
    """One feature set in one scope: the declared rule, the null calibration, the K fit.

    Runs in a worker process. ``cluster.internal_curves`` and ``cluster.apply_k_rule`` are
    the functions every other module applies, and ``structure.kmeans_profile``,
    ``structure.simulate`` and ``structure.null_calibration`` are composed exactly as
    ``structure.main`` composes them, so an ablation meets the headline result on
    identical terms. The K partition is the rule's own k=K fit: ``cluster._kmeans`` with
    the same tag draws the same seed and returns the same labels.
    """
    with threadpool_limits(limits=1):
        curves = C.internal_curves(x, tag)
        rule = C.apply_k_rule(curves)
        labels = C._kmeans(x, K, tag).labels_
        observed = S.kmeans_profile(x, config.RANDOM_STATE)
        calibration = S.null_calibration(x, tag, observed, S.simulate(x, NULL), NULL)

    record = {
        "n_features": int(x.shape[1]),
        "chosen_k": int(rule["chosen_k"]),
        "rule_branch": rule["rule_branch"],
        "favoured_by_criterion": rule["favoured_by_criterion"],
        "silhouette_by_k": {str(k): v for k, v in curves["silhouette"].items()},
        "null_at_k": calibration["by_k"][str(K)],
        "null_observed_best_k": int(calibration["observed_best_k"]),
        "null_separation_ratio": calibration["separation_ratio"],
        "null_by_k": calibration["by_k"],
        "smaller_cluster_share": round(float(np.bincount(labels).min() / labels.shape[0]), 4),
    }
    return record, labels


def run_cells(scopes: dict[str, pd.DataFrame]) -> dict[tuple[str, str], tuple[dict, np.ndarray]]:
    """Every feature set in every scope, in parallel, gathered in a fixed order."""
    jobs = [(name, scope, cols) for name, _, _, cols in ablations() for scope in scopes]
    stream = Parallel(n_jobs=N_JOBS, return_as="generator")(
        delayed(measure_cell)(scopes[scope][cols].to_numpy(float), _tag(scope, name))
        for name, scope, cols in jobs
    )
    cells = {}
    for (name, scope, _), (record, labels) in zip(jobs, stream, strict=True):
        cells[(scope, name)] = (record, labels)
        at_k = record["null_at_k"]
        print(
            f"  {scope:12s} {name:18s} d={record['n_features']:>2}  k={record['chosen_k']}  "
            f"silhouette {at_k['observed']:.4f}  null {at_k['null_mean']:.4f}  "
            f"z {at_k['z_against_null']}",
            flush=True,
        )
    return cells


# --------------------------------------------------------------------------------------
# Reading the cells
# --------------------------------------------------------------------------------------


def _z(record: dict) -> float | None:
    return record["null_at_k"]["z_against_null"]


def _sortable(z: float | None) -> float:
    return -np.inf if z is None else z


def survives(record: dict) -> bool:
    """The rule still returns K and the observed silhouette at K clears every simulation."""
    return record["chosen_k"] == K and bool(record["null_at_k"]["exceeds_every_simulation"])


def carries_alone(record: dict) -> bool:
    """The rule returns K and the K silhouette sits above the null by the declared margin."""
    z = _z(record)
    return record["chosen_k"] == K and z is not None and z > CARRIES_ALONE_MIN_Z


def _brief(record: dict) -> dict:
    return {
        "z": _z(record),
        "chosen_k": record["chosen_k"],
        "exceeds_every_simulation": record["null_at_k"]["exceeds_every_simulation"],
        "ari_vs_full_at_k": record["ari_vs_full_at_k"],
    }


def summarise_scope(entry: dict) -> dict:
    """One scope read off: survival, the weakest and least similar removals, what carries alone.

    The index against the full partition is listed for every family beside the carrying
    lists, because a family can split a scope into two modes on its own that are not the
    paper's two modes, and the index is what tells those apart.
    """
    loo = entry["leave_one_out"]
    koo = entry["keep_one_only"]
    weakest = min(loo, key=lambda f: (_sortable(_z(loo[f])), f))
    least_similar = min(loo, key=lambda f: (loo[f]["ari_vs_full_at_k"], f))
    strongest = max(koo, key=lambda f: (_sortable(_z(koo[f])), f))
    return {
        "full_z": _z(entry["full"]),
        "full_chosen_k": entry["full"]["chosen_k"],
        "k_survives_every_leave_one_out": all(survives(r) for r in loo.values()),
        "leave_one_out_survives": {f: survives(r) for f, r in loo.items()},
        "leave_one_out_ari_vs_full": {f: r["ari_vs_full_at_k"] for f, r in loo.items()},
        "weakest_leave_one_out": {"family": weakest, **_brief(loo[weakest])},
        "least_similar_leave_one_out": {"family": least_similar, **_brief(loo[least_similar])},
        "families_carrying_alone": [f for f in koo if carries_alone(koo[f])],
        "families_not_carrying_alone": [f for f in koo if not carries_alone(koo[f])],
        "keep_one_only_ari_vs_full": {f: r["ari_vs_full_at_k"] for f, r in koo.items()},
        "strongest_single_family": {"family": strongest, **_brief(koo[strongest])},
    }


def assemble_scope(scope: str, frame: pd.DataFrame, cells: dict) -> dict:
    """Gather one scope's cells and score every ablation's K partition against the full one."""
    full, full_labels = cells[(scope, "full")]
    entry: dict = {"n": int(len(frame)), "full": full, "leave_one_out": {}, "keep_one_only": {}}
    for name, kind, family, _ in ablations():
        if kind == "full":
            continue
        record, labels = cells[(scope, name)]
        record["ari_vs_full_at_k"] = round(float(adjusted_rand_score(full_labels, labels)), 4)
        entry[kind][family] = record
    entry["summary"] = summarise_scope(entry)
    return entry


def _names(items: list[str]) -> str:
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _verdict(summary: dict) -> str:
    weakest = summary["weakest_leave_one_out_any_scope"]
    if summary["k_survives_every_leave_one_out_in_every_scope"]:
        head = (
            f"The two modes survive the removal of every family in every scope. The weakest "
            f"case is removing {weakest['family']} from {weakest['scope']}, where the rule "
            f"still returns k={K} and the k={K} silhouette sits {weakest['z']} null standard "
            f"deviations above the clusterless mean, with an adjusted Rand index of "
            f"{weakest['ari_vs_full_at_k']} against the full-set partition."
        )
    else:
        failures = [f"{f['family']} in {f['scope']}" for f in summary["leave_one_out_failures"]]
        head = (
            f"The two modes do not survive every ablation: removing {_names(failures)} breaks "
            f"the k={K} result, so the claim depends on that block of features."
        )
    everywhere = summary["families_carrying_alone_in_every_scope"]
    nowhere = summary["families_carrying_alone_in_no_scope"]
    alone = (
        f" Alone, {_names(everywhere)} carry the structure in every scope"
        if everywhere
        else " Alone, no family carries the structure in every scope"
    )
    never = (
        f" and {_names(nowhere)} carry it in none"
        if nowhere
        else " and every family carries it in at least one scope"
    )
    return f"{head}{alone}{never}, at the z above {CARRIES_ALONE_MIN_Z:g} threshold."


def summarise(results: dict) -> dict:
    scopes = results["scopes"]
    families = list(FAMILY_COLUMNS)
    per_scope = {name: entry["summary"] for name, entry in scopes.items()}

    weakest_scope = min(
        per_scope, key=lambda s: (_sortable(per_scope[s]["weakest_leave_one_out"]["z"]), s)
    )
    least_scope = min(
        per_scope,
        key=lambda s: (per_scope[s]["least_similar_leave_one_out"]["ari_vs_full_at_k"], s),
    )
    failures = [
        {"scope": s, "family": f}
        for s, v in per_scope.items()
        for f, ok in v["leave_one_out_survives"].items()
        if not ok
    ]
    summary = {
        "k_survives_every_leave_one_out_in_every_scope": not failures,
        "leave_one_out_failures": failures,
        "weakest_leave_one_out_any_scope": {
            "scope": weakest_scope,
            **per_scope[weakest_scope]["weakest_leave_one_out"],
        },
        "least_similar_leave_one_out_any_scope": {
            "scope": least_scope,
            **per_scope[least_scope]["least_similar_leave_one_out"],
        },
        "families_carrying_alone_by_scope": {
            s: v["families_carrying_alone"] for s, v in per_scope.items()
        },
        "families_carrying_alone_in_every_scope": [
            f
            for f in families
            if all(f in v["families_carrying_alone"] for v in per_scope.values())
        ],
        "families_carrying_alone_in_no_scope": [
            f
            for f in families
            if not any(f in v["families_carrying_alone"] for v in per_scope.values())
        ],
    }
    summary["verdict"] = _verdict(summary)
    return summary


# --------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------


def figure_ablation(results: dict) -> None:
    """One panel per scope: the z of the K silhouette with each family removed and alone.

    Leave-one-out and keep-one-only rows carry different marker shapes as well as
    different colours, the full set is a diamond in ink, and its z is the dashed reference
    every other row is read against. The rule's choice is text labelled only where it
    departs from K, so a row that kept its separation but lost the rule is visible.
    """
    scopes = list(results["scopes"])
    families = list(FAMILY_COLUMNS)
    rows = ["full set"] + [f"without {f}" for f in families] + [f"{f} only" for f in families]
    y = np.arange(len(rows))[::-1]
    kinds = (
        ("leave_one_out", 1, "o", P.CATEGORICAL[0], "leave one family out"),
        ("keep_one_only", 1 + len(families), "^", P.CATEGORICAL[1], "keep one family only"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(P.WIDTH_FULL, 5.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, (ax, scope) in enumerate(zip(axes, scopes, strict=True)):
        entry = results["scopes"][scope]
        full_z = _z(entry["full"])
        ax.axvline(0.0, color=P.INK_MUTED, linewidth=0.8, zorder=1, label="null mean")
        ax.axvline(
            full_z,
            color=P.INK_SECONDARY,
            linewidth=1.2,
            linestyle=(0, (4, 3)),
            zorder=1,
            label="_nolegend_",
        )
        ax.scatter(
            [full_z], [y[0]], marker="D", s=26, color=P.INK_PRIMARY, zorder=5, label="full set"
        )
        for kind, offset, marker, colour, text in kinds:
            records = [entry[kind][f] for f in families]
            zs = np.array([np.nan if _z(r) is None else _z(r) for r in records])
            ys = y[offset : offset + len(families)]
            ax.scatter(zs, ys, marker=marker, s=26, color=colour, zorder=5, label=text)
            P.annotate_points(
                ax, zs, ys, [f"k={r['chosen_k']}" if r["chosen_k"] != K else None for r in records]
            )
        xlabel = f"z of the k={K} silhouette against the Gaussian null" if i >= 2 else ""
        P.style_axis(ax, xlabel, "", scope)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(rows)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Two-mode separation with each feature family removed, or kept alone",
        fontsize=P.BASE_FONT_PT,
        fontweight="bold",
    )
    P.save_figure(fig, "ablation")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def _print_summary(results: dict) -> None:
    print()
    for scope, entry in results["scopes"].items():
        s = entry["summary"]
        weakest = s["weakest_leave_one_out"]
        least = s["least_similar_leave_one_out"]
        print(f"{scope:14s} n={entry['n']:>5,}  full set k={s['full_chosen_k']} z={s['full_z']}")
        print(
            f"  leave one out: k={K} survives all {s['k_survives_every_leave_one_out']}; "
            f"weakest without {weakest['family']} z={weakest['z']} k={weakest['chosen_k']} "
            f"ARI vs full {weakest['ari_vs_full_at_k']}; least similar without "
            f"{least['family']} ARI {least['ari_vs_full_at_k']}"
        )
        print(
            f"  keep one only: carries alone {_names(s['families_carrying_alone'])}; "
            f"does not {_names(s['families_not_carrying_alone'])}"
        )
    print()
    print("verdict:", results["summary"]["verdict"])


def main() -> None:
    P.use_style()
    scopes = load_scopes()
    print(
        f"ablation: {len(ablations())} feature sets in {len(scopes)} scopes, "
        f"{N_SIMULATIONS} simulations per cell",
        flush=True,
    )
    cells = run_cells(scopes)

    results = {
        "question": "Does the two-mode result depend on any one block of features?",
        "n_features_full": len(A.OUTFIELD_CORE),
        "k": K,
        "null": NULL,
        "n_simulations": N_SIMULATIONS,
        "execution": (
            "Each cell is one feature set in one scope, run as an independent single "
            "threaded process. Seeds derive from config.RANDOM_STATE and the cell's tag, "
            "never from the schedule, so the cells reproduce in any order."
        ),
        "thresholds": {
            "survives": (
                f"the rule returns k={K} and the observed silhouette at k={K} exceeds every "
                "simulation"
            ),
            "carries_alone_min_z": CARRIES_ALONE_MIN_Z,
            "carries_alone": (
                f"the rule returns k={K} and the z of the k={K} silhouette against the null "
                "exceeds carries_alone_min_z"
            ),
        },
        "families": {
            name: {"features": FAMILIES[name], "columns": FAMILY_COLUMNS[name]} for name in FAMILIES
        },
        "scopes": {scope: assemble_scope(scope, frame, cells) for scope, frame in scopes.items()},
    }
    results["summary"] = summarise(results)

    figure_ablation(results)
    with open(config.METRICS / "ablation.json", "w") as fh:
        json.dump(results, fh, indent=2)

    _print_summary(results)
    print("ablation complete")


if __name__ == "__main__":
    main()
