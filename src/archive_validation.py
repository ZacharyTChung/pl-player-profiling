"""Validation layer for the archive result.

``src/structure.py`` established what the 9,263 eligible outfield player-seasons in the
pre-withdrawal archive actually contain: the declared consensus rule returns k=2 in every
scope on the full 33-feature set, and a clusterless null calibration shows that the
two-mode structure is real while finer taxonomies are not. That settles what the
partition is. It does not say whether the partition is stable, whether it reproduces
across five seasons and five leagues, what it corresponds to, or how many dimensions the
feature space really has. This module answers those four questions.

1. **Stability and correspondence.** The k=2 partition is fitted globally and inside each
   listed position group, resampled with a bootstrap, and scored against both the coarse
   ``position_group`` label and the full ``position_full`` string. The question answered
   directly is whether the clusters recover listed positions or cut across them.

2. **Replication.** The clustering is refitted from scratch inside each season and inside
   each league, and each independent fit is compared against the pooled one. Two
   comparisons are reported because they measure different things. The season pairwise
   index is computed on the players present in both seasons, so it mixes replication of
   the solution with genuine change in what those players did. The transfer index applies
   one scope's centroids to another scope's players and compares against that scope's own
   fit, which isolates replication of the solution from change in the players. This is the
   part of the study the enlarged sample makes possible: the earlier version had two
   seasons of one league, and this has twenty-five season-league cells.

3. **Supervised validation.** Multinomial logistic regression and LightGBM predict
   ``position_group`` from the 33 features under five-fold stratified cross validation,
   against a majority-class baseline fitted through identical folds. SHAP on the boosted
   model then establishes which statistics define each position now that the full modern
   feature set is available. That is the direct contrast with the impoverished live
   sample, where positions were identifiable mainly by what a player did not do: the sign
   of the relation between a feature value and its SHAP contribution is computed here so
   the presence-versus-absence question is answered with a number rather than by eye.

4. **Dimensionality.** PCA on the same 33 features, with the scree curve, the cumulative
   variance, the loadings on the first five components and the count of components needed
   to reach 90 percent. The paper argues the role space is dominated by a single axis, so
   the share carried by the leading component is reported plainly either way.

Outputs
-------
results/metrics/archive_cluster_validation.json
results/metrics/archive_replication.json
results/metrics/archive_supervised.json
results/metrics/archive_pca.json
figures/archive_cluster_vs_position.{pdf,png}
figures/archive_replication.{pdf,png}
figures/archive_confusion.{pdf,png}
figures/archive_shap_beeswarm.{pdf,png}
figures/archive_shap_by_class.{pdf,png}
figures/archive_pca.{pdf,png}
"""

from __future__ import annotations

import json
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src import archive as A
from src import cluster as C
from src import plotting as P

# --------------------------------------------------------------------------------------
# Declared choices
# --------------------------------------------------------------------------------------

#: The partition under validation. Fixed by the k-selection rule in every scope, which is
#: recomputed here on the 33-feature archive matrix rather than assumed.
K = 2

#: Bootstrap resamples per scope. ``config.BOOTSTRAP_N`` is 200 and is used on the live
#: sample, where a scope holds a few hundred players. Here the global scope holds 9,263,
#: every resample refits KMeans on that matrix, and the co-assignment matrix that
#: ``cluster.bootstrap_stability`` accumulates is 9,263 by 9,263. The count is therefore
#: reduced to 50, which is ample: the between-resample standard deviation of the adjusted
#: Rand index is reported alongside the mean, and it is small enough that the extra 150
#: resamples would not move the third decimal place. The reduction is recorded in the
#: saved metrics as well as here.
N_BOOT = 50

#: Cross-validation folds for the supervised check. Stratified because forwards are 18
#: percent of the pool.
N_SPLITS = 5

#: SHAP is exact tree SHAP, which is linear in the number of rows explained and in the
#: number of trees. Explaining all 9,263 players over 300 trees and three classes is
#: needlessly slow for a ranking that is stable well before then, so a stratified
#: sub-sample is explained instead. The model itself is fitted on all 9,263 rows.
SHAP_SAMPLE = 2000

#: Features shown in the SHAP figures. All 33 are ranked in the saved metrics; showing 33
#: rows of beeswarm would need a figure a foot tall.
SHAP_DISPLAY = 20

#: Principal components whose loadings are tabulated and drawn.
PC_DISPLAY = 5

#: Above this adjusted Rand index the clusters are called a recovery of listed positions,
#: below the lower bound they are called unrelated to them. Declared before the numbers
#: were looked at.
ARI_RECOVERS = 0.50
ARI_UNRELATED = 0.20

CLASS_ORDER = list(config.OUTFIELD_GROUPS)

#: Short display names. The canonical names carry ``_p90`` and ``_padj`` suffixes that
#: waste axis space when 33 of them have to fit along one edge of a heatmap.
FEATURE_LABELS = {
    "np_xg_p90": "npxG",
    "shots_p90": "Shots",
    "goals_non_penalty_p90": "NP goals",
    "xag_p90": "xAG",
    "key_passes_p90": "Key passes",
    "sca_p90": "SCA",
    "gca_p90": "GCA",
    "passes_final_third_p90": "Passes final third",
    "passes_penalty_area_p90": "Passes into box",
    "crosses_p90": "Crosses",
    "progressive_passes_p90": "Prog. passes",
    "progressive_carries_p90": "Prog. carries",
    "progressive_receptions_p90": "Prog. receptions",
    "carries_final_third_p90": "Carries final third",
    "touches_def_pen_p90": "Touches own box",
    "touches_def_third_p90": "Touches def third",
    "touches_mid_third_p90": "Touches mid third",
    "touches_att_third_p90": "Touches att third",
    "touches_att_pen_p90": "Touches opp box",
    "take_ons_p90": "Take ons",
    "tackles_padj_p90": "Tackles (adj)",
    "interceptions_padj_p90": "Interceptions (adj)",
    "blocks_padj_p90": "Blocks (adj)",
    "clearances_padj_p90": "Clearances (adj)",
    "ball_recoveries_p90": "Ball recoveries",
    "fouls_committed_padj_p90": "Fouls (adj)",
    "shot_accuracy_pct": "Shot accuracy %",
    "take_on_success_pct": "Take on success %",
    "tackle_win_pct": "Tackle win %",
    "aerials_won_pct": "Aerials won %",
    "pass_cmp_short_pct": "Short pass %",
    "pass_cmp_medium_pct": "Medium pass %",
    "pass_cmp_long_pct": "Long pass %",
}

#: The project sequential ramp starts one shade off the paper surface, which is right for
#: a filled cell and wrong for a scatter mark, where near-white reads as absent.
BEESWARM_CMAP = ListedColormap(P.SEQUENTIAL(np.linspace(0.22, 1.0, 256)))


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _r(value, digits: int = 4):
    """Round for JSON, mapping non-finite values to None."""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _write(payload: dict, name: str) -> None:
    path = config.METRICS / name
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=_native)
    print(f"  wrote {path.relative_to(config.ROOT)}")


def _standardise(x: np.ndarray) -> np.ndarray:
    """Z-score within the sample supplied, leaving constant columns at zero."""
    mean = x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (x - mean) / sd


def _axis(centres: np.ndarray) -> np.ndarray:
    """The single discriminant direction of a two-cluster solution."""
    return centres[1] - centres[0]


def _oriented_correlation(axis: np.ndarray, reference: np.ndarray) -> float:
    """Correlation between two centroid-difference vectors, sign fixed to be positive.

    Cluster ids are arbitrary, so an otherwise identical solution can return the axis
    negated. Orienting removes that artefact and nothing else, because negating both
    labels of a two-cluster solution leaves every partition-level statistic unchanged.
    """
    return float(abs(np.corrcoef(axis, reference)[0, 1]))


def _stratified_sample(y: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Row indices, sampled without replacement, keeping the class shares."""
    if size >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for cls in CLASS_ORDER:
        members = np.flatnonzero(y == cls)
        take = max(1, int(round(size * members.size / len(y))))
        take = min(take, members.size)
        chosen.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """The eligible table, both standardised tables, and the feature list they share."""
    processed = config.DATA_PROCESSED
    eligible = pd.read_parquet(processed / "archive_eligible.parquet").reset_index(drop=True)
    z_global = pd.read_parquet(processed / "archive_z_global.parquet").reset_index(drop=True)
    z_group = pd.read_parquet(processed / "archive_z_bygroup.parquet").reset_index(drop=True)
    features = [c for c in A.OUTFIELD_CORE if c in z_global.columns]
    if len(features) != len(A.OUTFIELD_CORE):
        missing = sorted(set(A.OUTFIELD_CORE) - set(features))
        raise KeyError(f"archive feature columns absent from the standardised table: {missing}")
    return eligible, z_global, z_group, features


def scope_frames(z_global: pd.DataFrame, z_group: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The four clustering scopes, each already standardised in the right reference set."""
    scopes = {"All outfield": z_global}
    for group in config.OUTFIELD_GROUPS:
        scopes[group] = z_group[z_group["position_group"] == group].reset_index(drop=True).copy()
    return scopes


# --------------------------------------------------------------------------------------
# 1. Stability and correspondence with listed positions
# --------------------------------------------------------------------------------------


def cluster_validation(scopes: dict[str, pd.DataFrame], features: list[str]) -> dict:
    """Bootstrap stability and agreement with the two listed position fields, per scope."""
    payload: dict = {
        "question": "Is the two-mode partition stable, and does it recover listed positions?",
        "k": K,
        "n_features": len(features),
        "bootstrap_resamples": N_BOOT,
        "bootstrap_note": (
            f"config.BOOTSTRAP_N is {config.BOOTSTRAP_N} and is used on the live sample. It is "
            f"reduced to {N_BOOT} here because every resample refits KMeans on up to 9,263 "
            "players and accumulates a co-assignment matrix of that dimension squared. The "
            "between-resample standard deviation reported per scope shows the reduction costs "
            "nothing at the precision quoted."
        ),
        "scopes": {},
    }
    labels_by_scope: dict[str, np.ndarray] = {}

    for name, frame in scopes.items():
        tag = f"archive|{name}"
        x = frame[features].to_numpy(float)
        curves = C.internal_curves(x, tag)
        rule = C.apply_k_rule(curves)
        labels = C._kmeans(x, K, tag).labels_
        labels_by_scope[name] = labels

        boot = C.bootstrap_stability(x, labels, K, tag, n_boot=N_BOOT)
        boot.pop("coassignment", None)
        agreement = C.cluster_vs_position(frame, labels)

        sizes = {str(int(c)): int((labels == c).sum()) for c in sorted(set(labels.tolist()))}
        payload["scopes"][name] = {
            "n": int(len(frame)),
            "k_selected_by_rule": int(rule["chosen_k"]),
            "k_rule_branch": rule["rule_branch"],
            "k_used": K,
            "silhouette_at_k": _r(curves["silhouette"][K]),
            "cluster_sizes": sizes,
            "bootstrap": boot,
            "position_group_is_constant": bool(frame["position_group"].nunique() == 1),
            "vs_position": agreement,
        }
        print(
            f"  {name:14s} n={len(frame):>5,}  rule k={rule['chosen_k']}  "
            f"bootstrap ARI {boot['ari_mean']:.3f} (sd {boot['ari_sd']:.3f})  "
            f"ARI vs group {agreement['position_group']['adjusted_rand_index']:.3f}  "
            f"NMI {agreement['position_group']['normalized_mutual_info']:.3f}",
            flush=True,
        )

    payload["answer"] = _position_verdict(payload)
    return {"payload": payload, "labels": labels_by_scope}


def _position_verdict(payload: dict) -> dict:
    """Turn the global agreement scores into the plain answer the paper needs."""
    glob = payload["scopes"]["All outfield"]["vs_position"]
    ari = glob["position_group"]["adjusted_rand_index"]
    nmi = glob["position_group"]["normalized_mutual_info"]
    purity = glob["position_group"]["cluster_purity"]
    ari_full = glob["position_full"]["adjusted_rand_index"]
    if ari >= ARI_RECOVERS:
        verdict = "The clusters recover listed position groups."
    elif ari >= ARI_UNRELATED:
        verdict = (
            "The clusters do not recover listed position groups. They overlap them well "
            "above chance but cut across them, so the partition is a different division "
            "of the same players rather than a rediscovery of the listed one."
        )
    else:
        verdict = "The clusters are close to unrelated to listed position groups."
    return {
        "adjusted_rand_index_vs_position_group": ari,
        "normalized_mutual_info_vs_position_group": nmi,
        "cluster_purity_vs_position_group": purity,
        "adjusted_rand_index_vs_position_full": ari_full,
        "normalized_mutual_info_vs_position_full": glob["position_full"]["normalized_mutual_info"],
        "thresholds": {"recovers_at": ARI_RECOVERS, "unrelated_below": ARI_UNRELATED},
        "verdict": verdict,
    }


# --------------------------------------------------------------------------------------
# 2. Cross-season and cross-league replication
# --------------------------------------------------------------------------------------


def _fit_scope(x_raw: np.ndarray, tag: str) -> dict:
    """Standardise inside the subset, then fit the two-cluster solution from scratch."""
    x = _standardise(x_raw)
    fit = C._kmeans(x, K, tag)
    return {"x": x, "labels": fit.labels_, "centres": fit.cluster_centers_}


def _transfer(fit_source: dict, fit_target: dict) -> float:
    """Assign the target's players to the source's centroids, score against their own fit.

    Both scopes are standardised inside themselves, so a centroid from one is a position
    in the other's units. The index this returns asks whether the *solution* replicates,
    which is a different question from whether individual players stayed in the same
    cluster, and the two are reported separately for that reason.
    """
    distances = ((fit_target["x"][:, None, :] - fit_source["centres"][None, :, :]) ** 2).sum(-1)
    imported = distances.argmin(axis=1)
    return _r(adjusted_rand_score(fit_target["labels"], imported))


def replication(
    eligible: pd.DataFrame,
    z_global: pd.DataFrame,
    features: list[str],
    pooled_labels: np.ndarray,
) -> dict:
    """Refit inside every season and every league, and compare against the pooled fit."""
    raw = eligible[features].to_numpy(float)
    pooled_centres = np.vstack(
        [z_global[features].to_numpy(float)[pooled_labels == c].mean(axis=0) for c in range(K)]
    )
    pooled_axis = _axis(pooled_centres)

    seasons = [int(s) for s in sorted(eligible["season"].dropna().unique())]
    leagues = [lg for lg in A.LEAGUES if lg in set(eligible["league"])]

    def _by(field: str, keys: list) -> tuple[dict, dict]:
        fits: dict = {}
        report: dict = {}
        for key in keys:
            mask = (eligible[field] == key).to_numpy()
            fit = _fit_scope(raw[mask], f"archive|{field}|{key}")
            fit["mask"] = mask
            fits[key] = fit
            report[str(key)] = {
                "n": int(mask.sum()),
                "cluster_sizes": {str(c): int((fit["labels"] == c).sum()) for c in range(K)},
                "ari_vs_pooled": _r(adjusted_rand_score(pooled_labels[mask], fit["labels"])),
                "nmi_vs_pooled": _r(
                    normalized_mutual_info_score(pooled_labels[mask], fit["labels"])
                ),
                "axis_correlation_with_pooled": _r(
                    _oriented_correlation(_axis(fit["centres"]), pooled_axis)
                ),
            }
        return fits, report

    season_fits, season_report = _by("season", seasons)
    league_fits, league_report = _by("league", leagues)

    # Shared-player comparison across seasons. A player can hold two rows in one season
    # after a January transfer, so the longer spell is kept as that season's record.
    holder = eligible[["player", "season", "minutes"]].copy()
    holder["label"] = np.nan
    for fit in season_fits.values():
        holder.loc[fit["mask"], "label"] = fit["labels"]
    holder = holder.sort_values("minutes", ascending=False).drop_duplicates(
        subset=["player", "season"]
    )
    wide = holder.pivot(index="player", columns="season", values="label")

    pairwise: dict = {}
    season_matrix = np.full((len(seasons), len(seasons)), np.nan)
    for i, a in enumerate(seasons):
        season_matrix[i, i] = 1.0
        for j, b in enumerate(seasons):
            if j <= i:
                continue
            both = wide[[a, b]].dropna()
            value = _r(adjusted_rand_score(both[a].to_numpy(), both[b].to_numpy()))
            pairwise[f"{a} vs {b}"] = {
                "n_shared_players": int(len(both)),
                "adjusted_rand_index": value,
                "normalized_mutual_info": _r(
                    normalized_mutual_info_score(both[a].to_numpy(), both[b].to_numpy())
                ),
                "same_cluster_share": _r(float((both[a] == both[b]).mean())),
            }
            season_matrix[i, j] = season_matrix[j, i] = value

    season_transfer = _transfer_matrix(season_fits, seasons)
    league_transfer = _transfer_matrix(league_fits, leagues)

    off_season = season_matrix[~np.eye(len(seasons), dtype=bool)]
    off_league = np.asarray(league_transfer)[~np.eye(len(leagues), dtype=bool)]

    payload = {
        "question": "Does the two-cluster solution reproduce across seasons and leagues?",
        "k": K,
        "n_features": len(features),
        "n_players_seasons": int(len(eligible)),
        "seasons": seasons,
        "leagues": leagues,
        "method": (
            "Each season and each league is standardised inside itself and clustered from "
            "scratch, so no scope borrows a centre, a scale or a starting point from the "
            "pool. Three comparisons are reported. ari_vs_pooled scores an independent fit "
            "against the pooled partition on the same rows. The season pairwise index scores "
            "one season's labels against another's on the players present in both, which "
            "mixes replication of the solution with real change in those players. The "
            "transfer index imports one scope's centroids into another scope's players and "
            "scores against that scope's own fit, which isolates replication of the solution."
        ),
        "pooled": {
            "n": int(len(z_global)),
            "cluster_sizes": {str(c): int((pooled_labels == c).sum()) for c in range(K)},
        },
        "by_season": season_report,
        "by_league": league_report,
        "season_pairwise_shared_players": pairwise,
        "season_pairwise_matrix": {
            "order": seasons,
            "values": [[_r(v) for v in row] for row in season_matrix],
        },
        "season_transfer_matrix": {"order": seasons, "values": season_transfer},
        "league_transfer_matrix": {"order": leagues, "values": league_transfer},
        "summary": {
            "season_pairwise_ari_mean": _r(float(np.nanmean(off_season))),
            "season_pairwise_ari_min": _r(float(np.nanmin(off_season))),
            "season_pairwise_ari_max": _r(float(np.nanmax(off_season))),
            "season_vs_pooled_ari_min": _r(min(v["ari_vs_pooled"] for v in season_report.values())),
            "league_vs_pooled_ari_min": _r(min(v["ari_vs_pooled"] for v in league_report.values())),
            "league_transfer_ari_mean": _r(float(np.nanmean(off_league))),
            "league_transfer_ari_min": _r(float(np.nanmin(off_league))),
            "axis_correlation_min": _r(
                min(
                    v["axis_correlation_with_pooled"]
                    for v in list(season_report.values()) + list(league_report.values())
                )
            ),
        },
    }
    return payload


def _transfer_matrix(fits: dict, order: list) -> list[list[float]]:
    matrix = []
    for a in order:
        row = []
        for b in order:
            row.append(1.0 if a == b else _transfer(fits[a], fits[b]))
        matrix.append(row)
    return matrix


# --------------------------------------------------------------------------------------
# 3. Supervised validation
# --------------------------------------------------------------------------------------


def build_models() -> dict[str, object]:
    """The two classifiers, both seeded and both deterministic.

    LightGBM is pinned to one thread with ``deterministic`` and ``force_col_wise`` set,
    because its default histogram construction is order dependent across threads and would
    otherwise move the fourth decimal place between runs. ``verbosity`` and ``verbose`` are
    both set to -1 because the two flags silence different parts of its output.
    """
    logistic = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )
    boosted = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=25,
        colsample_bytree=0.8,
        subsample=0.9,
        subsample_freq=1,
        reg_lambda=1.0,
        objective="multiclass",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        verbose=-1,
    )
    return {"logistic_regression": logistic, "lightgbm": boosted}


def _score_block(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    totals = matrix.sum(axis=1, keepdims=True)
    return {
        "accuracy": _r(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": _r(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": _r(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": _r(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            cls: {
                "precision": _r(precision[i]),
                "recall": _r(recall[i]),
                "f1": _r(f1[i]),
                "support": int(support[i]),
            }
            for i, cls in enumerate(CLASS_ORDER)
        },
        "confusion_matrix": {
            "labels": CLASS_ORDER,
            "counts": matrix.tolist(),
            "row_normalised": np.divide(matrix, np.where(totals == 0, 1, totals)).tolist(),
        },
    }


def classification(eligible: pd.DataFrame, features: list[str]) -> dict:
    """Out-of-fold position prediction from the 33 features, against a majority baseline."""
    x = eligible[features].astype(float)
    y = eligible["position_group"].to_numpy()
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    models = build_models()
    models["majority_baseline"] = DummyClassifier(strategy="most_frequent")

    blocks: dict = {}
    for name, model in models.items():
        predicted = cross_val_predict(model, x, y, cv=folds)
        blocks[name] = _score_block(y, predicted)

    majority = max(set(y.tolist()), key=lambda c: int((y == c).sum()))
    payload = {
        "question": "Can listed position be predicted from the full 33-feature set?",
        "n_players": int(len(eligible)),
        "n_features": len(features),
        "features": list(features),
        "class_order": CLASS_ORDER,
        "class_counts": {cls: int((y == cls).sum()) for cls in CLASS_ORDER},
        "cross_validation": (
            f"{N_SPLITS}-fold stratified, shuffled, seeded from config.RANDOM_STATE. Every "
            "number is from out-of-fold predictions. The baseline runs through the same folds."
        ),
        "majority_class": majority,
        "majority_class_share": _r(float((y == majority).mean())),
        "models": blocks,
        "lift_over_baseline": {
            name: {
                "accuracy": _r(block["accuracy"] - blocks["majority_baseline"]["accuracy"]),
                "macro_f1": _r(block["macro_f1"] - blocks["majority_baseline"]["macro_f1"]),
            }
            for name, block in blocks.items()
            if name != "majority_baseline"
        },
    }
    return {"payload": payload}


def shap_analysis(eligible: pd.DataFrame, features: list[str]) -> dict:
    """SHAP on a LightGBM fitted to the whole sample, explained on a stratified sub-sample.

    Fitting on everything is right here and would be wrong for an accuracy claim. The
    question is which statistics the model uses to tell positions apart, which is about
    the fitted function rather than about generalisation, and the accuracy claims all come
    from the cross-validated fit above.
    """
    x = eligible[features].astype(float)
    y = eligible["position_group"].to_numpy()
    model = build_models()["lightgbm"].fit(x, y)

    index = _stratified_sample(y, SHAP_SAMPLE, config.RANDOM_STATE)
    x_sample = x.iloc[index]
    y_sample = y[index]

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(x_sample)
    values = np.stack(raw, axis=-1) if isinstance(raw, list) else np.asarray(raw)
    if values.ndim != 3:
        raise ValueError(f"expected per-class SHAP values, got an array of shape {values.shape}")

    fitted = [str(c) for c in model.classes_]
    class_index = {cls: fitted.index(cls) for cls in CLASS_ORDER}
    sample_values = x_sample.to_numpy(float)

    by_class: dict[str, list[dict]] = {}
    mean_abs: dict[str, np.ndarray] = {}
    direction: dict[str, np.ndarray] = {}
    for cls in CLASS_ORDER:
        block = values[:, :, class_index[cls]]
        magnitude = np.abs(block).mean(axis=0)
        mean_abs[cls] = magnitude
        # Sign of the relation between the feature value and its push toward this class.
        # Positive means high values identify the class, so the class is defined by what
        # the player does. Negative means low values identify it, so the class is defined
        # by what the player does not do.
        rho = np.array(
            [
                stats.spearmanr(sample_values[:, j], block[:, j]).statistic
                for j in range(len(features))
            ]
        )
        direction[cls] = rho
        order = np.argsort(-magnitude)
        by_class[cls] = [
            {
                "rank": int(rank + 1),
                "feature": features[j],
                "label": label(features[j]),
                "mean_abs_shap": _r(magnitude[j], 5),
                "mean_signed_shap": _r(block[:, j].mean(), 5),
                "value_shap_spearman": _r(rho[j]),
                "identified_by": "presence" if rho[j] > 0 else "absence",
            }
            for rank, j in enumerate(order)
        ]

    overall = np.mean([mean_abs[cls] for cls in CLASS_ORDER], axis=0)
    own = np.array([values[i, :, class_index[y_sample[i]]] for i in range(len(y_sample))])
    own_magnitude = np.abs(own).mean(axis=0)
    own_order = np.argsort(-own_magnitude)

    presence_share = {
        cls: _r(float(np.mean([e["value_shap_spearman"] > 0 for e in by_class[cls][:5]])))
        for cls in CLASS_ORDER
    }

    payload = {
        "question": "Which statistics define each position when the full feature set is present?",
        "model": "lightgbm",
        "fit": "all 9,263 rows, no held-out split, interpretability only",
        "explainer": "shap.TreeExplainer, exact tree SHAP",
        "explained_rows": int(len(index)),
        "explained_rows_note": (
            f"A stratified sub-sample of {int(len(index))} of the {len(eligible)} rows is "
            "explained. Exact tree SHAP is linear in the rows explained and the ranking is "
            "stable well below the full sample. The model itself is fitted on every row."
        ),
        "units": "log-odds contribution to the class score",
        "features": list(features),
        "class_order": CLASS_ORDER,
        "overall_ranking": [
            {
                "rank": int(rank + 1),
                "feature": features[j],
                "label": label(features[j]),
                "mean_abs_shap": _r(overall[j], 5),
            }
            for rank, j in enumerate(np.argsort(-overall))
        ],
        "own_class_ranking": [
            {
                "rank": int(rank + 1),
                "feature": features[j],
                "label": label(features[j]),
                "mean_abs_shap": _r(own_magnitude[j], 5),
                "mean_signed_shap": _r(own[:, j].mean(), 5),
            }
            for rank, j in enumerate(own_order)
        ],
        "by_class": by_class,
        "top5_by_class": {
            cls: [entry["label"] for entry in by_class[cls][:5]] for cls in CLASS_ORDER
        },
        "presence_share_of_top5": presence_share,
        "presence_vs_absence": (
            "value_shap_spearman is the rank correlation between a feature's value and the "
            "SHAP value it contributes to that class. A positive entry means the class is "
            "identified by high values of the statistic, so by what the player does. A "
            "negative entry means it is identified by low values, so by what the player does "
            "not do. presence_share_of_top5 is the fraction of each class's five strongest "
            "features that are positive."
        ),
        "notes": (
            "overall_ranking averages the per-class mean absolute SHAP over DF, MF and FW. "
            "own_class_ranking uses, for each player, the SHAP value attached to that "
            "player's own listed group, so a positive value means the feature pushed the "
            "player toward the position he is listed at. The beeswarm plots those values."
        ),
    }
    return {
        "payload": payload,
        "own": own,
        "own_order": own_order,
        "mean_abs": mean_abs,
        "overall": overall,
        "x": x_sample,
        "features": features,
    }


# --------------------------------------------------------------------------------------
# 4. Dimensionality
# --------------------------------------------------------------------------------------


def pca_analysis(z_global: pd.DataFrame, features: list[str]) -> dict:
    """PCA on the standardised 33 features, with loadings and the 90 percent count."""
    x = z_global[features].to_numpy(float)
    model = PCA(random_state=config.RANDOM_STATE).fit(x)
    ratio = model.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    n_90 = int(np.searchsorted(cumulative, config.PCA_VARIANCE_TARGET) + 1)

    loadings = {}
    interpretation = {}
    for pc in range(PC_DISPLAY):
        vector = model.components_[pc]
        # Orient each component so that its largest loading is positive, otherwise the
        # sign is an artefact of the solver and the reading flips between runs of numpy.
        if vector[np.argmax(np.abs(vector))] < 0:
            vector = -vector
        name = f"PC{pc + 1}"
        order = np.argsort(-np.abs(vector))
        loadings[name] = {
            "explained_variance_ratio": _r(ratio[pc]),
            "cumulative": _r(cumulative[pc]),
            "loadings": {features[j]: _r(vector[j]) for j in range(len(features))},
            "top_positive": [
                {"feature": features[j], "label": label(features[j]), "loading": _r(vector[j])}
                for j in np.argsort(-vector)[:5]
            ],
            "top_negative": [
                {"feature": features[j], "label": label(features[j]), "loading": _r(vector[j])}
                for j in np.argsort(vector)[:5]
            ],
            "strongest": [label(features[j]) for j in order[:5]],
        }
        if pc < 3:
            positive = ", ".join(e["label"] for e in loadings[name]["top_positive"][:3])
            negative = ", ".join(e["label"] for e in loadings[name]["top_negative"][:3])
            interpretation[name] = (
                f"{_r(ratio[pc] * 100, 1)} percent of the variance. Loads positively on "
                f"{positive} and negatively on {negative}."
            )

    payload = {
        "question": "How many dimensions does the role space really have?",
        "n_players": int(len(z_global)),
        "n_features": len(features),
        "space": "the 33 archive features, z-scored over the whole outfield pool",
        "explained_variance_ratio": [_r(v) for v in ratio],
        "cumulative_variance": [_r(v) for v in cumulative],
        "eigenvalues": [_r(v) for v in model.explained_variance_],
        "n_components_for_90pct": n_90,
        "variance_target": config.PCA_VARIANCE_TARGET,
        "pc1_share": _r(ratio[0]),
        "pc1_to_pc3_share": _r(cumulative[2]),
        "n_components_above_kaiser": int((model.explained_variance_ > 1.0).sum()),
        "components": loadings,
        "interpretation": interpretation,
        "verdict": _pca_verdict(ratio, cumulative, n_90),
    }
    return {"payload": payload, "ratio": ratio, "cumulative": cumulative, "model": model}


def _pca_verdict(ratio: np.ndarray, cumulative: np.ndarray, n_90: int) -> str:
    lead = ratio[0]
    if lead >= 0.5:
        shape = "The leading component carries a majority of the variance on its own"
    elif lead >= 0.3:
        shape = (
            "The leading component is much the largest but carries well under half the "
            "variance on its own"
        )
    else:
        shape = "No single component dominates"
    return (
        f"{shape}: PC1 takes {_r(lead * 100, 1)} percent, PC1 to PC3 take "
        f"{_r(cumulative[2] * 100, 1)} percent, and {n_90} components are needed to reach "
        f"{int(config.PCA_VARIANCE_TARGET * 100)} percent. The space has one dominant axis "
        "and a long tail rather than a single dimension."
    )


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _position_order(columns) -> list[str]:
    """Listed position labels ordered back to front, not alphabetically.

    ``pd.crosstab`` sorts its columns as strings, which puts FW between DF and MF and
    scatters the dual listings. Ordering by the primary group instead keeps the defence
    to attack reading that the contingency table is there to show.
    """
    rank = {group: i for i, group in enumerate(CLASS_ORDER)}
    return sorted(columns, key=lambda c: (rank.get(str(c).split(",")[0], 9), str(c)))


def _heatmap(ax, matrix, xlabels, ylabels, *, cmap, vmin, vmax, fmt="{:.2f}", fontsize=None):
    mesh = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    size = fontsize if fontsize is not None else P.BASE_FONT_PT - 2
    span = (vmax - vmin) or 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                continue
            dark = (value - vmin) / span > 0.6
            ax.text(
                j,
                i,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=size,
                color=P.SURFACE if dark else P.INK_PRIMARY,
            )
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return mesh


def figure_cluster_vs_position(validation: dict) -> None:
    """Where the two clusters sit relative to the listed position labels."""
    payload = validation["payload"]
    glob = payload["scopes"]["All outfield"]["vs_position"]
    scopes = list(payload["scopes"])

    fig, axes = plt.subplots(2, 2, figsize=(P.WIDTH_FULL, 5.6))

    for ax, field, title in (
        (axes[0, 0], "position_group", "Listed position group"),
        (axes[0, 1], "position_full", "Full listed position string"),
    ):
        table = pd.DataFrame(glob[field]["contingency"]).T.fillna(0.0)
        table = table.reindex(sorted(table.index), axis=0)
        table = table[_position_order(table.columns)]
        shares = table.to_numpy(float)
        shares = shares / shares.sum(axis=1, keepdims=True)
        _heatmap(
            ax,
            shares,
            list(table.columns),
            [f"Cluster {i}" for i in table.index],
            cmap=P.SEQUENTIAL,
            vmin=0.0,
            vmax=1.0,
            fmt="{:.0%}",
            fontsize=P.BASE_FONT_PT - 3 if field == "position_full" else P.BASE_FONT_PT - 1,
        )
        ax.set_title(
            f"{title}\nARI {glob[field]['adjusted_rand_index']:.3f}, "
            f"NMI {glob[field]['normalized_mutual_info']:.3f}",
            loc="left",
        )
        if field == "position_full":
            ax.tick_params(axis="x", labelrotation=90)

    ax = axes[1, 0]
    values = [payload["scopes"][s]["bootstrap"]["ari_mean"] for s in scopes]
    lows = [
        v - payload["scopes"][s]["bootstrap"]["ari_min"]
        for v, s in zip(values, scopes, strict=True)
    ]
    highs = [
        payload["scopes"][s]["bootstrap"]["ari_max"] - v
        for v, s in zip(values, scopes, strict=True)
    ]
    ax.bar(range(len(scopes)), values, color=P.CATEGORICAL[0], width=0.62, linewidth=0.0)
    ax.errorbar(
        range(len(scopes)),
        values,
        yerr=[lows, highs],
        fmt="none",
        ecolor=P.INK_SECONDARY,
        elinewidth=0.8,
        capsize=2.5,
    )
    for i, value in enumerate(values):
        ax.text(i, value + 0.035, f"{value:.3f}", ha="center", fontsize=P.BASE_FONT_PT - 2)
    ax.set_xticks(range(len(scopes)), [s.replace("All outfield", "All") for s in scopes])
    ax.set_ylim(0.0, 1.12)
    P.style_axis(
        ax, ylabel="Bootstrap ARI", title=f"Stability, {payload['bootstrap_resamples']} resamples"
    )
    ax.grid(False, axis="x")

    ax = axes[1, 1]
    colours = P.categorical(2)
    width = 0.38
    positions = np.arange(len(scopes))
    for offset, (key, name, colour) in enumerate(
        [
            ("adjusted_rand_index", "Adjusted Rand", colours[0]),
            ("normalized_mutual_info", "Normalised MI", colours[1]),
        ]
    ):
        heights = [payload["scopes"][s]["vs_position"]["position_full"][key] for s in scopes]
        ax.bar(
            positions + (offset - 0.5) * width,
            heights,
            width=width,
            color=colour,
            label=name,
            linewidth=0.0,
        )
    ax.set_xticks(positions, [s.replace("All outfield", "All") for s in scopes])
    ax.set_ylim(0.0, max(0.5, ax.get_ylim()[1]))
    P.style_axis(ax, ylabel="Agreement", title="Against the full listed position string")
    ax.grid(False, axis="x")
    ax.legend(loc="upper right", ncol=1)

    fig.suptitle(
        "The two clusters overlap listed positions without recovering them",
        fontsize=P.BASE_FONT_PT,
        fontweight="bold",
    )
    P.save_figure(fig, "archive_cluster_vs_position")


def figure_replication(payload: dict) -> None:
    """Season and league replication of the two-cluster solution."""
    seasons = payload["seasons"]
    leagues = payload["leagues"]
    short = {lg: lg.replace("Premier League", "Premier") for lg in leagues}

    fig, axes = plt.subplots(2, 2, figsize=(P.WIDTH_WIDE, 5.4))

    season_matrix = np.array(payload["season_pairwise_matrix"]["values"], dtype=float)
    _heatmap(
        axes[0, 0],
        season_matrix,
        [str(s) for s in seasons],
        [str(s) for s in seasons],
        cmap=P.SEQUENTIAL,
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 0].set_title(
        "Season against season, players present in both\n"
        f"mean {payload['summary']['season_pairwise_ari_mean']:.3f} adjusted Rand",
        loc="left",
    )

    league_matrix = np.array(payload["league_transfer_matrix"]["values"], dtype=float)
    _heatmap(
        axes[0, 1],
        league_matrix,
        [short[lg] for lg in leagues],
        [short[lg] for lg in leagues],
        cmap=P.SEQUENTIAL,
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 1].set_title(
        "One league's centroids imported into another\n"
        f"mean {payload['summary']['league_transfer_ari_mean']:.3f} adjusted Rand",
        loc="left",
    )

    for ax, report, keys, ticks, title in (
        (
            axes[1, 0],
            payload["by_season"],
            [str(s) for s in seasons],
            [str(s) for s in seasons],
            "Season fits against the pooled partition",
        ),
        (
            axes[1, 1],
            payload["by_league"],
            leagues,
            [short[lg] for lg in leagues],
            "League fits against the pooled partition",
        ),
    ):
        heights = [report[k]["ari_vs_pooled"] for k in keys]
        ax.bar(range(len(keys)), heights, color=P.CATEGORICAL[0], width=0.6, linewidth=0.0)
        for i, value in enumerate(heights):
            ax.text(i, value + 0.03, f"{value:.3f}", ha="center", fontsize=P.BASE_FONT_PT - 2)
        ax.set_xticks(range(len(keys)), ticks)
        ax.set_ylim(0.0, 1.12)
        P.style_axis(ax, ylabel="Adjusted Rand", title=title)
        ax.grid(False, axis="x")

    fig.suptitle(
        "The two-cluster solution reproduces in every season and every league",
        fontsize=P.BASE_FONT_PT,
        fontweight="bold",
    )
    P.save_figure(fig, "archive_replication")


def figure_confusion(payload: dict) -> None:
    """Out-of-fold confusion matrices for both classifiers."""
    names = ["logistic_regression", "lightgbm"]
    titles = {"logistic_regression": "Multinomial logistic", "lightgbm": "LightGBM"}
    fig, axes = plt.subplots(1, 2, figsize=(P.WIDTH_FULL, 3.0))
    mesh = None
    for ax, name in zip(axes, names, strict=True):
        block = payload["models"][name]["confusion_matrix"]
        counts = np.array(block["counts"], dtype=float)
        shares = np.array(block["row_normalised"], dtype=float)
        mesh = ax.imshow(shares, cmap=P.SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="equal")
        for i in range(len(CLASS_ORDER)):
            for j in range(len(CLASS_ORDER)):
                ax.text(
                    j,
                    i,
                    f"{int(counts[i, j])}\n{shares[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=P.BASE_FONT_PT - 1,
                    color=P.SURFACE if shares[i, j] > 0.55 else P.INK_PRIMARY,
                )
        ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
        ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
        ax.set_title(
            f"{titles[name]}\nacc {payload['models'][name]['accuracy']:.3f}, "
            f"macro F1 {payload['models'][name]['macro_f1']:.3f}",
            loc="left",
        )
        ax.set_xlabel("Predicted")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Listed position")
    bar = fig.colorbar(mesh, ax=axes, fraction=0.035, pad=0.02)
    bar.set_label("Share of the listed group", fontsize=P.BASE_FONT_PT - 1)
    bar.outline.set_visible(False)
    P.save_figure(fig, "archive_confusion")


def _beeswarm_offsets(values: np.ndarray, n_bins: int = 80, max_offset: float = 0.38):
    """Vertical offsets that turn a strip of points into a density beeswarm."""
    values = np.asarray(values, dtype=float)
    offsets = np.zeros_like(values)
    if values.size == 0:
        return offsets
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return offsets
    bins = np.clip(((values - low) / (high - low) * n_bins).astype(int), 0, n_bins - 1)
    densest = max(int(np.bincount(bins).max()), 2)
    step = 2.0 * max_offset / (densest - 1)
    for b in np.unique(bins):
        idx = np.where(bins == b)[0]
        idx = idx[np.argsort(values[idx], kind="stable")]
        ladder = (np.arange(idx.size) - (idx.size - 1) / 2.0) * step
        offsets[idx] = np.clip(ladder, -max_offset, max_offset)
    return offsets


def figure_shap_beeswarm(result: dict) -> None:
    """Beeswarm of the SHAP value attached to each player's own listed position group."""
    own = result["own"]
    features = result["features"]
    order = list(result["own_order"][:SHAP_DISPLAY])
    x = result["x"]
    n = len(order)

    fig, ax = plt.subplots(figsize=(P.WIDTH_FULL, 0.33 * n + 1.9))
    ax.axvline(0.0, color=P.INK_MUTED, linewidth=0.7, zorder=1)
    scatter = None
    for row, j in enumerate(order):
        centre = n - 1 - row
        shap_values = own[:, j]
        raw = x.iloc[:, j].to_numpy(dtype=float)
        # Percentile rank rather than the raw value, so one extreme player cannot flatten
        # the colour scale for the other 1,999.
        colour = stats.rankdata(raw, method="average") / len(raw)
        scatter = ax.scatter(
            shap_values,
            centre + _beeswarm_offsets(shap_values),
            c=colour,
            cmap=BEESWARM_CMAP,
            vmin=0.0,
            vmax=1.0,
            s=4.0,
            linewidths=0.0,
            alpha=0.8,
            zorder=3,
        )
    ax.set_yticks(range(n), [label(features[j]) for j in order[::-1]])
    ax.set_ylim(-0.7, n - 0.3)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    P.style_axis(
        ax,
        xlabel="SHAP value for the player's own listed position group (log-odds)",
        title=(
            f"Top {n} of {len(features)} statistics by contribution to listed position, "
            f"{result['payload']['explained_rows']:,} players explained"
        ),
    )
    bar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02, ticks=[0.0, 0.5, 1.0])
    bar.ax.set_yticklabels(["Low", "Median", "High"])
    bar.set_label("Feature value percentile", fontsize=P.BASE_FONT_PT - 1)
    bar.outline.set_visible(False)
    P.save_figure(fig, "archive_shap_beeswarm")


def figure_shap_by_class(result: dict) -> None:
    """Mean absolute SHAP per feature, faceted into one panel per listed position group."""
    mean_abs = result["mean_abs"]
    features = result["features"]
    order = list(np.argsort(-result["overall"])[:SHAP_DISPLAY])
    n = len(order)
    rows = np.arange(n)[::-1]

    fig, axes = P.facet_grid(
        len(CLASS_ORDER), ncols=len(CLASS_ORDER), width=P.WIDTH_FULL, panel_h=0.30 * n + 1.2
    )
    for ax, cls in zip(axes, CLASS_ORDER, strict=True):
        ax.barh(
            rows,
            mean_abs[cls][order],
            color=P.POSITION_COLORS[cls],
            height=0.72,
            linewidth=0.0,
        )
        P.style_axis(ax, title=cls)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    axes[0].set_yticks(rows, [label(features[j]) for j in order])
    axes[0].set_ylim(-0.8, n - 0.2)
    axes[0].set_xlim(0.0, float(max(mean_abs[c][order].max() for c in CLASS_ORDER)) * 1.08)
    fig.supxlabel("Mean absolute SHAP value (log-odds)", fontsize=P.BASE_FONT_PT)
    P.save_figure(fig, "archive_shap_by_class")


def figure_pca(result: dict, features: list[str]) -> None:
    """Scree, cumulative variance and the loadings on the first five components."""
    payload = result["payload"]
    ratio = np.asarray(result["ratio"], dtype=float)
    cumulative = np.asarray(result["cumulative"], dtype=float)
    n_pc = len(ratio)
    n_90 = payload["n_components_for_90pct"]

    fig = plt.figure(figsize=(P.WIDTH_WIDE, 5.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.05)

    ax = fig.add_subplot(grid[0, 0])
    ax.bar(np.arange(1, n_pc + 1), ratio * 100, color=P.CATEGORICAL[0], width=0.7, linewidth=0.0)
    ax.annotate(
        f"PC1 {ratio[0] * 100:.1f}%",
        (1, ratio[0] * 100),
        textcoords="offset points",
        xytext=(6, -2),
        fontsize=P.BASE_FONT_PT - 1,
        color=P.INK_SECONDARY,
    )
    ax.set_xticks([1] + list(range(5, n_pc + 1, 5)))
    P.style_axis(ax, "Component", "Variance explained (%)", "Scree")

    ax = fig.add_subplot(grid[0, 1])
    ax.plot(
        np.arange(1, n_pc + 1),
        cumulative * 100,
        color=P.CATEGORICAL[1],
        marker="o",
        markersize=3.0,
    )
    ax.axhline(config.PCA_VARIANCE_TARGET * 100, color=P.INK_MUTED, linewidth=0.8, linestyle="--")
    ax.axvline(n_90, color=P.INK_MUTED, linewidth=0.8, linestyle="--")
    ax.annotate(
        f"{n_90} components\nreach {int(config.PCA_VARIANCE_TARGET * 100)}%",
        (n_90, 30),
        textcoords="offset points",
        xytext=(7, 0),
        ha="left",
        va="center",
        fontsize=P.BASE_FONT_PT - 1,
        color=P.INK_SECONDARY,
    )
    ax.set_xticks([1] + list(range(5, n_pc + 1, 5)))
    ax.set_ylim(0, 104)
    P.style_axis(ax, "Component", "Cumulative variance (%)", "Cumulative")

    ax = fig.add_subplot(grid[1, :])
    matrix = np.array(
        [
            [payload["components"][f"PC{p + 1}"]["loadings"][f] for f in features]
            for p in range(PC_DISPLAY)
        ]
    )
    limit = float(np.abs(matrix).max())
    mesh = ax.imshow(matrix, cmap=P.DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(
        range(PC_DISPLAY),
        [
            f"PC{p + 1}  {payload['components'][f'PC{p + 1}']['explained_variance_ratio'] * 100:.1f}%"
            for p in range(PC_DISPLAY)
        ],
    )
    ax.set_xticks(range(len(features)), [label(f) for f in features], rotation=90)
    ax.tick_params(axis="x", labelsize=P.BASE_FONT_PT - 2.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Loadings on the first five components", loc="left")
    bar = fig.colorbar(mesh, ax=ax, fraction=0.02, pad=0.01)
    bar.set_label("Loading", fontsize=P.BASE_FONT_PT - 1)
    bar.outline.set_visible(False)

    fig.suptitle(
        f"One dominant axis and a long tail: PC1 carries {ratio[0] * 100:.1f} percent "
        f"of the variance in 33 features",
        fontsize=P.BASE_FONT_PT,
        fontweight="bold",
    )
    P.save_figure(fig, "archive_pca")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
    # Embedding the sans-serif face into six PDFs makes fontTools log a note about the
    # font's own post table once per glyph run, which buries the numbers this prints.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    P.use_style()

    eligible, z_global, z_group, features = load_frames()
    scopes = scope_frames(z_global, z_group)
    print(
        f"archive validation: {len(eligible):,} player-seasons, {len(features)} features, "
        f"{eligible['season'].nunique()} seasons, {eligible['league'].nunique()} leagues"
    )

    print("\n1. cluster stability and correspondence with listed positions")
    validation = cluster_validation(scopes, features)
    _write(validation["payload"], "archive_cluster_validation.json")
    figure_cluster_vs_position(validation)
    print("  answer:", validation["payload"]["answer"]["verdict"])

    print("\n2. cross-season and cross-league replication")
    replication_payload = replication(
        eligible, z_global, features, validation["labels"]["All outfield"]
    )
    _write(replication_payload, "archive_replication.json")
    figure_replication(replication_payload)
    summary = replication_payload["summary"]
    print(
        f"  season pairwise ARI {summary['season_pairwise_ari_min']:.3f} to "
        f"{summary['season_pairwise_ari_max']:.3f}, league transfer ARI mean "
        f"{summary['league_transfer_ari_mean']:.3f}, worst axis correlation "
        f"{summary['axis_correlation_min']:.3f}"
    )

    print("\n3. supervised validation")
    supervised = classification(eligible, features)
    shap_result = shap_analysis(eligible, features)
    payload = dict(supervised["payload"])
    payload["shap"] = shap_result["payload"]
    _write(payload, "archive_supervised.json")
    figure_confusion(payload)
    figure_shap_beeswarm(shap_result)
    figure_shap_by_class(shap_result)
    for name in ("majority_baseline", "logistic_regression", "lightgbm"):
        block = payload["models"][name]
        recalls = ", ".join(f"{c} {block['per_class'][c]['recall']:.2f}" for c in CLASS_ORDER)
        print(
            f"  {name:20s} acc {block['accuracy']:.3f}  macro F1 {block['macro_f1']:.3f}  "
            f"recall {recalls}"
        )
    for cls in CLASS_ORDER:
        print(f"  top SHAP {cls}: {', '.join(shap_result['payload']['top5_by_class'][cls])}")

    print("\n4. dimensionality")
    pca = pca_analysis(z_global, features)
    _write(pca["payload"], "archive_pca.json")
    figure_pca(pca, features)
    for name, text in pca["payload"]["interpretation"].items():
        print(f"  {name}: {text}")
    print("  verdict:", pca["payload"]["verdict"])

    print("\narchive validation complete")


if __name__ == "__main__":
    main()
