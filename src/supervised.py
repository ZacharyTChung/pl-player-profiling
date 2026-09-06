"""Phase 7: supervised validation of the surviving feature set.

The clustering phase asks which groupings the data volunteer on their own. It selected
k=2 in every scope and agrees with the listed position groups at an adjusted Rand index
of only 0.226. That result is ambiguous on its own: it could mean that listed positions
are a poor description of what players do, or it could mean that the eighteen features
left after the January 2026 Opta withdrawal simply do not carry enough signal to
separate three position groups. This module asks the complementary supervised question
and settles that ambiguity. If a classifier that is told the answer can recover position
from these features, the signal exists and the clustering was choosing not to use it. If
the classifier cannot, the features themselves are the limit.

Four analyses are run on the primary season.

1. **Position classification.** Multinomial logistic regression and LightGBM predict
   ``position_group`` from the eighteen ``features.OUTFIELD_CORE`` per-90 columns under
   five-fold stratified cross validation. Every reported number comes from out-of-fold
   predictions, never from the training sample. The classes are unbalanced, roughly 136
   defenders, 181 midfielders and 49 forwards, so a majority-class baseline is fitted
   through the identical folds and macro F1 is treated as the headline score because it
   weights the 49 forwards equally with the 181 midfielders.

2. **SHAP.** SHAP values \\citep{lundberg2017unified} are computed on a LightGBM model
   fitted to the full sample. This step is interpretability only, it is not an accuracy
   claim, and the accuracy claims all come from the cross validated fit above. The
   global beeswarm plots, for every player, the SHAP value attached to that player's own
   listed position group, so a positive value means the feature pushed the player toward
   the position he is actually listed at. The per-class figure gives the standard mean
   absolute SHAP ranking within each of DF, MF and FW.

3. **Misclassification against the hybrid list.** The players the classifier places in
   the wrong group are cross referenced against the players the clustering flagged as
   hybrids. Two methods with different objectives, one unsupervised and one supervised,
   failing on the same players is evidence that those players genuinely sit between
   positions rather than evidence that either method is broken. Because the hybrid flag
   already covers a large share of the squad pool, the raw overlap rate is meaningless
   on its own, so the expected overlap under independence, the enrichment ratio and a
   Fisher exact test are reported alongside it.

4. **Finishing above role.** Ridge regression predicts ``np_xg_p90`` from
   ``features.NON_SHOOTING_FEATURES`` only, so the fitted value is the shot volume a
   player's off-ball and creative role would ordinarily generate and the residual is the
   part of shot quality that role does not explain.

   **Caveat, stated once here, computed in code and repeated in the saved metrics.** The
   residual is only interesting to the extent that the fitted component is both large and
   honest, and here it is large but not honest. The cross validated R squared on the
   specified feature set is 0.89, which looks excellent and is not. Understat defines
   xGChain as the total expected goals of every possession a player was involved in,
   including the possessions he finished himself, while xGBuildup is the same quantity
   with shots and key passes removed. The difference between the two therefore contains
   the player's own non-penalty expected goals almost by construction: across the primary
   season it correlates with the target at 0.90, so it is excluded from the feature set.
   Dropping xGChain takes the cross validated R squared from 0.89 to 0.53, and the exact
   values for both fits are computed at run time and written into
   ``results/metrics/finishing_above_role.json``. Neither leaderboard is a measurement of
   finishing ability. The specified one ranks players whose own shots are an unusual share
   of the possessions they touch, and the leak-free one ranks players whose shot volume is
   unusual given a model that explains about half the variance, which still leaves ample
   room for team attacking volume, penalty box occupancy and single season noise on 450 to
   3000 minutes. Both are reported, with the ablation, because the paper commits to
   reporting analyses that fail rather than deleting them.

Outputs
-------
results/metrics/position_classification.json
results/metrics/shap_importance.json
results/metrics/misclassified_vs_hybrid.json
results/metrics/finishing_above_role.json
figures/confusion_matrix.{pdf,png}
figures/shap_beeswarm.{pdf,png}
figures/shap_by_class.{pdf,png}
figures/finishing_above_role.{pdf,png}
"""

from __future__ import annotations

import json
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from src import features as F
from src import plotting
from src.descriptive import label

#: Class order used in every table, matrix and figure. Alphabetical order would put FW
#: in the middle, which reads badly on a confusion matrix whose off-diagonal story is
#: about the defender to forward axis.
CLASS_ORDER = config.OUTFIELD_GROUPS

#: Cross validation is stratified because forwards are only 13 percent of the pool and
#: an unstratified fold could easily hold six of them.
N_SPLITS = 5

#: Ridge penalty grid, chosen by an inner generalised cross validation on each training
#: fold so the outer score stays honest.
RIDGE_ALPHAS = np.logspace(-3, 3, 25)

#: Length of each half of the finishing residual leaderboard.
LEADERBOARD_N = 15

#: Fallback if the hybrid metrics file ever stops recording its own threshold.
DEFAULT_POSTERIOR_SPLIT = 0.6

#: Understat's xGChain counts every possession the player was involved in, including the
#: ones he finished himself, and xGBuildup is the same figure with shots and key passes
#: removed. Their difference therefore carries the player's own non-penalty expected
#: goals, which is the regression target. Both columns are kept in the specified fit
#: because they are what features.NON_SHOOTING_FEATURES declares, and the ablation below
#: measures how much of the fit they are responsible for.
LEAKY_FEATURES = ["xg_chain_p90", "xg_buildup_p90"]
LEAK_FREE_FEATURES = [c for c in F.NON_SHOOTING_FEATURES if c not in LEAKY_FEATURES]

#: The project sequential ramp begins one shade off the paper surface, which is correct
#: for a filled cell, where near-white reads as empty, and wrong for a scatter mark, where
#: it reads as absent. The bottom of the ramp is trimmed for the beeswarm marks only. The
#: ramp is used unmodified everywhere it fills an area.
BEESWARM_CMAP = ListedColormap(plotting.SEQUENTIAL(np.linspace(0.22, 1.0, 256)))

#: Recorded in the finishing metrics so the caveat travels with the numbers.
FINISHING_CAVEAT = (
    "This regression is not a measurement of finishing ability and its headline R squared "
    "is not evidence that it is a good model. Understat's xGChain includes the expected "
    "goals of possessions the player finished himself and xGBuildup excludes shots and key "
    "passes, so the difference between the two columns contains the target almost by "
    "construction. See the leakage block for the correlation and for the ablation R "
    "squared with xGChain removed. The specified leaderboard ranks players whose own shots "
    "are an unusual share of the possessions they touch. The leak-free leaderboard ranks "
    "players whose shot volume is unusual given a model that leaves roughly half the "
    "variance unexplained, which is still ample room for team attacking volume, penalty "
    "box occupancy and single season noise on 450 to 3000 minutes. Read neither as a "
    "ranking of talent."
)


def _r(value, digits: int = 4):
    """Round for JSON, mapping non-finite values to None."""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _write(payload: dict, name: str) -> None:
    with open(config.METRICS / name, "w") as fh:
        json.dump(payload, fh, indent=2)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def load_eligible(season: str) -> pd.DataFrame:
    """Per-90 features plus context, one row per eligible outfield player."""
    return pd.read_parquet(config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet")


def load_clusters(season: str) -> pd.DataFrame:
    return pd.read_parquet(config.DATA_PROCESSED / f"clusters_{season}.parquet")


def design_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix = frame[columns].astype(float)
    if matrix.isna().any().any():
        missing = matrix.columns[matrix.isna().any()].tolist()
        raise ValueError(f"missing values in supervised design matrix: {missing}")
    return matrix


# --------------------------------------------------------------------------------------
# 1. Position classification
# --------------------------------------------------------------------------------------


def build_models() -> dict[str, object]:
    """The two classifiers, both seeded and both deterministic.

    LightGBM is pinned to a single thread with ``deterministic`` and ``force_col_wise``
    set, because its default histogram construction is order dependent across threads
    and would otherwise change the fourth decimal place between runs.
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
        num_leaves=15,
        max_depth=5,
        min_child_samples=15,
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
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, np.where(row_totals == 0, 1, row_totals))
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
            "labels": list(CLASS_ORDER),
            "orientation": "rows are listed position, columns are predicted position",
            "counts": matrix.astype(int).tolist(),
            "row_normalised": [[_r(v, 4) for v in row] for row in normalised],
        },
    }


def classify_positions(frame: pd.DataFrame, season: str) -> dict:
    """Cross validated position classification with a majority-class baseline."""
    x = design_matrix(frame, F.OUTFIELD_CORE)
    y = frame["position_group"].to_numpy()
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    baseline_pred = cross_val_predict(
        DummyClassifier(strategy="most_frequent"), x, y, cv=folds, n_jobs=1
    )
    baseline = _score_block(y, baseline_pred)
    baseline["strategy"] = "most_frequent"
    baseline["majority_class"] = str(pd.Series(y).value_counts().idxmax())

    models: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    probabilities: dict[str, pd.DataFrame] = {}

    for name, estimator in build_models().items():
        proba = cross_val_predict(estimator, x, y, cv=folds, method="predict_proba", n_jobs=1)
        classes = list(np.unique(y))
        proba_frame = pd.DataFrame(proba, columns=classes, index=frame.index)[CLASS_ORDER]
        pred = proba_frame.columns.to_numpy()[np.argmax(proba_frame.to_numpy(), axis=1)]
        block = _score_block(y, pred)
        block["per_fold"] = {
            "accuracy": [
                _r(v)
                for v in cross_val_score(estimator, x, y, cv=folds, scoring="accuracy", n_jobs=1)
            ],
            "macro_f1": [
                _r(v)
                for v in cross_val_score(estimator, x, y, cv=folds, scoring="f1_macro", n_jobs=1)
            ],
        }
        block["per_fold"]["accuracy_mean"] = _r(np.mean(block["per_fold"]["accuracy"]))
        block["per_fold"]["accuracy_sd"] = _r(np.std(block["per_fold"]["accuracy"], ddof=1))
        block["per_fold"]["macro_f1_mean"] = _r(np.mean(block["per_fold"]["macro_f1"]))
        block["per_fold"]["macro_f1_sd"] = _r(np.std(block["per_fold"]["macro_f1"], ddof=1))
        block["accuracy_over_baseline"] = _r(block["accuracy"] - baseline["accuracy"])
        block["macro_f1_over_baseline"] = _r(block["macro_f1"] - baseline["macro_f1"])
        models[name] = block
        predictions[name] = pred
        probabilities[name] = proba_frame

    # Standardised multinomial coefficients, a linear complement to the SHAP ranking.
    logistic = build_models()["logistic_regression"].fit(x, y)
    coefficients = logistic.named_steps["clf"].coef_
    fitted_classes = list(logistic.named_steps["clf"].classes_)
    coefficient_block = {
        cls: {
            feature: _r(coefficients[fitted_classes.index(cls), j])
            for j, feature in enumerate(F.OUTFIELD_CORE)
        }
        for cls in CLASS_ORDER
    }

    counts = pd.Series(y).value_counts()
    payload = {
        "season": season,
        "target": "position_group",
        "features": list(F.OUTFIELD_CORE),
        "n_features": len(F.OUTFIELD_CORE),
        "n_players": int(len(frame)),
        "class_counts": {cls: int(counts.get(cls, 0)) for cls in CLASS_ORDER},
        "class_shares": {cls: _r(counts.get(cls, 0) / len(frame)) for cls in CLASS_ORDER},
        "cross_validation": {
            "scheme": "stratified k-fold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": config.RANDOM_STATE,
            "scoring_source": "out-of-fold predictions pooled over the five folds",
        },
        "baseline": baseline,
        "models": models,
        "logistic_standardised_coefficients": coefficient_block,
        "headline_metric": "macro_f1",
        "notes": (
            "Macro F1 is the headline because the classes are unbalanced and accuracy "
            "rewards a model for getting midfielders right. All model scores are pooled "
            "out-of-fold scores, so nothing here is measured on training data."
        ),
    }
    return {"payload": payload, "y": y, "predictions": predictions, "probabilities": probabilities}


# --------------------------------------------------------------------------------------
# 2. SHAP
# --------------------------------------------------------------------------------------


def shap_analysis(frame: pd.DataFrame, season: str) -> dict:
    """SHAP values for a LightGBM model fitted to the whole sample.

    Fitting on everything is correct here and would be wrong for an accuracy claim. The
    question is which of the surviving statistics the model uses to tell the positions
    apart, and that question is about the fitted function, not about generalisation.
    """
    x = design_matrix(frame, F.OUTFIELD_CORE)
    y = frame["position_group"].to_numpy()
    model = build_models()["lightgbm"].fit(x, y)

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(x)
    values = np.stack(raw, axis=-1) if isinstance(raw, list) else np.asarray(raw)
    if values.ndim != 3:
        raise ValueError(f"expected per-class SHAP values, got array of shape {values.shape}")

    fitted_classes = [str(c) for c in model.classes_]
    class_index = {cls: fitted_classes.index(cls) for cls in CLASS_ORDER}

    by_class: dict[str, list[dict]] = {}
    mean_abs = {}
    for cls in CLASS_ORDER:
        block = values[:, :, class_index[cls]]
        magnitude = np.abs(block).mean(axis=0)
        signed = block.mean(axis=0)
        mean_abs[cls] = magnitude
        order = np.argsort(-magnitude)
        by_class[cls] = [
            {
                "rank": int(rank + 1),
                "feature": F.OUTFIELD_CORE[j],
                "label": label(F.OUTFIELD_CORE[j]),
                "mean_abs_shap": _r(magnitude[j], 5),
                "mean_signed_shap": _r(signed[j], 5),
            }
            for rank, j in enumerate(order)
        ]

    overall = np.mean([mean_abs[cls] for cls in CLASS_ORDER], axis=0)
    overall_order = np.argsort(-overall)
    overall_ranking = [
        {
            "rank": int(rank + 1),
            "feature": F.OUTFIELD_CORE[j],
            "label": label(F.OUTFIELD_CORE[j]),
            "mean_abs_shap": _r(overall[j], 5),
        }
        for rank, j in enumerate(overall_order)
    ]

    own = np.array([values[i, :, class_index[y[i]]] for i in range(len(y))])
    own_magnitude = np.abs(own).mean(axis=0)
    own_order = np.argsort(-own_magnitude)
    own_ranking = [
        {
            "rank": int(rank + 1),
            "feature": F.OUTFIELD_CORE[j],
            "label": label(F.OUTFIELD_CORE[j]),
            "mean_abs_shap": _r(own_magnitude[j], 5),
            "mean_signed_shap": _r(own[:, j].mean(), 5),
        }
        for rank, j in enumerate(own_order)
    ]

    payload = {
        "season": season,
        "model": "lightgbm",
        "fit": "full sample, no held-out split, interpretability only",
        "explainer": "shap.TreeExplainer, exact tree SHAP",
        "units": "log-odds contribution to the class score",
        "n_players": int(len(frame)),
        "features": list(F.OUTFIELD_CORE),
        "class_order": list(CLASS_ORDER),
        "overall_ranking": overall_ranking,
        "own_class_ranking": own_ranking,
        "by_class": by_class,
        "top3_by_class": {
            cls: [entry["label"] for entry in by_class[cls][:3]] for cls in CLASS_ORDER
        },
        "notes": (
            "overall_ranking averages the per-class mean absolute SHAP over DF, MF and FW. "
            "own_class_ranking uses, for each player, the SHAP value attached to that "
            "player's own listed position group, so its sign is readable: positive means "
            "the feature pushed the player toward the position he is listed at. The "
            "beeswarm figure plots the own-class values."
        ),
    }
    return {
        "payload": payload,
        "values": values,
        "own": own,
        "own_order": own_order,
        "mean_abs": mean_abs,
        "overall": overall,
        "x": x,
    }


# --------------------------------------------------------------------------------------
# 3. Misclassification against the hybrid list
# --------------------------------------------------------------------------------------


def hybrid_flags(season: str) -> dict:
    """Rebuild the full hybrid flag set and check it against the clustering metrics.

    ``results/metrics/hybrid_players.json`` names only the leading rows of the hybrid
    table, so the whole flagged set is reconstructed from ``clusters_{season}.parquet``
    using the definition the same file records. The reconstruction is then verified
    against the counts stored by the clustering phase, and this function raises if the
    two ever disagree, so the cross reference below can never quietly drift out of sync
    with the clustering it is supposed to be compared against.
    """
    with open(config.METRICS / "hybrid_players.json") as fh:
        source = json.load(fh)
    block = source["seasons"][season]
    definition = block["definition"]

    match = re.search(r"([0-9]*\.?[0-9]+)", definition["posterior_split"])
    threshold = float(match.group(1)) if match else DEFAULT_POSTERIOR_SPLIT

    clusters = load_clusters(season).copy()
    dominant = clusters.groupby("cluster_global")["position_group"].agg(
        lambda s: s.value_counts().idxmax()
    )
    clusters["cluster_dominant_group"] = clusters["cluster_global"].map(dominant)
    clusters["disagrees"] = clusters["cluster_dominant_group"] != clusters["position_group"]
    clusters["split"] = clusters["gmm_max_posterior"] < threshold
    clusters["is_hybrid"] = clusters["disagrees"] | clusters["split"]

    recovered = {
        "n_players": int(len(clusters)),
        "n_disagreeing": int(clusters["disagrees"].sum()),
        "n_posterior_split": int(clusters["split"].sum()),
        "n_both": int((clusters["disagrees"] & clusters["split"]).sum()),
        "n_flagged": int(clusters["is_hybrid"].sum()),
    }
    stored = block["counts"]
    if any(recovered[k] != stored[k] for k in recovered if k in stored):
        raise AssertionError(
            f"hybrid reconstruction disagrees with hybrid_players.json: {recovered} vs {stored}"
        )

    reason = np.where(
        clusters["disagrees"] & clusters["split"],
        "both",
        np.where(clusters["split"], "posterior split", "cluster disagreement"),
    )
    clusters["hybrid_reason"] = np.where(clusters["is_hybrid"], reason, "not flagged")
    return {
        "table": clusters,
        "definition": definition,
        "counts": recovered,
        "threshold": threshold,
        "named_top_players": [row["player"] for row in block["top_players"]],
    }


def _overlap_block(misclassified: pd.Series, is_hybrid: pd.Series, named: set[str]) -> dict:
    """Overlap between a misclassified set and the hybrid set, with a null comparison."""
    n = int(len(misclassified))
    a = int((misclassified & is_hybrid).sum())
    b = int((misclassified & ~is_hybrid).sum())
    c = int((~misclassified & is_hybrid).sum())
    d = int((~misclassified & ~is_hybrid).sum())
    n_mis = a + b
    n_hyb = a + c
    expected = n_mis * n_hyb / n if n else 0.0
    odds, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    return {
        "n_players": n,
        "n_misclassified": n_mis,
        "misclassification_rate": _r(n_mis / n) if n else None,
        "n_hybrid": n_hyb,
        "hybrid_base_rate": _r(n_hyb / n) if n else None,
        "n_overlap": a,
        "overlap_rate_of_misclassified": _r(a / n_mis) if n_mis else None,
        "overlap_rate_of_hybrids": _r(a / n_hyb) if n_hyb else None,
        "expected_overlap_if_independent": _r(expected, 2),
        "enrichment_ratio": _r(a / expected) if expected else None,
        "fisher_odds_ratio": _r(odds),
        "fisher_p_one_sided": _r(pvalue, 6),
        "contingency": {
            "misclassified_and_hybrid": a,
            "misclassified_not_hybrid": b,
            "hybrid_not_misclassified": c,
            "neither": d,
        },
        "n_overlap_with_named_hybrid_table": int(
            sum(1 for p in misclassified.index[misclassified] if p in named)
        ),
    }


def misclassification_vs_hybrid(frame: pd.DataFrame, classification: dict, season: str) -> dict:
    """Cross reference classifier errors against the clustering hybrid flags."""
    hybrids = hybrid_flags(season)
    flags = hybrids["table"].set_index("player")
    if set(flags.index) != set(frame["player"]):
        raise AssertionError("cluster table and eligible table cover different players")

    listed = frame.set_index("player")["position_group"]
    order = listed.index
    is_hybrid = flags.loc[order, "is_hybrid"]
    named = set(hybrids["named_top_players"])

    wrong: dict[str, pd.Series] = {}
    models: dict[str, dict] = {}
    for name, pred in classification["predictions"].items():
        predicted = pd.Series(pred, index=order)
        wrong[name] = predicted.ne(listed)
        block = _overlap_block(wrong[name], is_hybrid, named)
        proba = classification["probabilities"][name].copy()
        proba.index = order
        rows = []
        for player in order[wrong[name].to_numpy()]:
            info = flags.loc[player]
            rows.append(
                {
                    "player": player,
                    "team": frame.set_index("player").loc[player, "team"],
                    "listed_position_group": listed.loc[player],
                    "listed_position_full": info["position_full"],
                    "predicted_position_group": predicted.loc[player],
                    "predicted_probability": _r(proba.loc[player].max()),
                    "minutes": int(frame.set_index("player").loc[player, "minutes"]),
                    "cluster_global": int(info["cluster_global"]),
                    "cluster_dominant_group": info["cluster_dominant_group"],
                    "gmm_max_posterior": _r(info["gmm_max_posterior"]),
                    "is_hybrid": bool(info["is_hybrid"]),
                    "hybrid_reason": info["hybrid_reason"],
                    "in_named_hybrid_table": player in named,
                }
            )
        rows.sort(key=lambda r: (not r["is_hybrid"], -r["predicted_probability"]))
        block["misclassified_players"] = rows
        block["players_in_both"] = [r["player"] for r in rows if r["is_hybrid"]]
        models[name] = block

    both_wrong = wrong["logistic_regression"] & wrong["lightgbm"]
    agreement = _overlap_block(both_wrong, is_hybrid, named)
    agreement["players_in_both"] = [
        p for p in order[both_wrong.to_numpy()] if bool(is_hybrid.loc[p])
    ]

    payload = {
        "season": season,
        "misclassified_definition": (
            "out-of-fold cross validated prediction differs from the listed position group"
        ),
        "hybrid_definition": hybrids["definition"],
        "hybrid_counts": hybrids["counts"],
        "hybrid_source": "results/metrics/hybrid_players.json, full set rebuilt from clusters",
        "models": models,
        "both_models_wrong": agreement,
        "interpretation": (
            "The hybrid flag already covers a large share of the pool, so a high raw "
            "overlap rate is expected even under independence. The enrichment ratio and "
            "the Fisher test are the numbers that carry evidence, because they compare "
            "the observed overlap against the overlap that random error would produce."
        ),
    }
    return {"payload": payload, "wrong": wrong, "is_hybrid": is_hybrid}


# --------------------------------------------------------------------------------------
# 4. Finishing above role
# --------------------------------------------------------------------------------------


def _ridge_fit(frame: pd.DataFrame, columns: list[str], y: np.ndarray, folds: KFold) -> dict:
    """One cross validated ridge fit, returning out-of-fold predictions and scores."""
    x = design_matrix(frame, columns)
    pipeline = Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS))])
    oof = cross_val_predict(pipeline, x, y, cv=folds, n_jobs=1)
    fold_r2 = cross_val_score(pipeline, x, y, cv=folds, scoring="r2", n_jobs=1)
    fitted = pipeline.fit(x, y)
    residual = y - oof
    table = pd.DataFrame(
        {
            "player": frame["player"].to_numpy(),
            "team": frame["team"].to_numpy(),
            "position_group": frame["position_group"].to_numpy(),
            "minutes": frame["minutes"].to_numpy(),
            "actual": y,
            "expected": oof,
            "residual": residual,
        }
    ).sort_values("residual", ascending=False, kind="stable")
    return {
        "columns": list(columns),
        "table": table,
        "cv_r2": float(r2_score(y, oof)),
        "summary": {
            "features": list(columns),
            "n_features": len(columns),
            "alpha_full_sample": _r(float(fitted.named_steps["ridge"].alpha_), 5),
            "cv_r2_out_of_fold": _r(r2_score(y, oof)),
            "cv_r2_fold_mean": _r(float(np.mean(fold_r2))),
            "cv_r2_fold_sd": _r(float(np.std(fold_r2, ddof=1))),
            "cv_r2_per_fold": [_r(v) for v in fold_r2],
            "in_sample_r2": _r(r2_score(y, fitted.predict(x))),
            "residual_sd": _r(float(residual.std(ddof=1))),
            "share_of_variance_unexplained": _r(1.0 - r2_score(y, oof)),
            "coefficients_standardised": {
                feature: _r(coef, 5)
                for feature, coef in zip(columns, fitted.named_steps["ridge"].coef_, strict=True)
            },
        },
    }


def _leaderboard_rows(sub: pd.DataFrame) -> list[dict]:
    return [
        {
            "player": row.player,
            "team": row.team,
            "position_group": row.position_group,
            "minutes": int(row.minutes),
            "np_xg_p90_actual": _r(row.actual),
            "np_xg_p90_expected": _r(row.expected),
            "residual": _r(row.residual),
        }
        for row in sub.itertuples()
    ]


def finishing_above_role(frame: pd.DataFrame, season: str) -> dict:
    """Ridge regression of npxG per 90 on non-shooting behaviour, plus a leakage ablation.

    xGChain is excluded from the feature set because it contains the target. Understat
    credits xGChain for the expected goals of every possession the player touched,
    including the ones he finished himself, while xGBuildup removes shots and key passes,
    so the difference between the two carries npxG almost by construction. The ablation
    that adds xGChain back is run beside the specified model and both R squared values are
    reported, so the inflation the leak would have produced stays visible.
    """
    y = frame["np_xg_p90"].astype(float).to_numpy()
    folds = KFold(n_splits=N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    specified = _ridge_fit(frame, list(F.NON_SHOOTING_FEATURES), y, folds)
    leak_free = _ridge_fit(frame, LEAK_FREE_FEATURES, y, folds)
    # xGChain is excluded from the specified feature set, so the informative ablation is
    # the other direction: add it back and show how much it inflates the fit.
    leaky = _ridge_fit(frame, [*F.NON_SHOOTING_FEATURES, "xg_chain_p90"], y, folds)

    chain_minus_buildup = (frame["xg_chain_p90"] - frame["xg_buildup_p90"]).to_numpy(dtype=float)
    leak_correlation = float(np.corrcoef(chain_minus_buildup, y)[0, 1])

    cv_r2 = specified["cv_r2"]
    leak_free_r2 = leak_free["cv_r2"]
    leaky_r2 = leaky["cv_r2"]
    payload = {
        "season": season,
        "target": "np_xg_p90",
        "features": list(F.NON_SHOOTING_FEATURES),
        "n_features": len(F.NON_SHOOTING_FEATURES),
        "n_players": int(len(frame)),
        "model": "ridge, alpha selected by generalised cross validation inside each fold",
        "cross_validation": {
            "scheme": "k-fold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": config.RANDOM_STATE,
        },
        "target_sd": _r(float(y.std(ddof=1))),
        "cv_r2_out_of_fold": _r(cv_r2),
        "cv_r2_fold_mean": specified["summary"]["cv_r2_fold_mean"],
        "cv_r2_fold_sd": specified["summary"]["cv_r2_fold_sd"],
        "cv_r2_per_fold": specified["summary"]["cv_r2_per_fold"],
        "cv_r2_out_of_fold_leak_free": _r(leak_free_r2),
        "in_sample_r2": specified["summary"]["in_sample_r2"],
        "residual_sd": specified["summary"]["residual_sd"],
        "share_of_variance_unexplained": _r(1.0 - cv_r2),
        "alpha_full_sample": specified["summary"]["alpha_full_sample"],
        "coefficients_standardised": specified["summary"]["coefficients_standardised"],
        "leakage": {
            "leaky_features": list(LEAKY_FEATURES),
            "explanation": (
                "Understat xGChain is the expected goals of every possession the player "
                "touched, including the possessions he finished himself, and xGBuildup is "
                "the same total with shots and key passes removed. Their difference "
                "therefore contains the player's own non-penalty expected goals, which is "
                "the regression target."
            ),
            "corr_chain_minus_buildup_with_target": _r(leak_correlation),
            "cv_r2_specified": _r(cv_r2),
            "cv_r2_with_xg_chain_added_back": _r(leaky_r2),
            "cv_r2_without_chain_and_buildup": _r(leak_free_r2),
            "r2_inflation_from_xg_chain": _r(leaky_r2 - cv_r2),
        },
        "caveat": FINISHING_CAVEAT,
        "caveat_with_value": (
            f"Cross validated R squared on the specified feature set is {cv_r2:.3f}. "
            f"xGChain is excluded from that feature set because Understat credits it for "
            f"possessions the player finished himself: its difference from xGBuildup "
            f"correlates with npxG per 90 at {leak_correlation:.3f}, and a model that "
            f"includes it reaches {leaky_r2:.3f} by reconstructing the target rather than "
            f"predicting it. With the leak removed the fit still leaves roughly half the "
            f"variance unexplained, so the leaderboard does not measure finishing ability."
        ),
        "top_15_above_role": _leaderboard_rows(specified["table"].head(LEADERBOARD_N)),
        "bottom_15_below_role": _leaderboard_rows(
            specified["table"].tail(LEADERBOARD_N).iloc[::-1]
        ),
        "leak_free_model": {
            **leak_free["summary"],
            "top_15_above_role": _leaderboard_rows(leak_free["table"].head(LEADERBOARD_N)),
            "bottom_15_below_role": _leaderboard_rows(
                leak_free["table"].tail(LEADERBOARD_N).iloc[::-1]
            ),
            "note": (
                "Both Understat chain columns removed. This is the fit to quote if a single "
                "number is wanted for how much of shot volume a non-shooting role explains."
            ),
        },
    }
    return {
        "payload": payload,
        "table": specified["table"],
        "cv_r2": cv_r2,
        "leak_free_table": leak_free["table"],
        "leak_free_r2": leak_free_r2,
        "leak_correlation": leak_correlation,
    }


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def figure_confusion_matrix(classification: dict) -> None:
    """Two-panel confusion matrix, one panel per classifier."""
    names = ["logistic_regression", "lightgbm"]
    titles = {"logistic_regression": "Multinomial logistic", "lightgbm": "LightGBM"}
    fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_FULL, 2.9))
    mesh = None
    for ax, name in zip(axes, names, strict=True):
        block = classification["payload"]["models"][name]["confusion_matrix"]
        counts = np.array(block["counts"], dtype=float)
        shares = np.array(block["row_normalised"], dtype=float)
        mesh = ax.imshow(shares, cmap=plotting.SEQUENTIAL, vmin=0.0, vmax=1.0, aspect="equal")
        for i in range(len(CLASS_ORDER)):
            for j in range(len(CLASS_ORDER)):
                ax.text(
                    j,
                    i,
                    f"{int(counts[i, j])}\n{shares[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=plotting.BASE_FONT_PT - 1,
                    color=plotting.SURFACE if shares[i, j] > 0.55 else plotting.INK_PRIMARY,
                )
        ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
        ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
        macro = classification["payload"]["models"][name]["macro_f1"]
        accuracy = classification["payload"]["models"][name]["accuracy"]
        ax.set_title(
            f"{titles[name]}\nacc {accuracy:.3f}, macro F1 {macro:.3f}",
            loc="left",
        )
        ax.set_xlabel("Predicted")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Listed position")
    bar = fig.colorbar(mesh, ax=axes, fraction=0.035, pad=0.02)
    bar.set_label("Share of the listed group", fontsize=plotting.BASE_FONT_PT - 1)
    bar.outline.set_visible(False)
    plotting.save_figure(fig, "confusion_matrix")


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


def figure_shap_beeswarm(shap_result: dict) -> None:
    """Beeswarm of the SHAP value attached to each player's own position group."""
    own = shap_result["own"]
    order = shap_result["own_order"]
    x = shap_result["x"]
    n_features = len(order)

    fig, ax = plt.subplots(figsize=(plotting.WIDTH_FULL, 0.34 * n_features + 1.5))
    ax.axvline(0.0, color=plotting.INK_MUTED, linewidth=0.7, zorder=1)
    scatter = None
    for row, j in enumerate(order):
        y_centre = n_features - 1 - row
        shap_values = own[:, j]
        raw = x.iloc[:, j].to_numpy(dtype=float)
        # Percentile rank rather than raw value, so one extreme player cannot flatten
        # the colour scale for the other 365.
        colour = stats.rankdata(raw, method="average") / len(raw)
        scatter = ax.scatter(
            shap_values,
            y_centre + _beeswarm_offsets(shap_values),
            c=colour,
            cmap=BEESWARM_CMAP,
            vmin=0.0,
            vmax=1.0,
            s=6.0,
            linewidths=0.0,
            alpha=0.85,
            zorder=3,
        )
    ax.set_yticks(range(n_features), [label(F.OUTFIELD_CORE[j]) for j in order[::-1]])
    ax.set_ylim(-0.7, n_features - 0.3)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    plotting.style_axis(
        ax,
        xlabel="SHAP value for the player's own listed position group (log-odds)",
        title="Which surviving statistics identify a player's position",
    )
    bar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02, ticks=[0.0, 0.5, 1.0])
    bar.ax.set_yticklabels(["Low", "Median", "High"])
    bar.set_label("Feature value percentile", fontsize=plotting.BASE_FONT_PT - 1)
    bar.outline.set_visible(False)
    plotting.save_figure(fig, "shap_beeswarm")


def figure_shap_by_class(shap_result: dict) -> None:
    """Mean absolute SHAP per feature, faceted into one panel per position group."""
    mean_abs = shap_result["mean_abs"]
    order = np.argsort(-shap_result["overall"])
    n_features = len(order)
    positions = np.arange(n_features)[::-1]

    fig, axes = plotting.facet_grid(
        len(CLASS_ORDER), ncols=len(CLASS_ORDER), width=plotting.WIDTH_FULL, panel_h=4.6
    )
    for ax, cls in zip(axes, CLASS_ORDER, strict=True):
        ax.barh(
            positions,
            mean_abs[cls][order],
            color=plotting.POSITION_COLORS[cls],
            height=0.72,
            linewidth=0.0,
        )
        plotting.style_axis(ax, title=cls)
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
    axes[0].set_yticks(positions, [label(F.OUTFIELD_CORE[j]) for j in order])
    axes[0].set_ylim(-0.8, n_features - 0.2)
    axes[0].set_xlim(0.0, float(max(mean_abs[c].max() for c in CLASS_ORDER)) * 1.06)
    fig.supxlabel("Mean absolute SHAP value (log-odds)", fontsize=plotting.BASE_FONT_PT)
    plotting.save_figure(fig, "shap_by_class")


def _fit_panel(ax, table: pd.DataFrame, title: str, *, legend: bool) -> None:
    for cls in CLASS_ORDER:
        sub = table[table["position_group"] == cls]
        ax.scatter(
            sub["expected"],
            sub["actual"],
            s=10,
            color=plotting.POSITION_COLORS[cls],
            marker=plotting.POSITION_MARKERS[cls],
            linewidths=0.0,
            alpha=0.8,
            label=cls,
        )
    limits = [
        float(min(table["expected"].min(), table["actual"].min())) - 0.02,
        float(max(table["expected"].max(), table["actual"].max())) + 0.02,
    ]
    ax.plot(limits, limits, color=plotting.INK_MUTED, linewidth=0.8, zorder=1)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    plotting.style_axis(
        ax, xlabel="Expected npxG/90 from role", ylabel="Actual npxG/90", title=title
    )
    if legend:
        plotting.legend_below(ax, ncol=3)


def figure_finishing_above_role(finishing: dict) -> None:
    """Both fits on the left, the residual leaderboard of the specified model on the right.

    The specified fit is shown above the leak-free fit rather than on its own, because a
    single tight scatter at R squared 0.89 would advertise a model that is largely
    reproducing its own target.
    """
    table = finishing["table"]
    fig = plt.figure(figsize=(plotting.WIDTH_WIDE, 6.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.2], height_ratios=[1.0, 1.0])
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0])
    ax_board = fig.add_subplot(grid[:, 1])

    _fit_panel(
        ax_top,
        table,
        f"Specified features, $R^2$ = {finishing['cv_r2']:.3f}\n"
        f"xGChain carries the target, r = {finishing['leak_correlation']:.2f}",
        legend=False,
    )
    _fit_panel(
        ax_bottom,
        finishing["leak_free_table"],
        f"Chain columns removed, $R^2$ = {finishing['leak_free_r2']:.3f}",
        legend=True,
    )

    board = pd.concat([table.head(LEADERBOARD_N), table.tail(LEADERBOARD_N)])
    positions = np.arange(len(board))[::-1]
    span = float(np.abs(board["residual"]).max())
    colours = plotting.DIVERGING(0.5 + 0.5 * board["residual"].to_numpy() / span)
    ax_board.barh(
        positions, board["residual"].to_numpy(), color=colours, height=0.74, linewidth=0.0
    )
    ax_board.axvline(0.0, color=plotting.INK_SECONDARY, linewidth=0.7)
    ax_board.set_yticks(positions, board["player"].tolist(), fontsize=plotting.BASE_FONT_PT - 2)
    ax_board.set_ylim(-0.8, len(board) - 0.2)
    ax_board.grid(True, axis="x")
    ax_board.grid(False, axis="y")
    plotting.style_axis(
        ax_board,
        xlabel="Residual npxG/90 (actual minus expected)",
        title="Top and bottom 15, specified model\nnot a finishing ranking",
    )
    plotting.save_figure(fig, "finishing_above_role")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    np.random.seed(config.RANDOM_STATE)
    plotting.use_style()

    season = config.SEASON_PRIMARY
    frame = load_eligible(season).reset_index(drop=True)

    classification = classify_positions(frame, season)
    _write(classification["payload"], "position_classification.json")

    shap_result = shap_analysis(frame, season)
    _write(shap_result["payload"], "shap_importance.json")

    cross_reference = misclassification_vs_hybrid(frame, classification, season)
    _write(cross_reference["payload"], "misclassified_vs_hybrid.json")

    finishing = finishing_above_role(frame, season)
    _write(finishing["payload"], "finishing_above_role.json")

    figure_confusion_matrix(classification)
    figure_shap_beeswarm(shap_result)
    figure_shap_by_class(shap_result)
    figure_finishing_above_role(finishing)

    baseline = classification["payload"]["baseline"]
    print(
        f"{season}: baseline (majority {baseline['majority_class']}) "
        f"accuracy {baseline['accuracy']:.3f} macro F1 {baseline['macro_f1']:.3f}",
        flush=True,
    )
    for name, block in classification["payload"]["models"].items():
        print(
            f"{season}: {name} accuracy {block['accuracy']:.3f} "
            f"macro F1 {block['macro_f1']:.3f} "
            f"(fold sd {block['per_fold']['macro_f1_sd']:.3f}) "
            f"recall DF/MF/FW "
            + "/".join(f"{block['per_class'][c]['recall']:.2f}" for c in CLASS_ORDER),
            flush=True,
        )
    print(
        "top SHAP features: "
        + "; ".join(
            f"{cls} {', '.join(shap_result['payload']['top3_by_class'][cls])}"
            for cls in CLASS_ORDER
        ),
        flush=True,
    )
    for name, block in cross_reference["payload"]["models"].items():
        print(
            f"{season}: {name} misclassified {block['n_misclassified']} of "
            f"{block['n_players']}, {block['n_overlap']} also hybrid "
            f"({block['overlap_rate_of_misclassified']:.1%}, expected "
            f"{block['expected_overlap_if_independent']:.1f}, enrichment "
            f"{block['enrichment_ratio']:.2f}, Fisher p {block['fisher_p_one_sided']:.4f})",
            flush=True,
        )
    finishing_payload = finishing["payload"]
    leak = finishing_payload["leakage"]
    print(
        f"{season}: finishing-above-role ridge cross validated R squared "
        f"{finishing_payload['cv_r2_out_of_fold']:.3f} on the specified features "
        f"(fold mean {finishing_payload['cv_r2_fold_mean']:.3f}, "
        f"sd {finishing_payload['cv_r2_fold_sd']:.3f}). xGChain is excluded because it "
        f"correlates with the target at "
        f"{leak['corr_chain_minus_buildup_with_target']:.3f}; adding it back inflates the "
        f"fit to {leak['cv_r2_with_xg_chain_added_back']:.3f}. Neither leaderboard "
        "measures finishing ability.",
        flush=True,
    )
    print("supervised complete")


if __name__ == "__main__":
    main()
