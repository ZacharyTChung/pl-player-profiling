"""Phase 8b. Goalkeepers, rebuilt on the pre-withdrawal archive.

Why this module exists alongside ``src/keepers.py``
---------------------------------------------------
``src/keepers.py`` analyses the live 2024-25 and 2025-26 pulls. In those pulls 23 of the
26 columns of FBref's advanced goalkeeping table are served with a header and no values,
so that module can only measure shot stopping by volume and workload, and it says so at
length. Every quantity it calls a proxy is a proxy because the data behind the real
quantity had been deleted.

This module runs the goalkeeper analysis the study actually specified, on the archive
described in ``src/archive.py``. The mirror predates the January 2026 withdrawal, so the
whole advanced block is present: post-shot expected goals, PSxG per shot on target, PSxG
minus goals allowed, cross stop rate, sweeper actions and their average distance, launch
percentage and average pass length. Coverage is audited on load rather than assumed, and
the audit is written to ``results/metrics/archive_keeper_availability.json`` because the
contrast between it and the post-withdrawal audit is one of the paper's claims.

Population
----------
Basic and advanced keeper tables joined on season, squad and player, restricted to
``archive.SEASONS`` (2018 to 2022, labelled by the year the season ends). 2023 is
excluded for the reason given in ``src/archive.py``: FBref restructured other tables that
season and the mirror did not follow the renames, so keeping it would make the archive
inconsistent across analyses. That leaves 1,039 keeper-seasons across the five major
European leagues, of which those with at least ``config.MIN_MINUTES`` minutes form the
analysis set.

Derived quantities
------------------
``gk_nineties`` is the keeper's own minutes divided by 90. Counting statistics are
divided by it, so a keeper who played half a season is compared on the same footing as
one who played all of it.

* ``gk_sota_p90``      shots on target against per 90
* ``gk_psxg_p90``      post-shot expected goals faced per 90
* ``gk_psxg_net_p90``  PSxG minus goals allowed per 90

Shot stopping, done properly
----------------------------
``gk_psxg_net_p90`` supersedes ``keepers.gk_saves_above_volume``, the volume-based proxy
the post-withdrawal module was forced to use. That proxy was the residual of a regression
of total saves on total shots on target faced, so it adjusted for how many shots a keeper
faced and not at all for how hard they were. A keeper whose opponents shot from poor
positions scored well on it through no skill of his own. PSxG minus goals allowed adjusts
for the placement, distance and type of every shot on target faced, which is the quantity
the proxy was standing in for. Both a season total, in goals, and a per 90 rate are
reported, because the total rewards playing time and the rate does not.

The five composite axes
-----------------------
Each is the mean of the z-scores of its components, standardised over the eligible pool.
Signs are stated because two of them are not obvious.

``shot_stopping``   ``gk_psxg_net_p90`` + ``gk_save_pct`` + ``gk_psxg_per_sot``
    PSxG per shot on target enters with a POSITIVE sign. It measures the difficulty of
    the shots a keeper faced, not his response to them, and raw save percentage is
    deflated by exactly that difficulty (the two correlate -0.67 here). Adding it back
    offsets the deflation, which is what a shot-quality adjustment is for. The choice is
    checked rather than asserted: with the positive sign the composite correlates 0.959
    with PSxG minus goals allowed per 90 and -0.241 with workload, and with the negative
    sign only 0.753 with PSxG net and -0.453 with workload. The positive sign therefore
    tracks the gold-standard measure more closely and carries less team strength.

``sweeping``        ``gk_sweeper_per90`` + ``gk_sweeper_distance``
    Higher means more defensive actions outside the penalty area, further from goal.

``distribution``    -``gk_launch_pct`` + -``gk_pass_length``
    Both components are NEGATED, so higher means SHORTER distribution: fewer passes
    longer than forty yards and a lower average pass length. The convention is chosen so
    that the axis reads as "plays out from the back", which is the way the role is
    normally described. A keeper at +1 on this axis is a short distributor, not a long
    one.

``cross_handling``  ``gk_crosses_stopped_pct``
    A single component, kept as its own axis because the study specified it as one and
    because it turns out to carry the most cluster separation of the five.

``workload``        ``gk_sota_p90`` + ``gk_goals_against_per90``
    Higher means busier and more often beaten. This is an exposure axis: it describes the
    defence in front of the keeper at least as much as the keeper, and it is treated as
    situational throughout.

Clustering, and the validation that sank the earlier attempt
------------------------------------------------------------
Two feature spaces are fitted and both are reported in full, following the convention of
``src/cluster.py``, which fits and reports a standardised and a PCA space side by side.

``full``
    All twelve keeper features. This is the space the study originally specified for
    keepers and the direct analogue of the outfield pipeline, so it carries the headline
    partition.
``technique``
    The four technique composites, which is the full set of five with ``workload``
    removed. Workload is exposure rather than behaviour, so a partition that excludes it
    is asking a different and narrower question: do keepers separate by what they do.

Both spaces are reduced by PCA to ``config.PCA_VARIANCE_TARGET`` of the variance and
clustered with KMeans, with k chosen by ``config.K_RULE`` over k = 2 to 8. The cap of 8
replaces ``config.K_MAX`` of 12: with 693 eligible keeper-seasons and at most twelve
features, twelve groups would leave cells too small for a centroid to mean anything, and
the silhouette curve is flat and falling well before eight in both spaces.

The diagnostic that sank the earlier goalkeeper clustering is then repeated on both. For
each partition:

* the eta squared of every feature against the cluster labels, averaged within a
  situational block (goals against per 90, shots on target against per 90, PSxG faced per
  90, PSxG per shot on target, clean sheet percentage) and within a technique block (PSxG
  net per 90, save percentage, sweeper actions per 90, sweeper distance, cross stop rate,
  launch percentage, average pass length);
* multinomial logistic regressions of cluster membership on each block alone, reported as
  McFadden pseudo R squared and as five-fold cross-validated accuracy, which is the
  "variance in cluster membership explained" quantity stated the other way round;
* the same two regressions restricted to the three variables named in the study design,
  goals against and shots faced against PSxG net, sweeper distance and launch percentage.

Archetypes are named only from a partition that passes this test, and the condition is
recorded in the output rather than left implicit.

Outputs
-------
``results/metrics/archive_keeper_availability.json``
``results/metrics/archive_keeper_composites.json``
``results/metrics/archive_keeper_clusters.json``
``results/metrics/archive_keeper_archetypes.json``
``figures/archive_keeper_shot_stopping``, ``archive_keeper_clusters``,
``archive_keeper_composites``, ``archive_keeper_radar``, ``archive_keeper_corr``

Run with ``uv run python -m src.archive_keepers``.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import config
from src import archive, plotting
from src import cluster as C

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

RAW = config.DATA_RAW / "archive"

#: The join key for the two keeper tables. Verified unique in both.
KEY = ["Season_End_Year", "Squad", "Player"]

#: Carried through for context and for labelling figures.
IDS = ["Season_End_Year", "Squad", "Comp", "Player", "Nation", "Pos", "Age"]

#: Identifier columns of the raw tables, excluded from the emptiness audit so that the
#: counts refer to statistics rather than to names and links.
RAW_IDS = {"Season_End_Year", "Squad", "Comp", "Player", "Nation", "Pos", "Age", "Born", "Url"}

#: Season totals converted to per 90 with the keeper's own minutes.
COUNTS_TO_P90 = {
    "gk_sota_p90": "gk_shots_on_target_against",
    "gk_psxg_p90": "gk_psxg",
    "gk_psxg_net_p90": "gk_psxg_net",
}

#: The twelve features of the ``full`` clustering space.
FEATURES = [
    "gk_save_pct",
    "gk_clean_sheet_pct",
    "gk_goals_against_per90",
    "gk_sota_p90",
    "gk_psxg_p90",
    "gk_psxg_per_sot",
    "gk_psxg_net_p90",
    "gk_crosses_stopped_pct",
    "gk_sweeper_per90",
    "gk_sweeper_distance",
    "gk_launch_pct",
    "gk_pass_length",
]

#: Short names, used on the heatmap and the radar so the tick labels do not clip.
SHORT = {
    "gk_save_pct": "save %",
    "gk_clean_sheet_pct": "clean sheet %",
    "gk_goals_against_per90": "GA/90",
    "gk_sota_p90": "SoTA/90",
    "gk_psxg_p90": "PSxG/90",
    "gk_psxg_per_sot": "PSxG/SoT",
    "gk_psxg_net_p90": "PSxG net/90",
    "gk_crosses_stopped_pct": "crosses stopped %",
    "gk_sweeper_per90": "sweeps/90",
    "gk_sweeper_distance": "sweep distance",
    "gk_launch_pct": "launch %",
    "gk_pass_length": "pass length",
}

#: Exposure. What the defence in front of the keeper handed him.
SITUATION_BLOCK = [
    "gk_goals_against_per90",
    "gk_sota_p90",
    "gk_psxg_p90",
    "gk_psxg_per_sot",
    "gk_clean_sheet_pct",
]

#: Behaviour. What the keeper did with it.
TECHNIQUE_BLOCK = [
    "gk_psxg_net_p90",
    "gk_save_pct",
    "gk_sweeper_per90",
    "gk_sweeper_distance",
    "gk_crosses_stopped_pct",
    "gk_launch_pct",
    "gk_pass_length",
]

#: The three variables of each kind named in the study design, used for a second, narrower
#: version of the same comparison.
SITUATION_CORE = ["gk_goals_against_per90", "gk_sota_p90"]
TECHNIQUE_CORE = ["gk_psxg_net_p90", "gk_sweeper_distance", "gk_launch_pct"]

#: composite -> list of (feature, sign). See the module docstring for the sign argument.
COMPOSITES: dict[str, list[tuple[str, int]]] = {
    "shot_stopping": [("gk_psxg_net_p90", 1), ("gk_save_pct", 1), ("gk_psxg_per_sot", 1)],
    "sweeping": [("gk_sweeper_per90", 1), ("gk_sweeper_distance", 1)],
    "distribution": [("gk_launch_pct", -1), ("gk_pass_length", -1)],
    "cross_handling": [("gk_crosses_stopped_pct", 1)],
    "workload": [("gk_sota_p90", 1), ("gk_goals_against_per90", 1)],
}
COMPOSITE_NAMES = list(COMPOSITES)
TECHNIQUE_COMPOSITES = [c for c in COMPOSITE_NAMES if c != "workload"]

COMPOSITE_DIRECTION = {
    "shot_stopping": "higher is a better shot stopper for the difficulty of shots faced",
    "sweeping": "higher is more, and further from goal, defensive action outside the box",
    "distribution": "higher is shorter distribution, fewer launches and a lower average length",
    "cross_handling": "higher is a larger share of crosses faced that were stopped",
    "workload": "higher is busier and more often beaten, an exposure axis",
}

#: The two clustering spaces. The first carries the headline partition.
SPACES: dict[str, list[str]] = {"full": FEATURES, "technique": TECHNIQUE_COMPOSITES}
PRIMARY_SPACE = "full"

#: k is chosen over this range rather than over ``config.K_RANGE``. See the docstring.
K_MAX_KEEPERS = 8
K_RANGE = list(range(config.K_MIN, K_MAX_KEEPERS + 1))

KMEANS_N_INIT = 25

#: A block is said to drive the partition when its mean eta squared is at least this
#: multiple of the other block's. Declared here rather than chosen after the fact.
BLOCK_DOMINANCE_RATIO = 1.5

N_RANKED = 10
CV_FOLDS = 5


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _r(x: float, nd: int = 6) -> float:
    return float(np.round(float(x), nd))


def _write_json(payload: dict, name: str) -> None:
    path = config.METRICS / name
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=C._native, sort_keys=False)
    print(f"  wrote {path.relative_to(config.ROOT)}")


def _label(row: pd.Series, *, short: bool = False, compact: bool = False) -> str:
    """A figure label. ``short`` drops the given names, ``compact`` also shortens the year."""
    player = str(row["Player"])
    if short or compact:
        player = player.split(" ")[-1] if " " in player else player
    season = int(row["Season_End_Year"])
    stamp = f"{season % 100:02d}" if compact else str(season)
    return f"{player} ({row['Squad']} {stamp})"


def _zscore(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    values = frame[cols].to_numpy(float)
    return pd.DataFrame(StandardScaler().fit_transform(values), columns=cols, index=frame.index)


def _eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """Share of the variance of ``values`` lying between the clusters."""
    values = np.asarray(values, dtype=float)
    grand = values.mean()
    total = float(((values - grand) ** 2).sum())
    if total <= 0:
        return 0.0
    between = sum(
        ((values[labels == c].mean() - grand) ** 2) * int((labels == c).sum())
        for c in np.unique(labels)
    )
    return float(between / total)


# --------------------------------------------------------------------------------------
# Loading and the coverage audit
# --------------------------------------------------------------------------------------


def _read(name: str) -> pd.DataFrame:
    frame = pd.read_parquet(RAW / f"{name}.parquet")
    frame["Season_End_Year"] = pd.to_numeric(frame["Season_End_Year"], errors="coerce")
    return frame[frame["Season_End_Year"].isin(archive.SEASONS)].copy()


def load_keepers() -> tuple[pd.DataFrame, dict]:
    """Join the two keeper tables, verify every mapped column, and return the audit.

    Every column named in ``archive.KEEPER_SOURCES`` is checked for presence before the
    join and for non-null coverage after it. A missing column raises rather than being
    silently dropped, because a silently dropped goalkeeping column is exactly the failure
    mode this whole module exists to document.
    """
    basic = _read("keepers")
    advanced = _read("keepers_adv")

    by_table = {"keepers": basic, "keepers_adv": advanced}
    absent = {
        canon: f"{table}.{column}"
        for canon, (table, column) in archive.KEEPER_SOURCES.items()
        if column not in by_table[table].columns
    }
    if absent:
        raise KeyError(f"archive keeper columns absent: {absent}")

    for name, frame in by_table.items():
        duplicated = int(frame.duplicated(KEY).sum())
        if duplicated:
            raise ValueError(f"{name} has {duplicated} duplicate season-squad-player rows")

    merged = basic[IDS].copy()
    for table, frame in by_table.items():
        rename = {
            column: canon
            for canon, (source, column) in archive.KEEPER_SOURCES.items()
            if source == table
        }
        merged = merged.merge(frame[KEY + list(rename)].rename(columns=rename), on=KEY, how="left")

    for column in merged.columns:
        if column not in {"Squad", "Comp", "Player", "Nation", "Pos"}:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")

    audit = {
        "source": "worldfootballR mirror of FBref, scraped before the January 2026 withdrawal",
        "seasons": archive.SEASONS,
        "excluded_seasons": {str(k): v for k, v in archive.EXCLUDED_SEASONS.items()},
        "leagues": sorted(merged["Comp"].dropna().unique().tolist()),
        "keeper_seasons": int(len(merged)),
        "rows_basic_table": int(len(basic)),
        "rows_advanced_table": int(len(advanced)),
        "rows_with_no_advanced_match": int(merged["gk_psxg"].isna().sum()),
        "raw_table_coverage": {
            name: _raw_table_coverage(frame) for name, frame in by_table.items()
        },
        "canonical_features": {
            canon: {
                "table": table,
                "column": column,
                "non_null": int(merged[canon].notna().sum()),
                "coverage": _r(float(merged[canon].notna().mean()), 4),
            }
            for canon, (table, column) in archive.KEEPER_SOURCES.items()
        },
        "all_canonical_columns_present": True,
    }
    return merged, audit


def _raw_table_coverage(frame: pd.DataFrame) -> dict:
    """Non-null counts for every statistic column of a raw keeper table."""
    stats = [c for c in frame.columns if c not in RAW_IDS]
    counts = {c: int(pd.to_numeric(frame[c], errors="coerce").notna().sum()) for c in stats}
    return {
        "rows": int(len(frame)),
        "stat_columns": len(stats),
        "columns_fully_empty": sorted(c for c, n in counts.items() if n == 0),
        "columns_with_any_value": int(sum(1 for n in counts.values() if n > 0)),
        "min_coverage": _r(min(counts.values()) / max(len(frame), 1), 4),
        "non_null_counts": counts,
    }


def prepare(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add per-90 rates and the composites, and return the eligible pool."""
    frame = merged.copy()
    frame["gk_nineties"] = frame["gk_minutes"] / 90.0
    for target, source in COUNTS_TO_P90.items():
        frame[target] = frame[source] / frame["gk_nineties"]

    eligible = frame[frame["gk_minutes"] >= config.MIN_MINUTES].copy()
    eligible = eligible.dropna(subset=FEATURES).reset_index(drop=True)

    z = _zscore(eligible, FEATURES)
    for name, parts in COMPOSITES.items():
        eligible[name] = np.mean([sign * z[col] for col, sign in parts], axis=0)

    report = {
        "min_minutes": config.MIN_MINUTES,
        "keeper_seasons": int(len(frame)),
        "eligible": int(len(eligible)),
        "eligible_share": _r(len(eligible) / len(frame), 4),
        "eligible_by_season": {
            str(int(k)): int(v)
            for k, v in eligible["Season_End_Year"].value_counts().sort_index().items()
        },
        "eligible_by_league": eligible["Comp"].value_counts().to_dict(),
        "feature_coverage_in_eligible_pool": {
            f: _r(float(eligible[f].notna().mean()), 4) for f in FEATURES
        },
        "unique_keepers": int(eligible["Player"].nunique()),
    }
    return eligible, report


# --------------------------------------------------------------------------------------
# Shot stopping
# --------------------------------------------------------------------------------------


def shot_stopping(eligible: pd.DataFrame) -> dict:
    """PSxG minus goals allowed, the measure the post-withdrawal module could not build.

    ``keepers.gk_saves_above_volume`` regressed total saves on total shots on target
    faced and called the residual saves above expectation. It adjusted for how many shots
    a keeper faced and not for how hard any of them was. This function supersedes it:
    PSxG minus goals allowed is FBref's own shot-quality adjusted figure, computed from
    the placement and difficulty of every shot on target faced, and it is reported here
    both as a season total in goals and as a per 90 rate.
    """
    totals = eligible["gk_psxg_net"]
    rates = eligible["gk_psxg_net_p90"]

    def _rank(column: str, ascending: bool) -> list[dict]:
        ordered = eligible.sort_values(column, ascending=ascending).head(N_RANKED)
        return [
            {
                "player": row["Player"],
                "squad": row["Squad"],
                "league": row["Comp"],
                "season_end_year": int(row["Season_End_Year"]),
                "psxg_net": _r(row["gk_psxg_net"], 2),
                "psxg_net_p90": _r(row["gk_psxg_net_p90"], 4),
                "psxg_faced": _r(row["gk_psxg"], 2),
                "goals_against": _r(row["gk_psxg"] - row["gk_psxg_net"], 2),
                "save_pct": _r(row["gk_save_pct"], 1),
                "minutes": int(row["gk_minutes"]),
            }
            for _, row in ordered.iterrows()
        ]

    def _describe(series: pd.Series) -> dict:
        return {
            "n": int(series.notna().sum()),
            "mean": _r(series.mean(), 4),
            "sd": _r(series.std(ddof=1), 4),
            "min": _r(series.min(), 4),
            "p05": _r(series.quantile(0.05), 4),
            "q1": _r(series.quantile(0.25), 4),
            "median": _r(series.median(), 4),
            "q3": _r(series.quantile(0.75), 4),
            "p95": _r(series.quantile(0.95), 4),
            "max": _r(series.max(), 4),
            "share_positive": _r(float((series > 0).mean()), 4),
        }

    return {
        "definition": (
            "post-shot expected goals faced minus goals allowed, in goals. Positive means "
            "the keeper conceded fewer goals than the difficulty of the shots on target he "
            "faced implied."
        ),
        "supersedes": (
            "keepers.gk_saves_above_volume, a residual of total saves regressed on total "
            "shots on target faced, which adjusted for shot count and not shot difficulty."
        ),
        "distribution_total": _describe(totals),
        "distribution_per90": _describe(rates),
        "correlation_total_with_per90": _r(float(totals.corr(rates)), 4),
        "correlation_per90_with_minutes": _r(float(rates.corr(eligible["gk_minutes"])), 4),
        "leaders_total": _rank("gk_psxg_net", ascending=False),
        "laggards_total": _rank("gk_psxg_net", ascending=True),
        "leaders_per90": _rank("gk_psxg_net_p90", ascending=False),
        "laggards_per90": _rank("gk_psxg_net_p90", ascending=True),
    }


# --------------------------------------------------------------------------------------
# Composites
# --------------------------------------------------------------------------------------


def composite_report(eligible: pd.DataFrame) -> dict:
    """Definitions, distributions, correlations and the sign check on shot stopping."""
    z = _zscore(eligible, FEATURES)
    checks = {}
    for sign, tag in ((1, "positive"), (-1, "negative")):
        candidate = (z["gk_psxg_net_p90"] + z["gk_save_pct"] + sign * z["gk_psxg_per_sot"]) / 3.0
        checks[tag] = {
            "corr_with_psxg_net_p90": _r(float(candidate.corr(z["gk_psxg_net_p90"])), 4),
            "corr_with_workload": _r(float(candidate.corr(eligible["workload"])), 4),
            "corr_with_goals_against_p90": _r(
                float(candidate.corr(z["gk_goals_against_per90"])), 4
            ),
        }

    values = eligible[COMPOSITE_NAMES]
    return {
        "definitions": {
            name: {
                "components": [
                    {"feature": col, "sign": sign, "source": archive.KEEPER_SOURCES.get(col)}
                    for col, sign in parts
                ],
                "construction": "unweighted mean of the component z-scores over the eligible pool",
                "direction": COMPOSITE_DIRECTION[name],
            }
            for name, parts in COMPOSITES.items()
        },
        "sign_convention_notes": {
            "distribution": (
                "launch percentage and average pass length both enter negated, so higher "
                "means shorter distribution. A keeper at +1 plays out from the back."
            ),
            "shot_stopping": (
                "PSxG per shot on target enters positively as a difficulty adjustment to "
                "raw save percentage, which it correlates with at "
                f"{_r(float(z['gk_save_pct'].corr(z['gk_psxg_per_sot'])), 3)}."
            ),
            "shot_stopping_sign_check": checks,
            "shot_stopping_sign_chosen": "positive",
        },
        "summary": {
            name: {
                "mean": _r(values[name].mean(), 6),
                "sd": _r(values[name].std(ddof=1), 4),
                "min": _r(values[name].min(), 4),
                "median": _r(values[name].median(), 4),
                "max": _r(values[name].max(), 4),
            }
            for name in COMPOSITE_NAMES
        },
        "correlations": {
            a: {b: _r(float(values[a].corr(values[b])), 4) for b in COMPOSITE_NAMES}
            for a in COMPOSITE_NAMES
        },
        "feature_correlations": {
            a: {b: _r(float(eligible[a].corr(eligible[b])), 4) for b in FEATURES} for a in FEATURES
        },
        "extremes": {
            name: {
                "highest": [_label(row) for _, row in eligible.nlargest(5, name).iterrows()],
                "lowest": [_label(row) for _, row in eligible.nsmallest(5, name).iterrows()],
            }
            for name in COMPOSITE_NAMES
        },
        "values": [
            {
                "player": row["Player"],
                "squad": row["Squad"],
                "season_end_year": int(row["Season_End_Year"]),
                **{name: _r(row[name], 4) for name in COMPOSITE_NAMES},
            }
            for _, row in eligible.iterrows()
        ],
    }


# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------


def fit_space(eligible: pd.DataFrame, columns: list[str], tag: str) -> dict:
    """PCA to the variance target, then KMeans with k from ``config.K_RULE``."""
    matrix = StandardScaler().fit_transform(eligible[columns].to_numpy(float))
    pca = PCA(n_components=config.PCA_VARIANCE_TARGET, random_state=config.RANDOM_STATE)
    projected = pca.fit_transform(matrix)

    curves: dict[str, dict[int, float]] = {
        "silhouette": {},
        "davies_bouldin": {},
        "calinski_harabasz": {},
    }
    for k in K_RANGE:
        labels = (
            KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=C._seed(tag, k))
            .fit(projected)
            .labels_
        )
        curves["silhouette"][k] = _r(silhouette_score(projected, labels))
        curves["davies_bouldin"][k] = _r(davies_bouldin_score(projected, labels))
        curves["calinski_harabasz"][k] = _r(calinski_harabasz_score(projected, labels))
    gaps, gap_se, _ = C.gap_statistic(projected, K_RANGE, tag)
    curves["gap"] = {k: _r(v) for k, v in gaps.items()}
    curves["gap_se"] = {k: _r(v) for k, v in gap_se.items()}

    selection = C.apply_k_rule(curves)
    k = selection["chosen_k"]
    labels = (
        KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=C._seed(tag, k))
        .fit(projected)
        .labels_
    )

    stability = C.bootstrap_stability(projected, labels, k, tag)
    stability.pop("coassignment", None)

    return {
        "space": tag,
        "columns": columns,
        "n": int(projected.shape[0]),
        "n_components": int(projected.shape[1]),
        "explained_variance_ratio": [_r(v, 4) for v in pca.explained_variance_ratio_],
        "explained_variance_total": _r(float(pca.explained_variance_ratio_.sum()), 4),
        "k_range": K_RANGE,
        "k_range_note": (
            f"capped at {K_MAX_KEEPERS} rather than config.K_MAX of {config.K_MAX} because "
            f"{projected.shape[0]} keeper-seasons cannot support twelve centroids"
        ),
        "curves": curves,
        "selection": selection,
        "chosen_k": int(k),
        "silhouette": curves["silhouette"][k],
        "davies_bouldin": curves["davies_bouldin"][k],
        "calinski_harabasz": curves["calinski_harabasz"][k],
        "cluster_sizes": {str(c): int((labels == c).sum()) for c in range(k)},
        "stability": stability,
        "labels": labels,
        "projected": projected,
    }


def situation_versus_technique(eligible: pd.DataFrame, labels: np.ndarray) -> dict:
    """The diagnostic that sank the earlier goalkeeper clustering, repeated here.

    Two directions on the same question. ``eta_squared`` asks how much of each feature's
    variance lies between clusters, averaged within each block. The logistic regressions
    ask the question the other way round, how much of the variation in cluster membership
    a block accounts for, reported as McFadden pseudo R squared on the full sample and as
    cross-validated accuracy so that an over-fitted block cannot win on the first measure
    alone.
    """
    per_feature = {f: _r(_eta_squared(eligible[f].to_numpy(), labels), 4) for f in FEATURES}
    situation_eta = float(np.mean([per_feature[f] for f in SITUATION_BLOCK]))
    technique_eta = float(np.mean([per_feature[f] for f in TECHNIQUE_BLOCK]))

    def _explain(columns: list[str]) -> dict:
        matrix = StandardScaler().fit_transform(eligible[columns].to_numpy(float))
        model = LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE)
        model.fit(matrix, labels)
        probability = model.predict_proba(matrix)[np.arange(len(labels)), labels]
        log_likelihood = float(np.log(np.clip(probability, 1e-12, None)).sum())
        share = np.bincount(labels) / len(labels)
        null_log_likelihood = float(np.log(share[labels]).sum())
        folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        accuracy = cross_val_score(
            LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE),
            matrix,
            labels,
            cv=folds,
        )
        return {
            "variables": columns,
            "mcfadden_pseudo_r2": _r(1.0 - log_likelihood / null_log_likelihood, 4),
            "cv_accuracy": _r(float(accuracy.mean()), 4),
            "cv_accuracy_sd": _r(float(accuracy.std(ddof=1)), 4),
        }

    baseline = _r(float(np.bincount(labels).max() / len(labels)), 4)
    blocks = {
        "situation_full": _explain(SITUATION_BLOCK),
        "technique_full": _explain(TECHNIQUE_BLOCK),
        "situation_core": _explain(SITUATION_CORE),
        "technique_core": _explain(TECHNIQUE_CORE),
    }
    ratio = technique_eta / situation_eta if situation_eta > 0 else float("inf")
    if ratio >= BLOCK_DOMINANCE_RATIO:
        verdict = "technique"
    elif ratio <= 1.0 / BLOCK_DOMINANCE_RATIO:
        verdict = "situation"
    else:
        verdict = "neither"

    return {
        "question": (
            "does cluster membership track the situation the keeper was placed in, or what "
            "the keeper did"
        ),
        "situation_block": SITUATION_BLOCK,
        "technique_block": TECHNIQUE_BLOCK,
        "eta_squared_by_feature": per_feature,
        "mean_eta_squared_situation": _r(situation_eta, 4),
        "mean_eta_squared_technique": _r(technique_eta, 4),
        "technique_to_situation_ratio": _r(ratio, 4),
        "dominance_ratio_threshold": BLOCK_DOMINANCE_RATIO,
        "majority_class_baseline_accuracy": baseline,
        "membership_explained": blocks,
        "verdict": verdict,
        "supports_archetypes": verdict == "technique",
    }


# --------------------------------------------------------------------------------------
# Archetypes
# --------------------------------------------------------------------------------------


def _profile(eligible: pd.DataFrame, labels: np.ndarray, cluster: int) -> dict:
    members = eligible[labels == cluster]
    return {
        "n": int(len(members)),
        "share": _r(len(members) / len(eligible), 4),
        "composites": {c: _r(members[c].mean(), 4) for c in COMPOSITE_NAMES},
        "raw_means": {f: _r(members[f].mean(), 3) for f in FEATURES},
    }


def _exemplars(
    eligible: pd.DataFrame, labels: np.ndarray, projected: np.ndarray, cluster: int, n: int = 6
) -> list[str]:
    """The members closest to their own centroid, which is what a centroid describes."""
    members = np.flatnonzero(labels == cluster)
    centre = projected[members].mean(axis=0)
    order = members[np.argsort(np.linalg.norm(projected[members] - centre, axis=1))[:n]]
    return [_label(eligible.iloc[i]) for i in order]


def name_archetypes(eligible: pd.DataFrame, fit: dict, validation: dict) -> dict:
    """Derive names from the composite profiles, not from football expectation.

    Each cluster is named from the composite axes on which it is furthest from zero, with
    the value quoted in the justification. No name is assigned unless the validation says
    the partition tracks technique, and the label vocabulary is not fixed in advance, so
    the familiar shot stopper, sweeper keeper and distributor trio appears only if the
    profiles produce it.
    """
    if not validation["supports_archetypes"]:
        return {
            "named": False,
            "reason": (
                "the validation reports that cluster membership tracks "
                f"{validation['verdict']} rather than goalkeeping technique, so the "
                "partition does not describe archetypes and none are named"
            ),
            "verdict": validation["verdict"],
        }

    labels = fit["labels"]
    projected = fit["projected"]
    records = {}
    for cluster in range(fit["chosen_k"]):
        profile = _profile(eligible, labels, cluster)
        ordered = sorted(profile["composites"].items(), key=lambda kv: abs(kv[1]), reverse=True)
        driver, driver_value = ordered[0]
        second, second_value = ordered[1]
        name = _name_from_profile(profile["composites"])
        raw = profile["raw_means"]
        pool = {f: _r(eligible[f].mean(), 3) for f in FEATURES}
        records[str(cluster)] = {
            "name": name,
            "n": profile["n"],
            "share": profile["share"],
            "composites": profile["composites"],
            "leading_axes": [
                {"axis": driver, "z": driver_value},
                {"axis": second, "z": second_value},
            ],
            "justification": (
                f"{profile['n']} keeper-seasons. The profile is led by {driver} at "
                f"{driver_value:+.2f} standard deviations and {second} at "
                f"{second_value:+.2f}. In football units the cluster stops "
                f"{raw['gk_crosses_stopped_pct']} percent of the crosses it faces against "
                f"{pool['gk_crosses_stopped_pct']} for the pool, makes "
                f"{raw['gk_sweeper_per90']} defensive actions outside the area per 90 "
                f"against {pool['gk_sweeper_per90']}, at an average "
                f"{raw['gk_sweeper_distance']} yards from goal against "
                f"{pool['gk_sweeper_distance']}, launches {raw['gk_launch_pct']} percent of "
                f"its passes against {pool['gk_launch_pct']}, and runs a PSxG minus goals "
                f"allowed of {raw['gk_psxg_net_p90']:+.3f} per 90 against "
                f"{pool['gk_psxg_net_p90']:+.3f}."
            ),
            "exemplars": _exemplars(eligible, labels, projected, cluster),
            "raw_means": profile["raw_means"],
            "pool_means": pool,
        }
    return {
        "named": True,
        "space": fit["space"],
        "k": fit["chosen_k"],
        "silhouette": fit["silhouette"],
        "bootstrap_ari_mean": fit["stability"]["ari_mean"],
        "verdict": validation["verdict"],
        "condition": (
            "named because the situation versus technique validation on this partition "
            f"returned mean eta squared {validation['mean_eta_squared_technique']} for the "
            f"technique block against {validation['mean_eta_squared_situation']} for the "
            "situational block"
        ),
        "vocabulary_note": (
            "names are read off the composite axes each cluster departs from zero on. The "
            "football expectation of a shot stopper, a sweeper keeper and a distributor was "
            "not imposed, and the partition does not reproduce it: shot stopping and "
            "distribution separate the clusters far less than off-line activity does."
        ),
        "archetypes": records,
    }


def _name_from_profile(composites: dict[str, float]) -> str:
    """A short name built from the axes the cluster actually departs from zero on."""
    off_line = (composites["sweeping"] + composites["cross_handling"]) / 2.0
    words = []
    if off_line >= 0.3:
        words.append("Sweeper claimer")
    elif off_line <= -0.3:
        words.append("Line keeper")
    else:
        words.append("Neutral positioner")

    if composites["distribution"] >= 0.35:
        words.append("short distributor")
    elif composites["distribution"] <= -0.35:
        words.append("long distributor")

    if composites["shot_stopping"] >= 0.35:
        words.append("strong shot stopper")
    elif composites["shot_stopping"] <= -0.35:
        words.append("weak shot stopper")
    return ", ".join(words)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _spread_labels(
    ax: plt.Axes,
    points: list[tuple[float, float, str]],
    *,
    max_labels: int = 10,
    x_gap: float = 0.34,
    y_gap: float = 0.075,
) -> None:
    """Direct-label a few points, rejecting any label that would sit on an accepted one.

    Candidates arrive in priority order. The exclusion zone is an axes-fraction rectangle
    rather than a circle because a label is far wider than it is tall, so the horizontal
    and vertical gaps that avoid a collision are not the same number. Labels are capped
    hard: this figure carries several hundred points and only a handful can be named.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    accepted: list[tuple[float, float]] = []
    for x, y, text in points:
        if len(accepted) >= max_labels:
            return
        fx = (x - x0) / (x1 - x0)
        fy = (y - y0) / (y1 - y0)
        if not (0.02 <= fx <= 0.98 and 0.02 <= fy <= 0.98):
            continue
        flip = fx > 0.55
        if any(abs(fx - px) < x_gap and abs(fy - py) < y_gap for px, py in accepted):
            continue
        accepted.append((fx, fy))
        ax.annotate(
            text,
            (x, y),
            textcoords="offset points",
            xytext=(-4.0, 4.0) if flip else (4.0, 4.0),
            ha="right" if flip else "left",
            fontsize=plotting.BASE_FONT_PT - 2.5,
            color=plotting.INK_PRIMARY,
            zorder=7,
        )


def figure_shot_stopping(eligible: pd.DataFrame) -> None:
    """Distribution of PSxG minus goals allowed, and the season-total ranking."""
    fig, axes = plt.subplots(
        1, 2, figsize=(plotting.WIDTH_WIDE, 4.6), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )
    ax = axes[0]
    values = eligible["gk_psxg_net"]
    ax.hist(values, bins=32, color=plotting.CATEGORICAL[0], alpha=0.85, edgecolor=plotting.SURFACE)
    ax.axvline(0.0, color=plotting.INK_PRIMARY, lw=1.0, ls=(0, (3, 2)))
    ax.axvline(values.mean(), color=plotting.CATEGORICAL[1], lw=1.2)
    ax.annotate(
        f"mean {values.mean():+.2f}\nsd {values.std(ddof=1):.2f}\n"
        f"{100 * (values > 0).mean():.0f}% above zero",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        fontsize=plotting.BASE_FONT_PT - 1,
        color=plotting.INK_SECONDARY,
    )
    plotting.style_axis(
        ax,
        xlabel="PSxG minus goals allowed (goals per season)",
        ylabel="keeper-seasons",
        title="Saves above expectation",
    )

    ax = axes[1]
    # Position 0 is the bottom of a horizontal bar chart, so the worst is listed first and
    # the ranking then reads monotonically from the best at the top to the worst at the
    # bottom rather than restarting in the middle.
    worst_first = eligible.nsmallest(N_RANKED, "gk_psxg_net")
    best_last = eligible.nlargest(N_RANKED, "gk_psxg_net").iloc[::-1]
    ranked = pd.concat([worst_first, best_last])
    positions = np.arange(len(ranked))
    colours = [
        plotting.CATEGORICAL[0] if v > 0 else plotting.CATEGORICAL[1] for v in ranked["gk_psxg_net"]
    ]
    ax.barh(positions, ranked["gk_psxg_net"], color=colours, height=0.72)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [_label(row, short=True) for _, row in ranked.iterrows()],
        fontsize=plotting.BASE_FONT_PT - 2.5,
    )
    ax.axvline(0.0, color=plotting.INK_PRIMARY, lw=0.8)
    ax.set_ylim(-0.8, len(ranked) - 0.2)
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    for pos, value in zip(positions, ranked["gk_psxg_net"], strict=True):
        ax.annotate(
            f"{value:+.1f}",
            (value, pos),
            textcoords="offset points",
            xytext=(3 if value > 0 else -3, 0),
            ha="left" if value > 0 else "right",
            va="center",
            fontsize=plotting.BASE_FONT_PT - 2.5,
            color=plotting.INK_SECONDARY,
        )
    span = float(np.abs(ranked["gk_psxg_net"]).max())
    ax.set_xlim(-span * 1.28, span * 1.28)
    plotting.style_axis(
        ax,
        xlabel="PSxG minus goals allowed (goals)",
        title=f"Best and worst {N_RANKED}, {archive.SEASONS[0]} to {archive.SEASONS[-1]}",
    )
    plotting.panel_labels(axes)
    plotting.save_figure(fig, "archive_keeper_shot_stopping")


def figure_clusters(eligible: pd.DataFrame, fits: dict[str, dict], validation: dict) -> None:
    """The two partitions in their own principal component planes, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(plotting.WIDTH_FULL, 3.1))
    titles = {
        "full": "all twelve keeper features",
        "technique": "technique composites only",
    }
    for ax, (space, fit) in zip(axes, fits.items(), strict=True):
        projected = fit["projected"]
        labels = fit["labels"]
        k = fit["chosen_k"]
        colours = plotting.categorical(k)
        marks = plotting.markers(k)
        for cluster in range(k):
            mask = labels == cluster
            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                s=7,
                c=colours[cluster],
                marker=marks[cluster],
                alpha=0.55,
                linewidths=0.0,
                label=f"cluster {cluster} (n={int(mask.sum())})",
            )
        for cluster in range(k):
            centre = projected[labels == cluster][:, :2].mean(axis=0)
            ax.scatter(
                *centre,
                s=110,
                facecolor=plotting.SURFACE,
                edgecolor=plotting.INK_PRIMARY,
                linewidths=1.0,
                marker=marks[cluster],
                zorder=6,
            )
            ax.annotate(
                str(cluster),
                centre,
                ha="center",
                va="center",
                fontsize=plotting.BASE_FONT_PT - 2,
                color=plotting.INK_PRIMARY,
                zorder=7,
            )
        verdict = validation[space]
        plotting.style_axis(ax, xlabel="PC1", ylabel="PC2", title=titles[space])
        ax.annotate(
            f"k={k}, silhouette {fit['silhouette']:.3f}\n"
            f"tracks {verdict['verdict']}: technique eta2 "
            f"{verdict['mean_eta_squared_technique']:.3f}, "
            f"situation eta2 {verdict['mean_eta_squared_situation']:.3f}",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=plotting.BASE_FONT_PT - 2.5,
            color=plotting.INK_SECONDARY,
        )
        plotting.legend_below(ax, ncol=2)
    plotting.panel_labels(axes)
    plotting.save_figure(fig, "archive_keeper_clusters")


def figure_composites(eligible: pd.DataFrame, fit: dict) -> None:
    """Scatter of the two composite axes that separate the clusters most."""
    labels = fit["labels"]
    separation = {name: _eta_squared(eligible[name].to_numpy(), labels) for name in COMPOSITE_NAMES}
    x_axis, y_axis = sorted(separation, key=separation.get, reverse=True)[:2]

    fig, ax = plt.subplots(figsize=(plotting.WIDTH_COLUMN, 4.3))
    k = fit["chosen_k"]
    colours = plotting.categorical(k)
    marks = plotting.markers(k)
    for cluster in range(k):
        mask = labels == cluster
        ax.scatter(
            eligible.loc[mask, x_axis],
            eligible.loc[mask, y_axis],
            s=9,
            c=colours[cluster],
            marker=marks[cluster],
            alpha=0.6,
            linewidths=0.0,
            label=f"cluster {cluster} (n={int(mask.sum())})",
        )
    ax.axhline(0.0, color=plotting.INK_MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.axvline(0.0, color=plotting.INK_MUTED, lw=0.7, ls=(0, (3, 2)))
    plotting.style_axis(
        ax,
        xlabel=f"{x_axis.replace('_', ' ')} (z)",
        ylabel=f"{y_axis.replace('_', ' ')} (z)",
        title="Keeper-seasons on the two most separating axes",
    )
    # Widen the frame before any label is placed, so that a name attached to an extreme
    # point has room to sit inside the axes instead of running over the spine.
    x_lo, x_hi = ax.get_xlim()
    ax.set_xlim(x_lo - 0.05 * (x_hi - x_lo), x_hi + 0.10 * (x_hi - x_lo))
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + 0.06 * (y_hi - y_lo))

    distance = np.hypot(eligible[x_axis], eligible[y_axis])
    order = np.argsort(distance.to_numpy())[::-1][:120]
    candidates = [
        (
            float(eligible[x_axis].iloc[i]),
            float(eligible[y_axis].iloc[i]),
            _label(eligible.iloc[i], compact=True),
        )
        for i in order
    ]
    _spread_labels(ax, candidates, max_labels=9)
    ax.annotate(
        "eta2 against the partition: "
        f"{x_axis.replace('_', ' ')} {separation[x_axis]:.2f}, "
        f"{y_axis.replace('_', ' ')} {separation[y_axis]:.2f}, "
        f"workload {separation['workload']:.2f}.\n"
        "Axes chosen for separation, not for football expectation. "
        "The most extreme keeper-seasons are named.",
        xy=(0.0, -0.30),
        xycoords="axes fraction",
        fontsize=plotting.BASE_FONT_PT - 2.5,
        color=plotting.INK_SECONDARY,
    )
    plotting.legend_below(ax, ncol=2)
    plotting.save_figure(fig, "archive_keeper_composites")


def figure_radar(eligible: pd.DataFrame, fit: dict, archetypes: dict) -> None:
    """The five composites, one polygon per cluster."""
    labels = fit["labels"]
    k = fit["chosen_k"]
    centroids = {
        cluster: np.array([eligible.loc[labels == cluster, c].mean() for c in COMPOSITE_NAMES])
        for cluster in range(k)
    }
    limit = float(max(1.0, np.ceil(max(np.abs(v).max() for v in centroids.values()) * 4.0) / 4.0))
    n = len(COMPOSITE_NAMES)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    closed = np.concatenate([angles, angles[:1]])

    # Park the radial tick labels on the bisector of the adjacent spoke pair carrying the
    # least ink, otherwise they sit underneath a plotted polygon and cannot be read.
    load = np.abs(np.array([centroids[c] for c in range(k)])).sum(axis=0)
    quiet = float(360.0 * (int(np.argmin(load + np.roll(load, -1))) + 0.5) / n)

    fig = plt.figure(figsize=(plotting.WIDTH_COLUMN, 3.0))
    fig.set_layout_engine("none")
    ax = fig.add_axes((0.06, 0.09, 0.52, 0.82), projection="polar")
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in COMPOSITE_NAMES], fontsize=plotting.BASE_FONT_PT - 2
    )
    ax.tick_params(axis="x", pad=2)
    ax.set_rlim(-limit, limit)
    ticks = [-limit / 2.0, 0.0, limit / 2.0]
    ax.set_rgrids(
        ticks,
        labels=[f"{t:+.1f}" if t else "0" for t in ticks],
        fontsize=plotting.BASE_FONT_PT - 3,
        color=plotting.INK_SECONDARY,
    )
    ax.set_rlabel_position(quiet)
    for text in ax.get_yticklabels():
        text.set_bbox({"facecolor": plotting.SURFACE, "edgecolor": "none", "pad": 0.8})
        text.set_zorder(7)
    ax.spines["polar"].set_color(plotting.GRID)
    ax.grid(color=plotting.GRID, linewidth=0.5)
    circle = np.linspace(0.0, 2.0 * np.pi, 181)
    ax.plot(circle, np.zeros_like(circle), color=plotting.INK_MUTED, lw=0.9, ls=(0, (3, 2)))

    colours = plotting.categorical(k)
    marks = plotting.markers(k)
    named = archetypes.get("archetypes", {})
    for cluster in range(k):
        values = np.concatenate([centroids[cluster], centroids[cluster][:1]])
        title = named.get(str(cluster), {}).get("name", f"cluster {cluster}")
        size = int((labels == cluster).sum())
        ax.plot(
            closed,
            values,
            color=colours[cluster],
            lw=1.6,
            marker=marks[cluster],
            markersize=3.4,
            label=f"{cluster}  {title} (n={size})",
        )
        ax.fill(closed, values, color=colours[cluster], alpha=0.12)
    fig.legend(
        loc="center left",
        bbox_to_anchor=(0.60, 0.52),
        ncol=1,
        fontsize=plotting.BASE_FONT_PT - 1,
    )
    plotting.panel_labels(fig.axes, letters="cd")
    plotting.save_figure(fig, "archive_keeper_radar")


def figure_corr(eligible: pd.DataFrame) -> None:
    """Correlation heatmap of the twelve recovered keeper features."""
    matrix = eligible[FEATURES].corr().to_numpy()
    names = [SHORT[f] for f in FEATURES]
    fig, ax = plt.subplots(figsize=(plotting.WIDTH_FULL, 5.4))
    image = ax.imshow(matrix, cmap=plotting.DIVERGING, vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=plotting.BASE_FONT_PT - 2)
    ax.set_yticklabels(names, fontsize=plotting.BASE_FONT_PT - 2)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}".replace("0.", "."),
                ha="center",
                va="center",
                fontsize=plotting.BASE_FONT_PT - 3,
                color=plotting.SURFACE if abs(value) > 0.55 else plotting.INK_PRIMARY,
            )
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Pearson correlation", fontsize=plotting.BASE_FONT_PT - 1)
    bar.ax.tick_params(labelsize=plotting.BASE_FONT_PT - 2)
    ax.set_title(
        f"Recovered goalkeeper features, {len(eligible)} eligible keeper-seasons", loc="left"
    )
    plotting.save_figure(fig, "archive_keeper_corr")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def main() -> None:
    plotting.use_style()
    np.random.seed(config.RANDOM_STATE)

    merged, audit = load_keepers()
    eligible, pool = prepare(merged)
    audit["eligible_pool"] = pool
    audit["contrast"] = (
        "The post-withdrawal pull of the same table serves 26 columns of which 23 hold no "
        "values. Here every column named in archive.KEEPER_SOURCES is populated for every "
        "eligible keeper-season."
    )
    _write_json(audit, "archive_keeper_availability.json")
    print(
        f"  keeper-seasons {pool['keeper_seasons']:,}, eligible {pool['eligible']:,} "
        f"at {config.MIN_MINUTES} minutes, {pool['unique_keepers']} distinct keepers"
    )
    worst = min(audit["canonical_features"].values(), key=lambda v: v["coverage"])
    print(f"  worst canonical coverage before filtering: {worst['column']} {worst['coverage']}")

    stopping = shot_stopping(eligible)
    composites = composite_report(eligible)
    composites["shot_stopping_measure"] = stopping
    _write_json(composites, "archive_keeper_composites.json")
    best = stopping["leaders_total"][0]
    worst_stopper = stopping["laggards_total"][0]
    print(
        f"  PSxG minus goals allowed: best {best['player']} {best['psxg_net']:+.1f}, "
        f"worst {worst_stopper['player']} {worst_stopper['psxg_net']:+.1f}"
    )

    fits: dict[str, dict] = {}
    validations: dict[str, dict] = {}
    for space, columns in SPACES.items():
        fit = fit_space(eligible, columns, f"archive_keeper_{space}")
        validation = situation_versus_technique(eligible, fit["labels"])
        fits[space] = fit
        validations[space] = validation
        print(
            f"  {space}: k={fit['chosen_k']} silhouette {fit['silhouette']:.4f} "
            f"bootstrap ARI {fit['stability']['ari_mean']:.3f} "
            f"verdict {validation['verdict']}"
        )

    supporting = [s for s in SPACES if validations[s]["supports_archetypes"]]
    archetype_space = supporting[0] if supporting else PRIMARY_SPACE
    archetypes = name_archetypes(eligible, fits[archetype_space], validations[archetype_space])
    archetypes["primary_space"] = PRIMARY_SPACE
    archetypes["primary_space_verdict"] = validations[PRIMARY_SPACE]["verdict"]
    archetypes["archetype_space"] = archetype_space
    _write_json(archetypes, "archive_keeper_archetypes.json")

    cluster_payload = {
        "spaces": {
            space: {
                **{k: v for k, v in fit.items() if k not in {"labels", "projected"}},
                "validation": validations[space],
                "cluster_profiles": {
                    str(c): _profile(eligible, fit["labels"], c) for c in range(fit["chosen_k"])
                },
                "exemplars": {
                    str(c): _exemplars(eligible, fit["labels"], fit["projected"], c)
                    for c in range(fit["chosen_k"])
                },
            }
            for space, fit in fits.items()
        },
        "primary_space": PRIMARY_SPACE,
        "k_rule": config.K_RULE,
        "assignments": [
            {
                "player": row["Player"],
                "squad": row["Squad"],
                "season_end_year": int(row["Season_End_Year"]),
                **{f"cluster_{space}": int(fits[space]["labels"][i]) for space in SPACES},
            }
            for i, (_, row) in enumerate(eligible.iterrows())
        ],
        "conclusion": (
            "The partition over all twelve features tracks "
            f"{validations[PRIMARY_SPACE]['verdict']}. The partition over the technique "
            f"composites tracks {validations['technique']['verdict']}."
        ),
    }
    _write_json(cluster_payload, "archive_keeper_clusters.json")

    figure_shot_stopping(eligible)
    figure_clusters(eligible, fits, validations)
    figure_composites(eligible, fits[archetype_space])
    figure_radar(eligible, fits[archetype_space], archetypes)
    figure_corr(eligible)

    if archetypes["named"]:
        for cluster, record in archetypes["archetypes"].items():
            print(f"  archetype {cluster}: {record['name']} (n={record['n']})")
    else:
        print(f"  no archetypes named: {archetypes['reason']}")
    print("archive keepers complete")


if __name__ == "__main__":
    main()
