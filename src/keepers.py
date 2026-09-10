"""Phase 8. Goalkeeper pipeline, rebuilt around what survived the FBref data withdrawal.

What this module can and cannot do
----------------------------------
The goalkeeper analysis originally specified for this paper rested on ``keeper_adv``,
the FBref advanced goalkeeping table. On 20 January 2026 Sports Reference removed every
Opta-derived statistic from FBref. The table is still served with its full 26 column
header, but 23 of those 26 columns contain no values at all in either season we pull.
The audit in :func:`audit_availability` counts this directly from our own parquet files
and writes the counts to ``results/metrics/keeper_data_availability.json``, because the
emptiness is evidence for the paper's central claim rather than a footnote.

Destroyed, and therefore impossible here:

* post-shot expected goals (``Expected__PSxG``), PSxG per shot on target
  (``Expected__PSxG/SoT``), PSxG plus/minus (``Expected__PSxG+/-``) and PSxG plus/minus
  per 90 (``Expected__/90``). This removes shot-quality adjusted shot stopping, which is
  the only defensible measure of saves above expectation;
* sweeping: defensive actions outside the penalty area (``Sweeper__#OPA``), the same per
  90 (``Sweeper__#OPA/90``) and their average distance from goal (``Sweeper__AvgDist``);
* distribution: passes attempted (``Passes__Att (GK)``), throws (``Passes__Thr``), launch
  percentage (``Passes__Launch%``), average pass length (``Passes__AvgLen``), launched
  passes completed and attempted with their completion rate (``Launched__Cmp``,
  ``Launched__Att``, ``Launched__Cmp%``), goal kicks attempted (``Goal Kicks__Att``),
  goal kick launch percentage (``Goal Kicks__Launch%``) and goal kick average length
  (``Goal Kicks__AvgLen``);
* cross handling: crosses faced (``Crosses__Opp``), crosses stopped (``Crosses__Stp``)
  and the stop rate (``Crosses__Stp%``);
* goals conceded by situation: free kicks (``Goals__FK``), corners (``Goals__CK``) and
  own goals (``Goals__OG``), which would have supported a set-piece conceding profile.

Aerial claims never had a public FBref column of their own for goalkeepers, so the
"aerial claims" axis is approximated in the literature by cross stopping, which is also
gone. That axis is therefore unavailable for the same reason.

What survives is the basic ``keeper`` table: playing time, goals against, shots on
target against, saves, save percentage, clean sheets, clean sheet percentage and
penalties faced, allowed and saved. Those describe two things only, how often a keeper
is asked to make a save and how often he makes it. The five composite axes originally
planned (shot stopping, sweeping, distribution, cross handling, aerial claims) therefore
collapse to two: shot stopping and workload. No proxy is invented for the other three.

Derived quantities, defined precisely
-------------------------------------
All per-90 quantities use ``gk_nineties = gk_minutes / 90``, taken from the goalkeeper's
own minutes rather than from team minutes, so a keeper who played half a season is
compared on the same footing as one who played all of it.

* ``gk_goals_against_p90``          goals against divided by nineties
* ``gk_shots_on_target_against_p90`` shots on target against divided by nineties
* ``gk_saves_p90``                  saves divided by nineties
* ``gk_clean_sheets_p90``           clean sheets divided by nineties
* ``gk_pens_faced_p90``             penalties faced divided by nineties
* ``gk_pens_saved_p90``             penalties saved divided by nineties
* ``gk_save_pct``                   FBref ``Performance__Save%``, recomputed as
  ``100 * saves / shots on target against`` when a keeper's rows are combined
* ``gk_clean_sheet_pct``            FBref ``Performance__CS%``, recomputed as
  ``100 * clean sheets / matches played`` when rows are combined
* ``gk_saves_above_volume``         residual from an ordinary least squares fit of total
  saves on total shots on target against, over eligible keepers within a season. Units
  are saves. Positive means the keeper made more saves than a keeper facing the same
  volume of shots on target would be expected to make
* ``gk_saves_above_volume_p90``     the same residual divided by nineties

``gk_saves_above_volume`` is the honest substitute for PSxG plus/minus, and it is a much
weaker instrument. PSxG plus/minus adjusts for the quality of every shot faced, so it
separates a keeper who saved hard shots from one who saved easy ones. The residual used
here adjusts only for the number of shots on target faced. A keeper whose opponents shot
from poor positions will look good on it through no skill of his own, and a keeper facing
a high share of close-range shots will look bad. Every number derived from it inherits
that confound, and the paper says so wherever it is quoted.

Composites
----------
Two axes, built by averaging z-scores computed within the season's eligible pool:

* ``gk_shot_stopping`` mean z of ``gk_save_pct``, ``gk_clean_sheet_pct`` and
  ``gk_saves_above_volume``
* ``gk_workload``      mean z of ``gk_shots_on_target_against_p90`` and
  ``gk_goals_against_p90``

Clean sheet percentage is a team outcome as much as a goalkeeping one, and goals against
per 90 is largely a property of the defence in front of the keeper. Both are kept because
they are what survived, and both are flagged in the metrics as team-confounded.

Population
----------
The analysis set is goalkeepers with at least ``config.MIN_MINUTES`` (450) minutes, which
is 33 of 44 in 2024-25 and 28 of 40 in 2025-26. Descriptives are reported for both the
full pool and the eligible pool; every model is fitted on the eligible pool only.

Run with ``uv run python -m src.keepers``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

import config
from src import features as F
from src import plotting

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Identifier columns in the raw keeper parquet files. Everything else is a stat column,
#: and the emptiness audit is reported over stat columns so the counts match the header
#: FBref publishes for the table itself.
_ID_COLS = ("league", "season", "team", "player", "nation", "pos", "age", "born")

#: Counting columns in the raw keeper table that are summed when a keeper's spells at
#: more than one club are combined into a single season row.
_KEEPER_COUNT_SOURCES = {
    "gk_matches": "Playing Time__MP",
    "gk_starts": "Playing Time__Starts",
    "gk_minutes": "Playing Time__Min",
    "gk_goals_against": "Performance__GA",
    "gk_shots_on_target_against": "Performance__SoTA",
    "gk_saves": "Performance__Saves",
    "gk_clean_sheets": "Performance__CS",
    "gk_pens_faced": "Penalty Kicks__PKatt",
    "gk_pens_allowed": "Penalty Kicks__PKA",
    "gk_pens_saved": "Penalty Kicks__PKsv",
}

#: Counting columns converted to per 90. KEEPER_COUNTS from features.py plus the two
#: penalty columns, which are informative precisely because they carry almost no signal.
_PER90_COUNTS = [*F.KEEPER_COUNTS, "gk_pens_faced", "gk_pens_saved"]

#: The feature set used for correlation, PCA and clustering. This is exactly
#: features.KEEPER_CORE, which is everything the surviving table supports.
CLUSTER_FEATURES = list(F.KEEPER_CORE)

#: Wider set used for the correlation heatmap, so the reader can see that the penalty
#: columns are near-independent noise and the rest is two dimensions wearing six labels.
CORR_FEATURES = [*CLUSTER_FEATURES, "gk_pens_faced_p90", "gk_pens_saved_p90"]

#: Raw surviving columns summarised in the descriptive metrics.
DESCRIPTIVE_COLUMNS = [
    "gk_matches",
    "gk_starts",
    "gk_minutes",
    "gk_nineties",
    "gk_goals_against",
    "gk_shots_on_target_against",
    "gk_saves",
    "gk_clean_sheets",
    "gk_pens_faced",
    "gk_pens_allowed",
    "gk_pens_saved",
    *CORR_FEATURES,
    "gk_saves_above_volume",
    "gk_saves_above_volume_p90",
]

#: k is capped below config.K_MAX. With 33 eligible keepers in the primary season and a
#: feature set that spans roughly two effective dimensions, a k above 6 would give
#: clusters averaging fewer than six members, which is too few for a centroid profile to
#: be interpretable or for a silhouette to be stable under resampling.
K_CAP = 6

#: Kaufman and Rousseeuw's published bands for interpreting an average silhouette width.
#: Declared here in advance so the verdict on goalkeeper structure is not chosen after
#: seeing the number.
SILHOUETTE_BANDS = [
    (0.71, "strong structure"),
    (0.51, "reasonable structure"),
    (0.26, "weak structure, could be artificial"),
    (-1.01, "no substantial structure"),
]

#: Number of extreme points direct-labelled at each end of a scatter.
N_LABEL_EXTREMES = 4

#: A cluster smaller than this cannot support a centroid profile that means anything with
#: 28 to 33 observations, so a partition containing one is reported as an artefact of the
#: algorithm rather than as evidence of goalkeeper types. Declared before fitting.
MIN_CLUSTER_SIZE = 5

#: Candidate label offsets in points, in preference order, used by the placer below.
_LABEL_OFFSETS = [
    (4.5, 3.0),
    (4.5, -8.0),
    (-4.5, 3.0),
    (-4.5, -8.0),
    (0.0, 6.5),
    (0.0, -11.0),
    (10.0, 0.0),
    (-10.0, 0.0),
    (8.0, 9.0),
    (-8.0, 9.0),
    (8.0, -13.0),
    (-8.0, -13.0),
]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _jsonable(value):
    """Convert numpy and pandas scalars into something json.dump accepts."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else round(value, 6)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if value is None or isinstance(value, str | int | bool):
        return value
    if pd.isna(value):
        return None
    return value


def _write_metrics(name: str, payload: dict) -> Path:
    path = config.METRICS / f"{name}.json"
    with open(path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    return path


def _short_name(full: str) -> str:
    """Surname only, so 33 direct labels do not collide on a text-width figure."""
    parts = str(full).split()
    return parts[-1] if parts else str(full)


def _extreme_labels(values: pd.Series, names: pd.Series, n: int = N_LABEL_EXTREMES) -> list:
    """Label the n highest and n lowest points and nothing else.

    Returns a list aligned with ``values``, holding a short name where the point is an
    extreme and ``None`` everywhere else, which is what plotting.annotate_points expects.
    """
    order = values.rank(method="first")
    keep = (order <= n) | (order > len(values) - n)
    return [_short_name(nm) if k else None for nm, k in zip(names, keep, strict=True)]


def _place_labels(fig, ax, xs, ys, labels, *, colors=None, all_points=None) -> None:
    """Direct-label chosen points, picking an offset for each that avoids collisions.

    A goalkeeper scatter has 28 to 33 points inside a text-width panel, so the naive
    fixed offset used by plotting.annotate_points puts surnames on top of each other and
    off the right edge. This chooses, for each label, the first candidate offset whose
    text box overlaps no already-placed box, no plotted marker and no panel edge, and
    falls back to the least bad candidate when every option collides. Offsets are decided
    in points and applied through ``textcoords="offset points"``, so the placement holds
    when the figure is rasterised at a different dpi.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    labels = list(labels)
    fontsize = plotting.BASE_FONT_PT - 2
    fig.canvas.draw()
    scale = fig.dpi / 72.0
    char_w = fontsize * 0.58 * scale
    line_h = fontsize * 1.05 * scale
    panel = ax.get_window_extent()

    def box(px, py, dx, dy, width):
        left = px + dx * scale if dx >= 0 else px + dx * scale - width
        bottom = py + dy * scale if dy >= 0 else py + dy * scale - line_h
        return (left, bottom, left + width, bottom + line_h)

    def overlap(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0.0, min(a[3], b[3]) - max(a[1], b[1])
        )

    occupied: list[tuple[float, float, float, float]] = []
    marker_pad = 4.0 * scale
    source = np.column_stack([xs, ys]) if all_points is None else np.asarray(all_points, float)
    markers_px = ax.transData.transform(source)
    for px, py in markers_px:
        occupied.append((px - marker_pad, py - marker_pad, px + marker_pad, py + marker_pad))

    items = [(i, text) for i, text in enumerate(labels) if text is not None and str(text) != "nan"]
    for i, text in items:
        px, py = ax.transData.transform((xs[i], ys[i]))
        width = char_w * len(str(text))
        best, best_cost, best_rank = _LABEL_OFFSETS[0], None, 0
        for rank, (dx, dy) in enumerate(_LABEL_OFFSETS):
            candidate = box(px, py, dx, dy, width)
            cost = sum(overlap(candidate, other) for other in occupied)
            outside = (
                max(0.0, panel.x0 - candidate[0])
                + max(0.0, candidate[2] - panel.x1)
                + max(0.0, panel.y0 - candidate[1])
                + max(0.0, candidate[3] - panel.y1)
            )
            cost += outside * 40.0
            if best_cost is None or cost < best_cost:
                best, best_cost, best_rank = (dx, dy), cost, rank
            if cost == 0:
                break
        dx, dy = best
        chosen = box(px, py, dx, dy, width)
        occupied.append(chosen)
        # A hairline leader is drawn whenever the label sits far from its own marker or
        # closer to somebody else's, so the reader is never left guessing whose it is.
        anchor = ((chosen[0] + chosen[2]) / 2, (chosen[1] + chosen[3]) / 2)
        gaps = np.hypot(markers_px[:, 0] - anchor[0], markers_px[:, 1] - anchor[1])
        is_own = np.hypot(markers_px[:, 0] - px, markers_px[:, 1] - py) <= 0.5
        others = gaps[~is_own]
        own = float(np.hypot(px - anchor[0], py - anchor[1]))
        ambiguous = bool(others.size and others.min() < own * 1.4 + 4.0 * scale)
        leader = (
            {
                "arrowstyle": "-",
                "linewidth": 0.4,
                "color": plotting.INK_MUTED,
                "shrinkA": 1.0,
                "shrinkB": 2.0,
            }
            if best_rank >= 4 or ambiguous
            else None
        )
        ax.annotate(
            str(text),
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=fontsize,
            color=plotting.INK_SECONDARY if colors is None else colors[i],
            zorder=6,
            arrowprops=leader,
        )


def _eta_squared(values: pd.Series, labels: np.ndarray) -> float:
    """Share of a variable's variance explained by cluster membership."""
    series = pd.to_numeric(values, errors="coerce")
    grand = series.mean()
    ss_total = float(((series - grand) ** 2).sum())
    if not np.isfinite(ss_total) or ss_total == 0:
        return float("nan")
    ss_between = 0.0
    for c in np.unique(labels):
        member = series[labels == c]
        ss_between += len(member) * (member.mean() - grand) ** 2
    return float(ss_between / ss_total)


def _describe(frame: pd.DataFrame, columns: list[str]) -> dict:
    out: dict[str, dict] = {}
    for col in columns:
        if col not in frame.columns:
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        valid = series.dropna()
        out[col] = {
            "n": int(valid.size),
            "n_missing": int(series.isna().sum()),
            "mean": float(valid.mean()) if valid.size else None,
            "std": float(valid.std(ddof=1)) if valid.size > 1 else None,
            "min": float(valid.min()) if valid.size else None,
            "q25": float(valid.quantile(0.25)) if valid.size else None,
            "median": float(valid.median()) if valid.size else None,
            "q75": float(valid.quantile(0.75)) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
        }
    return out


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def _silhouette_verdict(score: float) -> str:
    for floor, label in SILHOUETTE_BANDS:
        if score >= floor:
            return label
    return "no substantial structure"


# --------------------------------------------------------------------------------------
# Step 1: emptiness audit
# --------------------------------------------------------------------------------------


def audit_availability() -> dict:
    """Count non-null values in every column of the raw keeper tables, both seasons.

    This is the evidence behind the claim that the advanced goalkeeping analysis is not
    merely inconvenient but impossible, so it counts every column rather than the ones we
    happen to want, and it reports identifier columns separately from stat columns.
    """
    report: dict = {
        "purpose": (
            "Direct count of populated cells in the FBref goalkeeper tables as pulled by "
            "this project. FBref still serves the full keeper_adv header after the "
            "January 2026 removal of Opta-derived statistics, so a column list alone "
            "does not reveal that the table is empty."
        ),
        "id_columns_excluded_from_stat_counts": list(_ID_COLS),
        "seasons": {},
    }
    for season in config.SEASONS:
        season_block: dict = {}
        for table in ("keeper", "keeper_adv"):
            path = config.DATA_RAW / season / f"players_{table}.parquet"
            frame = pd.read_parquet(path)
            stat_cols = [c for c in frame.columns if c not in _ID_COLS]
            non_null = {c: int(frame[c].notna().sum()) for c in stat_cols}
            empty = sorted(c for c, v in non_null.items() if v == 0)
            partial = sorted(c for c, v in non_null.items() if 0 < v < len(frame))
            season_block[table] = {
                "path": str(path.relative_to(config.ROOT)),
                "rows": int(len(frame)),
                "total_columns": int(frame.shape[1]),
                "stat_columns": len(stat_cols),
                "all_null_stat_columns": len(empty),
                "usable_stat_columns": len(stat_cols) - len(empty),
                "fully_populated_stat_columns": int(
                    sum(1 for v in non_null.values() if v == len(frame))
                ),
                "partially_populated_stat_columns": partial,
                "non_null_counts": non_null,
                "all_null_column_names": empty,
            }
        season_block["keeper_adv_empty_fraction"] = round(
            season_block["keeper_adv"]["all_null_stat_columns"]
            / season_block["keeper_adv"]["stat_columns"],
            4,
        )
        report["seasons"][season] = season_block

    adv = {s: b["keeper_adv"] for s, b in report["seasons"].items()}
    empty_sets = {s: set(b["all_null_column_names"]) for s, b in adv.items()}
    common = set.intersection(*empty_sets.values()) if empty_sets else set()
    report["summary"] = {
        "keeper_adv_stat_columns": {s: b["stat_columns"] for s, b in adv.items()},
        "keeper_adv_all_null": {s: b["all_null_stat_columns"] for s, b in adv.items()},
        "keeper_adv_usable": {s: b["usable_stat_columns"] for s, b in adv.items()},
        "keeper_adv_empty_in_both_seasons": sorted(common),
        "keeper_adv_survivors": sorted(set(adv[config.SEASON_PRIMARY]["non_null_counts"]) - common),
        "keeper_table_all_null": {
            s: b["keeper"]["all_null_stat_columns"] for s, b in report["seasons"].items()
        },
    }
    report["analyses_made_impossible"] = {
        "shot_stopping_vs_shot_quality": [
            "Expected__PSxG",
            "Expected__PSxG/SoT",
            "Expected__PSxG+/-",
            "Expected__/90",
        ],
        "sweeping": ["Sweeper__#OPA", "Sweeper__#OPA/90", "Sweeper__AvgDist"],
        "distribution": [
            "Passes__Att (GK)",
            "Passes__Thr",
            "Passes__Launch%",
            "Passes__AvgLen",
            "Launched__Cmp",
            "Launched__Att",
            "Launched__Cmp%",
            "Goal Kicks__Att",
            "Goal Kicks__Launch%",
            "Goal Kicks__AvgLen",
        ],
        "cross_handling_and_aerial_claims": ["Crosses__Opp", "Crosses__Stp", "Crosses__Stp%"],
        "conceding_by_situation": ["Goals__FK", "Goals__CK", "Goals__OG"],
    }
    return report


# --------------------------------------------------------------------------------------
# Step 2: build the analysis frame
# --------------------------------------------------------------------------------------


def load_keepers(season: str) -> pd.DataFrame:
    """One row per goalkeeper for a season, with the surviving keeper statistics.

    The processed ``keepers_all_{season}.parquet`` carries identity, playing time and the
    outfield-style per-90 columns but none of the goalkeeping columns, so the raw keeper
    table is read here and aggregated across clubs the same way preprocess aggregates
    outfield players: counting statistics are summed and rates are recomputed from their
    components rather than averaged, because averaging rates over unequal spells is wrong.
    """
    raw = pd.read_parquet(config.DATA_RAW / season / "players_keeper.parquet")
    rename = {src: canon for canon, src in _KEEPER_COUNT_SOURCES.items()}
    keep = raw[["player", "team", *rename]].rename(columns=rename)
    for col in _KEEPER_COUNT_SOURCES:
        keep[col] = pd.to_numeric(keep[col], errors="coerce")

    ordered = keep.sort_values("gk_minutes", ascending=False)
    agg = ordered.groupby("player", as_index=False)[list(_KEEPER_COUNT_SOURCES)].sum()
    teams = (
        ordered.groupby("player")["team"]
        .agg(lambda s: " / ".join(dict.fromkeys(s)))
        .reset_index()
        .rename(columns={"team": "gk_team"})
    )
    squads = ordered.groupby("player", as_index=False).size().rename(columns={"size": "gk_squads"})
    agg = agg.merge(teams, on="player", how="left").merge(squads, on="player", how="left")

    agg["gk_nineties"] = agg["gk_minutes"] / 90.0
    agg["gk_save_pct"] = np.where(
        agg["gk_shots_on_target_against"] > 0,
        100.0 * agg["gk_saves"] / agg["gk_shots_on_target_against"],
        np.nan,
    )
    agg["gk_clean_sheet_pct"] = np.where(
        agg["gk_matches"] > 0, 100.0 * agg["gk_clean_sheets"] / agg["gk_matches"], np.nan
    )
    nineties = agg["gk_nineties"].replace(0, np.nan)
    for col in _PER90_COUNTS:
        agg[f"{col}_p90"] = agg[col] / nineties

    processed = pd.read_parquet(config.DATA_PROCESSED / f"keepers_all_{season}.parquet")
    context = processed[["player", "team", "age", "nation", "team_points_per_match"]].copy()
    out = context.merge(agg, on="player", how="inner")
    out.insert(0, "season", season)
    out["eligible"] = out["gk_minutes"] >= config.MIN_MINUTES
    return out.sort_values("gk_minutes", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Step 3: shot stopping, the weak substitute for PSxG plus/minus
# --------------------------------------------------------------------------------------


def fit_saves_on_sota(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Regress total saves on total shots on target against, return residuals and fit.

    The residual is the shot-stopping index used throughout this module. It answers only
    "did this keeper make more saves than the volume of shots on target he faced would
    predict", which is a strictly weaker question than the one PSxG plus/minus answers,
    because it holds shot volume constant and not shot quality. A keeper facing tame
    long-range efforts and a keeper facing one-on-ones are treated identically here.
    """
    x = frame["gk_shots_on_target_against"].to_numpy(dtype=float)
    y = frame["gk_saves"].to_numpy(dtype=float)
    n = len(x)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    resid = y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Standard error of the slope, so the paper can report the fit rather than assert it.
    dof = n - 2
    mse = ss_res / dof if dof > 0 else float("nan")
    sxx = float(np.sum((x - x.mean()) ** 2))
    se_slope = float(np.sqrt(mse / sxx)) if sxx > 0 and np.isfinite(mse) else float("nan")

    residuals = pd.Series(resid, index=frame.index, name="gk_saves_above_volume")
    minutes_corr = (
        float(np.corrcoef(residuals, frame["gk_minutes"])[0, 1]) if n > 2 else float("nan")
    )
    info = {
        "model": "OLS, total saves ~ total shots on target against",
        "n": int(n),
        "slope": float(slope),
        "slope_std_error": se_slope,
        "intercept": float(intercept),
        "r_squared": float(r2),
        "residual_std_error": float(np.sqrt(mse)) if np.isfinite(mse) else None,
        "residual_min": float(residuals.min()),
        "residual_max": float(residuals.max()),
        "residual_corr_with_minutes": minutes_corr,
        "interpretation": (
            "The slope is the league-average save rate. The residual is saves above the "
            "number a keeper facing that many shots on target would be expected to make."
        ),
        "r_squared_note": (
            "The R squared is very high because saves and shots on target against are "
            "close to an accounting identity: every shot on target is either saved or "
            "conceded. The fit quality is therefore not evidence that the model is good. "
            "The informative quantity is the residual spread, which is the residual "
            "standard error reported above, roughly "
            f"{100 * np.sqrt(ss_res / n) / y.mean():.1f} percent of a keeper's mean save "
            "total across the eligible pool."
        ),
        "caveat": (
            "This is a much weaker instrument than PSxG plus/minus. It adjusts for shot "
            "volume only, never for shot quality, so a keeper whose opponents shoot from "
            "poor positions scores well without saving anything difficult. PSxG, "
            "PSxG/SoT and PSxG+/- are all empty in keeper_adv for both seasons, which is "
            "why the substitution is necessary."
        ),
    }
    if abs(minutes_corr) > 0.25:
        info["minutes_confound_note"] = (
            "The residual correlates with minutes played at r = "
            f"{minutes_corr:.2f}, so gk_saves_above_volume_p90 is reported alongside it."
        )
    return residuals, info


def savepct_vs_workload(frame: pd.DataFrame) -> dict:
    """Relate save percentage to shots on target against per 90.

    Tests the standard claim that keepers at weaker clubs face more shots and therefore
    post different save percentages, using only surviving columns.
    """
    x = frame["gk_shots_on_target_against_p90"]
    y = frame["gk_save_pct"]
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    n = int(mask.sum())
    pearson = float(x.corr(y)) if n > 2 else float("nan")
    spearman = float(x.corr(y, method="spearman")) if n > 2 else float("nan")
    slope, intercept = np.polyfit(x.to_numpy(float), y.to_numpy(float), 1)
    # Two-sided t test on the Pearson correlation, exact under bivariate normality.
    if n > 2 and abs(pearson) < 1:
        t_stat = pearson * np.sqrt((n - 2) / (1 - pearson**2))
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 2))
    else:
        t_stat, p_value = float("nan"), float("nan")
    direction = "higher" if slope > 0 else "lower"
    if p_value < 0.05:
        verdict = (
            f"Busier keepers post {direction} save percentages: each extra shot on target "
            f"faced per 90 goes with {abs(slope):.2f} percentage points of save percentage "
            f"in that direction, significant at the 5 percent level "
            f"(r = {pearson:.3f}, p = {p_value:.3f}, n = {n})."
        )
    else:
        verdict = (
            "Save percentage does not depend on workload in this season. The point "
            f"estimate is {direction} save percentages for busier keepers, at "
            f"{abs(slope):.2f} percentage points per extra shot on target faced per 90, "
            "but it is not distinguishable from no relationship "
            f"(r = {pearson:.3f}, p = {p_value:.3f}, n = {n})."
        )
    return {
        "n": n,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "t_statistic": float(t_stat),
        "p_value": p_value,
        "ols_slope_savepct_per_extra_sota_per90": float(slope),
        "ols_intercept": float(intercept),
        "verdict": verdict,
    }


# --------------------------------------------------------------------------------------
# Step 4: clustering
# --------------------------------------------------------------------------------------


def _gap_statistic(data: np.ndarray, k: int, rng: np.random.Generator, b: int) -> float:
    """Tibshirani gap statistic against a uniform reference over the data bounding box."""

    def dispersion(sample: np.ndarray) -> float:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10).fit(sample)
        return float(km.inertia_)

    lo, hi = data.min(axis=0), data.max(axis=0)
    observed = np.log(max(dispersion(data), 1e-12))
    reference = np.empty(b, dtype=float)
    for i in range(b):
        draw = rng.uniform(lo, hi, size=data.shape)
        reference[i] = np.log(max(dispersion(draw), 1e-12))
    return float(reference.mean() - observed)


def _apply_k_rule(rows: list[dict]) -> tuple[int, dict]:
    """Apply config.K_RULE, the same consensus rule used by the outfield pipeline.

    Smallest k whose silhouette is within ``silhouette_tolerance`` of the best observed
    silhouette and which ranks in the top ``top_n_rank`` on at least ``min_criteria`` of
    silhouette, Calinski-Harabasz and gap. Davies-Bouldin, lower being better, breaks ties.
    """
    rule = config.K_RULE
    best_sil = max(r["silhouette"] for r in rows)

    def ranks(key: str, higher_is_better: bool = True) -> dict[int, int]:
        order = sorted(rows, key=lambda r: r[key], reverse=higher_is_better)
        return {r["k"]: i + 1 for i, r in enumerate(order)}

    rank_sil = ranks("silhouette")
    rank_ch = ranks("calinski_harabasz")
    rank_gap = ranks("gap")
    for row in rows:
        row["rank_silhouette"] = rank_sil[row["k"]]
        row["rank_calinski_harabasz"] = rank_ch[row["k"]]
        row["rank_gap"] = rank_gap[row["k"]]
        row["criteria_in_top_n"] = int(
            sum(
                r <= rule["top_n_rank"]
                for r in (row["rank_silhouette"], row["rank_calinski_harabasz"], row["rank_gap"])
            )
        )
        row["within_silhouette_tolerance"] = bool(
            best_sil - row["silhouette"] <= rule["silhouette_tolerance"]
        )
        row["passes_rule"] = bool(
            row["within_silhouette_tolerance"] and row["criteria_in_top_n"] >= rule["min_criteria"]
        )

    passing = [r for r in rows if r["passes_rule"]]
    fallback = False
    if not passing:
        passing = [r for r in rows if r["within_silhouette_tolerance"]] or rows
        fallback = True
    smallest = min(r["k"] for r in passing)
    tied = [r for r in passing if r["k"] == smallest]
    chosen = min(tied, key=lambda r: r["davies_bouldin"])["k"]
    return chosen, {
        "rule": rule,
        "best_silhouette_observed": best_sil,
        "fallback_used": fallback,
        "fallback_note": (
            "No k satisfied both clauses of the consensus rule, so the smallest k inside "
            "the silhouette tolerance was taken and Davies-Bouldin broke the tie."
            if fallback
            else None
        ),
    }


def cluster_keepers(frame: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Standardise, PCA to the variance target, then KMeans with the consensus k rule."""
    matrix = frame[CLUSTER_FEATURES].to_numpy(dtype=float)
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=config.PCA_VARIANCE_TARGET, random_state=config.RANDOM_STATE)
    scores = pca.fit_transform(scaled)

    k_values = [k for k in config.K_RANGE if k <= K_CAP and k < len(frame)]
    rng = np.random.default_rng(config.RANDOM_STATE)
    rows: list[dict] = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10).fit(scores)
        labels = km.labels_
        labels_by_k[k] = labels
        rows.append(
            {
                "k": int(k),
                "silhouette": float(silhouette_score(scores, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(scores, labels)),
                "davies_bouldin": float(davies_bouldin_score(scores, labels)),
                "gap": _gap_statistic(scores, k, rng, config.GAP_STATISTIC_B),
                "inertia": float(km.inertia_),
                "cluster_sizes": sorted(np.bincount(labels, minlength=k).tolist()),
            }
        )

    chosen_k, rule_info = _apply_k_rule(rows)
    labels = labels_by_k[chosen_k]
    chosen_row = next(r for r in rows if r["k"] == chosen_k)
    sil = chosen_row["silhouette"]

    # Order clusters by mean workload so the label ids are stable and comparable across
    # seasons rather than being whatever KMeans happened to emit.
    workload = frame["gk_shots_on_target_against_p90"].to_numpy(dtype=float)
    means = {c: float(workload[labels == c].mean()) for c in np.unique(labels)}
    remap = {old: new for new, (old, _) in enumerate(sorted(means.items(), key=lambda kv: kv[1]))}
    labels = np.array([remap[c] for c in labels])

    info = {
        "n_keepers": int(len(frame)),
        "features": CLUSTER_FEATURES,
        "n_features": len(CLUSTER_FEATURES),
        "k_range_searched": k_values,
        "k_cap": K_CAP,
        "k_cap_justification": (
            f"config.K_MAX is {config.K_MAX}, which is set for the outfield pool of several "
            "hundred players. With at most 33 eligible goalkeepers and a feature set that "
            "spans roughly two effective dimensions, k above 6 gives clusters averaging "
            "fewer than six members, too few for an interpretable centroid profile or a "
            "stable silhouette, so k is capped at 6 here."
        ),
        "pca_n_components": int(pca.n_components_),
        "pca_variance_target": config.PCA_VARIANCE_TARGET,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "pca_loadings_pc1": dict(zip(CLUSTER_FEATURES, pca.components_[0], strict=True)),
        "pca_loadings_pc2": (
            dict(zip(CLUSTER_FEATURES, pca.components_[1], strict=True))
            if pca.n_components_ > 1
            else None
        ),
        "candidates": rows,
        "k_selection": rule_info,
        "chosen_k": int(chosen_k),
        "silhouette": sil,
        "calinski_harabasz": chosen_row["calinski_harabasz"],
        "davies_bouldin": chosen_row["davies_bouldin"],
        "gap": chosen_row["gap"],
        "silhouette_band": _silhouette_verdict(sil),
        "cluster_sizes": {int(c): int((labels == c).sum()) for c in np.unique(labels)},
        "cluster_label_ordering": (
            "clusters are renumbered by ascending mean shots on target against per 90"
        ),
    }

    sizes = list(info["cluster_sizes"].values())
    smallest_cluster = int(min(sizes))
    passes_size = smallest_cluster >= MIN_CLUSTER_SIZE
    passes_sil = sil >= 0.51

    # What is the partition actually separating? If cluster membership explains more of
    # the variance in minutes played and team strength than in save percentage, the
    # partition is recovering the situation a keeper is in rather than how he keeps goal.
    eta = {
        col: _eta_squared(frame[col], labels)
        for col in [
            "gk_save_pct",
            "gk_saves_above_volume",
            "gk_clean_sheet_pct",
            "gk_shots_on_target_against_p90",
            "gk_goals_against_p90",
            "gk_minutes",
            "team_points_per_match",
        ]
    }
    situational = max(eta["gk_minutes"], eta["team_points_per_match"])
    tracks_situation = situational >= eta["gk_save_pct"]

    info["min_cluster_size"] = smallest_cluster
    info["min_cluster_size_floor"] = MIN_CLUSTER_SIZE
    info["has_undersized_cluster"] = not passes_size
    info["variance_explained_by_cluster"] = eta
    info["partition_tracks_situation_not_technique"] = bool(tracks_situation)
    info["structure_is_meaningful"] = bool(passes_sil and passes_size and not tracks_situation)
    reasons = []
    if not passes_sil:
        reasons.append(
            f"the average silhouette width is {sil:.3f}, in the band "
            f"'{_silhouette_verdict(sil)}' on the Kaufman and Rousseeuw scale declared in "
            "advance"
        )
    if not passes_size:
        reasons.append(
            f"the smallest cluster holds {smallest_cluster} keepers, below the floor of "
            f"{MIN_CLUSTER_SIZE} declared before fitting, so at least one centroid is an "
            "average of a handful of players"
        )
    if tracks_situation:
        reasons.append(
            "cluster membership explains more of the variance in minutes played and team "
            f"points per match (eta squared up to {situational:.2f}) than in save "
            f"percentage ({eta['gk_save_pct']:.2f}), so the partition separates keepers by "
            "the situation they work in rather than by how they keep goal"
        )
    if reasons:
        info["verdict"] = (
            f"Negative result. KMeans at k = {chosen_k} returns a partition, but it is not "
            "evidence for goalkeeper types, because " + "; and ".join(reasons) + ". With "
            "shot stopping and workload the only surviving information, the eligible "
            "keepers should be read as a continuum cut arbitrarily, not as archetypes."
        )
    else:
        info["verdict"] = (
            f"Average silhouette width at k = {chosen_k} is {sil:.3f}, the smallest cluster "
            f"holds {smallest_cluster} keepers, and the partition separates keepers on "
            "goalkeeping rather than situational variables. Reported as a positive result."
        )
    scores_frame = pd.DataFrame(
        scores[:, : min(2, scores.shape[1])],
        index=frame.index,
        columns=[f"gk_pc{i + 1}" for i in range(min(2, scores.shape[1]))],
    )
    info["_pc_scores"] = scores_frame
    return pd.Series(labels, index=frame.index, name="gk_cluster"), info


# --------------------------------------------------------------------------------------
# Step 5: composites
# --------------------------------------------------------------------------------------


def build_composites(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Two axes, because only two are supported by the surviving columns."""
    shot_parts = ["gk_save_pct", "gk_clean_sheet_pct", "gk_saves_above_volume"]
    workload_parts = ["gk_shots_on_target_against_p90", "gk_goals_against_p90"]

    z = pd.DataFrame(index=frame.index)
    for col in [*shot_parts, *workload_parts]:
        z[f"z_{col}"] = _zscore(frame[col])

    out = pd.DataFrame(index=frame.index)
    out["gk_shot_stopping"] = z[[f"z_{c}" for c in shot_parts]].mean(axis=1)
    out["gk_workload"] = z[[f"z_{c}" for c in workload_parts]].mean(axis=1)

    corr = float(out["gk_shot_stopping"].corr(out["gk_workload"]))
    info = {
        "axes_built": ["gk_shot_stopping", "gk_workload"],
        "axes_not_buildable": ["sweeping", "distribution", "cross_handling", "aerial_claims"],
        "why_not_buildable": (
            "Every input column for those four axes is all-null in keeper_adv in both "
            "seasons. No proxy is substituted, because no surviving column measures "
            "anything a keeper does with the ball at his feet or outside his six yard box."
        ),
        "gk_shot_stopping": {
            "definition": "mean of within-season z-scores of " + ", ".join(shot_parts),
            "components": shot_parts,
            "caveat": (
                "Clean sheet percentage is a team outcome as much as a goalkeeping one, "
                "and the saves-above-volume residual is unadjusted for shot quality. The "
                "axis therefore measures shot stopping only as well as those two allow."
            ),
        },
        "gk_workload": {
            "definition": "mean of within-season z-scores of " + ", ".join(workload_parts),
            "components": workload_parts,
            "caveat": (
                "Goals against per 90 is mostly a property of the defence in front of the "
                "keeper, so this axis describes the team situation rather than the player."
            ),
        },
        "correlation_between_axes": corr,
        "component_correlations": {
            c: {d: float(z[f"z_{c}"].corr(z[f"z_{d}"])) for d in [*shot_parts, *workload_parts]}
            for c in [*shot_parts, *workload_parts]
        },
    }
    return pd.concat([z, out], axis=1), info


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def figure_corr_heatmap(by_season: dict[str, pd.DataFrame]) -> None:
    seasons = list(by_season)
    fig, axes = plt.subplots(
        1, len(seasons), figsize=(plotting.WIDTH_FULL, 3.5), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    labels = [c.replace("gk_", "").replace("_p90", "/90").replace("_", " ") for c in CORR_FEATURES]
    mesh = None
    for ax, season in zip(axes, seasons, strict=True):
        corr = by_season[season][CORR_FEATURES].corr()
        mesh = ax.imshow(corr.to_numpy(), cmap=plotting.DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(season, loc="left")
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = corr.iat[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}".replace("0.", ".").replace("1.00", "1"),
                    ha="center",
                    va="center",
                    fontsize=plotting.BASE_FONT_PT - 3.5,
                    color=plotting.INK_PRIMARY if abs(v) < 0.6 else plotting.SURFACE,
                )
    for ax in axes[1:]:
        ax.set_yticklabels([])
    cbar = fig.colorbar(mesh, ax=axes.tolist(), shrink=0.7, pad=0.02)
    cbar.set_label("Pearson r", fontsize=plotting.BASE_FONT_PT - 1)
    cbar.outline.set_visible(False)
    plotting.save_figure(fig, "keeper_corr_heatmap")


def figure_saves_vs_sota(by_season: dict[str, pd.DataFrame], fits: dict[str, dict]) -> None:
    seasons = list(by_season)
    fig, axes = plt.subplots(
        1, len(seasons), figsize=(plotting.WIDTH_FULL, 3.0), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, season in zip(axes, seasons, strict=True):
        frame = by_season[season]
        fit = fits[season]
        ax.scatter(
            frame["gk_shots_on_target_against"],
            frame["gk_saves"],
            s=26,
            color=plotting.CATEGORICAL[0],
            edgecolor=plotting.SURFACE,
            linewidth=0.5,
            zorder=3,
        )
        grid = np.linspace(
            frame["gk_shots_on_target_against"].min(),
            frame["gk_shots_on_target_against"].max(),
            50,
        )
        ax.plot(
            grid,
            fit["intercept"] + fit["slope"] * grid,
            color=plotting.INK_SECONDARY,
            linewidth=1.2,
            zorder=2,
        )
        ax.margins(x=0.14, y=0.10)
        plotting.style_axis(
            ax,
            xlabel="Shots on target against",
            ylabel="Saves" if ax is axes[0] else "",
            title=f"{season}   R2 = {fit['r_squared']:.3f}",
        )
        labels = _extreme_labels(frame["gk_saves_above_volume"], frame["player"], n=3)
        # The fitted line is fed in as occupancy so no surname is dropped on top of it.
        line = np.column_stack([grid, fit["intercept"] + fit["slope"] * grid])
        _place_labels(
            fig,
            ax,
            frame["gk_shots_on_target_against"],
            frame["gk_saves"],
            labels,
            all_points=np.vstack(
                [np.column_stack([frame["gk_shots_on_target_against"], frame["gk_saves"]]), line]
            ),
        )
    plotting.save_figure(fig, "keeper_saves_vs_sota")


def figure_savepct_vs_workload(by_season: dict[str, pd.DataFrame], rel: dict[str, dict]) -> None:
    seasons = list(by_season)
    fig, axes = plt.subplots(
        1, len(seasons), figsize=(plotting.WIDTH_FULL, 3.0), sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for ax, season in zip(axes, seasons, strict=True):
        frame = by_season[season]
        x = frame["gk_shots_on_target_against_p90"]
        y = frame["gk_save_pct"]
        ax.scatter(
            x,
            y,
            s=26,
            color=plotting.CATEGORICAL[2],
            marker="^",
            edgecolor=plotting.SURFACE,
            linewidth=0.5,
            zorder=3,
        )
        info = rel[season]
        grid = np.linspace(x.min(), x.max(), 50)
        ax.plot(
            grid,
            info["ols_intercept"] + info["ols_slope_savepct_per_extra_sota_per90"] * grid,
            color=plotting.INK_SECONDARY,
            linewidth=1.2,
            linestyle=(0, (4, 2)),
            zorder=2,
        )
        # Label the busiest, the quietest and the save percentage extremes.
        keep = set(_extreme_labels(x, frame["player"], n=2)) | set(
            _extreme_labels(y, frame["player"], n=2)
        )
        keep.discard(None)
        labels = [_short_name(p) if _short_name(p) in keep else None for p in frame["player"]]
        ax.margins(x=0.16, y=0.12)
        plotting.style_axis(
            ax,
            xlabel="Shots on target against per 90",
            ylabel="Save percentage" if ax is axes[0] else "",
            title=f"{season}   r = {info['pearson_r']:.2f}, p = {info['p_value']:.2f}",
        )
        line = np.column_stack(
            [grid, info["ols_intercept"] + info["ols_slope_savepct_per_extra_sota_per90"] * grid]
        )
        _place_labels(
            fig,
            ax,
            x,
            y,
            labels,
            all_points=np.vstack([np.column_stack([x, y]), line]),
        )
    plotting.save_figure(fig, "keeper_savepct_vs_workload")


def figure_clusters_pca(by_season: dict[str, pd.DataFrame], cluster_info: dict[str, dict]) -> None:
    seasons = list(by_season)
    max_k = max(cluster_info[s]["chosen_k"] for s in seasons)
    if max_k <= plotting.MAX_CATEGORICAL:
        colors = plotting.categorical(max_k)
        marks = plotting.markers(max_k)
        fig, axes = plt.subplots(
            1, len(seasons), figsize=(plotting.WIDTH_FULL, 3.2), constrained_layout=True
        )
        axes = np.atleast_1d(axes)
        for ax, season in zip(axes, seasons, strict=True):
            frame = by_season[season]
            info = cluster_info[season]
            for c in sorted(frame["gk_cluster"].unique()):
                sub = frame[frame["gk_cluster"] == c]
                ax.scatter(
                    sub["gk_pc1"],
                    sub["gk_pc2"],
                    s=30,
                    color=colors[c],
                    marker=marks[c],
                    edgecolor=plotting.SURFACE,
                    linewidth=0.5,
                    label=f"cluster {c} (n={len(sub)})",
                    zorder=3,
                )
            ax.margins(x=0.18, y=0.14)
            plotting.style_axis(
                ax,
                xlabel="PC1",
                ylabel="PC2" if ax is axes[0] else "",
                title=f"{season}   k = {info['chosen_k']}, silhouette = {info['silhouette']:.2f}",
            )
            # Clusters are keyed by direct centroid labels in the matching colour rather
            # than by a legend, because the cluster ids of two seasons are unrelated and
            # a shared legend strip below the panels reads as though they were the same.
            marker_x = frame["gk_pc1"].to_numpy(float)
            marker_y = frame["gk_pc2"].to_numpy(float)
            key_x, key_y, key_text, key_colour = [], [], [], []
            for c in sorted(frame["gk_cluster"].unique()):
                sub = frame[frame["gk_cluster"] == c]
                key_x.append(float(sub["gk_pc1"].mean()))
                key_y.append(float(sub["gk_pc2"].mean()))
                key_text.append(f"C{c}, n={len(sub)}")
                key_colour.append(colors[c])
                ax.scatter(
                    key_x[-1],
                    key_y[-1],
                    s=70,
                    facecolor="none",
                    edgecolor=colors[c],
                    linewidth=1.1,
                    marker=marks[c],
                    zorder=5,
                )
            extremes = _extreme_labels(frame["gk_pc1"], frame["player"], n=2)
            # Centroid keys are placed first so they win the uncluttered offsets, then
            # the two extreme players per panel are fitted around them in one pass.
            _place_labels(
                fig,
                ax,
                [*key_x, *marker_x],
                [*key_y, *marker_y],
                [*key_text, *extremes],
                colors=[*key_colour, *([plotting.INK_SECONDARY] * len(marker_x))],
                all_points=np.column_stack([marker_x, marker_y]),
            )
    else:
        # More than four clusters, so identity is carried by small multiples rather than
        # by a fifth hue, per the palette rule in plotting.py.
        panels = [(s, c) for s in seasons for c in sorted(by_season[s]["gk_cluster"].unique())]
        fig, axes = plotting.facet_grid(len(panels), ncols=max_k, panel_h=1.6)
        for ax, (season, c) in zip(axes, panels, strict=True):
            frame = by_season[season]
            ax.scatter(frame["gk_pc1"], frame["gk_pc2"], s=12, color=plotting.CONTEXT_GREY)
            sub = frame[frame["gk_cluster"] == c]
            ax.scatter(sub["gk_pc1"], sub["gk_pc2"], s=18, color=plotting.CATEGORICAL[0], zorder=3)
            ax.set_title(f"{season} cluster {c} (n={len(sub)})", loc="left")
    plotting.save_figure(fig, "keeper_clusters_pca")


def figure_composites(by_season: dict[str, pd.DataFrame]) -> None:
    seasons = list(by_season)
    colors = plotting.categorical(2)
    marks = plotting.markers(2)
    fig, axes = plt.subplots(
        1,
        len(seasons),
        figsize=(plotting.WIDTH_FULL, 3.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for i, (ax, season) in enumerate(zip(axes, seasons, strict=True)):
        frame = by_season[season]
        ax.axhline(0, color=plotting.GRID, linewidth=0.8, zorder=1)
        ax.axvline(0, color=plotting.GRID, linewidth=0.8, zorder=1)
        ax.scatter(
            frame["gk_workload"],
            frame["gk_shot_stopping"],
            s=28,
            color=colors[i],
            marker=marks[i],
            edgecolor=plotting.SURFACE,
            linewidth=0.5,
            zorder=3,
        )
        # Label the four corners of the plane plus the shot-stopping extremes.
        distance = np.hypot(frame["gk_workload"], frame["gk_shot_stopping"])
        keep = set(_extreme_labels(frame["gk_shot_stopping"], frame["player"], n=2))
        corner = distance.nlargest(3).index
        keep |= {_short_name(p) for p in frame.loc[corner, "player"]}
        keep.discard(None)
        labels = [_short_name(p) if _short_name(p) in keep else None for p in frame["player"]]
        ax.margins(x=0.18, y=0.14)
        plotting.style_axis(
            ax,
            xlabel="Workload composite (z)",
            ylabel="Shot-stopping composite (z)" if i == 0 else "",
            title=season,
        )
        _place_labels(fig, ax, frame["gk_workload"], frame["gk_shot_stopping"], labels)
    plotting.save_figure(fig, "keeper_composites")


def figure_shot_stopping_ranking(by_season: dict[str, pd.DataFrame]) -> None:
    seasons = list(by_season)
    tallest = max(len(by_season[s]) for s in seasons)
    fig, axes = plt.subplots(
        1,
        len(seasons),
        figsize=(plotting.WIDTH_FULL, max(4.2, 0.16 * tallest + 1.0)),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    limit = max(abs(by_season[s]["gk_saves_above_volume"]).max() for s in seasons)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    for ax, season in zip(axes, seasons, strict=True):
        frame = by_season[season].sort_values("gk_saves_above_volume")
        values = frame["gk_saves_above_volume"].to_numpy()
        y = np.arange(len(frame))
        ax.barh(
            y,
            values,
            color=[plotting.DIVERGING(norm(v)) for v in values],
            edgecolor=plotting.INK_SECONDARY,
            linewidth=0.3,
            height=0.75,
            zorder=3,
        )
        ax.set_yticks(y, [_short_name(p) for p in frame["player"]])
        ax.tick_params(axis="y", length=0)
        ax.axvline(0, color=plotting.INK_SECONDARY, linewidth=0.7, zorder=4)
        ax.set_ylim(-0.8, len(frame) - 0.2)
        ax.grid(axis="y", visible=False)
        plotting.style_axis(
            ax, xlabel="Saves above volume expectation", title=f"{season} (n={len(frame)})"
        )
    plotting.save_figure(fig, "keeper_shot_stopping_ranking")


def figure_composite_distributions(by_season: dict[str, pd.DataFrame]) -> None:
    seasons = list(by_season)
    axes_names = [
        ("gk_shot_stopping", "Shot-stopping composite"),
        ("gk_workload", "Workload composite"),
    ]
    fig, axs = plt.subplots(
        1, 2, figsize=(plotting.WIDTH_FULL, 3.0), sharey=True, constrained_layout=True
    )
    colors = plotting.categorical(2)
    rng = np.random.default_rng(config.RANDOM_STATE)
    for ax, (col, nice) in zip(axs, axes_names, strict=True):
        data = [by_season[s][col].to_numpy() for s in seasons]
        parts = ax.violinplot(data, positions=range(len(seasons)), widths=0.7, showextrema=False)
        for body, colour in zip(parts["bodies"], colors, strict=True):
            body.set_facecolor(colour)
            body.set_alpha(0.22)
            body.set_edgecolor(colour)
            body.set_linewidth(0.8)
        bp = ax.boxplot(
            data,
            positions=range(len(seasons)),
            widths=0.18,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": plotting.INK_PRIMARY, "linewidth": 1.2},
            boxprops={"facecolor": plotting.SURFACE, "edgecolor": plotting.INK_SECONDARY},
            whiskerprops={"color": plotting.INK_SECONDARY},
            capprops={"color": plotting.INK_SECONDARY},
        )
        del bp
        for i, (values, colour) in enumerate(zip(data, colors, strict=True)):
            jitter = rng.uniform(-0.12, 0.12, size=values.size)
            ax.scatter(
                i + 0.28 + jitter,
                values,
                s=9,
                color=colour,
                alpha=0.75,
                linewidth=0,
                zorder=4,
            )
        ax.axhline(0, color=plotting.GRID, linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(seasons)), seasons)
        plotting.style_axis(ax, ylabel="z" if col == "gk_shot_stopping" else "", title=nice)
        ax.grid(axis="x", visible=False)
    plotting.save_figure(fig, "keeper_composite_distributions")


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def analyse_season(season: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (all keepers, eligible analysis frame, per-season metric blocks)."""
    everyone = load_keepers(season)
    eligible = everyone[everyone["eligible"]].copy().reset_index(drop=True)

    residuals, fit_info = fit_saves_on_sota(eligible)
    eligible["gk_saves_above_volume"] = residuals
    eligible["gk_saves_above_volume_p90"] = residuals / eligible["gk_nineties"]

    composites, comp_info = build_composites(eligible)
    eligible = pd.concat([eligible, composites], axis=1)

    labels, cluster_info = cluster_keepers(eligible)
    pc_scores = cluster_info.pop("_pc_scores")
    eligible["gk_cluster"] = labels
    eligible = pd.concat([eligible, pc_scores], axis=1)
    if "gk_pc2" not in eligible.columns:
        eligible["gk_pc2"] = 0.0

    profile_cols = [
        *CLUSTER_FEATURES,
        "gk_saves_above_volume",
        "gk_shot_stopping",
        "gk_workload",
        "gk_minutes",
        "team_points_per_match",
    ]
    cluster_info["centroids"] = {
        int(c): {
            "n": int((eligible["gk_cluster"] == c).sum()),
            "members": sorted(eligible.loc[eligible["gk_cluster"] == c, "player"].tolist()),
            "means": {
                col: float(eligible.loc[eligible["gk_cluster"] == c, col].mean())
                for col in profile_cols
            },
        }
        for c in sorted(eligible["gk_cluster"].unique())
    }

    ranking = eligible.sort_values("gk_saves_above_volume", ascending=False)
    comp_info["shot_stopping_index_ranking"] = [
        {
            "rank": i + 1,
            "player": r.player,
            "team": r.team,
            "gk_minutes": int(r.gk_minutes),
            "gk_saves_above_volume": float(r.gk_saves_above_volume),
            "gk_saves_above_volume_p90": float(r.gk_saves_above_volume_p90),
            "gk_save_pct": float(r.gk_save_pct),
            "gk_shot_stopping": float(r.gk_shot_stopping),
        }
        for i, r in enumerate(ranking.itertuples())
    ]
    comp_composite = eligible.sort_values("gk_shot_stopping", ascending=False)
    comp_info["shot_stopping_composite_ranking"] = [
        {"rank": i + 1, "player": r.player, "gk_shot_stopping": float(r.gk_shot_stopping)}
        for i, r in enumerate(comp_composite.itertuples())
    ]
    comp_info["saves_on_sota_regression"] = fit_info
    comp_info["min_minutes"] = config.MIN_MINUTES
    comp_info["n_eligible"] = int(len(eligible))
    comp_info["n_all_keepers"] = int(len(everyone))

    descriptive = {
        "n_all_keepers": int(len(everyone)),
        "n_eligible": int(len(eligible)),
        "min_minutes": config.MIN_MINUTES,
        "all_keepers": _describe(everyone, DESCRIPTIVE_COLUMNS),
        "eligible_keepers": _describe(eligible, DESCRIPTIVE_COLUMNS),
        "correlations_eligible": {
            a: {b: float(eligible[a].corr(eligible[b])) for b in CORR_FEATURES}
            for a in CORR_FEATURES
        },
        "savepct_vs_workload": savepct_vs_workload(eligible),
        "surviving_feature_count": len(CLUSTER_FEATURES),
        "note": (
            "Every column summarised here comes from the basic keeper table. The advanced "
            "table contributes nothing because 23 of its 26 stat columns are empty."
        ),
    }
    return (
        everyone,
        eligible,
        {"descriptive": descriptive, "clusters": cluster_info, "composites": comp_info},
    )


def main() -> None:
    plotting.use_style()

    availability = audit_availability()
    _write_metrics("keeper_data_availability", availability)
    for season in config.SEASONS:
        adv = availability["seasons"][season]["keeper_adv"]
        print(
            f"{season}: keeper_adv {adv['all_null_stat_columns']}/{adv['stat_columns']} stat "
            f"columns entirely empty, {adv['usable_stat_columns']} usable",
            flush=True,
        )

    eligible_by_season: dict[str, pd.DataFrame] = {}
    all_by_season: dict[str, pd.DataFrame] = {}
    descriptive: dict[str, dict] = {}
    clusters: dict[str, dict] = {}
    composites: dict[str, dict] = {}

    for season in config.SEASONS:
        everyone, eligible, blocks = analyse_season(season)
        all_by_season[season] = everyone
        eligible_by_season[season] = eligible
        descriptive[season] = blocks["descriptive"]
        clusters[season] = blocks["clusters"]
        composites[season] = blocks["composites"]

        keep_cols = [
            "season",
            "player",
            "team",
            "gk_team",
            "gk_squads",
            "age",
            "nation",
            "gk_matches",
            "gk_starts",
            "gk_minutes",
            "gk_nineties",
            "gk_goals_against",
            "gk_shots_on_target_against",
            "gk_saves",
            "gk_clean_sheets",
            "gk_pens_faced",
            "gk_pens_allowed",
            "gk_pens_saved",
            *CORR_FEATURES,
            "gk_saves_above_volume",
            "gk_saves_above_volume_p90",
            "gk_shot_stopping",
            "gk_workload",
            "gk_pc1",
            "gk_pc2",
            "gk_cluster",
        ]
        out_path = config.DATA_PROCESSED / f"keepers_analysis_{season}.parquet"
        eligible[keep_cols].to_parquet(out_path, index=False)

        fit = composites[season]["saves_on_sota_regression"]
        info = clusters[season]
        print(
            f"{season}: n={len(everyone)} eligible={len(eligible)} "
            f"saves~SoTA R2={fit['r_squared']:.3f} k={info['chosen_k']} "
            f"silhouette={info['silhouette']:.3f} ({info['silhouette_band']})",
            flush=True,
        )

    workload_r = {s: descriptive[s]["savepct_vs_workload"]["pearson_r"] for s in config.SEASONS}
    descriptive["_shared"] = {
        "features": CLUSTER_FEATURES,
        "correlation_features": CORR_FEATURES,
        "savepct_vs_workload_across_seasons": {
            "pearson_r_by_season": workload_r,
            "sign_agrees": bool(len({r > 0 for r in workload_r.values() if np.isfinite(r)}) == 1),
            "verdict": (
                "Neither season shows a significant association between save percentage "
                "and shots on target against per 90"
                + (
                    ", and the sign of the correlation flips between the two seasons ("
                    + ", ".join(f"{s}: r = {r:.3f}" for s, r in workload_r.items())
                    + "), so busier goalkeepers are not shown to save at a different rate."
                    if len({r > 0 for r in workload_r.values() if np.isfinite(r)}) > 1
                    else ", so busier goalkeepers are not shown to save at a different rate."
                )
            ),
        },
        "derived_definitions": {
            "gk_nineties": "gk_minutes / 90",
            "gk_goals_against_p90": "gk_goals_against / gk_nineties",
            "gk_shots_on_target_against_p90": "gk_shots_on_target_against / gk_nineties",
            "gk_saves_p90": "gk_saves / gk_nineties",
            "gk_clean_sheets_p90": "gk_clean_sheets / gk_nineties",
            "gk_pens_faced_p90": "gk_pens_faced / gk_nineties",
            "gk_pens_saved_p90": "gk_pens_saved / gk_nineties",
            "gk_save_pct": "100 * gk_saves / gk_shots_on_target_against",
            "gk_clean_sheet_pct": "100 * gk_clean_sheets / gk_matches",
        },
    }
    chosen_ks = {s: clusters[s]["chosen_k"] for s in config.SEASONS}
    replicates = len(set(chosen_ks.values())) == 1
    clusters["_shared"] = {
        "seasons_compared": list(config.SEASONS),
        "chosen_k_by_season": chosen_ks,
        "k_replicates_across_seasons": replicates,
        "silhouette_by_season": {s: clusters[s]["silhouette"] for s in config.SEASONS},
        "min_cluster_size_by_season": {s: clusters[s]["min_cluster_size"] for s in config.SEASONS},
        "structure_meaningful_by_season": {
            s: clusters[s]["structure_is_meaningful"] for s in config.SEASONS
        },
        "overall_verdict": (
            "Goalkeeper clustering is a positive result: k replicates across seasons and "
            "both partitions clear the silhouette, cluster size and interpretability "
            "checks declared in advance."
            if replicates and all(clusters[s]["structure_is_meaningful"] for s in config.SEASONS)
            else "Goalkeeper clustering is a negative result in this study. "
            + (
                ""
                if replicates
                else "The consensus k rule selects a different k in each season ("
                + ", ".join(f"{s}: k = {k}" for s, k in chosen_ks.items())
                + "), so the partition does not replicate. "
            )
            + "The surviving keeper columns carry shot stopping and workload and nothing "
            "else, and neither season yields groups that are simultaneously well "
            "separated, large enough to profile and separated on goalkeeping rather than "
            "on minutes played and team strength. The clusters are reported for "
            "completeness and are not given archetype names."
        ),
    }
    composites["_shared"] = {
        "axes_planned": [
            "shot_stopping",
            "sweeping",
            "distribution",
            "cross_handling",
            "aerial_claims",
        ],
        "axes_delivered": ["shot_stopping", "workload"],
        "reason": (
            "Three of the five planned axes and the shot-quality half of the fourth "
            "depend entirely on keeper_adv columns that FBref emptied in January 2026. "
            "Workload replaces none of them: it is a description of the situation a "
            "keeper works in, reported because it is the second thing the surviving "
            "columns measure, not as a substitute for what was lost."
        ),
    }

    _write_metrics("keeper_descriptive", descriptive)
    _write_metrics("keeper_clusters", clusters)
    _write_metrics("keeper_composites", composites)

    figure_corr_heatmap(eligible_by_season)
    figure_saves_vs_sota(
        eligible_by_season,
        {s: composites[s]["saves_on_sota_regression"] for s in config.SEASONS},
    )
    figure_savepct_vs_workload(
        eligible_by_season, {s: descriptive[s]["savepct_vs_workload"] for s in config.SEASONS}
    )
    figure_clusters_pca(eligible_by_season, clusters)
    figure_composites(eligible_by_season)
    figure_shot_stopping_ranking(eligible_by_season)
    figure_composite_distributions(eligible_by_season)

    print("keepers complete: 4 metric files, 7 figures", flush=True)


if __name__ == "__main__":
    main()
