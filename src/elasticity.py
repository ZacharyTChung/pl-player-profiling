"""How sure are the possession elasticities?

The possession adjustment rests on five exponents, fitted by regressing log team
defensive volume on log opponent possession across the archive's team-seasons, and the
paper's claim is that all five sit below one, so the conventional adjustment with an
exponent of one overcorrects. Those exponents are point estimates. Four of them are far
from one, but clearances at roughly 0.96 is close enough that the claim may not survive
an interval, and a paper that criticises unevidenced claims cannot make that one without
evidence.

This module supplies the missing uncertainty, three ways:

* a **bootstrap over team-seasons**, resampled with replacement and refitted exactly as
  :func:`src.archive.fit_elasticities` fits them. It gives each exponent a percentile
  interval, a standard error and a one-sided test against the conventional value of one.
* **refits within each league and within each season**, each with its own bootstrap. The
  paper hedges that the exponents should not be carried to a competition with a wider
  possession spread without refitting; this measures whether that hedge is needed.
* a **check on the functional form**. A log-log fit assumes the elasticity is the same
  at every level of possession. Adding a quadratic term in log opponent possession and
  bootstrapping it asks whether that is true, and the fitted curve is read off at the
  low and high ends of the observed range so a non-constant slope is reported in the
  units the paper uses.

The outcome is written to the metrics directory whatever it is. If clearances cannot be
told apart from one, the paper has to say so, and this is where it finds out.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src import archive as A
from src import plotting as P

#: Bootstrap resamples of team-seasons. Two thousand is enough for the ends of a 95
#: percent percentile interval to settle at the third decimal.
N_BOOTSTRAP = 2000

#: Two-sided coverage of the reported percentile interval.
CONFIDENCE = 0.95

#: The conventional exponent, and the null hypothesis throughout.
UNIT_ELASTICITY = config.PADJ_UNIT_ELASTICITY

#: Quantiles of opponent possession at which the quadratic model's implied elasticity is
#: evaluated, so a slope that changes across the range is reported at both ends of it.
RANGE_QUANTILES = (0.05, 0.95)

LABELS = {
    "tackles": "Tackles",
    "interceptions": "Interceptions",
    "blocks": "Blocks",
    "clearances": "Clearances",
    "fouls_committed": "Fouls committed",
}

#: Short league names for the figure, keyed by the archive's competition label.
SHORT = {
    "Premier League": "England",
    "La Liga": "Spain",
    "Bundesliga": "Germany",
    "Serie A": "Italy",
    "Ligue 1": "France",
}


def team_seasons() -> pd.DataFrame:
    """One row per team-season with opponent possession and each defensive rate.

    Built exactly as :func:`src.archive.fit_elasticities` builds its regression frame:
    player rows summed to the club, merged with the team possession table, divided by
    nineties. The full-sample fit on this frame therefore reproduces the archive's
    exponents to floating point, which :func:`main` asserts rather than assumes. The
    competition label is carried along for the per-league refits; a club plays in exactly
    one competition per season, so taking the first is taking the only.
    """
    poss = A.team_possession()
    players, _ = A.load_players()
    agg = players.groupby(["Season_End_Year", "Squad"], as_index=False).agg(
        {**{c: "sum" for c in A.DEFENSIVE_COUNTS}, "minutes": "sum", "Comp": "first"}
    )
    agg = agg.merge(poss, on=["Season_End_Year", "Squad"], how="inner")
    agg["nineties"] = agg["minutes"] / 90.0
    agg["opponent"] = 100.0 - agg["team_possession"]
    for count in A.DEFENSIVE_COUNTS:
        agg[count] = agg[count] / agg["nineties"]
    agg = agg.rename(columns={"Season_End_Year": "season", "Comp": "league"})
    return agg.reset_index(drop=True)


def _logs(frame: pd.DataFrame, count: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Log opponent possession, log rate, and the rows the archive's fit would keep."""
    rate = frame[count].to_numpy(dtype=float)
    opp = frame["opponent"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lx, ly = np.log(opp), np.log(rate)
    valid = np.isfinite(lx) & np.isfinite(ly)
    return lx, ly, valid


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(x, y, 1)[0])


def bootstrap_indices(n: int) -> np.ndarray:
    """The resample matrix, drawn once per scope so every count is refitted on the same draws."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    return rng.integers(0, n, size=(N_BOOTSTRAP, n))


def bootstrap_elasticities(frame: pd.DataFrame) -> dict[str, dict]:
    """Percentile intervals for each elasticity from resampled team-seasons.

    The frame is resampled whole, so the five counts are refitted on the same draw, and
    any row the positivity filter would drop is dropped inside the draw. That is what
    refitting :func:`src.archive.fit_elasticities` on a resampled archive would do.

    Two quantities are reported against the conventional exponent. The share of resamples
    at or above one reads the percentile distribution directly. The p-value is the
    bootstrap test of the null that the elasticity is at least one: the resampled slopes
    are recentred on one and the p-value is the share of them at or below the observed
    estimate. With a symmetric bootstrap distribution the two agree, so a gap between
    them is a sign of skew rather than of anything else.
    """
    idx = bootstrap_indices(len(frame))
    lo_q, hi_q = 100.0 * (1.0 - CONFIDENCE) / 2.0, 100.0 * (1.0 + CONFIDENCE) / 2.0
    out = {}
    for count in A.DEFENSIVE_COUNTS:
        lx, ly, valid = _logs(frame, count)
        alpha = _slope(lx[valid], ly[valid])
        boots = np.empty(N_BOOTSTRAP)
        for b, rows in enumerate(idx):
            keep = valid[rows]
            boots[b] = _slope(lx[rows][keep], ly[rows][keep])
        low, high = np.percentile(boots, [lo_q, hi_q])
        se = float(boots.std(ddof=1))
        recentred = boots - alpha + UNIT_ELASTICITY
        out[count] = {
            "elasticity": round(alpha, 4),
            "n_team_seasons": int(valid.sum()),
            "bootstrap_se": round(se, 4),
            "ci_low": round(float(low), 4),
            "ci_high": round(float(high), 4),
            "interval_excludes_one": bool(high < UNIT_ELASTICITY or low > UNIT_ELASTICITY),
            "share_at_or_above_one": round(float(np.mean(boots >= UNIT_ELASTICITY)), 4),
            "p_value_at_least_one": round(float(np.mean(recentred <= alpha)), 4),
            "distance_from_one_in_se": round((alpha - UNIT_ELASTICITY) / se, 2) if se > 0 else None,
        }
    return out


def _possession_spread(frame: pd.DataFrame) -> dict:
    opp = frame["opponent"].to_numpy(dtype=float)
    return {
        "min": round(float(opp.min()), 2),
        "max": round(float(opp.max()), 2),
        "sd": round(float(opp.std(ddof=1)), 3),
    }


def by_scope(frame: pd.DataFrame, column: str, values: list) -> dict[str, dict]:
    """Refit and bootstrap inside each league or each season."""
    out = {}
    for value in values:
        sub = frame[frame[column] == value].reset_index(drop=True)
        out[str(value)] = {
            "n_team_seasons": int(len(sub)),
            "opponent_possession": _possession_spread(sub),
            "per_count": bootstrap_elasticities(sub),
        }
    return out


def heterogeneity(pooled: dict, scopes: dict) -> dict[str, dict]:
    """How far the within-scope exponents move, and whether their intervals meet the pooled one.

    The range of point estimates is the plain measure. The overlap flags answer the
    transport question: a scope whose interval does not reach the pooled interval is one
    where the pooled exponent would be the wrong exponent.
    """
    out = {}
    for count in A.DEFENSIVE_COUNTS:
        p = pooled[count]
        estimates = {s: scopes[s]["per_count"][count]["elasticity"] for s in scopes}
        overlap = {
            s: bool(
                scopes[s]["per_count"][count]["ci_low"] <= p["ci_high"]
                and scopes[s]["per_count"][count]["ci_high"] >= p["ci_low"]
            )
            for s in scopes
        }
        contains = {
            s: bool(
                scopes[s]["per_count"][count]["ci_low"]
                <= p["elasticity"]
                <= scopes[s]["per_count"][count]["ci_high"]
            )
            for s in scopes
        }
        lowest = min(estimates, key=lambda s: estimates[s])
        highest = max(estimates, key=lambda s: estimates[s])
        out[count] = {
            "estimates": estimates,
            "min": estimates[lowest],
            "max": estimates[highest],
            "range": round(estimates[highest] - estimates[lowest], 4),
            "scope_of_min": lowest,
            "scope_of_max": highest,
            "interval_overlaps_pooled": overlap,
            "all_intervals_overlap_pooled": all(overlap.values()),
            "pooled_estimate_inside_every_interval": all(contains.values()),
            "n_estimates_at_or_above_one": int(
                sum(v >= UNIT_ELASTICITY for v in estimates.values())
            ),
            "n_intervals_excluding_one": int(
                sum(scopes[s]["per_count"][count]["interval_excludes_one"] for s in scopes)
            ),
        }
    return out


def _adjusted_r2(y: np.ndarray, fitted: np.ndarray, n_terms: int) -> float:
    resid = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    n = len(y)
    return 1.0 - (resid / total) * (n - 1) / (n - n_terms - 1)


def functional_form(frame: pd.DataFrame) -> dict[str, dict]:
    """Does a quadratic in log opponent possession beat the constant-elasticity line?

    Log possession is centred before the square is taken so the two terms are not nearly
    collinear; the quadratic coefficient is unaffected by the centring and the linear
    coefficient becomes the elasticity at the centre of the range. The elasticity the
    quadratic implies is then read off at the low and high ends of the observed range,
    since a curve that is flat in the middle can still bend at the edges where the
    transport hedge bites.
    """
    idx = bootstrap_indices(len(frame))
    lo_q, hi_q = 100.0 * (1.0 - CONFIDENCE) / 2.0, 100.0 * (1.0 + CONFIDENCE) / 2.0
    out = {}
    for count in A.DEFENSIVE_COUNTS:
        lx, ly, valid = _logs(frame, count)
        centre = float(lx[valid].mean())
        z, y = lx[valid] - centre, ly[valid]
        linear = np.polyfit(z, y, 1)
        quad = np.polyfit(z, y, 2)
        adj_linear = _adjusted_r2(y, np.polyval(linear, z), 1)
        adj_quad = _adjusted_r2(y, np.polyval(quad, z), 2)

        # Read the curve's slope where the data actually sit rather than at the centre.
        q_low, q_high = np.quantile(lx[valid], RANGE_QUANTILES)
        ends = np.array([q_low, q_high]) - centre

        c2 = np.empty(N_BOOTSTRAP)
        at_ends = np.empty((N_BOOTSTRAP, 2))
        for b, rows in enumerate(idx):
            keep = valid[rows]
            coef = np.polyfit(lx[rows][keep] - centre, ly[rows][keep], 2)
            c2[b] = coef[0]
            at_ends[b] = coef[1] + 2.0 * coef[0] * ends
        c2_low, c2_high = np.percentile(c2, [lo_q, hi_q])
        low_ci = np.percentile(at_ends[:, 0], [lo_q, hi_q])
        high_ci = np.percentile(at_ends[:, 1], [lo_q, hi_q])

        out[count] = {
            "adjusted_r2_linear": round(adj_linear, 4),
            "adjusted_r2_quadratic": round(adj_quad, 4),
            "adjusted_r2_change": round(adj_quad - adj_linear, 4),
            "quadratic_coefficient": round(float(quad[0]), 4),
            "quadratic_ci_low": round(float(c2_low), 4),
            "quadratic_ci_high": round(float(c2_high), 4),
            "quadratic_excludes_zero": bool(c2_high < 0.0 or c2_low > 0.0),
            "opponent_possession_low": round(float(np.exp(q_low)), 2),
            "opponent_possession_high": round(float(np.exp(q_high)), 2),
            "elasticity_at_low_possession": round(float(quad[1] + 2.0 * quad[0] * ends[0]), 4),
            "elasticity_at_low_ci": [round(float(v), 4) for v in low_ci],
            "elasticity_at_high_possession": round(float(quad[1] + 2.0 * quad[0] * ends[1]), 4),
            "elasticity_at_high_ci": [round(float(v), 4) for v in high_ci],
        }
    return out


def _prose(counts: list[str]) -> str:
    """Join statistic names the way a sentence would."""
    names = [LABELS[c].lower() for c in counts]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def figure_uncertainty(pooled: dict, leagues: dict) -> None:
    counts = sorted(A.DEFENSIVE_COUNTS, key=lambda c: pooled[c]["elasticity"])
    ypos = np.arange(len(counts))[::-1]
    fig, ax = plt.subplots(figsize=(P.WIDTH_FULL, 3.1))

    ax.axvline(
        UNIT_ELASTICITY, color=P.INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2
    )
    for y, count in zip(ypos, counts, strict=True):
        block = pooled[count]
        ax.plot(
            [block["ci_low"], block["ci_high"]],
            [y, y],
            color=P.CATEGORICAL[0],
            linewidth=1.8,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.plot(
            block["elasticity"],
            y,
            marker="o",
            markersize=7.0,
            color=P.CATEGORICAL[0],
            linestyle="none",
            zorder=5,
        )
        ax.text(
            block["ci_high"] + 0.025,
            y,
            f"{block['elasticity']:.2f} [{block['ci_low']:.2f}, {block['ci_high']:.2f}]",
            va="center",
            fontsize=P.BASE_FONT_PT - 2,
            color=P.INK_SECONDARY,
        )

    # Five leagues is one more than the palette allows, so league is a marker shape in a
    # single second colour. The circle is reserved for the pooled estimate.
    for j, league in enumerate(A.LEAGUES):
        xs = [leagues[league]["per_count"][c]["elasticity"] for c in counts]
        ax.plot(
            xs,
            ypos + 0.3,
            marker=P.MARKERS[j + 1],
            markersize=3.8,
            color=P.CATEGORICAL[1],
            markeredgecolor=P.SURFACE,
            markeredgewidth=0.4,
            linestyle="none",
            zorder=4,
        )

    upper = max(
        [pooled[c]["ci_high"] for c in counts]
        + [leagues[lg]["per_count"][c]["elasticity"] for lg in A.LEAGUES for c in counts]
        + [UNIT_ELASTICITY]
    )
    ax.set_xlim(0.0, upper + 0.4)
    ax.set_ylim(-0.6, len(counts) - 0.2)
    ax.set_yticks(ypos, [LABELS[c] for c in counts], fontsize=P.BASE_FONT_PT - 1)
    ax.grid(False, axis="y")
    P.style_axis(
        ax,
        "Elasticity of team defensive volume to opponent possession",
        "",
        f"Possession elasticities, {N_BOOTSTRAP:,} bootstrap resamples of team-seasons",
    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=P.CATEGORICAL[0],
            marker="o",
            markersize=6.0,
            linewidth=1.8,
            label="pooled, 95 percent interval",
        ),
        plt.Line2D(
            [0],
            [0],
            color=P.INK_SECONDARY,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            label="conventional exponent of one",
        ),
    ]
    for j, league in enumerate(A.LEAGUES):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=P.CATEGORICAL[1],
                marker=P.MARKERS[j + 1],
                markersize=4.0,
                linestyle="none",
                label=SHORT[league],
            )
        )
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    P.save_figure(fig, "elasticity_uncertainty")


def sweeper_exposure() -> dict:
    """Are goalkeeper sweeper actions an exposure count like tackles, or a tactical one?

    The paper possession adjusts every outfield defensive count on the argument that a
    tackle cannot happen while your own side has the ball. A sweeper action is also a
    count, so the same question has to be asked of it rather than assumed either way.
    The fit is the one used for the outfield counts: log team sweeper actions per ninety
    on log opponent possession, one row per team-season, with the same bootstrap.
    """
    adv = pd.read_parquet(A.RAW / "keepers_adv.parquet")
    base = pd.read_parquet(A.RAW / "keepers.parquet")
    keys = ["Season_End_Year", "Squad", "Player"]
    d = adv.merge(base[[*keys, "Min_Playing"]], on=keys, how="left")
    d = d[d["Season_End_Year"].isin(A.SEASONS)]
    team = d.groupby(["Season_End_Year", "Squad"], as_index=False).agg(
        actions=("#OPA_Sweeper", "sum"), minutes=("Min_Playing", "sum")
    )
    team = team.merge(A.team_possession(), on=["Season_End_Year", "Squad"], how="inner")
    team["rate"] = team["actions"] / (team["minutes"] / 90.0)
    team["opp"] = 100.0 - team["team_possession"]
    sub = team[(team["rate"] > 0) & (team["opp"] > 0)].dropna(subset=["rate", "opp"])
    x, y = np.log(sub["opp"].to_numpy()), np.log(sub["rate"].to_numpy())
    alpha = float(np.polyfit(x, y, 1)[0])
    rng = np.random.default_rng(config.RANDOM_STATE)
    draws = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(x), len(x))
        draws[i] = np.polyfit(x[idx], y[idx], 1)[0]
    lo_q, hi_q = 100.0 * (1.0 - CONFIDENCE) / 2.0, 100.0 * (1.0 + CONFIDENCE) / 2.0
    lo, hi = np.percentile(draws, [lo_q, hi_q])
    busiest = sub.sort_values("rate", ascending=False).head(3)
    return {
        "question": "Do sweeper actions fall with a team's own possession, as tackles do?",
        "n_team_seasons": int(len(sub)),
        "elasticity_to_opponent_possession": round(alpha, 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "log_log_correlation": round(float(np.corrcoef(x, y)[0, 1]), 4),
        "interval_excludes_zero": bool(hi < 0 or lo > 0),
        "sign": "negative" if alpha < 0 else "positive",
        "busiest_sweeping_team_seasons": [
            f"{r.Squad} {int(r.Season_End_Year)} ({r.team_possession:.1f} percent possession)"
            for r in busiest.itertuples()
        ],
        "reading": (
            "Sweeper actions rise with a team's own possession rather than falling with it, "
            "so they are a tactical count that tracks line height, not an exposure count, "
            "and adjusting them for possession would remove the signal rather than a bias."
            if alpha < 0
            else "Sweeper actions fall with a team's own possession like the outfield counts."
        ),
    }


def main() -> None:
    P.use_style()
    frame = team_seasons()

    # The archive's own fit is the number the paper prints, so agreement is asserted.
    archive = A.fit_elasticities()["alphas"]
    pooled = bootstrap_elasticities(frame)
    gap = max(abs(pooled[c]["elasticity"] - archive[c]) for c in A.DEFENSIVE_COUNTS)
    if gap > 1e-4:
        raise ValueError(f"pooled refit disagrees with archive.fit_elasticities by {gap:.6f}")

    leagues = by_scope(frame, "league", A.LEAGUES)
    seasons = by_scope(frame, "season", A.SEASONS)
    league_spread = heterogeneity(pooled, leagues)
    season_spread = heterogeneity(pooled, seasons)
    form = functional_form(frame)

    figure_uncertainty(pooled, leagues)

    excludes = [c for c in A.DEFENSIVE_COUNTS if pooled[c]["interval_excludes_one"]]
    ambiguous = [c for c in A.DEFENSIVE_COUNTS if not pooled[c]["interval_excludes_one"]]
    bends = [c for c in A.DEFENSIVE_COUNTS if form[c]["quadratic_excludes_zero"]]
    transport = [
        c for c in A.DEFENSIVE_COUNTS if not league_spread[c]["all_intervals_overlap_pooled"]
    ]
    results = {
        "question": "Are the five possession elasticities distinguishable from one?",
        "method": {
            "fit": (
                "ordinary least squares of log team defensive volume per ninety on log "
                "opponent possession, one row per team-season, as in archive.fit_elasticities"
            ),
            "bootstrap": (
                "team-seasons resampled with replacement, all five counts refitted on each "
                "draw; percentile interval, standard error, share of draws at or above one, "
                "and a one-sided p-value from the draws recentred on one"
            ),
            "scopes": "the same bootstrap repeated inside each league and inside each season",
            "functional_form": (
                "a quadratic in centred log opponent possession against the line, compared "
                "by adjusted R squared and by the bootstrap interval of the squared term"
            ),
        },
        "n_bootstrap": N_BOOTSTRAP,
        "confidence": CONFIDENCE,
        "seed": config.RANDOM_STATE,
        "n_team_seasons": int(len(frame)),
        "opponent_possession": _possession_spread(frame),
        "max_gap_to_archive_fit": round(gap, 6),
        "pooled": pooled,
        "by_league": leagues,
        "by_season": seasons,
        "heterogeneity": {"league": league_spread, "season": season_spread},
        "functional_form": form,
        "sweeper": sweeper_exposure(),
        "summary": {
            "all_pooled_below_one": all(
                pooled[c]["elasticity"] < UNIT_ELASTICITY for c in A.DEFENSIVE_COUNTS
            ),
            "intervals_excluding_one": excludes,
            "not_distinguishable_from_one": ambiguous,
            "largest_league_range": max(league_spread[c]["range"] for c in A.DEFENSIVE_COUNTS),
            "counts_with_a_league_interval_missing_the_pooled_one": transport,
            "quadratic_term_matters": bends,
            "verdict": (
                "Every pooled interval excludes one, so the sub-proportional claim holds "
                "for all five statistics."
                if not ambiguous
                else f"The interval for {_prose(ambiguous)} reaches one, so the "
                f"sub-proportional claim is established for {_prose(excludes)} and is "
                f"not established for {_prose(ambiguous)}."
            ),
        },
    }
    with open(config.METRICS / "elasticity.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"elasticity: {len(frame)} team-seasons, {N_BOOTSTRAP:,} bootstrap resamples")
    for count in A.DEFENSIVE_COUNTS:
        p, h = pooled[count], league_spread[count]
        print(
            f"  {count:16s} {p['elasticity']:.3f} [{p['ci_low']:.3f}, {p['ci_high']:.3f}]"
            f"  se {p['bootstrap_se']:.3f}  p(>=1) {p['p_value_at_least_one']:.3f}"
            f"  excludes one {str(p['interval_excludes_one']):5s}"
            f"  leagues {h['min']:.3f} to {h['max']:.3f}"
            f"  overlap {str(h['all_intervals_overlap_pooled'])}"
        )
    print("functional form, quadratic in log opponent possession:")
    for count in A.DEFENSIVE_COUNTS:
        f = form[count]
        print(
            f"  {count:16s} squared term {f['quadratic_coefficient']:+.3f}"
            f" [{f['quadratic_ci_low']:+.3f}, {f['quadratic_ci_high']:+.3f}]"
            f"  adj R2 change {f['adjusted_r2_change']:+.4f}"
            f"  elasticity {f['elasticity_at_low_possession']:.2f} at"
            f" {f['opponent_possession_low']:.0f} to {f['elasticity_at_high_possession']:.2f}"
            f" at {f['opponent_possession_high']:.0f} percent"
        )
    print("verdict:", results["summary"]["verdict"])
    print("elasticity complete")


if __name__ == "__main__":
    main()
