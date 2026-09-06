"""Invariants for the merged analysis tables."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import config
from src import features as F

SEASONS = config.SEASONS


@pytest.fixture(scope="module")
def metrics() -> dict:
    path = config.METRICS / "preprocess.json"
    if not path.exists():
        pytest.skip("run `uv run python -m src.preprocess` first")
    return json.loads(path.read_text())


def _load(kind: str, season: str) -> pd.DataFrame:
    path = config.DATA_PROCESSED / f"{kind}_{season}.parquet"
    if not path.exists():
        pytest.skip(f"missing {path.name}; run preprocess first")
    return pd.read_parquet(path)


@pytest.mark.parametrize("season", SEASONS)
def test_no_duplicate_players_in_aggregated(season: str) -> None:
    df = _load("aggregated", season)
    dupes = df[df.duplicated(subset=["season", "player"], keep=False)]
    assert dupes.empty, f"duplicate player rows: {dupes['player'].tolist()[:10]}"


@pytest.mark.parametrize("season", SEASONS)
def test_per_squad_rows_unique_by_team_and_player(season: str) -> None:
    df = _load("per_squad", season)
    assert not df.duplicated(subset=["season", "team", "player"]).any()


@pytest.mark.parametrize("season", SEASONS)
def test_no_nans_in_features_after_imputation(season: str) -> None:
    df = _load("outfield_eligible", season)
    present = [c for c in F.OUTFIELD_CORE if c in df.columns]
    nan_cols = [c for c in present if df[c].isna().any()]
    assert not nan_cols, f"NaNs remain in {nan_cols}"


@pytest.mark.parametrize("season", SEASONS)
def test_per90_values_are_physically_plausible(season: str) -> None:
    """Per-90 rates must be non-negative and within a generous football ceiling."""
    df = _load("outfield_eligible", season)
    ceilings = {
        "np_xg_p90": 2.0,
        "xa_p90": 2.0,
        "key_passes_p90": 10.0,
        "shots_p90": 12.0,
        "shots_on_target_p90": 8.0,
        "goals_non_penalty_p90": 3.0,
        "assists_p90": 3.0,
        "crosses_p90": 15.0,
        "interceptions_p90": 10.0,
        "tackles_won_p90": 10.0,
        "fouls_committed_p90": 10.0,
        # Adjusted rates carry the same ceilings widened by the largest plausible
        # adjustment factor, which is bounded by the league's possession spread.
        "interceptions_padj_p90": 20.0,
        "tackles_won_padj_p90": 20.0,
        "fouls_committed_padj_p90": 20.0,
        "offsides_p90": 5.0,
        "cards_yellow_p90": 2.0,
    }
    for col, ceiling in ceilings.items():
        if col not in df.columns:
            continue
        assert df[col].min() >= 0, f"{col} has negative values"
        assert df[col].max() <= ceiling, f"{col} max {df[col].max():.2f} exceeds {ceiling}"


@pytest.mark.parametrize("season", SEASONS)
def test_percentage_columns_are_bounded(season: str) -> None:
    df = _load("outfield_eligible", season)
    if "shot_accuracy_pct" in df.columns:
        assert df["shot_accuracy_pct"].between(0, 100).all()
    if "goals_per_shot" in df.columns:
        assert df["goals_per_shot"].between(0, 1).all()


@pytest.mark.parametrize("season", SEASONS)
def test_eligible_rows_match_minutes_filter(season: str, metrics: dict) -> None:
    """Row count must equal the aggregated outfield pool minus the minutes filter."""
    outfield = _load("aggregated", season)
    outfield = outfield[outfield["position_group"].isin(config.OUTFIELD_GROUPS)]
    expected = int((outfield["minutes"] >= config.MIN_MINUTES).sum())
    eligible = _load("outfield_eligible", season)
    assert len(eligible) == expected
    assert metrics[season]["rows_outfield_eligible"] == expected


@pytest.mark.parametrize("season", SEASONS)
def test_all_eligible_players_clear_the_threshold(season: str) -> None:
    df = _load("outfield_eligible", season)
    assert df["minutes"].min() >= config.MIN_MINUTES


@pytest.mark.parametrize("season", SEASONS)
def test_position_groups_are_valid(season: str) -> None:
    df = _load("aggregated", season)
    groups = set(df["position_group"].dropna().unique())
    assert groups <= set(config.POSITION_GROUPS), f"unexpected groups: {groups}"
    eligible = _load("outfield_eligible", season)
    assert set(eligible["position_group"].unique()) <= set(config.OUTFIELD_GROUPS)


@pytest.mark.parametrize("season", SEASONS)
def test_standardised_tables_are_actually_standardised(season: str) -> None:
    df = _load("outfield_z_global", season)
    present = [c for c in F.OUTFIELD_CORE if c in df.columns]
    means = df[present].mean().abs()
    stds = df[present].std(ddof=0)
    assert (means < 1e-9).all(), "global z-scores are not centred"
    assert ((stds - 1.0).abs() < 1e-9).all(), "global z-scores are not unit variance"


@pytest.mark.parametrize("season", SEASONS)
def test_group_standardisation_is_within_position(season: str) -> None:
    df = _load("outfield_z_bygroup", season)
    present = [c for c in F.OUTFIELD_CORE if c in df.columns]
    by_group = df.groupby("position_group")[present].mean().abs()
    assert (by_group < 1e-9).all().all(), "per-group z-scores are not centred within group"


@pytest.mark.parametrize("season", SEASONS)
def test_understat_match_rate_is_high(season: str, metrics: dict) -> None:
    """A low match rate would silently starve the xG features."""
    rate = metrics[season]["understat_match"]["match_rate"]
    assert rate > 0.90, f"Understat match rate fell to {rate:.1%}"


@pytest.mark.parametrize("season", SEASONS)
def test_no_ambiguous_name_matches_were_accepted(season: str, metrics: dict) -> None:
    """Token-subset matching must never resolve a name to more than one candidate."""
    assert metrics[season]["understat_match"]["ambiguous_token_matches"] == []


@pytest.mark.parametrize("season", SEASONS)
def test_multi_club_players_are_aggregated_not_duplicated(season: str) -> None:
    per_squad = _load("per_squad", season)
    aggregated = _load("aggregated", season)
    movers = per_squad["player"].value_counts()
    movers = movers[movers > 1]
    if movers.empty:
        pytest.skip("no mid-season transfers in this season")
    for name in movers.index[:5]:
        rows = aggregated[aggregated["player"] == name]
        assert len(rows) == 1
        assert rows["n_squads"].iloc[0] >= 2
        summed = per_squad.loc[per_squad["player"] == name, "minutes"].sum()
        assert abs(rows["minutes"].iloc[0] - summed) < 1e-6


@pytest.mark.parametrize("season", SEASONS)
def test_possession_adjustment_is_correctly_scaled(season: str) -> None:
    """Each adjusted rate must equal the raw rate times the recorded factor."""
    df = _load("outfield_eligible", season)
    assert "possession_exposure_ratio" in df.columns
    assert "team_possession" in df.columns
    expected = config.PADJ_REFERENCE_POSSESSION / (100.0 - df["team_possession"])
    assert (df["possession_exposure_ratio"] - expected).abs().max() < 1e-9
    for count in config.PADJ_COUNTS:
        raw, adj, fac = f"{count}_p90", f"{count}_padj_p90", f"{count}_padj_factor"
        if raw in df.columns and adj in df.columns:
            assert (df[adj] - df[raw] * df[fac]).abs().max() < 1e-9
            # Each feature carries its own fitted elasticity, so its factor is the
            # exposure ratio raised to that exponent and must move with it monotonically.
            assert df[fac].corr(df["possession_exposure_ratio"]) > 0.99


@pytest.mark.parametrize("season", SEASONS)
def test_adjustment_factor_is_bounded_and_centred(season: str) -> None:
    """A factor far from one would mean the reference possession is miscalibrated."""
    df = _load("outfield_eligible", season)
    for count in config.PADJ_COUNTS:
        fac = df[f"{count}_padj_factor"]
        assert fac.between(0.5, 2.0).all(), f"{count}: implausible factors"
        assert 0.9 < fac.mean() < 1.15, f"{count}: factor mean {fac.mean():.3f} is not near one"


@pytest.mark.parametrize("season", SEASONS)
def test_adjustment_moves_defensive_rates_toward_team_neutrality(season: str) -> None:
    """The point of the adjustment: leave defensive rates roughly team-neutral.

    The target is a correlation with team possession near zero, not merely a smaller one
    than before. Where a raw correlation is already negligible there is nothing to
    remove, and demanding further reduction would only be testing noise, so the
    condition is satisfied either by improving on the raw value or by landing inside a
    near-zero band.
    """
    df = _load("outfield_eligible", season)
    neutral_band = 0.10
    for count in config.PADJ_COUNTS:
        raw, adj = f"{count}_p90", f"{count}_padj_p90"
        if raw not in df.columns or adj not in df.columns:
            continue
        raw_corr = abs(df[raw].corr(df["team_possession"]))
        adj_corr = abs(df[adj].corr(df["team_possession"]))
        assert adj_corr < max(raw_corr, neutral_band), (
            f"{count}: possession correlation {raw_corr:.3f} -> {adj_corr:.3f}, "
            f"outside the neutral band of {neutral_band}"
        )


def test_adjustment_reduces_the_confound_in_the_primary_season() -> None:
    """In the season the paper is about, every adjusted rate must beat its raw form.

    This is the claim the paper actually makes, so it is asserted directly rather than
    inferred from the weaker per-season condition above.
    """
    df = _load("outfield_eligible", config.SEASON_PRIMARY)
    for count in config.PADJ_COUNTS:
        raw_corr = abs(df[f"{count}_p90"].corr(df["team_possession"]))
        adj_corr = abs(df[f"{count}_padj_p90"].corr(df["team_possession"]))
        assert adj_corr < raw_corr, f"{count}: {raw_corr:.3f} -> {adj_corr:.3f}"


def test_fitted_elasticities_are_below_unity() -> None:
    """The finding that motivates the whole approach.

    A textbook adjustment assumes an elasticity of one. If the fitted values were near
    one, the standard formula would be right and this machinery unnecessary; they are
    not, and an exponent of one demonstrably overcorrects.
    """
    path = config.METRICS / "preprocess.json"
    if not path.exists():
        pytest.skip("run preprocess first")
    report = json.loads(path.read_text())[config.SEASON_PRIMARY]["possession_adjustment"]
    elasticities = report["elasticity"]["elasticities"]
    assert set(elasticities) == set(config.PADJ_COUNTS)
    for count, alpha in elasticities.items():
        assert 0.0 < alpha < 1.0, f"{count}: elasticity {alpha} is not strictly inside (0, 1)"
