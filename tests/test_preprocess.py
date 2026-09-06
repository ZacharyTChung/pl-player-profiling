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
