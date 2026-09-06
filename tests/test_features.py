"""Guard the feature definitions against FBref's empty-shell columns.

FBref still serves the full schema for tables whose Opta-derived values were deleted in
January 2026. A feature list built from the published schema therefore looks correct but
yields an all-NaN matrix, so every feature is checked against real values here.
"""

from __future__ import annotations

import pandas as pd
import pytest

import config
from src import features as F

SEASONS = config.SEASONS


@pytest.mark.parametrize("season", SEASONS)
def test_every_fbref_source_column_exists(season: str) -> None:
    for canon, (table, raw) in F.FBREF_SOURCES.items():
        path = config.DATA_RAW / season / f"players_{table}.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing; run ingest first")
        cols = pd.read_parquet(path).columns
        assert raw in cols, f"{canon}: {raw!r} absent from players_{table}"


@pytest.mark.parametrize("season", SEASONS)
def test_every_fbref_source_column_has_data(season: str) -> None:
    """The central guard: reject any column that FBref serves but no longer populates."""
    empty = []
    for canon, (table, raw) in F.FBREF_SOURCES.items():
        path = config.DATA_RAW / season / f"players_{table}.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} missing; run ingest first")
        series = pd.read_parquet(path)[raw]
        if series.notna().sum() == 0:
            empty.append(f"{canon} ({table}.{raw})")
    assert not empty, f"features mapped to empty FBref columns: {empty}"


@pytest.mark.parametrize("season", SEASONS)
def test_every_understat_source_column_has_data(season: str) -> None:
    path = config.DATA_RAW / season / "understat_players.parquet"
    if not path.exists():
        pytest.skip("understat pull missing")
    df = pd.read_parquet(path)
    for canon, raw in F.UNDERSTAT_SOURCES.items():
        assert raw in df.columns, f"{canon}: {raw!r} absent from understat"
        assert df[raw].notna().sum() > 0, f"{canon}: {raw!r} is empty"


@pytest.mark.parametrize("season", SEASONS)
def test_keeper_sources_have_data(season: str) -> None:
    empty = []
    for canon, (table, raw) in F.KEEPER_SOURCES.items():
        path = config.DATA_RAW / season / f"players_{table}.parquet"
        if not path.exists():
            pytest.skip("keeper pull missing")
        df = pd.read_parquet(path)
        assert raw in df.columns, f"{canon}: {raw!r} absent"
        if df[raw].notna().sum() == 0:
            empty.append(canon)
    assert not empty, f"keeper features mapped to empty columns: {empty}"


@pytest.mark.parametrize("season", SEASONS)
def test_outfield_core_present_in_processed(season: str) -> None:
    path = config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet"
    if not path.exists():
        pytest.skip("run preprocess first")
    df = pd.read_parquet(path)
    missing = [c for c in F.OUTFIELD_CORE if c not in df.columns]
    assert not missing, f"OUTFIELD_CORE columns missing after preprocess: {missing}"


@pytest.mark.parametrize("season", SEASONS)
def test_outfield_core_has_variance(season: str) -> None:
    """A zero-variance feature would break standardisation and contribute nothing."""
    path = config.DATA_PROCESSED / f"outfield_eligible_{season}.parquet"
    if not path.exists():
        pytest.skip("run preprocess first")
    df = pd.read_parquet(path)
    flat = [c for c in F.OUTFIELD_CORE if c in df.columns and df[c].std(ddof=0) == 0]
    assert not flat, f"zero-variance features: {flat}"


def test_derived_feature_lists_are_consistent() -> None:
    assert [f"{c}_p90" for c in F.OUTFIELD_COUNTS] + F.OUTFIELD_RATES == F.OUTFIELD_CORE
    assert [f"{c}_p90" for c in F.KEEPER_COUNTS] + F.KEEPER_RATES == F.KEEPER_CORE
    assert len(set(F.OUTFIELD_CORE)) == len(F.OUTFIELD_CORE), "duplicate feature names"


def test_radar_and_headline_axes_are_real_features() -> None:
    known = set(F.OUTFIELD_CORE)
    assert set(F.RADAR_AXES) <= known, set(F.RADAR_AXES) - known
    assert set(F.HEADLINE_METRICS) <= known, set(F.HEADLINE_METRICS) - known


def test_non_shooting_features_exclude_shooting() -> None:
    """The finishing-above-role regression must not see shooting volume."""
    banned = {
        "shots_p90",
        "shots_on_target_p90",
        "np_xg_p90",
        "goals_non_penalty_p90",
        "shot_accuracy_pct",
        "goals_per_shot",
    }
    assert not (set(F.NON_SHOOTING_FEATURES) & banned)


def test_team_strength_columns_are_not_features() -> None:
    """Team quality describes the squad, not the player's role."""
    banned = {"team_points_per_match", "team_plus_minus_per90", "on_off"}
    assert not (set(F.OUTFIELD_CORE) & banned)
    assert banned <= set(F.CONTEXT_COLS)
