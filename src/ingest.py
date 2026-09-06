"""FBref ingestion.

soccerdata 1.9.1 is the only release whose FBref reader clears Cloudflare (it uses the
seleniumbase driver), but the same rewrite that introduced the driver also removed six
season stat tables that this study depends on: passing, passing_types,
goal_shot_creation, defense, possession and keeper_adv.

``FBrefFull`` restores them. It reuses soccerdata's own downloader, cache and table
parser; only the allowed stat_type list, the URL slug map and the table-id map are
re-declared. No raw request loop is written against fbref.com.

The URL slugs and table ids were verified empirically against live FBref pages, which
is how the ``goal_shot_creation -> stats_gca`` mismatch was found (soccerdata 1.8.8
looked for ``stats_goal_shot_creation``, which does not exist).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import config

os.environ.setdefault("SOCCERDATA_DIR", str(config.HTML_CACHE))

import pandas as pd  # noqa: E402
import soccerdata as sd  # noqa: E402
from lxml import etree, html  # noqa: E402
from soccerdata._common import standardize_colnames  # noqa: E402
from soccerdata._config import TEAMNAME_REPLACEMENTS  # noqa: E402
from soccerdata.fbref import (  # noqa: E402
    FBREF_API,
    _concat,
    _fix_nation_col,
    _parse_table,
)

# stat_type -> URL path segment on fbref.com
PAGE_SLUG = {
    "standard": "stats",
    "shooting": "shooting",
    "passing": "passing",
    "passing_types": "passing_types",
    "goal_shot_creation": "gca",
    "defense": "defense",
    "possession": "possession",
    "playing_time": "playingtime",
    "misc": "misc",
    "keeper": "keepers",
    "keeper_adv": "keepersadv",
}

# stat_type -> the token FBref uses inside table ids. Identical to stat_type except GCA.
TABLE_KEY = dict.fromkeys(PAGE_SLUG)
TABLE_KEY.update({k: k for k in PAGE_SLUG})
TABLE_KEY["goal_shot_creation"] = "gca"


def _find_table(tree, table_id: str):
    """Locate a table by id, whether it is live in the DOM or inside an HTML comment.

    FBref serves the player tables inside comment nodes and the squad tables live, so
    both paths are needed.
    """
    found = tree.xpath(f"//table[@id='{table_id}']")
    if found:
        return found[0]

    parser = etree.HTMLParser(recover=True)
    for comment in tree.xpath("//comment()"):
        if table_id not in (comment.text or ""):
            continue
        sub = etree.fromstring(comment.text, parser)
        if sub is None:
            continue
        hits = sub.xpath(f"//table[@id='{table_id}']")
        if hits:
            return hits[0]
    raise ValueError(f"table id {table_id!r} not found in page")


class FBrefFull(sd.FBref):
    """FBref reader with the full set of season stat tables restored."""

    @classmethod
    def _all_leagues(cls) -> dict[str, str]:
        """Delegate league resolution to the parent.

        ``BaseReader._all_leagues`` filters the league dictionary by ``cls.__name__``,
        so a subclass would otherwise resolve to no valid leagues at all.
        """
        return sd.FBref._all_leagues()

    def _season_pages(self, page: str):
        seasons = self.read_seasons()
        for (lkey, skey), season in seasons.iterrows():
            base = "/".join(season.url.split("/")[:-1])
            tail = season.url.split("/")[-1]
            url = f"{FBREF_API}{base}/{page}/{tail}"
            # Player and squad tables share a page, so they share one cache entry.
            filepath = self.data_dir / f"page_{page}_{lkey}_{skey}.html"
            yield lkey, skey, url, filepath

    def read_player_season_stats(self, stat_type: str = "standard") -> pd.DataFrame:
        if stat_type not in PAGE_SLUG:
            raise TypeError(f"Invalid stat_type {stat_type!r}; expected one of {list(PAGE_SLUG)}")

        page = PAGE_SLUG[stat_type]
        table_id = f"stats_{TABLE_KEY[stat_type]}"

        frames = []
        for lkey, skey, url, filepath in self._season_pages(page):
            tree = html.parse(self.get(url, filepath))
            for elem in tree.xpath("//td[@data-stat='comp_level']//span"):
                elem.getparent().remove(elem)
            df_table = _parse_table(_find_table(tree, table_id))
            df_table[("Unnamed: league", "league")] = lkey
            df_table[("Unnamed: season", "season")] = skey
            frames.append(_fix_nation_col(df_table))

        df = _concat(frames, key=["league", "season"])
        df = df[df.Player != "Player"]
        for drop_col in ("Matches", "Rk"):
            if drop_col in df.columns.get_level_values(0):
                df = df.drop(drop_col, axis=1, level=0)
        return (
            df.rename(columns={"Squad": "team"})
            .replace({"team": TEAMNAME_REPLACEMENTS})
            .pipe(standardize_colnames, cols=["Player", "Nation", "Pos", "Age", "Born"])
            .set_index(["league", "season", "team", "player"])
            .sort_index()
        )

    def read_team_season_stats(
        self, stat_type: str = "standard", opponent_stats: bool = False
    ) -> pd.DataFrame:
        if stat_type not in PAGE_SLUG:
            raise ValueError(f"Invalid stat_type {stat_type!r}; expected one of {list(PAGE_SLUG)}")

        side = "against" if opponent_stats else "for"
        table_id = f"stats_squads_{TABLE_KEY[stat_type]}_{side}"
        page = PAGE_SLUG[stat_type]

        frames = []
        for lkey, skey, url, filepath in self._season_pages(page):
            tree = html.parse(self.get(url, filepath))
            html_table = _find_table(tree, table_id)
            df_table = _parse_table(html_table)
            df_table["league"] = lkey
            df_table["season"] = skey
            frames.append(df_table)

        return (
            _concat(frames, key=["league", "season"])
            .rename(columns={"Squad": "team", "# Pl": "players_used"})
            .replace({"team": TEAMNAME_REPLACEMENTS})
            .set_index(["league", "season", "team"])
            .sort_index()
        )


def flatten_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Flatten FBref's two-level headers to ``group__stat`` deterministically.

    Returns the flattened frame and the original two-level names for the manifest, so
    nothing about the source header is lost.
    """
    original = []
    flat = []
    for col in df.columns:
        if isinstance(col, tuple):
            group, stat = (str(x) for x in col[:2])
            group = "" if group.startswith("Unnamed") else group.strip()
            stat = "" if stat.startswith("Unnamed") else stat.strip()
            name = f"{group}__{stat}" if group and stat else (group or stat)
        else:
            group, stat, name = "", str(col), str(col)
        original.append({"group": group, "stat": stat, "flat": name})
        flat.append(name)

    out = df.copy()
    out.columns = flat
    # Disambiguate any collisions rather than silently dropping a column.
    if len(set(flat)) != len(flat):
        seen: dict[str, int] = {}
        deduped = []
        for name in flat:
            seen[name] = seen.get(name, 0) + 1
            deduped.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
        out.columns = deduped
        for rec, name in zip(original, deduped, strict=True):
            rec["flat"] = name
    return out, original


def ingest_season(season: str, force: bool = False) -> dict:
    """Pull every player and team stat table for one season and cache it as parquet."""
    out_dir = config.DATA_RAW / season
    out_dir.mkdir(parents=True, exist_ok=True)

    fb = FBrefFull(leagues=config.LEAGUE, seasons=season)
    manifest = {
        "season": season,
        "league": config.LEAGUE,
        "soccerdata_version": sd.__version__,
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "tables": {},
    }

    for scope, stat_types, reader in (
        ("players", config.PLAYER_STAT_TYPES, fb.read_player_season_stats),
        ("teams", config.TEAM_STAT_TYPES, fb.read_team_season_stats),
    ):
        for stat_type in stat_types:
            path = out_dir / f"{scope}_{stat_type}.parquet"
            dict_path = config.DATA_RAW / f"column_dictionary_{scope}_{stat_type}.csv"
            if path.exists() and not force:
                df_cached = pd.read_parquet(path)
                manifest["tables"][f"{scope}_{stat_type}"] = {
                    "rows": int(df_cached.shape[0]),
                    "cols": int(df_cached.shape[1]),
                    "cached": True,
                }
                print(f"  cached  {scope}/{stat_type}: {df_cached.shape}", flush=True)
                continue

            df = reader(stat_type=stat_type)
            flat, original = flatten_columns(df.reset_index())
            flat.to_parquet(path, index=False)
            pd.DataFrame(original).to_csv(dict_path, index=False)

            manifest["tables"][f"{scope}_{stat_type}"] = {
                "rows": int(flat.shape[0]),
                "cols": int(flat.shape[1]),
                "fbref_columns": [f"{r['group']}|{r['stat']}" for r in original],
                "flat_columns": list(flat.columns),
                "cached": False,
            }
            print(f"  pulled  {scope}/{stat_type}: {flat.shape}", flush=True)

    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def ingest_understat_season(season: str, force: bool = False) -> dict:
    """Pull Understat player season stats.

    FBref's Opta-derived columns were deleted in January 2026, which removed every
    expected-goals and progression statistic from the site. Understat is an independent
    xG source that still publishes npxG, xA, key passes, xGChain and xGBuildup, so it
    supplies the shot-quality and creation half of the feature set.
    """
    out_dir = config.DATA_RAW / season
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "understat_players.parquet"

    if path.exists() and not force:
        df = pd.read_parquet(path)
        print(f"  cached  understat/players: {df.shape}", flush=True)
        return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "cached": True}

    us = sd.Understat(leagues=config.LEAGUE, seasons=season)
    df = us.read_player_season_stats().reset_index()
    df.to_parquet(path, index=False)
    print(f"  pulled  understat/players: {df.shape}", flush=True)
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "cached": False,
    }


def main() -> None:
    for season in config.SEASONS:
        print(f"=== {season} ===", flush=True)
        manifest = ingest_season(season)
        manifest["understat"] = ingest_understat_season(season)
        with open(config.DATA_RAW / season / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)
    print("ingestion complete")


if __name__ == "__main__":
    main()
