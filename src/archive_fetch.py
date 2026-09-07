"""Download the pre-withdrawal FBref mirror that the primary analysis is built on.

FBref removed its Opta-derived columns in January 2026. The worldfootballR project has
been mirroring those tables since they first appeared, and publishes the snapshots as
serialised R data frames in a public repository. This module fetches them once, converts
them to parquet under ``data/raw/archive/`` and records provenance, so that the primary
sample is reproducible from a stated URL rather than from files of unknown origin sitting
in the working tree.

The mirror is a third party scrape rather than a primary source, and the paper says so.
It is used because the primary source no longer serves the data and the mirror predates
its removal. Every file is hashed on download and the hashes are written to the manifest,
so a later change upstream is detectable rather than silent.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pyreadr

import config

#: Upstream directory. Pinned to a commit rather than a branch would be preferable, but the
#: repository rewrites these files in place on every scrape and keeps no tags, so the
#: manifest's per-file SHA-256 is what makes a given run identifiable after the fact.
BASE = (
    "https://github.com/JaseZiv/worldfootballR_data/raw/master/data/fb_big5_advanced_season_stats"
)

#: Local table name mapped to the upstream file name.
TABLES = {
    "standard": "big5_player_standard.rds",
    "shooting": "big5_player_shooting.rds",
    "passing": "big5_player_passing.rds",
    "passing_types": "big5_player_passing_types.rds",
    "gca": "big5_player_gca.rds",
    "defense": "big5_player_defense.rds",
    "possession": "big5_player_possession.rds",
    "misc": "big5_player_misc.rds",
    "playing_time": "big5_player_playing_time.rds",
    "keepers": "big5_player_keepers.rds",
    "keepers_adv": "big5_player_keepers_adv.rds",
    "team_standard": "big5_team_standard.rds",
    "team_possession": "big5_team_possession.rds",
}

RAW = config.DATA_RAW / "archive"
CACHE = RAW / "rds"


def _download(name: str, filename: str) -> tuple[Path, str]:
    """Fetch one .rds, reusing the cached copy when one is already present."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / filename
    if not dest.exists():
        url = f"{BASE}/{filename}"
        print(f"  downloading {filename}", flush=True)
        with urllib.request.urlopen(url) as response:  # noqa: S310, https URL is a literal
            dest.write_bytes(response.read())
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    return dest, digest


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "worldfootballR_data, a third party mirror of FBref advanced season tables",
        "source_url": BASE,
        "why": (
            "FBref withdrew its Opta-derived columns in January 2026. The mirror's snapshot "
            "predates the withdrawal and is the only public route to the full feature set."
        ),
        "retrieved": datetime.now(UTC).strftime("%Y-%m-%d"),
        "tables": {},
    }

    for name, filename in TABLES.items():
        path, digest = _download(name, filename)
        frame = next(iter(pyreadr.read_r(str(path)).values()))
        out = RAW / f"{name}.parquet"
        frame.to_parquet(out, index=False)
        manifest["tables"][name] = {
            "upstream_file": filename,
            "sha256": digest,
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "seasons": sorted(int(v) for v in frame["Season_End_Year"].dropna().unique())
            if "Season_End_Year" in frame.columns
            else None,
        }
        print(f"{name:16s} {frame.shape[0]:>6,} rows x {frame.shape[1]:>3} cols", flush=True)

    with open(RAW / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {len(TABLES)} tables and a manifest to {RAW}")


if __name__ == "__main__":
    main()
