"""Does the stability result survive the fact that player-seasons are nested?

The unit of analysis is the player-season, and the same player supplies up to five of them.
A bootstrap that resamples rows therefore treats correlated observations as independent,
and the stability it reports is too optimistic by an unknown amount. The paper has so far
acknowledged that and answered it by design, refitting every season and every league
independently rather than adjusting the pooled fit.

This module answers it directly instead. It runs the identical bootstrap twice on the same
partition: once resampling player-seasons, which is what the paper reports elsewhere and
what the genre does, and once resampling whole players and taking every season a drawn
player contributed, which respects the nesting. The comparison says how much the
exchangeability assumption was worth.

Two things are worth knowing before reading the numbers. A player block bootstrap draws
fewer distinct rows than a row bootstrap, because drawing a player with four seasons
duplicates four correlated rows at once, so its resamples are smaller and more redundant
and its index should be expected to fall a little for that reason alone. And the question
this answers is about the observed data only: the clusterless nulls are independent rows by
construction, so there is no player to block on and the null comparison in
``src/structure.py`` is unaffected.

Writes ``results/metrics/nesting.json``.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import config
from src import archive_validation as V
from src import cluster as C

#: Resamples per bootstrap, matched to the count ``src/archive_validation.py`` uses on this
#: sample so the two indices are comparable rather than merely similar.
N_BOOT = V.N_BOOT

K = V.K


def _r(value: float, digits: int = 4) -> float:
    return float(np.round(float(value), digits))


def block_bootstrap(
    x: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    k: int,
    tag: str,
    n_boot: int = N_BOOT,
) -> dict:
    """Resample whole groups with replacement, refit, and score against the reference.

    A drawn group contributes every one of its rows, so the resample respects whatever
    correlation those rows share. The adjusted Rand index is computed on the distinct rows
    the draw contains, exactly as the row bootstrap computes it on the distinct rows it
    draws, which is what makes the two numbers comparable.
    """
    rng = np.random.default_rng(C._seed("block-bootstrap", tag, k))
    keys, inverse = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(inverse == g) for g in range(keys.size)]

    aris: list[float] = []
    sizes: list[int] = []
    for b in range(n_boot):
        drawn = rng.integers(0, keys.size, size=keys.size)
        rows = np.concatenate([rows_by_group[g] for g in drawn])
        unique, first = np.unique(rows, return_index=True)
        if unique.size <= k:
            continue
        fit = KMeans(
            n_clusters=k, n_init=C.KMEANS_N_INIT, random_state=C._seed(tag, "block", k, b)
        ).fit(x[rows])
        aris.append(float(adjusted_rand_score(reference[unique], fit.labels_[first])))
        sizes.append(int(unique.size))

    return {
        "n_bootstrap": len(aris),
        "ari_mean": _r(float(np.mean(aris))),
        "ari_sd": _r(float(np.std(aris, ddof=1))),
        "ari_min": _r(float(np.min(aris))),
        "ari_max": _r(float(np.max(aris))),
        "distinct_rows_mean": int(round(float(np.mean(sizes)))),
    }


def run() -> dict:
    eligible, z_global, z_group, features = V.load_frames()
    scopes = V.scope_frames(z_global, z_group)

    payload: dict = {
        "question": (
            "Player-seasons are nested inside players. Does resampling players rather than "
            "player-seasons change what the bootstrap says about the two-mode partition?"
        ),
        "k": K,
        "n_bootstrap": N_BOOT,
        "unit": "player-season",
        "block": "player",
        "note": (
            "A player block bootstrap draws fewer distinct rows than a row bootstrap of the "
            "same nominal size, because one drawn player contributes all of his seasons at "
            "once, so a small fall in the index is expected from the reduced effective "
            "sample alone and is not by itself evidence that the row bootstrap was "
            "misleading. The clusterless nulls are independent rows by construction, so the "
            "null calibration is unaffected."
        ),
        "scopes": {},
    }

    for name, frame in scopes.items():
        tag = f"archive|{name}"
        x = frame[features].to_numpy(float)
        labels = C._kmeans(x, K, tag).labels_
        players = frame["Player"].to_numpy()

        rows = C.bootstrap_stability(x, labels, K, tag, n_boot=N_BOOT)
        rows.pop("coassignment", None)
        rows.pop("per_cluster", None)
        block = block_bootstrap(x, labels, players, K, tag, n_boot=N_BOOT)

        payload["scopes"][name] = {
            "n_player_seasons": int(len(frame)),
            "n_players": int(np.unique(players).size),
            "seasons_per_player_mean": _r(len(frame) / np.unique(players).size, 2),
            "row_bootstrap": rows,
            "player_block_bootstrap": block,
            "difference": _r(rows["ari_mean"] - block["ari_mean"]),
        }
        print(
            f"  {name:14s} n={len(frame):>5,} players={np.unique(players).size:>5,}  "
            f"row {rows['ari_mean']:.3f}  block {block['ari_mean']:.3f}  "
            f"difference {rows['ari_mean'] - block['ari_mean']:+.3f}"
        )

    means = [s["player_block_bootstrap"]["ari_mean"] for s in payload["scopes"].values()]
    diffs = [s["difference"] for s in payload["scopes"].values()]
    payload["summary"] = {
        "block_ari_min_over_scopes": _r(min(means)),
        "block_ari_max_over_scopes": _r(max(means)),
        "largest_fall_against_row_bootstrap": _r(max(diffs)),
        "verdict": (
            "The partition is as stable under a bootstrap that respects the nesting as "
            "under one that ignores it, so the exchangeability assumption the row bootstrap "
            "makes is not what produced the reported stability."
            if max(diffs) < 0.05
            else "Resampling players rather than player-seasons materially lowers the index."
        ),
    }
    return payload


def main() -> None:
    payload = run()
    path = config.METRICS / "nesting.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {path}")
    print(payload["summary"]["verdict"])


if __name__ == "__main__":
    main()
