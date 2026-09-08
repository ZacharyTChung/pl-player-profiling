"""Generate the LaTeX tables and the macro definitions the paper reads at build time.

The paper is forbidden from containing a hard-coded number. Every quantity it cites is
defined here as a macro in ``results/macros.tex``, and every table it includes is written
here as a ``booktabs`` fragment in ``results/tables``. If an analysis has not been run,
the macro it would have produced is absent, the paper fails to compile, and the
inconsistency is caught rather than published.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src import features as F

# --------------------------------------------------------------------------------------
# Loading and lookup
# --------------------------------------------------------------------------------------

_MISSING = object()


def load(name: str) -> dict | None:
    """Read a metrics file, returning None when the producing stage has not run."""
    path = config.METRICS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_results(name: str) -> dict | None:
    path = config.RESULTS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def season_block(obj: Any, season: str) -> Any:
    """Return the block for a season, whether the file nests by season or not.

    The analysis modules were written independently, so some metric files are keyed by
    season at the top level and others are flat with a ``season`` field. Both shapes are
    accepted rather than requiring every module to be rewritten.
    """
    if not isinstance(obj, dict):
        return None
    if season in obj:
        return obj[season]
    if obj.get("season") == season:
        return obj
    return None


def dig(obj: Any, *path: str | int, default: Any = _MISSING) -> Any:
    """Walk a nested structure, tolerating absent branches."""
    cur = obj
    for key in path:
        if cur is None:
            break
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            cur = None
            break
    if cur is None:
        if default is _MISSING:
            return None
        return default
    return cur


# --------------------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------------------

_TEX_ESCAPES = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(text: Any) -> str:
    out = []
    for ch in str(text):
        out.append(_TEX_ESCAPES.get(ch, ch))
    return "".join(out)


def num(value: Any, places: int = 3) -> str:
    """Format a number for the text. Integers stay integers."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return tex_escape(value)
    if f.is_integer() and abs(f) < 1e6:
        return f"{int(f):,}"
    return f"{f:.{places}f}"


def pct(value: Any, places: int = 1) -> str:
    """Format a fraction in 0..1 as a percentage number, without the sign."""
    if value is None:
        return "n/a"
    f = float(value)
    if f <= 1.0:
        f *= 100.0
    return f"{f:.{places}f}"


# --------------------------------------------------------------------------------------
# Macro emission
# --------------------------------------------------------------------------------------


class Macros:
    """Collects macro definitions and writes them once, checking for collisions."""

    def __init__(self) -> None:
        self._defs: dict[str, str] = {}
        self._skipped: list[str] = []

    def add(self, name: str, value: Any, places: int = 3) -> None:
        if value is None:
            self._skipped.append(name)
            return
        self._defs[name] = num(value, places) if not isinstance(value, str) else tex_escape(value)

    def add_pct(self, name: str, value: Any, places: int = 1) -> None:
        if value is None:
            self._skipped.append(name)
            return
        self._defs[name] = pct(value, places)

    def add_raw(self, name: str, text: str) -> None:
        self._defs[name] = text

    def add_year(self, name: str, value: Any) -> None:
        """Years must not carry a thousands separator."""
        if value is None:
            self._skipped.append(name)
            return
        self._defs[name] = str(int(value))

    def add_pvalue(self, name: str, value: Any, floor: float = 1e-4) -> None:
        """Report a p-value, but never as a bare zero.

        A p-value printed as 0 claims more than any finite test can, so anything below
        the floor is reported as being below it.
        """
        if value is None:
            self._skipped.append(name)
            return
        p = float(value)
        self._defs[name] = f"<{floor:g}" if p < floor else f"{p:.4f}"

    @property
    def skipped(self) -> list[str]:
        return sorted(self._skipped)

    def write(self, path: Path) -> int:
        lines = [
            "% Generated by src/tables.py. Do not edit by hand.",
            "% Every number cited by the paper is defined here.",
            "",
        ]
        for name in sorted(self._defs):
            lines.append(f"\\newcommand{{\\{name}}}{{{self._defs[name]}}}")
        lines.append("")
        path.write_text("\n".join(lines))
        return len(self._defs)


# --------------------------------------------------------------------------------------
# Table writers
# --------------------------------------------------------------------------------------


def _write_table(name: str, body: str) -> Path:
    path = config.TABLES / f"{name}.tex"
    path.write_text(body.rstrip() + "\n")
    return path


def table_data_availability(keeper_avail: dict | None) -> None:
    """Non-null column counts per raw table, the evidence for the withdrawal claim."""
    rows = []
    for season in config.SEASONS:
        season_dir = config.DATA_RAW / season
        for stat in config.PLAYER_STAT_TYPES:
            path = season_dir / f"players_{stat}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            ids = {"league", "season", "team", "player", "nation", "pos", "age", "born"}
            cols = [c for c in df.columns if c not in ids]
            empty = sum(1 for c in cols if df[c].notna().sum() == 0)
            rows.append(
                {
                    "season": season,
                    "table": stat.replace("_", " "),
                    "stat_cols": len(cols),
                    "empty": empty,
                    "usable": len(cols) - empty,
                }
            )
    if not rows:
        return
    df = pd.DataFrame(rows)
    primary = df[df.season == config.SEASON_PRIMARY]
    repl = df[df.season == config.SEASON_REPLICATION].set_index("table")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Populated columns in each FBref player table. The tables are still "
        r"served with complete headers, so the empty counts are the number of columns "
        r"that resolve correctly and contain no values.}",
        r"\label{tab:availability}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & \multicolumn{2}{c}{"
        + config.SEASON_PRIMARY
        + r"} & \multicolumn{2}{c}{"
        + config.SEASON_REPLICATION
        + r"} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Table & Columns & Empty & Columns & Empty \\",
        r"\midrule",
    ]
    for _, r in primary.iterrows():
        other = repl.loc[r["table"]] if r["table"] in repl.index else None
        o_cols = int(other["stat_cols"]) if other is not None else 0
        o_empty = int(other["empty"]) if other is not None else 0
        lines.append(
            f"{tex_escape(r['table'])} & {r['stat_cols']} & {r['empty']} & {o_cols} & {o_empty} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("data_availability", "\n".join(lines))


def table_feature_dictionary() -> None:
    """The full outfield feature set with its source and definition."""
    definitions = {
        "np_xg_p90": ("Understat", "Non-penalty expected goals"),
        "xa_p90": ("Understat", "Expected assists"),
        "key_passes_p90": ("Understat", "Passes leading directly to a shot"),
        "xg_chain_p90": ("Understat", "Expected goals of possessions the player was involved in"),
        "xg_buildup_p90": ("Understat", "As xGChain, excluding shots and key passes"),
        "shots_p90": ("FBref", "Shots taken, excluding penalties"),
        "shots_on_target_p90": ("FBref", "Shots on target"),
        "goals_non_penalty_p90": ("FBref", "Goals excluding penalties"),
        "assists_p90": ("FBref", "Assists"),
        "crosses_p90": ("FBref", "Crosses attempted"),
        "interceptions_padj_p90": ("FBref", "Interceptions, possession adjusted"),
        "tackles_won_padj_p90": ("FBref", "Tackles won, possession adjusted"),
        "fouls_committed_padj_p90": ("FBref", "Fouls committed, possession adjusted"),
        "fouls_drawn_p90": ("FBref", "Fouls drawn"),
        "offsides_p90": ("FBref", "Times caught offside"),
        "cards_yellow_p90": ("FBref", "Yellow cards"),
        "shot_accuracy_pct": ("FBref", "Shots on target as a share of shots, a rate"),
        "goals_per_shot": ("FBref", "Goals per shot, a rate"),
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{The outfield feature set. Counting statistics are per ninety minutes. "
        r"The two rate features are used as published.}",
        r"\label{tab:features}",
        r"\begin{tabular}{llp{7.2cm}}",
        r"\toprule",
        r"Feature & Source & Definition \\",
        r"\midrule",
    ]
    for feat in F.OUTFIELD_CORE:
        source, definition = definitions.get(feat, ("", ""))
        lines.append(f"{tex_escape(feat)} & {source} & {definition} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("feature_dictionary", "\n".join(lines))


def table_archive_feature_dictionary() -> None:
    """The primary feature set, with the FBref table each column was read from.

    The reduced sample has its own dictionary because its features are a different set
    drawn from a different pull, and conflating the two would misdescribe both.
    """
    from src import archive as A

    definitions = {
        "np_xg": "Non-penalty expected goals",
        "shots": "Shots taken, excluding penalties",
        "goals_non_penalty": "Goals excluding penalties",
        "xag": "Expected assisted goals, the expected goals of shots a pass created",
        "key_passes": "Passes leading directly to a shot",
        "sca": "Shot creating actions, the two offensive actions before a shot",
        "gca": "Goal creating actions, the two offensive actions before a goal",
        "passes_final_third": "Completed passes into the final third",
        "passes_penalty_area": "Completed passes into the penalty area",
        "crosses": "Crosses attempted",
        "progressive_passes": "Completed passes moving the ball substantially toward goal",
        "progressive_carries": "Carries moving the ball substantially toward goal",
        "progressive_receptions": "Passes received in a progressive position",
        "carries_final_third": "Carries into the final third",
        "touches_def_pen": "Touches in the defending team's own penalty area",
        "touches_def_third": "Touches in the defensive third",
        "touches_mid_third": "Touches in the middle third",
        "touches_att_third": "Touches in the attacking third",
        "touches_att_pen": "Touches in the opposition penalty area",
        "take_ons": "Take-ons attempted",
        "tackles": "Tackles, possession adjusted",
        "interceptions": "Interceptions, possession adjusted",
        "blocks": "Blocks of a pass or shot, possession adjusted",
        "clearances": "Clearances, possession adjusted",
        "ball_recoveries": "Loose balls recovered",
        "fouls_committed": "Fouls committed, possession adjusted",
        "shot_accuracy_pct": "Shots on target as a share of shots",
        "take_on_success_pct": "Take-ons completed as a share of those attempted",
        "tackle_win_pct": "Duels won as a share of those contested",
        "aerials_won_pct": "Aerial duels won as a share of those contested",
        "pass_cmp_short_pct": "Short passes completed as a share of those attempted",
        "pass_cmp_medium_pct": "Medium passes completed as a share of those attempted",
        "pass_cmp_long_pct": "Long passes completed as a share of those attempted",
    }

    rows = []
    for feature in A.OUTFIELD_CORE:
        base = feature[:-4] if feature.endswith("_p90") else feature
        base = base[:-5] if base.endswith("_padj") else base
        source = A.SOURCES.get(base)
        table = source[0] if source else "derived"
        definition = definitions.get(base)
        if definition is None:
            raise KeyError(f"no definition for archive feature {feature!r}; add one")
        suffix = " (per 90)" if feature.endswith("_p90") else ""
        rows.append(
            f"{tex_escape(feature.replace('_', ' '))} & {tex_escape(table)} & "
            f"{tex_escape(definition)}{suffix} \\\\"
        )

    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{The primary feature set, with the FBref table each column was read "
            r"from. Counting statistics are per ninety minutes played; the seven rate "
            r"features are used as published. Five defensive counts are possession "
            r"adjusted, as described in Section~\ref{sec:methods:padj}.}",
            r"\label{tab:archivefeatures}",
            r"\begin{tabular}{llp{7.0cm}}",
            r"\toprule",
            r"Feature & FBref table & Definition \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    _write_table("archive_feature_dictionary", body)


def table_ablation(ab: dict | None) -> None:
    """Standard scores against the clusterless null under every feature-family ablation."""
    if not ab:
        return
    scopes = [("All outfield", "All"), ("DF", "DF"), ("MF", "MF"), ("FW", "FW")]
    families = list((ab.get("families") or {}).keys())
    if not families:
        return

    def cell(scope: str, mode: str, fam: str) -> str:
        entry = dig(ab, "scopes", scope, mode, fam) or {}
        z = dig(entry, "null_at_k", "z_against_null")
        k = entry.get("chosen_k")
        if z is None:
            return "--"
        mark = "" if k == 2 else r"$^{\dagger}$"
        return f"{float(z):.1f}{mark}"

    head = " & ".join(tag for _, tag in scopes)
    rows = []
    rows.append(r"\multicolumn{5}{l}{\emph{Full feature set}} \\")
    full = " & ".join(
        f"{float(dig(ab, 'scopes', scope, 'full', 'null_at_k', 'z_against_null')):.1f}"
        for scope, _ in scopes
    )
    rows.append(f"all {dig(ab, 'n_features_full')} features & {full} \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\emph{Leave one family out}} \\")
    for fam in families:
        rows.append(
            f"without {tex_escape(fam)} & "
            + " & ".join(cell(scope, "leave_one_out", fam) for scope, _ in scopes)
            + " \\\\"
        )
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{5}{l}{\emph{One family alone}} \\")
    for fam in families:
        rows.append(
            f"{tex_escape(fam)} only & "
            + " & ".join(cell(scope, "keep_one_only", fam) for scope, _ in scopes)
            + " \\\\"
        )

    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Standard score of the observed two-cluster silhouette against the "
            r"clusterless Gaussian null, by scope, under every feature-family ablation. A "
            r"dagger marks a cell where the pre-declared rule chose a number of clusters other "
            r"than two.}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            f"Feature set & {head} \\\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    _write_table("ablation", body)


def table_outcomes_extremes(oc: dict | None) -> None:
    """The team-seasons at the two ends of the attacking-mode share, against real results."""
    if not oc:
        return
    rows = []
    for side, title in (
        ("highest_attacking_share", "Highest attacking-mode share"),
        ("lowest_attacking_share", "Lowest attacking-mode share"),
    ):
        entries = dig(oc, "extremes", side) or []
        if not entries:
            continue
        rows.append(r"\multicolumn{6}{l}{\emph{" + title + r"}} \\")
        for e in entries:
            pos = e.get("league_position", e.get("position", ""))
            rows.append(
                f"{tex_escape(e.get('team', ''))} & {tex_escape(e.get('league', ''))} & "
                f"{e.get('season', '')} & {float(e.get('attacking_share', 0)):.3f} & "
                f"{float(e.get('points_per_match', 0)):.2f} & {pos} \\\\"
            )
    if not rows:
        return
    body = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{The five team-seasons with the highest and the five with the lowest "
            r"minutes-weighted share of outfield minutes in the attacking mode, with their real "
            r"points per match and final league position.}",
            r"\label{tab:outcomes}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Club & League & Season & Share & Points per match & Position \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    _write_table("outcomes_extremes", body)


def table_umap_grid(grid: dict | None) -> None:
    if not grid:
        return
    season = season_block(grid, config.SEASON_PRIMARY) or grid
    entries = None
    # "grid" holds the swept parameter values, not the outcomes, so it must not be
    # mistaken for the results list.
    for key in ("results", "configurations", "runs"):
        candidate = dig(season, key)
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            entries = candidate
            break
    if entries is None and isinstance(season, list):
        entries = season
    if not entries:
        raise ValueError("umap_grid.json contains no per-configuration results list")

    rows = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        rows.append(
            {
                "n": e.get("n_neighbors", e.get("n_neighbours")),
                "d": e.get("min_dist"),
                "t": e.get("trustworthiness"),
            }
        )
    rows = [r for r in rows if r["n"] is not None and r["t"] is not None]
    if not rows:
        return
    rows.sort(key=lambda r: -float(r["t"]))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{UMAP hyperparameter grid, ranked by trustworthiness. Selection was by "
        r"trustworthiness alone and was decided before any embedding was inspected.}",
        r"\label{tab:umapgrid}",
        r"\begin{tabular}{rrr}",
        r"\toprule",
        r"Neighbours & Minimum distance & Trustworthiness \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(f"{int(r['n'])} & {num(r['d'], 2)} & {num(r['t'], 4)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("umap_grid", "\n".join(lines))


def table_divergence(div: dict | None) -> None:
    if not div:
        return
    season = dig(div, config.SEASON_PRIMARY)
    top = None
    for key in ("top_divergent", "top10", "top_pairs"):
        if isinstance(season, dict) and key in season:
            top = season[key]
            break
    if not top:
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{The ten feature pairs whose Spearman correlation differs most across "
        r"position groups. Every pair changes sign between at least two groups.}",
        r"\label{tab:divergence}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Feature A & Feature B & DF & MF & FW & Spread \\",
        r"\midrule",
    ]
    for r in top[:10]:
        g = r.get("by_group", {})
        lines.append(
            f"{tex_escape(r.get('label_a', r.get('feature_a')))} & "
            f"{tex_escape(r.get('label_b', r.get('feature_b')))} & "
            f"{num(g.get('DF'), 2)} & {num(g.get('MF'), 2)} & {num(g.get('FW'), 2)} & "
            f"{num(r.get('spread'), 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("correlation_divergence", "\n".join(lines))


def table_summary_by_position(summary: dict | None) -> None:
    if not summary:
        return
    season = dig(summary, config.SEASON_PRIMARY)
    groups = dig(season, "groups")
    if not groups:
        return
    metrics = F.HEADLINE_METRICS
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Headline per-90 statistics by position group in the earlier of the two "
        r"reduced-sample seasons, reported as mean with standard deviation in parentheses.}",
        r"\label{tab:summary}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Statistic & DF & MF & FW \\",
        r"\midrule",
    ]
    for m in metrics:
        cells = []
        for grp in ("DF", "MF", "FW"):
            stats = dig(groups, grp, m) or {}
            mean, sd = stats.get("mean"), stats.get("std", stats.get("sd"))
            cells.append(f"{num(mean, 2)} ({num(sd, 2)})" if mean is not None else "n/a")
        lines.append(f"{tex_escape(m)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("summary_by_position", "\n".join(lines))


def table_k_selection(sel: dict | None) -> None:
    if not sel:
        return
    season = dig(sel, "seasons", config.SEASON_PRIMARY)
    if not season:
        return
    lines = [
        r"\begin{longtable}{llrrrrr}",
        r"\caption{Cluster selection criteria for every candidate number of clusters, by "
        r"scope. The consensus rule is applied to these values.}\\",
        r"\label{tab:kselection}\\",
        r"\toprule",
        r"Scope & $k$ & Silhouette & Davies-Bouldin & Calinski-Harabasz & Gap \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Scope & $k$ & Silhouette & Davies-Bouldin & Calinski-Harabasz & Gap \\",
        r"\midrule",
        r"\endhead",
    ]
    wrote = False
    for scope in ("global", "DF", "MF", "FW"):
        curves = dig(season, scope, "spaces", "standardised", "curves")
        if not isinstance(curves, dict):
            continue
        chosen = dig(season, scope, "chosen_k")
        # curves is {criterion: {k: value}}, so pivot on the k keys of the silhouette curve.
        for k in sorted(curves.get("silhouette", {}), key=int):
            mark = r"$\ast$" if str(chosen) == str(k) else ""
            lines.append(
                f"{scope}{mark} & {k} & "
                f"{num(dig(curves, 'silhouette', k), 3)} & "
                f"{num(dig(curves, 'davies_bouldin', k), 3)} & "
                f"{num(dig(curves, 'calinski_harabasz', k), 1)} & "
                f"{num(dig(curves, 'gap', k), 3)} \\\\"
            )
            wrote = True
    lines += [r"\bottomrule", r"\end{longtable}"]
    if wrote:
        _write_table("k_selection", "\n".join(lines))


def write_reduced_membership() -> None:
    """Publish the reduced sample's archetype assignments as CSV.

    The paper cites this file rather than printing a roster of every player, so it has to
    exist and be regenerated with everything else.
    """
    frames = []
    for season in config.SEASONS:
        path = config.DATA_PROCESSED / f"archetypes_{season}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True).sort_values(
        ["season", "team", "player"], kind="stable"
    )
    out = config.RESULTS / "membership_reduced.csv"
    frame.to_csv(out, index=False)
    print(f"wrote {out.relative_to(config.ROOT)} ({len(frame):,} rows)")


def table_possession_adjustment(pre: dict | None) -> None:
    """Elasticities and the confound before and after adjustment."""
    padj = dig(season_block(pre, config.SEASON_PRIMARY), "possession_adjustment")
    removal = dig(padj, "confound_removal")
    if not removal:
        return
    labels = {
        "interceptions": "Interceptions",
        "tackles_won": "Tackles won",
        "fouls_committed": "Fouls committed",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Possession adjustment. The elasticity is fitted from team totals. The "
        r"final three columns give the correlation between each per-90 rate and team "
        r"possession: unadjusted, adjusted with the fitted elasticity, and adjusted with "
        r"the conventional exponent of one. A value near zero is the goal.}",
        r"\label{tab:padj}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Correlation with team possession} \\",
        r"\cmidrule(lr){3-5}",
        r"Statistic & Elasticity & Unadjusted & Adjusted & Exponent one \\",
        r"\midrule",
    ]
    for count, label in labels.items():
        b = removal.get(count)
        if not b:
            continue
        lines.append(
            f"{label} & {num(b.get('elasticity_used'), 3)} & "
            f"{num(b.get('corr_raw_with_possession'), 3)} & "
            f"{num(b.get('corr_adjusted_with_possession'), 3)} & "
            f"{num(b.get('corr_unit_elasticity_with_possession'), 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("possession_adjustment", "\n".join(lines))


def table_minutes_sensitivity(pre: dict | None, sens: dict | None) -> None:
    """Eligible counts and, when available, clustering outcomes at each threshold."""
    if not pre:
        return
    counts = dig(pre, config.SEASON_PRIMARY, "minutes_sensitivity_counts") or {}
    if not counts:
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Sensitivity of the analysis to the minutes threshold in the earlier of the two "
        r"reduced-sample seasons. Agreement is the adjusted Rand index against the partition at the "
        r"threshold actually used, computed on the players common to both pools.}",
        r"\label{tab:sensitivity}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Minutes & Eligible & $k$ & Silhouette & Bootstrap ARI & Agreement \\",
        r"\midrule",
    ]
    for threshold in config.MIN_MINUTES_SENSITIVITY:
        row = dig(sens, str(threshold)) or {}
        agree = dig(
            sens, "agreement_with_baseline", str(threshold), "adjusted_rand_index_vs_baseline"
        )
        agree_cell = "baseline" if threshold == config.MIN_MINUTES else num(agree, 3)
        lines.append(
            f"{threshold} & {row.get('n_eligible', counts.get(str(threshold), 'n/a'))} & "
            f"{row.get('chosen_k', 'n/a')} & {num(row.get('silhouette'), 3)} & "
            f"{num(row.get('bootstrap_ari_mean'), 3)} & {agree_cell} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write_table("minutes_sensitivity", "\n".join(lines))


# --------------------------------------------------------------------------------------
# Macro assembly
# --------------------------------------------------------------------------------------


def build_macros() -> Macros:
    m = Macros()
    pre = load("preprocess")
    norm = load("normality")
    div = load("correlation_divergence")
    emb = load("embedding_comparison")
    grid = load("umap_grid")
    pca = load("pca_loadings")
    sel = load("cluster_selection")
    stab = load("cluster_stability")
    vspos = load("cluster_vs_position")
    keep_avail = load("keeper_data_availability")

    primary, repl = config.SEASON_PRIMARY, config.SEASON_REPLICATION

    m.add_raw("SeasonPrimary", primary)
    m.add_raw("SeasonReplication", repl)
    m.add("MinMinutes", config.MIN_MINUTES)
    m.add("NFeatures", len(F.OUTFIELD_CORE))
    m.add("BootstrapN", config.BOOTSTRAP_N)
    m.add("GapB", config.GAP_STATISTIC_B)

    # Population
    m.add("NPlayersRawPrimary", dig(pre, primary, "understat_match", "fbref_rows"))
    m.add("NPlayersRawRepl", dig(pre, repl, "understat_match", "fbref_rows"))
    m.add("NOutfieldPrimary", dig(pre, primary, "rows_outfield"))
    m.add("NEligiblePrimary", dig(pre, primary, "rows_outfield_eligible"))
    m.add("NEligibleRepl", dig(pre, repl, "rows_outfield_eligible"))
    m.add("NKeepersPrimary", dig(pre, primary, "rows_keepers"))
    m.add("NKeepersRepl", dig(pre, repl, "rows_keepers"))
    m.add("NMultiClubPrimary", dig(pre, primary, "players_multi_club"))
    # Eligible counts, not pool counts: these are quoted in the text as the analysis
    # sample and must sum to NEligiblePrimary.
    for grp in ("DF", "MF", "FW"):
        m.add(f"N{grp}Primary", dig(pre, primary, "eligible_position_group_counts", grp))
        m.add(f"N{grp}Pool", dig(pre, primary, "position_group_counts", grp))
    m.add_pct("MatchRatePrimary", dig(pre, primary, "understat_match", "match_rate"))
    m.add_pct("MatchRateRepl", dig(pre, repl, "understat_match", "match_rate"))

    # Normality
    m.add("NFeaturesRejectNormal", dig(norm, primary, "n_features_rejecting_normality_shapiro_005"))
    m.add("SkewMeanAbs", dig(norm, primary, "skew_mean_abs"), places=2)
    m.add("SkewMax", dig(norm, primary, "skew_max"), places=2)
    m.add_raw("SkewMaxFeature", tex_escape(dig(norm, primary, "skew_max_feature") or ""))

    # Correlation divergence
    m.add("CorrSpreadMax", dig(div, primary, "spread_max"), places=2)
    m.add("CorrSpreadMedian", dig(div, primary, "spread_median"), places=2)

    # Embeddings
    for key, macro in (("pca", "PCA"), ("umap", "UMAP"), ("tsne", "TSNE")):
        block = dig(emb, primary, "embeddings", key) or {}
        m.add(f"Trust{macro}", block.get("trustworthiness"), places=3)
        m.add(f"Cont{macro}", block.get("continuity"), places=3)
        m.add(f"Purity{macro}", block.get("knn_position_purity"), places=3)
    m.add("PurityBaseline", dig(emb, primary, "purity_baseline_random"), places=3)
    m.add("KNNPurityK", dig(emb, primary, "k"))

    best = dig(grid, primary, "best") or {}
    m.add("UMAPBestNeighbors", best.get("n_neighbors"))
    m.add("UMAPBestMinDist", best.get("min_dist"), places=2)
    m.add("UMAPBestTrust", dig(grid, primary, "best_trustworthiness"), places=4)
    span = dig(grid, primary, "trustworthiness_range")
    if isinstance(span, list | tuple) and len(span) == 2:
        m.add("UMAPTrustMin", span[0], places=4)
        m.add("UMAPTrustMax", span[1], places=4)

    # PCA variance
    for i in range(1, 4):
        m.add(
            f"PCVar{['One', 'Two', 'Three'][i - 1]}",
            dig(pca, primary, "explained_variance_ratio", i - 1),
            places=3,
        )

    # Clustering
    for scope in ("global", "DF", "MF", "FW"):
        label = "Global" if scope == "global" else scope
        m.add(f"K{label}", dig(sel, "seasons", primary, scope, "chosen_k"))
        m.add(f"BootARI{label}", dig(stab, "seasons", primary, scope, "ari_mean"), places=3)
        m.add(f"BootARISD{label}", dig(stab, "seasons", primary, scope, "ari_sd"), places=3)
        m.add(f"BootARIMin{label}", dig(stab, "seasons", primary, scope, "ari_min"), places=3)
    m.add("ARIvsPosition", dig(vspos, "answer", "adjusted_rand_index_vs_position_group"), places=3)
    m.add("PurityVsPosition", dig(vspos, "answer", "cluster_purity_vs_position_group"), places=3)
    m.add(
        "NMIvsPosition",
        dig(vspos, "seasons", primary, "global", "position_group", "normalized_mutual_info"),
        places=3,
    )
    m.add(
        "ARIvsPositionFull",
        dig(vspos, "seasons", primary, "global", "position_full", "adjusted_rand_index"),
        places=3,
    )
    m.add(
        "ARIWithinFW",
        dig(vspos, "seasons", primary, "FW", "position_full", "adjusted_rand_index"),
        places=3,
    )

    # Algorithm comparison. HDBSCAN is the strongest corroboration available, because it
    # is free to declare that no clusters exist, and within position groups it does.
    algo = load("cluster_algorithms")
    scopes = dig(algo, "seasons", primary, "scopes") or {}
    for scope in ("global", "DF", "MF", "FW"):
        label = "Global" if scope == "global" else scope
        m.add(
            f"Silhouette{label}",
            dig(scopes, scope, "kmeans", "silhouette", "standardised"),
            places=3,
        )
        m.add(
            f"HDBSCANNoise{label}",
            dig(scopes, scope, "hdbscan", "standardised", "5", "noise_fraction"),
            places=3,
        )
        m.add(
            f"HDBSCANClusters{label}",
            dig(scopes, scope, "hdbscan", "standardised", "5", "n_clusters"),
        )
    m.add(
        "PCASensitivityARI", dig(scopes, "global", "kmeans", "ari_standardised_vs_pca90"), places=3
    )
    m.add("PCANinetyComponents", dig(scopes, "global", "pca90", "n_components"))
    m.add(
        "GMMvsKMeansARI",
        dig(algo, "seasons", primary, "gmm_posterior_fit", "ari_vs_kmeans"),
        places=3,
    )

    hyb = load("hybrid_players")
    counts = dig(hyb, "seasons", primary, "counts") or {}
    m.add("NHybridPlayers", counts.get("n_disagreeing"))
    m.add("NPosteriorSplit", counts.get("n_posterior_split"))
    m.add("NHybridFlagged", counts.get("n_flagged"))

    # Goalkeeper availability
    m.add("NKeeperAdvEmpty", dig(keep_avail, "summary", "keeper_adv_all_null", primary))
    m.add("NKeeperAdvStatCols", dig(keep_avail, "summary", "keeper_adv_stat_columns", primary))
    m.add("NKeeperAdvUsable", dig(keep_avail, "summary", "keeper_adv_usable", primary))

    # Goalkeeper analysis
    kdesc, kclust, kcomp = (
        load("keeper_descriptive"),
        load("keeper_clusters"),
        load("keeper_composites"),
    )
    m.add("NKeepersEligible", dig(kdesc, primary, "n_eligible"))
    m.add("NKeepersEligibleRepl", dig(kdesc, repl, "n_eligible"))
    m.add("KeeperFeatureCount", dig(kdesc, primary, "surviving_feature_count"))
    m.add("KeeperSavesRSq", dig(kcomp, primary, "saves_on_sota_regression", "r_squared"), places=3)
    m.add(
        "KeeperSavesResidSE",
        dig(kcomp, primary, "saves_on_sota_regression", "residual_std_error"),
        places=2,
    )
    m.add("KeeperWorkloadR", dig(kdesc, primary, "savepct_vs_workload", "pearson_r"), places=3)
    m.add("KeeperWorkloadP", dig(kdesc, primary, "savepct_vs_workload", "p_value"), places=3)
    m.add("KeeperWorkloadRRepl", dig(kdesc, repl, "savepct_vs_workload", "pearson_r"), places=3)
    m.add("KeeperKPrimary", dig(kclust, primary, "chosen_k"))
    m.add("KeeperKRepl", dig(kclust, repl, "chosen_k"))
    m.add("KeeperSilhouettePrimary", dig(kclust, primary, "silhouette"), places=3)
    m.add("KeeperSilhouetteRepl", dig(kclust, repl, "silhouette"), places=3)
    m.add("KeeperMinClusterRepl", dig(kclust, repl, "min_cluster_size"))
    eta = dig(kclust, primary, "variance_explained_by_cluster") or {}
    if isinstance(eta, dict):
        m.add("KeeperEtaTeamPoints", eta.get("team_points_per_match"), places=2)
        m.add("KeeperEtaSavePct", eta.get("gk_save_pct"), places=2)
    m.add("KeeperAxesBuilt", len(dig(kcomp, primary, "axes_built") or []))
    m.add("KeeperAxesLost", len(dig(kcomp, primary, "axes_not_buildable") or []))

    # Supervised validation
    clf = load("position_classification")
    m.add("BaselineAccuracy", dig(season_block(clf, primary), "baseline", "accuracy"), places=3)
    m.add("BaselineMacroF", dig(season_block(clf, primary), "baseline", "macro_f1"), places=3)
    models = dig(season_block(clf, primary), "models") or {}
    for key, block in models.items():
        label = "LGBM" if "light" in key or "gbm" in key or "boost" in key else "Logit"
        m.add(f"{label}Accuracy", block.get("accuracy"), places=3)
        m.add(f"{label}MacroF", block.get("macro_f1"), places=3)
    mis = load("misclassified_vs_hybrid")
    for key, block in (dig(season_block(mis, primary), "models") or {}).items():
        label = "LGBM" if ("light" in key or "gbm" in key) else "Logit"
        m.add(f"Overlap{label}", block.get("overlap_rate_of_misclassified"), places=3)
        m.add(f"Enrichment{label}", block.get("enrichment_ratio"), places=2)
        m.add(f"Misclassified{label}", block.get("n_misclassified"))

    # Archetype sizes, keyed by the generated labels so the paper can cite them.
    arche = load_results("archetypes")
    _ordinal = {"0": "Zero", "1": "One"}
    for entry in dig(season_block(arche, primary), "archetypes") or []:
        label = str(entry.get("label", ""))
        group, _, idx = label.partition("-")
        suffix = _ordinal.get(idx)
        if suffix and group:
            m.add(f"N{group}{suffix}", entry.get("n_players"))
    fin = load("finishing_above_role")
    m.add("FinishingRSq", dig(season_block(fin, primary), "cv_r2_out_of_fold"), places=3)
    m.add(
        "FinishingRSqLeaky",
        dig(season_block(fin, primary), "leakage", "cv_r2_with_xg_chain_added_back"),
        places=3,
    )
    m.add(
        "FinishingLeakCorr",
        dig(season_block(fin, primary), "leakage", "corr_chain_minus_buildup_with_target"),
        places=3,
    )

    # Team signatures
    teams = load("team_signatures")
    m.add("NTeams", dig(season_block(teams, primary), "n_teams"))
    m.add("NStyleGroups", dig(season_block(teams, primary), "n_style_groups"))
    m.add(
        "TeamPCOneRho",
        dig(season_block(teams, primary), "pca1_centroid_vs_league_position", "spearman_rho"),
        places=3,
    )
    m.add(
        "TeamPCOneRhoRepl",
        dig(season_block(teams, repl), "pca1_centroid_vs_league_position", "spearman_rho"),
        places=3,
    )

    # Cross-season replication
    rep = load("cross_season_replication")
    m.add("NCommonPlayers", dig(rep, "n_common_eligible"))
    for scope in ("global", "DF", "MF", "FW"):
        label = "Global" if scope == "global" else scope
        m.add(f"CrossSeasonARI{label}", dig(rep, "scopes", scope, "adjusted_rand_index"), places=3)
        m.add(f"CrossSeasonChanged{label}", dig(rep, "scopes", scope, "n_changed_cluster"))

    # Empirical Bayes shrinkage
    shr = load("shrinkage")
    m.add("ShrinkARIGlobal", dig(shr, "headline", "adjusted_rand_index_global"), places=3)
    m.add("ShrinkChangedGlobal", dig(shr, "headline", "n_changed_global"))
    for grp in ("DF", "MF", "FW"):
        m.add(
            f"ShrinkARI{grp}",
            dig(shr, "headline", "adjusted_rand_index_within_group", grp),
            places=3,
        )
        m.add(f"ShrinkChanged{grp}", dig(shr, "headline", "n_changed_within_group", grp))

    # Possession adjustment
    padj = dig(season_block(pre, primary), "possession_adjustment") or {}
    _suffix = {
        "interceptions": "Interceptions",
        "tackles_won": "Tackles",
        "fouls_committed": "Fouls",
    }
    m.add("PadjRefPossession", padj.get("reference_possession"), places=0)
    m.add("PadjTeamSeasons", dig(padj, "elasticity", "n_team_seasons"))
    m.add("PadjPossessionMin", padj.get("team_possession_min"), places=1)
    m.add("PadjPossessionMax", padj.get("team_possession_max"), places=1)
    for count, suffix in _suffix.items():
        m.add(f"Elasticity{suffix}", dig(padj, "elasticity", "elasticities", count), places=3)
        m.add(
            f"PadjOvercorrect{suffix}",
            dig(padj, "elasticity", "per_feature", count, "unit_elasticity_overcorrection_factor"),
            places=1,
        )
        block = dig(padj, "confound_removal", count) or {}
        m.add(f"PadjCorrRaw{suffix}", block.get("corr_raw_with_possession"), places=3)
        m.add(f"PadjCorrAdj{suffix}", block.get("corr_adjusted_with_possession"), places=3)
        m.add(f"PadjCorrUnit{suffix}", block.get("corr_unit_elasticity_with_possession"), places=3)

    # Minutes threshold sensitivity
    sens = load("minutes_sensitivity")
    if sens:
        sils = [dig(sens, str(t), "silhouette") for t in config.MIN_MINUTES_SENSITIVITY]
        aris = [dig(sens, str(t), "bootstrap_ari_mean") for t in config.MIN_MINUTES_SENSITIVITY]
        sils = [v for v in sils if v is not None]
        aris = [v for v in aris if v is not None]
        if sils:
            m.add("SensSilhouetteLow", min(sils), places=3)
            m.add("SensSilhouetteHigh", max(sils), places=3)
        if aris:
            m.add("SensARILow", min(aris), places=3)
            m.add("SensARIHigh", max(aris), places=3)
        agree = dig(sens, "agreement_with_baseline") or {}
        vals = {k: v.get("adjusted_rand_index_vs_baseline") for k, v in agree.items()}
        if vals:
            lo_key = min(vals, key=lambda k: int(k))
            hi_key = max(vals, key=lambda k: int(k))
            m.add("SensAgreeLow", vals[lo_key], places=3)
            m.add("SensAgreeHigh", vals[hi_key], places=3)

    # Three dimensional structure
    td = load("threed")
    m.add("ThreeDVarTwo", dig(td, "positions", "cumulative_two_components"), places=3)
    m.add("ThreeDVarThree", dig(td, "positions", "cumulative_three_components"), places=3)
    for i, name in enumerate(("One", "Two", "Three"), start=1):
        m.add(
            f"ThreeDTeamPC{name}",
            dig(td, "teams", "corr_with_league_position", f"pc{i}"),
            places=3,
        )
    for grp in ("DF", "MF", "FW"):
        m.add(
            f"Cosine{grp}", dig(td, "archetype_axes", "within_group_centroid_cosine", grp), places=2
        )

    # Symmetric adjustment counterfactual
    sym = load("symmetric_adjustment")
    _beta = {
        "xg_buildup": "XGBuildup",
        "xg_chain": "XGChain",
        "assists": "Assists",
        "goals_non_penalty": "Goals",
        "xa": "XA",
        "np_xg": "NPXG",
        "key_passes": "KeyPasses",
        "shots": "Shots",
        "crosses": "Crosses",
    }
    for canon, suffix in _beta.items():
        m.add(
            f"Beta{suffix}",
            dig(sym, "attacking_elasticities", "per_feature", canon, "elasticity"),
            places=2,
        )
    m.add("SymCorrBaseline", dig(sym, "baseline", "team_pc1_vs_league_position"), places=3)
    m.add(
        "SymCorrAdjusted",
        dig(sym, "symmetric_counterfactual", "team_pc1_vs_league_position"),
        places=3,
    )
    m.add(
        "SymARI",
        dig(sym, "symmetric_counterfactual", "adjusted_rand_index_vs_baseline"),
        places=3,
    )
    m.add("SymSilhouetteBase", dig(sym, "baseline", "silhouette"), places=3)
    m.add("SymSilhouetteAdj", dig(sym, "symmetric_counterfactual", "silhouette"), places=3)

    # Age and archetype
    ages = load("age_archetype")
    for grp in ("DF", "MF", "FW"):
        block = dig(ages, "by_position_group", grp) or {}
        m.add(f"AgeTrendRho{grp}", dig(block, "age_trend", "spearman_rho"), places=3)
        m.add(f"AgeTrendP{grp}", dig(block, "age_trend", "p_value"), places=3)
        m.add(
            f"AgeGap{grp}",
            dig(block, "kruskal_age_difference", "mean_difference_years"),
            places=2,
        )
        m.add(f"AgeKruskalP{grp}", dig(block, "kruskal_age_difference", "p_value"), places=3)
        m.add(f"AgeChiP{grp}", dig(block, "association", "p_value"), places=3)
    m.add("AgeGlobalP", dig(ages, "global_cluster", "association", "p_value"), places=3)

    # Role drift
    drift = load("role_drift")
    m.add_raw("DriftTeam", tex_escape(dig(drift, "team") or ""))
    m.add("DriftMatches", dig(drift, "n_matches_scraped"))
    m.add("DriftPlayers", dig(drift, "n_players_with_windows"))
    m.add("DriftWindow", dig(drift, "window_matches"))
    m.add("DriftFeatures", len(dig(drift, "match_features") or []))
    m.add("DriftSwitchers", dig(drift, "n_players_switching"))
    m.add_pct("DriftSwitchShare", dig(drift, "share_of_players_switching"), places=0)
    m.add(
        "DriftReducedARI",
        dig(drift, "reduced_space", "adjusted_rand_index_vs_main_partition"),
        places=3,
    )

    # Multi-league replication
    lg = load("league_comparison")
    rows = dig(lg, "leagues") or []
    short = {
        "ENG-Premier League": "Eng",
        "ESP-La Liga": "Esp",
        "GER-Bundesliga": "Ger",
        "ITA-Serie A": "Ita",
        "FRA-Ligue 1": "Fra",
    }
    for row in rows:
        tag = short.get(row.get("league"), "")
        if not tag:
            continue
        m.add(f"LgK{tag}", row.get("chosen_k"))
        m.add(f"LgSil{tag}", row.get("silhouette"), places=3)
        m.add(f"LgBootARI{tag}", row.get("bootstrap_ari_mean"), places=3)
        m.add(f"LgARIPos{tag}", row.get("ari_vs_position_group"), places=3)
        m.add(f"LgN{tag}", row.get("n_eligible"))
    m.add("NLeagues", len(rows))
    if rows:
        m.add("LgSilMin", min(r["silhouette"] for r in rows), places=3)
        m.add("LgSilMax", max(r["silhouette"] for r in rows), places=3)
        m.add("LgBootARIMin", min(r["bootstrap_ari_mean"] for r in rows), places=3)
        m.add("LgBootARIMax", max(r["bootstrap_ari_mean"] for r in rows), places=3)
        m.add("LgARIPosMin", min(r["ari_vs_position_group"] for r in rows), places=3)
        m.add("LgARIPosMax", max(r["ari_vs_position_group"] for r in rows), places=3)
    prep = dig(lg, "preparation") or {}
    els = [v for info in prep.values() for v in (info.get("elasticities") or {}).values()]
    if els:
        m.add("LgElastMin", min(els), places=3)
        m.add("LgElastMax", max(els), places=3)
        m.add("LgElastCount", len(els) + len(config.PADJ_COUNTS))

    # The archive sample: the paper's primary dataset
    arch = load("archive")
    m.add("ArcRows", dig(arch, "rows_all"))
    m.add("ArcEligible", dig(arch, "rows_eligible"))
    m.add("ArcKeepers", dig(arch, "rows_keepers"))
    m.add("ArcFeatures", dig(arch, "n_features"))
    m.add("ArcNSeasons", len(dig(arch, "seasons") or []))
    m.add("ArcNLeagues", len(dig(arch, "leagues") or []))
    seasons = dig(arch, "seasons") or []
    if seasons:
        m.add_year("ArcSeasonFirst", seasons[0])
        m.add_year("ArcSeasonLast", seasons[-1])
    for grp in ("DF", "MF", "FW"):
        m.add(f"ArcN{grp}", dig(arch, "eligible_by_position", grp))
    for count, tag in (
        ("tackles", "Tackles"),
        ("interceptions", "Interceptions"),
        ("blocks", "Blocks"),
        ("clearances", "Clearances"),
        ("fouls_committed", "Fouls"),
    ):
        m.add(f"ArcElast{tag}", dig(arch, "elasticities", count), places=3)

    # Structure: is the space clustered or continuous?
    st = load("structure")
    scopes = dig(st, "scopes") or {}
    name_map = {"All outfield": "All", "DF": "DF", "MF": "MF", "FW": "FW"}
    for scope, tag in name_map.items():
        cal = dig(scopes, scope, "null_calibration") or {}
        m.add(f"StrObs{tag}", cal.get("observed_best_silhouette"), places=3)
        m.add(f"StrNull{tag}", cal.get("null_best_silhouette_mean"), places=3)
        m.add(f"StrRatio{tag}", cal.get("separation_ratio"), places=3)
        m.add(f"StrZ{tag}", dig(cal, "by_k", "2", "z_against_null"), places=1)
        dip = dig(scopes, scope, "dip_tests", "PC1") or {}
        m.add(f"StrDip{tag}", dip.get("dip"), places=4)
        m.add_pvalue(f"StrDipP{tag}", dip.get("p_value"))
        hd = dig(scopes, scope, "density", "50") or {}
        m.add(f"StrNoise{tag}", hd.get("noise_fraction"), places=3)
    m.add("StrSimulations", dig(scopes, "All outfield", "null_calibration", "n_simulations"))
    m.add("StrRatioMin", dig(st, "summary", "separation_ratio_min"), places=3)
    m.add("StrRatioMax", dig(st, "summary", "separation_ratio_max"), places=3)

    # Goalkeepers on the primary sample
    kav = load("archive_keeper_availability")
    karc = load("archive_keeper_archetypes")
    kclu = load("archive_keeper_clusters")
    m.add("ArcKeeperSeasons", dig(kav, "keeper_seasons"))
    cov = dig(kav, "canonical_features") or {}
    rates = (
        [v.get("coverage") for v in cov.values() if isinstance(v, dict) and v.get("coverage")]
        if isinstance(cov, dict)
        else []
    )
    if rates:
        m.add("ArcKeeperCoverageMin", min(rates), places=3)
    if isinstance(cov, dict) and cov:
        m.add("KeeperFeatureCountArchive", len(cov))
    m.add("ArcKeeperK", dig(karc, "k"))
    m.add("ArcKeeperSil", dig(karc, "silhouette"), places=3)
    m.add("ArcKeeperBootARI", dig(karc, "bootstrap_ari_mean"), places=3)
    m.add_raw("ArcKeeperPrimaryVerdict", tex_escape(dig(karc, "primary_space_verdict") or ""))
    spaces = dig(kclu, "spaces") or {}
    for key, tag in (("full", "Full"), ("technique", "Tech")):
        sp = spaces.get(key) or {}
        m.add(f"ArcKeeperSil{tag}", sp.get("silhouette"), places=3)
        val = sp.get("validation") or {}
        for side in ("situation", "technique"):
            eta = dig(val, "mean_eta_squared", side) or dig(val, f"mean_eta_squared_{side}")
            m.add(f"ArcKeeperEta{tag}{side.capitalize()}", eta, places=3)
    arche = dig(karc, "archetypes") or {}
    if isinstance(arche, dict):
        for idx, (_label, entry) in enumerate(sorted(arche.items())):
            tag = ["Zero", "One"][idx] if idx < 2 else str(idx)
            m.add_raw(f"ArcKeeperName{tag}", tex_escape(entry.get("name", "")))
            m.add(f"ArcKeeperN{tag}", entry.get("n_players") or entry.get("n"))

    # Archetypes on the primary sample
    aa = load_results("archive_archetypes")
    _tag = {
        "DF-0": "DFZero",
        "DF-1": "DFOne",
        "MF-0": "MFZero",
        "MF-1": "MFOne",
        "FW-0": "FWZero",
        "FW-1": "FWOne",
    }
    for entry in dig(aa, "archetypes") or []:
        tag = _tag.get(str(entry.get("label")))
        if not tag:
            continue
        m.add_raw(f"ArcName{tag}", tex_escape(entry.get("name", "")))
        m.add(f"ArcN{tag}", entry.get("n_players"))

    ast = load("archive_archetype_stability")
    cs = dig(ast, "consecutive_seasons") or {}
    m.add("ArcTracked", cs.get("n_tracked_players"))
    m.add("ArcTrackedTwo", cs.get("players_with_two_or_more_seasons"))
    m.add_pct("ArcSwitchRate", cs.get("mean_switch_rate_within_group"), places=1)
    pairs = cs.get("pairs") or []
    aris = [p.get("adjusted_rand_index") for p in pairs if isinstance(p, dict)]
    aris = [a for a in aris if a is not None]
    if aris:
        m.add("ArcSeasonARIMin", min(aris), places=3)
        m.add("ArcSeasonARIMax", max(aris), places=3)
        m.add("ArcSeasonARIMean", sum(aris) / len(aris), places=3)
    m.add("ArcLeagueARI", dig(ast, "league_refits", "overall_mean_adjusted_rand_index"), places=3)
    m.add(
        "ArcSeasonRefitARI", dig(ast, "season_refits", "overall_mean_adjusted_rand_index"), places=3
    )
    for grp in ("DF", "MF", "FW"):
        boot = dig(ast, "bootstrap_resampling", grp) or {}
        val = boot.get("ari_mean") if isinstance(boot, dict) else None
        m.add(f"ArcBoot{grp}", val, places=3)

    # Possession elasticities on the primary sample
    det = dig(arch, "elasticity_detail") or {}
    _elast_tag = {
        "tackles": "Tackles",
        "interceptions": "Interceptions",
        "blocks": "Blocks",
        "clearances": "Clearances",
        "fouls_committed": "Fouls",
    }
    seasons_seen = set()
    for key, tag in _elast_tag.items():
        entry = det.get(key) or {}
        m.add(f"ArcElastCorr{tag}", entry.get("log_log_correlation"), places=3)
        if entry.get("n_team_seasons"):
            seasons_seen.add(int(entry["n_team_seasons"]))
    if len(seasons_seen) == 1:
        m.add("ArcPadjTeamSeasons", seasons_seen.pop())
    vals = [v for v in (dig(arch, "elasticities") or {}).values() if v is not None]
    if vals:
        m.add("ArcElastCount", len(vals))
        m.add("ArcElastMax", max(vals), places=3)
        m.add("ArcElastMin", min(vals), places=3)

    # Uncertainty on the possession elasticities. The paper's claim changed on seeing these:
    # four intervals exclude one and clearances does not, so the wording is per statistic.
    el = load("elasticity")
    _el_tag = {
        "tackles": "Tackles",
        "interceptions": "Interceptions",
        "blocks": "Blocks",
        "clearances": "Clearances",
        "fouls_committed": "Fouls",
    }
    m.add("ElastNBootstrap", dig(el, "n_bootstrap"))
    for key, tag in _el_tag.items():
        pooled = dig(el, "pooled", key) or {}
        m.add(f"ElastCI{tag}Low", pooled.get("ci_low"), places=3)
        m.add(f"ElastCI{tag}High", pooled.get("ci_high"), places=3)
        m.add(f"ElastSE{tag}", pooled.get("bootstrap_se"), places=3)
        m.add_pvalue(f"ElastPOne{tag}", pooled.get("p_value_at_least_one"))
        m.add_pct(f"ElastShareOne{tag}", pooled.get("share_at_or_above_one"))
        het = dig(el, "heterogeneity", "league", key) or {}
        m.add(f"ElastLeagueMin{tag}", het.get("min"), places=3)
        m.add(f"ElastLeagueMax{tag}", het.get("max"), places=3)
        m.add_raw(f"ElastLeagueMinName{tag}", tex_escape(het.get("scope_of_min", "")))
        m.add_raw(f"ElastLeagueMaxName{tag}", tex_escape(het.get("scope_of_max", "")))
        m.add(f"ElastLeaguesAboveOne{tag}", het.get("n_estimates_at_or_above_one"))
        ff = dig(el, "functional_form", key) or {}
        m.add(f"ElastQuadDeltaR{tag}", ff.get("adjusted_r2_change"), places=3)
        m.add(f"ElastAtLow{tag}", ff.get("elasticity_at_low_possession"), places=2)
        m.add(f"ElastAtHigh{tag}", ff.get("elasticity_at_high_possession"), places=2)
    sw = dig(el, "sweeper") or {}
    m.add("ElastSweeper", sw.get("elasticity_to_opponent_possession"), places=3)
    m.add("ElastCISweeperLow", sw.get("ci_low"), places=3)
    m.add("ElastCISweeperHigh", sw.get("ci_high"), places=3)
    m.add("ElastSweeperTeamSeasons", sw.get("n_team_seasons"))
    summ = dig(el, "summary") or {}
    m.add("NElastExcludeOne", len(summ.get("intervals_excluding_one") or []))
    m.add("NElastQuadMatters", len(summ.get("quadratic_term_matters") or []))
    ff_any = dig(el, "functional_form", "fouls_committed") or {}
    m.add("ElastPossLow", ff_any.get("opponent_possession_low"), places=1)
    m.add("ElastPossHigh", ff_any.get("opponent_possession_high"), places=1)

    # Feature-family ablation on the primary sample.
    ab = load("ablation")
    m.add("AblSimulations", dig(ab, "n_simulations"))
    m.add("AblFamilies", len(dig(ab, "families") or {}))
    absum = dig(ab, "summary") or {}
    weakest = absum.get("weakest_leave_one_out_any_scope") or {}
    m.add_raw("AblWeakScope", tex_escape(weakest.get("scope", "")))
    m.add_raw("AblWeakFamily", tex_escape(weakest.get("family", "")))
    m.add("AblWeakZ", weakest.get("z"), places=2)
    m.add("AblWeakK", weakest.get("chosen_k"))
    m.add("AblWeakARI", weakest.get("ari_vs_full_at_k"), places=3)
    least = absum.get("least_similar_leave_one_out_any_scope") or {}
    m.add_raw("AblLeastScope", tex_escape(least.get("scope", "")))
    m.add_raw("AblLeastFamily", tex_escape(least.get("family", "")))
    m.add("AblLeastARI", least.get("ari_vs_full_at_k"), places=3)
    m.add("AblLeastZ", least.get("z"), places=2)
    every = absum.get("families_carrying_alone_in_every_scope") or []
    m.add("AblNCarryEvery", len(every))
    m.add_raw("AblCarryEvery", tex_escape(", ".join(every)))
    none = absum.get("families_carrying_alone_in_no_scope") or []
    m.add("AblNCarryNone", len(none))
    m.add_raw("AblCarryNone", tex_escape(", ".join(none) if none else "none"))
    m.add_raw(
        "AblSurvivesAll",
        "every" if absum.get("k_survives_every_leave_one_out_in_every_scope") else "not every",
    )
    for scope, tag in (("All outfield", "All"), ("DF", "DF"), ("MF", "MF"), ("FW", "FW")):
        sc = dig(ab, "scopes", scope) or {}
        m.add(f"AblFullZ{tag}", dig(sc, "full", "null_at_k", "z_against_null"), places=2)
        ss = sc.get("summary") or {}
        wk = ss.get("weakest_leave_one_out") or {}
        m.add(f"AblWeakZ{tag}", wk.get("z"), places=2)
        m.add_raw(f"AblWeakFamily{tag}", tex_escape(wk.get("family", "")))
        m.add(f"AblNCarry{tag}", len(ss.get("families_carrying_alone") or []))
        nc = ss.get("families_not_carrying_alone") or []
        m.add_raw(f"AblNotCarry{tag}", tex_escape(", ".join(nc) if nc else "none"))

    # Which single families reproduce the paper's own partition, not merely some partition.
    _ab_scopes = (("All outfield", "All"), ("DF", "DF"), ("MF", "MF"), ("FW", "FW"))
    territory, leave_aris = [], []
    for scope, tag in _ab_scopes:
        ss = dig(ab, "scopes", scope, "summary") or {}
        ko = ss.get("keep_one_only_ari_vs_full") or {}
        lo = ss.get("leave_one_out_ari_vs_full") or {}
        leave_aris.extend(v for v in lo.values() if v is not None)
        for fam in ("shooting", "creation", "progression", "territory", "defending", "passing"):
            m.add(f"AblKeepARI{fam.capitalize()}{tag}", ko.get(fam), places=3)
            m.add(
                f"AblKeepZ{fam.capitalize()}{tag}",
                dig(ab, "scopes", scope, "keep_one_only", fam, "null_at_k", "z_against_null"),
                places=2,
            )
        if ko.get("territory") is not None:
            territory.append(ko["territory"])
    if territory:
        m.add("AblTerritoryARIMin", min(territory), places=3)
        m.add("AblTerritoryARIMax", max(territory), places=3)
    if leave_aris:
        m.add("AblLeaveARIMin", min(leave_aris), places=3)
        m.add("AblLeaveARIMax", max(leave_aris), places=3)
    cells = 0
    beats = 0
    for scope, _ in _ab_scopes:
        sc = dig(ab, "scopes", scope) or {}
        for mode in ("leave_one_out", "keep_one_only"):
            for entry in (sc.get(mode) or {}).values():
                cells += 1
                beats += bool(dig(entry, "null_at_k", "exceeds_every_simulation"))
        cells += 1
        beats += bool(dig(sc, "full", "null_at_k", "exceeds_every_simulation"))
    m.add("AblCells", cells)
    m.add("AblCellsBeatNull", beats)

    # Three nulls, the null-data bootstrap and the dip effect size. The paper cites the
    # hardest null everywhere, so those are the numbers the reader can check.
    st = load("structure")
    stsum = dig(st, "summary") or {}
    m.add("StrNulls", len(dig(st, "nulls") or {}))
    _sc = (("All outfield", "All"), ("DF", "DF"), ("MF", "MF"), ("FW", "FW"))
    _nulltag = {
        "gaussian": "Gauss",
        "gaussian_copula": "Copula",
        "uniform_principal_box": "Uniform",
    }
    _nullname = {
        "gaussian": "Gaussian",
        "gaussian_copula": "Gaussian copula",
        "uniform_principal_box": "uniform",
    }
    for scope, tag in _sc:
        k2 = dig(stsum, "k2_against_three_nulls", scope) or {}
        for null, nt in _nulltag.items():
            m.add(f"StrZ{nt}{tag}", dig(k2, "z_by_null", null), places=2)
        m.add(f"StrMinZ{tag}", k2.get("min_z"), places=2)
        m.add_raw(f"StrHardest{tag}", _nullname.get(k2.get("hardest_null", ""), ""))
        dp = dig(stsum, "dip_pc1_against_three_nulls", scope) or {}
        m.add(f"StrDipObs{tag}", dp.get("observed_dip"), places=4)
        m.add(f"StrDipMinZ{tag}", dp.get("min_z"), places=2)
        m.add_raw(f"StrDipHardest{tag}", _nullname.get(dp.get("hardest_null", ""), ""))
        for null, nt in _nulltag.items():
            m.add(f"StrDipZ{nt}{tag}", dig(dp, "z_by_null", null), places=2)
        nb = dig(st, "scopes", scope, "null_bootstrap") or {}
        m.add(f"StrBootObs{tag}", dig(nb, "observed", "ari_mean"), places=3)
        for null, nt in _nulltag.items():
            m.add(f"StrBootNull{nt}{tag}", dig(nb, "by_null", null, "ari_mean"), places=3)
    m.add("StrMinZOverall", stsum.get("k2_min_z_over_scopes_and_nulls"), places=2)
    m.add_raw(
        "StrBeatsEverywhere",
        "every"
        if stsum.get("k2_exceeds_every_simulation_under_every_null_everywhere")
        else "not every",
    )
    bs = stsum.get("bootstrap") or {}
    m.add("StrBootScreen", bs.get("stability_screen_ari"), places=1)
    m.add("StrBootObsMin", bs.get("observed_ari_min_over_scopes"), places=3)
    m.add("StrBootObsMax", bs.get("observed_ari_max_over_scopes"), places=3)
    m.add("StrBootNullMin", bs.get("null_ari_min_over_scopes_and_nulls"), places=3)
    m.add("StrBootNullMax", bs.get("null_ari_max_over_scopes_and_nulls"), places=3)
    m.add("StrBootNullMinSingle", bs.get("null_ari_min_single_dataset"), places=3)
    m.add("StrBootGapMin", bs.get("observed_minus_null_min"), places=3)
    m.add("StrBootGapMax", bs.get("observed_minus_null_max"), places=3)
    m.add_raw(
        "StrBootNullPassesScreen",
        "every" if bs.get("null_passes_stability_screen_everywhere") else "not every",
    )
    m.add(
        "StrBootDatasets",
        dig(st, "scopes", "All outfield", "null_bootstrap", "n_datasets_per_null"),
    )
    m.add("StrBootResamples", dig(st, "scopes", "All outfield", "null_bootstrap", "n_bootstrap"))
    dips = [dig(stsum, "dip_pc1_against_three_nulls", sc, "min_z") for sc, _ in _sc]
    dips = [v for v in dips if v is not None]
    if dips:
        m.add("StrDipMinZOverall", min(dips), places=2)
        m.add("StrDipMinZOverallPositive", min(v for v in dips if v > 0), places=2)

    # The only test against real results: composition on the two-mode axis against points.
    oc = load("outcomes")
    m.add("OutN", dig(oc, "composition", "n_team_seasons"))
    m.add_pct("OutCoverageMin", dig(oc, "composition", "coverage_min"))
    m.add_pct("OutCoverageMedian", dig(oc, "composition", "coverage_median"))
    m.add("OutGFOnPitch", dig(oc, "outcomes_audit", "n_goals_for_from_on_pitch"))
    m.add("OutGFFallback", dig(oc, "outcomes_audit", "n_goals_for_from_player_total"))
    m.add("OutOffSchedule", len(dig(oc, "outcomes_audit", "credited_results_off_schedule") or []))
    for meas, tag in (("attacking_share", "Share"), ("axis_mean", "Axis")):
        for tgt, tt in (("points_per_match", ""), ("goal_difference_per_match", "GD")):
            c = dig(oc, "correlations", "pooled", meas, tgt) or {}
            m.add(f"OutRho{tag}{tt}", c.get("rho"), places=3)
            m.add(f"OutRho{tag}{tt}Low", c.get("ci_low"), places=3)
            m.add(f"OutRho{tag}{tt}High", c.get("ci_high"), places=3)
    # Within-league spread of the share correlation, whatever the nesting order.
    by_league = dig(oc, "correlations", "by_league") or {}
    rhos = {}
    for k1, v1 in by_league.items():
        c = dig(v1, "attacking_share", "points_per_match", "rho")
        if c is None:
            c = dig(v1, "points_per_match", "rho") if k1 == "attacking_share" else None
        if c is not None:
            rhos[k1] = c
    if not rhos and "attacking_share" in by_league:
        for lg, v in by_league["attacking_share"].items():
            c = dig(v, "points_per_match", "rho")
            if c is not None:
                rhos[lg] = c
    if rhos:
        lo_l, hi_l = min(rhos, key=rhos.get), max(rhos, key=rhos.get)
        m.add("OutRhoLeagueMin", rhos[lo_l], places=2)
        m.add("OutRhoLeagueMax", rhos[hi_l], places=2)
        m.add_raw("OutRhoLeagueMinName", tex_escape(lo_l))
        m.add_raw("OutRhoLeagueMaxName", tex_escape(hi_l))
    for tgt, tt in (("points_per_match", ""), ("goal_difference_per_match", "GD")):
        for scheme, st in (("season", "Season"), ("league", "League"), ("team", "Club")):
            models = dig(oc, "ridge", tgt, scheme, "models") or {}
            for name, mt in (
                ("possession", "Poss"),
                ("composition", "Comp"),
                ("both", "Both"),
                ("two_mode", "TwoMode"),
                ("archetypes", "Arche"),
                ("possession_and_two_mode", "PossTwoMode"),
                ("possession_and_archetypes", "PossArche"),
            ):
                m.add(f"OutRSq{mt}{tt}{st}", dig(models, name, "r2_out_of_sample"), places=3)
            incs = dig(oc, "ridge", tgt, scheme, "increments") or {}
            for name, it in (
                ("both_over_possession", "Inc"),
                ("possession_and_two_mode_over_possession", "IncTwoMode"),
                ("both_over_possession_and_two_mode", "IncArche"),
                ("possession_and_archetypes_over_possession", "IncArcheOnly"),
            ):
                e = incs.get(name) or {}
                m.add(f"Out{it}{tt}{st}", e.get("delta_r2"), places=3)
                m.add(f"Out{it}{tt}{st}Low", e.get("ci_low"), places=3)
                m.add(f"Out{it}{tt}{st}High", e.get("ci_high"), places=3)
    for name, kt in (
        ("cluster_full_share", "Full"),
        ("cluster_technique_share", "Tech"),
        ("shot_stopping", "Stop"),
    ):
        c = dig(oc, "keepers", "tests", name, "points_per_match") or {}
        if not c:
            # The shot stopping test may live under a differently named key.
            for kk, vv in (dig(oc, "keepers", "tests") or {}).items():
                if kt == "Stop" and ("psxg" in kk.lower() or "stop" in kk.lower()):
                    c = dig(vv, "points_per_match") or {}
        m.add(f"OutKeep{kt}", c.get("rho"), places=3)
        m.add(f"OutKeep{kt}Low", c.get("ci_low"), places=3)
        m.add(f"OutKeep{kt}High", c.get("ci_high"), places=3)
    m.add_pvalue(
        "OutKeepTechP",
        dig(oc, "keepers", "tests", "cluster_technique_share", "points_per_match", "p_value"),
    )
    for side, st in (("highest_attacking_share", "Top"), ("lowest_attacking_share", "Bottom")):
        rows = dig(oc, "extremes", side) or []
        for i, e in enumerate(rows[:3]):
            m.add_raw(
                f"Out{st}{['One', 'Two', 'Three'][i]}",
                tex_escape(f"{e.get('team')} {e.get('season')}"),
            )
    table_outcomes_extremes(oc)

    # Do the two modes simply recover listed position?
    acv = load("archive_cluster_validation")
    m.add("ArcARIvsPosition", dig(acv, "answer", "adjusted_rand_index_vs_position_group"), places=3)
    m.add(
        "ArcNMIvsPosition", dig(acv, "answer", "normalized_mutual_info_vs_position_group"), places=3
    )
    m.add_pct("ArcPurityVsPosition", dig(acv, "answer", "cluster_purity_vs_position_group"))
    m.add(
        "ArcARIvsPositionFull", dig(acv, "answer", "adjusted_rand_index_vs_position_full"), places=3
    )
    m.add("ArcSilhouetteAll", dig(acv, "scopes", "All outfield", "silhouette_at_k"), places=3)
    m.add("ArcBootAll", dig(acv, "scopes", "All outfield", "bootstrap", "ari_mean"), places=3)
    cont = dig(acv, "scopes", "All outfield", "vs_position", "position_group", "contingency") or {}
    # Mode 0 is the defensive pole, mode 1 the attacking pole; the midfield straddles them.
    for mode, tag in (("0", "Def"), ("1", "Att")):
        cell = cont.get(mode) or {}
        total = sum(cell.values()) or None
        for grp in ("DF", "MF", "FW"):
            m.add(f"ArcMode{tag}{grp}", cell.get(grp))
        if total:
            m.add_pct(f"ArcMode{tag}Share", cell.get("MF", 0) / total)
    if cont.get("0") and cont.get("1"):
        for grp in ("DF", "MF", "FW"):
            lo, hi = cont["0"].get(grp, 0), cont["1"].get(grp, 0)
            if lo + hi:
                m.add_pct(f"ArcSplit{grp}", max(lo, hi) / (lo + hi))

    # Replication of the two-mode solution across seasons and leagues
    rep = load("archive_replication")
    m.add("RepN", dig(rep, "n_players_seasons"))
    m.add("RepSeasonPooledMin", dig(rep, "summary", "season_vs_pooled_ari_min"), places=3)
    m.add("RepLeaguePooledMin", dig(rep, "summary", "league_vs_pooled_ari_min"), places=3)
    m.add("RepAxisCorrMin", dig(rep, "summary", "axis_correlation_min"), places=4)
    m.add("RepTransferMean", dig(rep, "summary", "league_transfer_ari_mean"), places=3)
    m.add("RepTransferMin", dig(rep, "summary", "league_transfer_ari_min"), places=3)
    m.add("RepPairMean", dig(rep, "summary", "season_pairwise_ari_mean"), places=3)
    m.add("RepPairMin", dig(rep, "summary", "season_pairwise_ari_min"), places=3)
    m.add("RepPairMax", dig(rep, "summary", "season_pairwise_ari_max"), places=3)
    m.add_pct("RepShareMean", dig(rep, "summary", "season_pairwise_same_cluster_share_mean"))
    m.add_pct("RepShareMin", dig(rep, "summary", "season_pairwise_same_cluster_share_min"))

    # Supervised lens on the archive sample
    sup = load("archive_supervised")
    m.add("SupN", dig(sup, "n_players"))
    m.add_pct("SupBaseline", dig(sup, "models", "majority_baseline", "accuracy"))
    for tag, key in (("Logit", "logistic_regression"), ("GBM", "lightgbm")):
        m.add_pct(f"Sup{tag}Acc", dig(sup, "models", key, "accuracy"))
        m.add(f"Sup{tag}F", dig(sup, "models", key, "macro_f1"), places=3)
    for grp in ("DF", "MF", "FW"):
        m.add(f"Sup{grp}F", dig(sup, "models", "lightgbm", "per_class", grp, "f1"), places=3)
        m.add_pct(f"Sup{grp}Recall", dig(sup, "models", "lightgbm", "per_class", grp, "recall"))
    counts = dig(sup, "models", "lightgbm", "confusion_matrix", "counts") or []
    labels = dig(sup, "models", "lightgbm", "confusion_matrix", "labels") or []
    if counts and labels:
        i = {lab: n for n, lab in enumerate(labels)}
        # The poles of the space almost never trade places; the middle trades with both.
        m.add("SupDFasFW", counts[i["DF"]][i["FW"]])
        m.add("SupFWasDF", counts[i["FW"]][i["DF"]])
        m.add("SupMFasDF", counts[i["MF"]][i["DF"]])
        m.add("SupMFasFW", counts[i["MF"]][i["FW"]])
        m.add("SupFWasMF", counts[i["FW"]][i["MF"]])
        m.add("SupDFasMF", counts[i["DF"]][i["MF"]])
        poles = counts[i["DF"]][i["FW"]] + counts[i["FW"]][i["DF"]]
        middle = (
            counts[i["MF"]][i["DF"]]
            + counts[i["MF"]][i["FW"]]
            + counts[i["DF"]][i["MF"]]
            + counts[i["FW"]][i["MF"]]
        )
        m.add("SupPoleErrors", poles)
        m.add("SupMiddleErrors", middle)
        m.add_pct("SupPoleShare", poles / (poles + middle) if poles + middle else None, places=2)
    # How many of each class's five strongest features identify it by presence rather
    # than by absence. The paper reads these out as counts, so they are counts here.
    shares = dig(sup, "shap", "presence_share_of_top5") or {}
    for grp in ("DF", "MF", "FW"):
        share = shares.get(grp)
        if share is not None:
            m.add(f"ShapPresence{grp}", round(share * 5), places=0)

    top = dig(sup, "shap", "overall_ranking") or []
    for n, entry in enumerate(top[:3]):
        m.add_raw(f"ShapArcTop{['One', 'Two', 'Three'][n]}", tex_escape(entry.get("label", "")))

    # Players the two-mode description fits worst
    aout = load("archive_outliers")
    m.add("ArcOutN", dig(aout, "n_reported"))
    m.add_pct("ArcOutAgreement", dig(aout, "detector_agreement"))
    players = dig(aout, "players") or []
    if players:
        m.add("ArcOutCompound", sum(1 for e in players if "," in str(e.get("position_full", ""))))
        m.add("ArcOutShortMinutes", sum(1 for e in players if (e.get("minutes") or 0) < 900))
        m.add("ArcOutDistinct", len({e.get("player") for e in players}))
        m.add_raw("ArcOutTop", tex_escape(players[0].get("player", "")))
        seen: list[str] = []
        for e in players:
            name = str(e.get("player", ""))
            if name and name not in seen:
                seen.append(name)
        m.add_raw("ArcOutNames", tex_escape(", ".join(seen[:3])))

    # How many dimensions the archive role space has
    apca = load("archive_pca")
    m.add_pct("ArcPCOne", dig(apca, "pc1_share"))
    m.add_pct("ArcPCThree", dig(apca, "pc1_to_pc3_share"))
    m.add("ArcPCNinety", dig(apca, "n_components_for_90pct"))
    m.add("ArcPCKaiser", dig(apca, "n_components_above_kaiser"))

    # Additional descriptive counts
    m.add("NPairsSignFlip", dig(div, primary, "n_pairs_sign_flip"))
    m.add("NPairsTotal", dig(div, primary, "n_pairs"))
    m.add("PCNinetyComponents", dig(pca, primary, "n_components_for_90pct"))
    m.add("PCCumThree", dig(pca, primary, "cumulative_explained_variance", 2), places=3)

    return m


def main() -> None:
    config.TABLES.mkdir(parents=True, exist_ok=True)

    # The palette validation record is provenance for a design decision the paper cites,
    # so it is written here rather than by a figure module: `make clean` removes it, and
    # nothing in the figure path was regenerating it.
    from src import plotting

    plotting.write_palette_provenance()

    table_data_availability(load("keeper_data_availability"))
    table_feature_dictionary()
    table_archive_feature_dictionary()
    table_ablation(load("ablation"))
    table_umap_grid(load("umap_grid"))
    table_divergence(load("correlation_divergence"))
    table_summary_by_position(load("descriptive_summary"))
    table_k_selection(load("cluster_selection"))
    write_reduced_membership()
    table_minutes_sensitivity(load("preprocess"), load("minutes_sensitivity"))
    table_possession_adjustment(load("preprocess"))

    macros = build_macros()
    count = macros.write(config.RESULTS / "macros.tex")

    written = sorted(p.name for p in config.TABLES.glob("*.tex"))
    print(f"wrote {count} macros to results/macros.tex")
    print(f"wrote {len(written)} tables: {', '.join(written)}")
    if macros.skipped:
        print(f"macros skipped because their metrics are absent: {', '.join(macros.skipped)}")


if __name__ == "__main__":
    main()
