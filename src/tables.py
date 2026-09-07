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


def table_cluster_membership() -> None:
    """Every eligible player with club, listed position, archetype and centroid distance."""
    path = config.DATA_PROCESSED / f"archetypes_{config.SEASON_PRIMARY}.parquet"
    if not path.exists():
        return
    df = pd.read_parquet(path).sort_values(["position_group", "archetype_name", "player"])
    lines = [
        r"\small",
        # Fixed-width wrapping columns: archetype names run to several words and would
        # otherwise push the table well past the text block.
        r"\begin{longtable}{p{3.4cm}p{2.5cm}p{1.0cm}p{5.0cm}r}",
        r"\caption{Cluster membership for every eligible outfield player in the earlier of the two "
        r"reduced-sample seasons.}\\",
        r"\label{tab:membership}\\",
        r"\toprule",
        r"Player & Club & Listed & Archetype & Distance \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Player & Club & Listed & Archetype & Distance \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"{tex_escape(r['player'])} & {tex_escape(r['team'])} & "
            f"{tex_escape(r['position_group'])} & {tex_escape(r['archetype_name'])} & "
            f"{num(r.get('distance_to_centroid'), 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}"]
    _write_table("cluster_membership", "\n".join(lines))


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
    table_umap_grid(load("umap_grid"))
    table_divergence(load("correlation_divergence"))
    table_summary_by_position(load("descriptive_summary"))
    table_k_selection(load("cluster_selection"))
    table_cluster_membership()
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
