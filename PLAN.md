# Plan: Premier League Player Profiling with Unsupervised Learning

## Context

Build a complete, reproducible sports-analytics research project from an empty directory to a
compiled LaTeX paper: *"Unsupervised Learning and Statistical Analysis of Premier League Player
Statistics: Data-Driven Role Discovery from the 2024-25 Season"* (Zachary Chung, Christian Chung).

The scientific goal is data-driven role discovery: instead of accepting FBref's listed positions,
derive player archetypes from per-90 statistical profiles using dimensionality reduction and
clustering, validate them with stability analysis and a supervised lens, extend to goalkeepers and
team-level stylistic signatures, and replicate across a second season.

Target location: **`~/pl-player-profiling`** (the current working directory is `/Users/chung`, the
home directory, which is not a suitable project root and is not a git repo).

Pre-flight reconnaissance is already done and surfaced two spec-breaking issues, resolved below.

---

## Verified pre-flight findings

| Check | Result |
|---|---|
| FBref via plain HTTP | 403 + Cloudflare challenge (`Just a moment`, `challenge-platform`) |
| `soccerdata` Cloudflare stack | seleniumbase `Driver()` + `wrapper-tls-requests`; needs real Chrome (present) |
| `soccerdata` 1.9.1 season stat types | **only 5**: standard, keeper, shooting, playing_time, misc |
| `soccerdata` 1.8.8 season stat types | **all 11** required, player *and* team, identical HTTP stack |
| `soccerdata` 1.8.8 `requires_python` | `<3.14,>=3.9` |
| System Python | 3.14.6 (**too new**, cannot run soccerdata) |
| FBref rate limit in library | 7 s/request, so full pull ≈ 32 requests ≈ 5-10 min |
| `read_player_match_stats` (1.8.8) | summary, keepers, passing, passing_types, defense, possession, misc |
| uv / TeX / pandas / sklearn / umap / lightgbm / shap / ruff | all **missing** |
| Chrome, brew, git, make | present |
| Disk | 111 GB free |

---

## Deviations from the master prompt, with rationale

These are deliberate and will be recorded in `CLAUDE.md` and in the paper's reproducibility note.

1. **Pin `soccerdata==1.8.8`, not latest.** 1.9.x removed 6 of the 11 required stat tables, which
   would delete roughly 25 of the ~35 `outfield_core` features (all progressive passing, all
   defensive actions, carries/touches/take-ons, SCA/GCA) and the entire `keeper_core` set. Without
   this pin the paper's central analysis is not possible.
2. **Python 3.12 is mandatory, not merely preferred.** System 3.14.6 is outside soccerdata's
   supported range. `uv python install 3.12` provides it.
3. **HDBSCAN from `sklearn.cluster.HDBSCAN`** (available since scikit-learn 1.3), not the standalone
   `hdbscan` package. Matches the prompt's wording.
4. **Gap statistic implemented in-repo** (`src/cluster.py`, Tibshirani et al. reference
   implementation with B=50 uniform reference sets over the PCA bounding box). Not available in
   scikit-learn.
5. **"Byte-similar PDF" is relaxed to deterministic content.** LaTeX embeds timestamps and file IDs.
   Reproducibility is enforced instead by SHA-256 hashing every figure and every value in
   `results/metrics/*.json` across two clean runs, plus `SOURCE_DATE_EPOCH` pinned in the Makefile.
6. **Phase 10 scope: three analyses**, per your selection: similarity search, cross-season
   replication, empirical Bayes shrinkage. Role drift and age×archetype are skipped and will be
   listed as skipped-with-reason in `CLAUDE.md` and as future work in the paper.
7. **MacTeX-no-gui requires your sudo password.** I cannot type it. Phase 0 will stop and hand you
   the exact one-line command to run with the `!` prefix.
8. **LightGBM needs `libomp`** on Apple Silicon (`brew install libomp`), added to setup.

---

## Phases

Each phase ends with a commit. `CLAUDE.md` status is updated as each completes.

### Phase 0: Feasibility spike and scaffolding (blocking, do first)

The entire project depends on one unproven assumption: that FBref is reachable from this machine.
Prove it before building anything.

1. `mkdir ~/pl-player-profiling`, `git init`, write `CLAUDE.md` and `PLAN.md` first (per ground rules).
2. Install `uv` via brew; `uv python install 3.12`; create `pyproject.toml` pinning
   `soccerdata==1.8.8`, pandas, numpy, scipy, scikit-learn>=1.3, umap-learn, lightgbm, shap,
   matplotlib, pyarrow, pytest, ruff. Generate `uv.lock`.
3. `brew install libomp`.
4. **Spike:** one call, `FBref(leagues="ENG-Premier League", seasons="2024-25").read_player_season_stats("standard")`.
   Confirm a real DataFrame with ~500+ rows and a two-level column index.
   - On success: proceed.
   - On CAPTCHA/IP block: switch to the approved **Chrome fallback** (load the ~32 FBref URLs in
     your real browser session at slow cadence via Claude-in-Chrome, save raw HTML into
     `data/raw/html/`, parse offline with soccerdata's own parser). No raw-request loop either way.
5. Hand you the MacTeX command; continue analysis while TeX installs.
6. Write `.claude/settings.json`: PostToolUse hook running `ruff check` + `ruff format --check` on
   edited `.py`; pre-commit hook grepping `paper/**/*.tex` for `—`, `–`, and the banned filler words
   (`delve`, `leverage`, `it is worth noting`, `in the realm of`) and failing the commit on a hit.

**Gate: do not proceed to Phase 1 until the spike returns real data.**

### Phase 1: Ingestion (`src/ingest.py`)
All 11 player tables and 11 team tables, seasons 2024-25 and 2025-26. Raw parquet straight to
`data/raw/{season}/{stat_type}.parquet` before any transformation. `manifest.json` per season with
row/column counts, FBref column names, timestamp. Deterministic `group__stat` flattening and
`data/raw/column_dictionary.csv` mapped to the FBref glossary (cited).

### Phase 2: Preprocessing (`src/preprocess.py`)
Merge on (player, squad, season); keep per-squad rows *and* build aggregated per-player rows
(aggregated is the default for analysis). 450-minute filter for clustering, retained in descriptives.
Per-90 conversion for counts, rates left as rates. Position parsing into `primary_position` and
`position_group`. Missingness report, drop >20% missing, median-impute within position group with a
flag column. Both global and position-group z-scoring saved. `tests/test_preprocess.py` per spec.

### Phase 3: Descriptives (`src/descriptive.py`)
Per-position summary table (12 headline metrics), Spearman correlation with hierarchically clustered
heatmap, position-stratified correlation matrices plus the top-10 most divergent correlations
(a genuine finding), KDE ridge plots, multivariate normality and skewness reporting.

### Phase 4: Dimensionality reduction (`src/reduce.py`)
PCA with scree, cumulative variance, PC1-PC5 loadings, biplot. UMAP sweep over
`n_neighbors ∈ {10,15,30,50}` × `min_dist ∈ {0.0,0.1,0.5}` selected by trustworthiness, full grid to
appendix. t-SNE at perplexity 30 for visual contrast only, explicitly not used for clustering.
Comparison figure plus numeric table (trustworthiness, continuity, kNN position-purity at k=10).
Position-stratified PCA fits, which the role-profiling section uses.

### Phase 5: Clustering and validation (`src/cluster.py`)
KMeans (primary), GMM (BIC), HDBSCAN (noise fraction), fit on both standardized and 90%-variance PCA
spaces, reporting which space each result uses. k selection over 2..12 on silhouette,
Davies-Bouldin, Calinski-Harabasz, and gap statistic.

**Consensus rule, declared in `config.py` in advance:** choose the smallest k whose silhouette is
within 0.02 of the maximum observed silhouette and which ranks top-3 on at least two of
{silhouette, Calinski-Harabasz, gap}, with Davies-Bouldin (lower better) as tiebreak.

Stability: 200 bootstrap resamples, mean ARI vs reference, consensus co-assignment matrix figure,
explicit call-out of low-stability clusters. Cluster-vs-position contingency with NMI and ARI.
Hybrid players: cluster/position disagreement plus split GMM posteriors, top-20 table.

### Phase 6: Archetypes (`src/profiles.py`)
Per cluster: centroid z-profile vs position-group mean, 8 most distinguishing features, 10
representative players, 3 boundary players. Radar charts on fixed 10 axes per position group plus a
small-multiples panel. Archetype names assigned **only after** inspecting feature profiles, with a
one-paragraph justification citing distinguishing features and 3-5 named players, written to
`results/archetypes.json` and pulled verbatim into the paper. Isolation-forest outliers, top 15 with
Mahalanobis distance and data-backed explanation.

### Phase 7: Supervised validation (`src/supervised.py`)
Logistic regression and LightGBM predicting `position_group`, 5-fold stratified CV, accuracy, macro
F1, confusion matrix. SHAP beeswarm and per-class importance for interpretability. Misclassified
players cross-referenced against the Phase 5 hybrid table (overlap is a finding). Ridge regression
of npxG/90 on non-shooting features, CV R², residuals as a heavily caveated "finishing above role"
leaderboard.

### Phase 8: Goalkeepers (`src/keepers.py`)
Independent pipeline on `keeper_core`, 450-minute minimum. Descriptives, correlation heatmap, the
PSxG+/- vs save% scatter with regression and labelled points, and the sweeper scatter (defensive
actions outside area vs average distance). PCA, UMAP, KMeans under the same k rule, expecting but not
forcing 2-4 clusters. Five composite z-axes (shot stopping, sweeping, distribution, cross handling,
aerial claims), radar charts, PSxG+/- per-90 ranking bars, box/violin plots, labelled UMAP.

### Phase 9: Team signatures (`src/teams.py`)
Minutes-weighted team profiles in PCA space and as archetype-composition shares. Hierarchical
clustering of teams with dendrogram, compared against final league position and team-level FBref
stats. The 20-panel 4×5 figure: each team's players highlighted in the shared position-stratified
UMAP with the league greyed out, consistent axes, single vector PDF.

### Phase 10: Novel analyses (`src/novel.py`) — three, as selected
1. **Similarity search.** Cosine similarity within position group, 5 nearest neighbours for 10
   marquee players, scouting-style table, discussion of whether neighbours make football sense.
2. **Cross-season replication.** Re-run clustering on 2025-26, ARI between seasons for players in
   both, archetype-membership stability, named role-changers.
3. **Empirical Bayes shrinkage.** Shrink per-90 rates toward position means by minutes played,
   re-cluster, report how many assignments change. Directly addresses per-90 noise.

### Phase 11: Figures and tables (`src/plotting.py`, `src/tables.py`)
Single matplotlib style, `constrained_layout`, colorblind-safe palette, 9pt fonts matching the paper,
PDF + 300dpi PNG, no seaborn theme leakage. `booktabs` `.tex` fragments. `results/macros.tex` with one
`\newcommand` per cited number. `tests/test_results_exist.py` parses `paper/**/*.tex`, extracts every
`\includegraphics` path and every macro used, asserts each exists; runs in `make paper`.

### Phase 12: Literature (subagent) → `paper/references.bib`
≥15 verified sources, each with a DOI or a stable URL actually fetched. Required coverage: xG origins
and validation; role clustering (Bialkowski, Decroos & Davis, Pappalardo PlayeRank, Aalbers & Van
Haaren); UMAP, t-SNE, HDBSCAN, gap statistic, silhouette, SHAP; FBref/StatsBomb provenance.
**No invented citations**; unverifiable sources are dropped, not guessed.

### Phase 13: The paper (`paper/`)
`\documentclass[11pt]{article}` with booktabs, graphicx, siunitx, natbib, hyperref, subcaption, and
`\input{../results/macros.tex}`. Sections split under `paper/sections/`. 10-14 pages plus appendix,
structured exactly as the prompt specifies. Abstract written last, 200 words, every number a macro.
Writing rules enforced by hook: no em/en dashes, no sentence-initial "And", no bullets in the body,
no banned filler. Build clean: zero undefined refs, zero missing figures, no overfull hbox >10pt.

### Phase 14: Verification and handoff
`make clean && make all` from scratch with cache. `pytest` green. Every figure rendered to PNG and
**visually inspected** with image reading, fixing overlaps and unreadable legends. Compiled PDF read
page by page with every number checked against `results/metrics/*.json`. `README.md` with setup,
targets, per-phase runtime, findings summary. `CLAUDE.md` marked complete with skipped analyses
listed. Final report: five most interesting findings, three weakest parts, next steps.

---

## Subagent orchestration

Coordination through files only, never chat. Main agent owns integration and the paper.

- **Literature subagent** — launched immediately at Phase 0, runs in parallel throughout, owns only
  `paper/references.bib`.
- **Outfield subagent** — Phases 3-7, owns `src/{descriptive,reduce,cluster,profiles,supervised}.py`.
- **Goalkeeper subagent** — Phase 8, owns `src/keepers.py`.
- **Figure-styling subagent** — Phase 11, owns `src/plotting.py` and the export helpers.

Contracts fixed before launch: `config.py` and `src/features.py` are written by the main agent first,
and every subagent reads feature lists and the seed from there. All numeric output goes to
`results/metrics/*.json`; no subagent writes to `paper/` except the literature one.

---

## Critical files

`config.py` (seasons, `MIN_MINUTES=450`, feature lists, `RANDOM_STATE=42`, k-selection rule),
`src/ingest.py`, `src/preprocess.py`, `src/features.py`, `src/cluster.py`, `src/tables.py`,
`results/macros.tex`, `paper/main.tex`, `Makefile`, `.claude/settings.json`, `CLAUDE.md`.

---

## Verification

1. `uv run pytest` — preprocessing invariants, feature integrity, and the figure/macro existence test.
2. `make clean && make all` on a clean tree with data cache present, twice; SHA-256 compare all
   figures and `results/metrics/*.json` between runs to prove determinism.
3. `ruff check && ruff format --check` clean across `src/`.
4. Writing-rule grep over `paper/**/*.tex` returns nothing.
5. `latexmk` completes with zero undefined references and no overfull hbox above 10pt.
6. Manual visual pass over every rendered PNG and a page-by-page read of the compiled PDF, checking
   each cited number against `results/metrics/`.

---

## Open risks

- **FBref access is the single point of failure.** Mitigated by the Phase 0 gate and the approved
  Chrome fallback. If both fail the project stops at Phase 0 and I report, per your ground rules.
- **soccerdata 1.8.8 is an older release**; if it fails against current FBref HTML, the fallback
  parser path may need per-table fixes. This is why raw HTML is cached before any transformation.
- **2025-26 replication assumes FBref has complete 2025-26 data.** The season ended May 2026, so it
  should be complete; verified during Phase 1 and reported if not.
- **Cluster structure may be weak.** If silhouette is low and bootstrap ARI is poor, the paper reports
  that honestly as a negative result rather than forcing archetypes.
