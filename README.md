# Role Discovery Under Data Scarcity

Reproducible analysis and paper: **"Role Discovery Under Data Scarcity: Unsupervised Player
Profiling in the Premier League After the Withdrawal of Public Advanced Football
Statistics"**, by Zachary Chung and Christian Chung.

On 20 January 2026 Sports Reference removed every Opta-derived statistic from FBref, after
the licence was terminated and deletion was demanded. The removal was retroactive. This
repository measures what player role structure survives that withdrawal, using the
statistics that remain plus an independent expected goals source.

## Why this is not a normal football analytics repo

The affected FBref tables are **still served, with complete headers and one row per
player. Only the values are gone.** A feature list built from the published schema
resolves correctly and produces an all-NaN matrix. Every feature in `src/features.py` was
therefore verified to contain non-null values in both seasons before inclusion, and
`tests/test_features.py` fails if that ever stops being true.

| FBref player table | Stat columns | Empty | Usable |
|---|---|---|---|
| `passing` | 21 | 19 | 2 |
| `possession` | 21 | 20 | 1 |
| `goal_shot_creation` | 17 | 16 | 1 |
| `defense` | 17 | 14 | 3 |
| `keeper_adv` | 26 | 23 | 3 |
| `passing_types` | 16 | 14 | 2 |
| `standard`, `shooting`, `keeper`, `misc`, `playing_time` | | 0 | all |

## Setup

```bash
brew install uv libomp          # libomp is required by LightGBM on Apple Silicon
uv sync --extra dev             # creates .venv on Python 3.12
```

A TeX distribution is needed only for the paper. Either works:

```bash
curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh    # per user, no sudo
tlmgr install booktabs siunitx natbib caption subcaption hyperref geometry cm-super multirow
# or
brew install --cask mactex-no-gui                              # system wide, needs sudo
```

The Makefile adds TinyTeX to `PATH` automatically and defers to a system TeX when present.

## Running it

```bash
make help        # list targets
make all         # everything through paper/main.pdf, using the committed scrape cache
make test        # pytest
make lint        # ruff check and format check
make clean       # remove derived artefacts, keep the scrape cache
```

`make data` is deliberately **not** part of `make all`. The scrape is slow and rate
limited, and the raw parquet files are committed, so the pipeline runs against the cache.

### Approximate runtimes

| Stage | Time | Notes |
|---|---|---|
| `data` | 15 to 20 min | Two seasons, 7 s per request, browser driven |
| `preprocess` | 5 s | |
| `descriptive` | 25 s | |
| `reduce` | 3 min | Twelve UMAP fits plus t-SNE |
| `cluster` | 2 min | 200 bootstraps and the gap statistic over four scopes |
| `profiles` | 40 s | |
| `supervised` | 1 min | LightGBM plus SHAP |
| `keepers` | 30 s | |
| `teams` | 30 s | |
| `novel` | 3 min | Shrinkage refits the clustering |
| `tables` | 10 s | |
| `paper` | 40 s | Three LaTeX passes plus bibtex |

## Layout

```
config.py          seeds, thresholds, the k selection rule, paths
src/ingest.py      FBref and Understat retrieval
src/preprocess.py  merge, filter, per-90, possession adjust, standardise
src/features.py    the feature sets, verified against observed values
src/*.py           one module per analysis phase
src/tables.py      generates results/macros.tex and results/tables/*.tex
data/raw/          untouched pulls, one parquet per table per season
results/metrics/   every computed number as JSON
paper/             main.tex, sections/, references.bib
```

## Reproducibility

Everything is seeded from `RANDOM_STATE = 42` in `config.py`. Dependencies are pinned by
`uv.lock`. The paper contains **no hard-coded numbers**: every quantity is a macro in
`results/macros.tex`, generated from the metrics files, and `tests/test_results_exist.py`
fails the build if the paper cites a macro that does not exist, includes a figure that was
not generated, inputs a table fragment that is missing, or writes a decimal literal into
results prose.

Two constraints are enforced mechanically rather than by review. A `PostToolUse` hook runs
`ruff` on every edited Python file, and a git `pre-commit` hook rejects em dashes, en
dashes, sentence-initial "And", bullet lists and a list of filler phrases anywhere in
`paper/**/*.tex`.

## Findings

1. **The withdrawal is a trap, not just a loss.** Tables resolve correctly and return
   nothing. Feature selection must be made against observed values.
2. **The standard possession adjustment overcorrects by 2 to 3 times.** Defensive counts
   must be corrected for how long a team spends without the ball, and the conventional
   formula assumes direct proportionality. Fitted against 40 team-seasons the elasticities
   are 0.53 (interceptions), 0.43 (tackles won) and 0.28 (fouls committed). Applying an
   exponent of 1 does not remove the confound, it flips its sign and enlarges it. Using
   the measured exponent takes the primary-season correlation with team possession from
   0.122 to 0.022, 0.111 to 0.018, and 0.102 to 0.037.
3. **Correlations reverse sign across positions.** 87 of 153 feature pairs change sign
   between defenders, midfielders and forwards, and all ten of the most divergent pairs do.
   Pooled correlation analysis of football data should be considered unsafe by default.
4. **The data supports one binary division.** The selection rule returns two clusters in
   every scope. HDBSCAN, free to find nothing, labels 100 percent of every position group
   as noise. The division found is attacking against defensive involvement, agreeing with
   listed positions at an adjusted Rand index of only 0.226, while a supervised model
   reaches 0.803 accuracy on the same features.
5. **That division is robust; the within-group ones are weaker, and unevenly so.** The
   global partition holds at 0.945 under shrinkage. The defender partition fails all three
   tests, falling to 0.453. The forward partition fails bootstrap stability and
   cross-season replication but partly survives shrinkage at 0.626. Midfielders pass all
   three.
6. **Goalkeeper clusters recover team strength, not goalkeeping.** Cluster membership
   explains 0.72 of the variance in team points per match against 0.25 in save percentage,
   so the partition is reported as a negative result and no archetypes are named.
7. **Club summaries are stable where players are not.** The minutes-weighted club position
   on the first principal component correlates with final league position at -0.771 and
   -0.784 across two independent seasons.

## Licence and data

Data is retrieved from FBref and Understat through `soccerdata`, which caches to disk and
rate limits requests. No raw request loop is written against either site. The cached HTML
under `data/raw/html` is gitignored and regenerable.
