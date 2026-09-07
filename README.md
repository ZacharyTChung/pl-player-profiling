# Two Modes, Not Many Types

Reproducible analysis and paper: **"Two Modes, Not Many Types: The Structure of Football
Player Roles in Season-Aggregate Data"**, by Zachary Chung and Christian Chung.

Published studies of football player roles routinely report a set of discovered archetypes.
A partitioning algorithm returns a partition whether or not the data contain groups, so the
usual evidence cannot distinguish a real taxonomy from a cut through a continuum. This
repository tests the existence question directly, on 13,822 player-seasons covering five
seasons of the big five European leagues described by 33 statistics.

**The result: two modes are real and finer taxonomies are not.** Calibrated against 25
simulated datasets matched on size, dimension and covariance but built to contain no groups
at all, the observed silhouette at k=2 exceeds every simulation in every scope, with standard
scores from 13.2 to 24.2, and Hartigan's dip test rejects unimodality on the leading component
at p < 0.0001. Beyond two modes the observed curve converges on the clusterless null, HDBSCAN
labels every point noise within each position group, and mixture BIC never settles.

## Why the null calibration is the point

Run k-means on a single multivariate Gaussian, which by construction contains no groups, and
it hands back tidy clusters with a respectable silhouette, a high bootstrap ARI and two
nameable centroids. Those three things are exactly what a published role taxonomy is usually
supported by, so none of them is evidence that groups exist. `src/structure.py` supplies the
missing comparison, and it is cheap: a few dozen refits on simulated data drawn from the
observed mean and covariance.

## The two samples

**Primary: a pre-withdrawal archive.** 13,822 player-seasons, five seasons ending 2018 to
2022, all five major European leagues, 33 features, 9,263 eligible outfield players and 1,035
goalkeepers. On 20 January 2026 Sports Reference removed every Opta-derived statistic from
FBref, retroactively, after the licence was terminated. The worldfootballR project mirrors
those tables and its snapshot predates the removal. `src/archive_fetch.py` downloads it from a
stated URL and records a SHA-256 per file, so the primary sample is reproducible rather than
assumed. The mirror is a third-party scrape and the paper says so; row counts, per-column and
per-season coverage are audited on load, and the season ending 2023 is excluded because its
carrying block is empty upstream.

**Secondary: the live reduced sample.** A direct pull for 2024-25 and 2025-26, after the
withdrawal, yielding 18 features for 366 Premier League players once Understat is merged in as
an independent expected goals source. This is the paper's robustness case. The main result
holding on both a 33-feature and an 18-feature description is worth more than it holding on
either alone.

The withdrawal is a trap and not only a loss: **the affected tables are still served, with
complete headers and one row per player. Only the values are gone.** A feature list built from
the published schema resolves correctly and produces an all-NaN matrix. Every feature in
`src/features.py` was verified against observed values, and `tests/test_features.py` fails if
that stops being true.

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
make help           # list targets
make all            # everything through paper/main.pdf, against the committed caches
make archive-fetch  # refresh the pre-withdrawal mirror from upstream
make data           # re-scrape the live sample
make submission     # flatten the paper into an arXiv ready tarball
make test           # pytest
make lint           # ruff check and format check
make clean          # remove derived artefacts, keep the caches
```

`make submission` exists because the repository layout is convenient to build from and
wrong to upload: `paper/figures` is a symlink, which does not survive a tarball, and
`main.tex` reaches into `../results` for the generated macros and tables, which arXiv will
not resolve. The target flattens both, ships only the figures the document includes, pins
the date so the submitted PDF is reproducible, compiles the staged copy from a bare
directory to prove it stands alone, and writes `SUBMISSION.md` with the title, authors and
plain-text abstract rendered from the same macros the PDF uses.

`make data` and `make archive-fetch` are deliberately **not** part of `make all`. Both write
files that are committed, so the pipeline runs against them. Everything downstream of those
two, including the whole archive chain, is regenerated by `make all`.

### Approximate runtimes

| Stage | Time | Notes |
|---|---|---|
| `archive-fetch` | 1 min | 13 files from the mirror, cached after the first run |
| `data` | 15 to 20 min | Live scrape, 7 s per request, browser driven, outside `make all` |
| `archive` | 40 s | Load, audit, possession adjust, standardise 13,822 rows |
| `structure` | 6 min | The null calibration, 25 simulations across four scopes |
| `archive_profiles` | 4 min | Archetypes, stability, outliers on the primary sample |
| `archive_keepers` | 1 min | Goalkeepers on the full 13-feature block |
| `archive_validation` | 12 min | Replication across 5 seasons and 5 leagues, LightGBM, SHAP, PCA |
| `preprocess` | 5 s | Reduced sample |
| `descriptive` | 25 s | |
| `reduce` | 3 min | Twelve UMAP fits plus t-SNE |
| `cluster` | 2 min | 200 bootstraps and the gap statistic over four scopes |
| `profiles` / `supervised` / `keepers` / `teams` | 40 s each | |
| `novel` | 3 min | Shrinkage refits the clustering |
| `sensitivity` | 2 min | Refits selection and stability at three thresholds |
| `threed` | 30 s | Four three dimensional scenes, two viewpoints each |
| `symmetric` | 2 min | Symmetric possession adjustment counterfactual |
| `drift` | 5 min | Scrapes 38 match reports, outside `make all` |
| `leagues` | 20 min | Live pipeline across the big five, scrapes, outside `make all` |
| `tables` | 10 s | |
| `paper` | 40 s | Three LaTeX passes plus bibtex |

## Layout

```
config.py                 seeds, thresholds, the k selection rule, paths
src/archive_fetch.py      download the pre-withdrawal mirror, hash it, write the manifest
src/archive.py            load and audit the archive, standardise, write the primary sample
src/structure.py          the null calibration: does the clustering find groups that exist?
src/archive_profiles.py   archetypes, stability and outliers on the primary sample
src/archive_keepers.py    goalkeepers on the full keeper block
src/archive_validation.py replication across seasons and leagues, supervised, PCA
src/ingest.py             live FBref and Understat retrieval
src/preprocess.py         merge, filter, per-90, possession adjust, standardise
src/features.py           the reduced feature set, verified against observed values
src/*.py                  one module per reduced-sample analysis
src/tables.py             generates results/macros.tex and results/tables/*.tex
data/raw/archive/         the mirror as parquet, plus rds/ and a hashed manifest
results/metrics/          every computed number as JSON
paper/                    main.tex, sections/, references.bib
```

## Reproducibility

Everything is seeded from `RANDOM_STATE = 42` in `config.py`. Dependencies are pinned by
`uv.lock`. The paper contains **no hard-coded numbers**: every quantity is a macro in
`results/macros.tex` generated from the metrics files, and `tests/test_results_exist.py` fails
the build if the paper cites a macro that does not exist, includes a figure that was not
generated, generates a figure nothing references, inputs a missing table fragment, writes a
decimal literal into results prose, or quotes counts and percentages that do not agree with
each other.

Two constraints are enforced mechanically rather than by review. A `PostToolUse` hook runs
`ruff` on every edited Python file, and a git `pre-commit` hook rejects em dashes, en dashes,
sentence-initial "And", bullet lists and a list of filler phrases anywhere in `paper/**/*.tex`.

## Findings

1. **The structure is real.** Against 25 clusterless simulations matched on size, dimension
   and covariance, the observed silhouette at k=2 exceeds every simulation in every scope,
   with standard scores from 13.2 to 24.2. The dip test rejects unimodality on the leading
   component at p < 0.0001 for the pool, defenders and midfielders. Football role data is not
   a featureless cloud.
2. **The structure is two modes and nothing finer.** Beyond k=2 the observed silhouette
   converges on the clusterless null, so the third and subsequent clusters a taxonomy would
   name are indistinguishable from what a partitioning algorithm extracts from structureless
   data. HDBSCAN assigns 86.5 percent of the pool and 100 percent of every position group to
   noise. Mixture BIC declines monotonically to the top of the candidate range.
3. **The modes are not the positions relabelled.** Agreement with listed position group is
   ARI 0.293, far above chance and far below recovery. 92.9 percent of defenders fall on one
   side and 99.8 percent of forwards on the other, while midfielders divide 1,920 to 1,846.
   The partition cuts across the listed categories at their busiest point.
4. **A supervised model finds the same geometry.** LightGBM reaches 93.4 percent accuracy
   against a 40.9 percent baseline, and yet routes all but 1.31 percent of its errors through
   midfield: 0 defenders are called forwards and 8 forwards are called defenders. That is the
   confusion structure of an ordered line, not of three categories.
5. **Two modes is not one dimension.** PC1 carries 39.9 percent of the variance, more than any
   other direction, but 14 of 33 components are needed to reach 90 percent. One dominant
   bimodal axis with a long continuous tail.
6. **It reproduces everywhere.** Refitted independently, all five seasons and all five leagues
   recover the pooled partition, worst case ARI 0.916, with the separating axis correlating
   with the pooled axis above 0.998 everywhere. Centroids fitted in one league assign another
   league's players at mean ARI 0.944. Season-against-season agreement decays with the gap
   between seasons while the transfer index stays flat, so that decay measures players
   changing rather than the solution failing: 91.6 percent of players present in two seasons
   land in the same mode in both.
7. **The conventional possession adjustment overcorrects.** Defensive counts are usually
   adjusted as though opportunity scales inversely with possession. Fitted over 490
   team-seasons the elasticities are 0.17 for tackles, 0.35 for interceptions, 0.44 for
   blocks, 0.44 for fouls and 0.96 for clearances, all below one. Applying an exponent of 1
   does not remove the confound, it flips its sign and enlarges it.
8. **The correction is genuinely one sided.** Attacking output responds to a team's own
   possession super-proportionally: shots 1.09, npxG 1.48, goals 1.63, assists 1.72, xGChain
   2.12, xGBuildup 2.44. Increasing returns to possession are a fact about football, so
   dividing them out removes signal. The symmetric counterfactual is computed anyway: it moves
   the club-to-league-position correlation from -0.861 to -0.610 at a cost of moving the
   partition to ARI 0.872.
9. **Goalkeepers divide by role assignment, not by ability.** Clustering on all available
   keeper features produces a partition explaining 0.386 of the variance in situational
   quantities against 0.109 in technique, which is a league table wearing a tactical
   vocabulary. Clustering on technique alone reverses that to 0.135 against 0.019, and what
   it finds is line keepers against sweeper claimers, not the familiar shot stopper, sweeper
   and distributor trio.
10. **The result survives halving the feature set.** The reduced sample, which lost expected
    goals, all passing and possession volume, chance creation and the entire advanced
    goalkeeping table, returns two clusters in every scope with agreement against listed
    positions still weak at 0.226. Richer data sharpens the description of the two modes and
    does not reveal more of them.
11. **Correlations reverse sign across positions.** 88 of 153 feature pairs change sign
    between defenders, midfielders and forwards, and all ten of the most divergent pairs do.
    Pooled correlation analysis of football data should be considered unsafe by default.
12. **Outliers are the players the listing system also declines to place.** 11 of the 15
    strongest outliers carry a compound listed position, against a much smaller share in the
    pool.

## Licence and data

The archive is retrieved from the worldfootballR public data mirror, with per-file SHA-256
digests recorded in `data/raw/archive/manifest.json`. Live data is retrieved from FBref and
Understat through `soccerdata`, which caches to disk and rate limits requests. No raw request
loop is written against either site. The cached HTML under `data/raw/html` is gitignored and
regenerable.
