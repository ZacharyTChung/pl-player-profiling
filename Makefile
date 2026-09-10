# Reproducible build for the Premier League player profiling paper.
#
# Every target is driven by a stamp file under .stamps, because each pipeline stage
# writes many outputs (figures, metrics, processed tables) rather than a single file.
# A stage reruns only when its own source or an upstream stamp is newer.
#
# `make all` from a clean tree, with the scrape cache present, regenerates every
# figure, table and the PDF.

SHELL     := /bin/bash
PY        := uv run python
STAMPS    := .stamps

# TinyTeX installs per user with no administrator rights. If it is present, put it on
# PATH so the paper builds without the caller having to arrange that. A system TeX such
# as MacTeX still takes precedence when one is already on PATH.
TINYTEX   := $(HOME)/Library/TinyTeX/bin/universal-darwin
export PATH := $(PATH):$(TINYTEX)

# Pin the timestamp LaTeX embeds so repeated builds differ only where content differs.
export SOURCE_DATE_EPOCH := 1735689600

CONFIG    := config.py src/features.py
PLOTTING  := src/plotting.py

RAW_MANIFESTS  := $(wildcard data/raw/*/manifest.json)
PROCESSED      := $(wildcard data/processed/*.parquet)

.PHONY: all data preprocess descriptive reduce cluster profiles supervised keepers \
        teams novel sensitivity threed symmetric age drift leagues paper-figures tables \
        paper review test lint format clean distclean help submission

all: paper

help:
	@echo "Targets:"
	@echo "  data        pull FBref and Understat (slow, cached, rate limited)"
	@echo "  preprocess  merge, filter, per-90, standardise"
	@echo "  descriptive summary statistics, correlations, distributions"
	@echo "  reduce      PCA, UMAP, t-SNE and the embedding comparison"
	@echo "  cluster     k selection, KMeans/GMM/HDBSCAN, stability"
	@echo "  profiles    archetype characterisation, radars, outliers"
	@echo "  supervised  position prediction and SHAP"
	@echo "  keepers     goalkeeper pipeline"
	@echo "  teams       team stylistic signatures and the 20 panel figure"
	@echo "  novel       similarity search, replication, empirical Bayes shrinkage"
	@echo "  sensitivity minutes threshold sensitivity"
	@echo "  threed      three dimensional views of the player and club spaces"
	@echo "  symmetric   symmetric possession adjustment counterfactual"
	@echo "  age         age against archetype"
	@echo "  drift       role drift within a season, one club, scrapes match reports"
	@echo "  leagues     repeat the pipeline across the big five, scrapes four leagues"
	@echo "  paper-figures composite main-text figures"
	@echo "  tables      LaTeX tables and results/macros.tex"
	@echo "  paper       compile paper/main.pdf and paper/supplementary.pdf"
	@echo "  review      double-spaced, de-identified manuscript for submission"
	@echo "  test        pytest"
	@echo "  all         everything through the PDF"
	@echo "  submission  flatten the paper into an arXiv ready tarball"
	@echo "  clean       remove derived artefacts, keep the scrape cache"
	@echo "  distclean   also remove the scrape cache (forces a full re-scrape)"

$(STAMPS):
	@mkdir -p $(STAMPS)

# --- Stage 1: ingestion -----------------------------------------------------------
# Deliberately not part of `all`. The scrape is slow and rate limited, and the raw
# parquet files are committed, so `make all` runs against the cache.
data: | $(STAMPS)
	$(PY) -m src.ingest
	@touch $(STAMPS)/data

# Fetch the pre-withdrawal mirror the primary sample is built on. Deliberately not part
# of `all`: the converted parquet files are committed, so `all` runs against them. Rerun
# this only to refresh from upstream, and expect the manifest digests to change if the
# mirror has been rescraped since.
archive-fetch: | $(STAMPS)
	$(PY) -m src.archive_fetch
	@touch $(STAMPS)/archive-fetch

# --- Stage 2: preprocessing -------------------------------------------------------
$(STAMPS)/preprocess: src/preprocess.py $(CONFIG) $(RAW_MANIFESTS) | $(STAMPS)
	$(PY) -m src.preprocess
	@touch $@
preprocess: $(STAMPS)/preprocess

# --- Stage 3: analysis ------------------------------------------------------------
$(STAMPS)/descriptive: src/descriptive.py $(CONFIG) $(PLOTTING) $(STAMPS)/preprocess
	$(PY) -m src.descriptive
	@touch $@
descriptive: $(STAMPS)/descriptive

$(STAMPS)/reduce: src/reduce.py $(CONFIG) $(PLOTTING) $(STAMPS)/preprocess
	$(PY) -m src.reduce
	@touch $@
reduce: $(STAMPS)/reduce

$(STAMPS)/cluster: src/cluster.py $(CONFIG) $(PLOTTING) $(STAMPS)/preprocess
	$(PY) -m src.cluster
	@touch $@
cluster: $(STAMPS)/cluster

$(STAMPS)/profiles: src/profiles.py $(CONFIG) $(PLOTTING) $(STAMPS)/cluster
	$(PY) -m src.profiles
	@touch $@
profiles: $(STAMPS)/profiles

$(STAMPS)/supervised: src/supervised.py $(CONFIG) $(PLOTTING) $(STAMPS)/cluster
	$(PY) -m src.supervised
	@touch $@
supervised: $(STAMPS)/supervised

$(STAMPS)/keepers: src/keepers.py $(CONFIG) $(PLOTTING) $(STAMPS)/preprocess
	$(PY) -m src.keepers
	@touch $@
keepers: $(STAMPS)/keepers

$(STAMPS)/teams: src/teams.py $(CONFIG) $(PLOTTING) $(STAMPS)/profiles $(STAMPS)/reduce
	$(PY) -m src.teams
	@touch $@
teams: $(STAMPS)/teams

# Not part of `all`: scrapes four further leagues. Their raw tables are cached under
# data/raw/leagues, so the analysis reruns without repeating the scrape.
$(STAMPS)/leagues: src/leagues.py $(CONFIG) $(PLOTTING) $(STAMPS)/preprocess src/cluster.py
	$(PY) -m src.leagues
	@touch $@
leagues: $(STAMPS)/leagues

# Not part of `all`: this target scrapes 38 match reports at seven seconds each. The
# assembled frame is cached under data/processed, so the analysis reruns without it.
$(STAMPS)/drift: src/drift.py $(CONFIG) $(PLOTTING) $(STAMPS)/cluster src/cluster.py
	$(PY) -m src.drift
	@touch $@
drift: $(STAMPS)/drift

$(STAMPS)/age: src/age.py $(CONFIG) $(PLOTTING) $(STAMPS)/profiles
	$(PY) -m src.age
	@touch $@
age: $(STAMPS)/age

$(STAMPS)/symmetric: src/symmetric.py $(CONFIG) $(PLOTTING) $(STAMPS)/preprocess src/cluster.py
	$(PY) -m src.symmetric
	@touch $@
symmetric: $(STAMPS)/symmetric

$(STAMPS)/threed: src/threed.py $(CONFIG) $(PLOTTING) $(STAMPS)/profiles $(STAMPS)/teams
	$(PY) -m src.threed
	@touch $@
threed: $(STAMPS)/threed $(STAMPS)/symmetric $(STAMPS)/age

$(STAMPS)/sensitivity: src/sensitivity.py $(CONFIG) $(PLOTTING) $(STAMPS)/preprocess src/cluster.py
	$(PY) -m src.sensitivity
	@touch $@
sensitivity: $(STAMPS)/sensitivity

$(STAMPS)/novel: src/novel.py $(CONFIG) $(PLOTTING) $(STAMPS)/cluster $(STAMPS)/profiles
	$(PY) -m src.novel
	@touch $@
novel: $(STAMPS)/novel

# --- Stage 3b: the primary sample -------------------------------------------------
# Everything above this line runs on the reduced live sample, which the paper reports as
# the robustness case. The chain below is the primary analysis.
ARCHIVE_RAW := $(wildcard data/raw/archive/*.parquet)

$(STAMPS)/archive: src/archive.py $(CONFIG) $(ARCHIVE_RAW) src/preprocess.py | $(STAMPS)
	$(PY) -m src.archive
	@touch $@
archive: $(STAMPS)/archive

$(STAMPS)/structure: src/structure.py $(CONFIG) $(PLOTTING) $(STAMPS)/archive
	$(PY) -m src.structure
	@touch $@
structure: $(STAMPS)/structure

$(STAMPS)/archive_profiles: src/archive_profiles.py $(CONFIG) $(PLOTTING) $(STAMPS)/archive
	$(PY) -m src.archive_profiles
	@touch $@
archive-profiles: $(STAMPS)/archive_profiles

$(STAMPS)/archive_keepers: src/archive_keepers.py $(CONFIG) $(PLOTTING) $(STAMPS)/archive
	$(PY) -m src.archive_keepers
	@touch $@
archive-keepers: $(STAMPS)/archive_keepers

$(STAMPS)/archive_validation: src/archive_validation.py $(CONFIG) $(PLOTTING) \
                              $(STAMPS)/archive_profiles
	$(PY) -m src.archive_validation
	@touch $@
archive-validation: $(STAMPS)/archive_validation

$(STAMPS)/elasticity: src/elasticity.py $(CONFIG) $(PLOTTING) $(STAMPS)/archive
	$(PY) -m src.elasticity
	@touch $@
elasticity: $(STAMPS)/elasticity

$(STAMPS)/ablation: src/ablation.py $(CONFIG) $(PLOTTING) $(STAMPS)/archive \
                    src/structure.py src/cluster.py
	$(PY) -m src.ablation
	@touch $@
ablation: $(STAMPS)/ablation

# The only test against real results: team-season composition on the two-mode axis
# against points and goal difference, out of sample, with possession as the baseline.
$(STAMPS)/outcomes: src/outcomes.py $(CONFIG) $(PLOTTING) $(STAMPS)/archive_profiles \
                    $(STAMPS)/archive_validation $(STAMPS)/archive_keepers
	$(PY) -m src.outcomes
	@touch $@
outcomes: $(STAMPS)/outcomes

# Composite figures for the main text. Reads the metrics the archive chain wrote and
# refits only the pooled two-cluster solution, so it is cheap and must run last.
$(STAMPS)/paper_figures: src/paper_figures.py $(CONFIG) $(PLOTTING) \
                         $(STAMPS)/archive_profiles $(STAMPS)/archive_validation
	$(PY) -m src.paper_figures
	@touch $@
paper-figures: $(STAMPS)/paper_figures

# --- Stage 4: tables and macros ---------------------------------------------------
ANALYSIS_STAMPS := $(STAMPS)/descriptive $(STAMPS)/reduce $(STAMPS)/cluster \
                   $(STAMPS)/profiles $(STAMPS)/supervised $(STAMPS)/keepers \
                   $(STAMPS)/teams $(STAMPS)/novel $(STAMPS)/sensitivity \
                   $(STAMPS)/threed $(STAMPS)/symmetric $(STAMPS)/age \
                   $(STAMPS)/archive $(STAMPS)/structure $(STAMPS)/archive_profiles \
                   $(STAMPS)/archive_keepers $(STAMPS)/archive_validation \
                   $(STAMPS)/elasticity $(STAMPS)/ablation $(STAMPS)/outcomes \
                   $(STAMPS)/paper_figures

results/macros.tex: src/tables.py $(CONFIG) $(ANALYSIS_STAMPS)
	$(PY) -m src.tables
tables: results/macros.tex

# --- Stage 6: the submission package ----------------------------------------------
# arXiv unpacks a flat tarball: it cannot follow paper/figures, which is a symlink, and
# it cannot resolve main.tex's reach into ../results. This target flattens both, ships
# only the figures the document includes, and compiles the staged copy from scratch so
# a failure surfaces here rather than after upload.
submission: paper/main.pdf
	$(PY) scripts/make_submission.py

# --- Stage 5: the paper -----------------------------------------------------------
TEX_SOURCES := paper/main.tex paper/supplementary.tex paper/preamble.tex \
               $(wildcard paper/sections/*.tex) paper/references.bib

# The article and the supplement are separate documents that reference each other
# through xr, so each has to see the other's .aux file. Compiling the pair twice in
# sequence is what resolves that: the first pass writes both .aux files, the second
# reads them.
paper/main.pdf: $(TEX_SOURCES) results/macros.tex
	@command -v latexmk >/dev/null 2>&1 || { \
	  echo "latexmk not found. Install a TeX distribution, either"; \
	  echo "  brew install --cask mactex-no-gui     (system wide, needs sudo)"; \
	  echo "  curl -sL https://yihui.org/tinytex/install-bin-unix.sh | sh   (per user)"; \
	  exit 1; }
	@.claude/hooks/check_prose.sh
	$(PY) -m pytest tests/test_results_exist.py -q
	cd paper && for pass in 1 2; do \
	  latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex || exit 1; \
	  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex || exit 1; \
	done
	@echo "--- build warnings ---"
	@grep -hE "(Undefined|Overfull|LaTeX Warning)" paper/main.log \
	  paper/supplementary.log || echo "none"
	@echo "--- lengths ---"
	@grep -o "Output written on main.pdf ([0-9]* pages" paper/main.log | tail -1
	@grep -o "Output written on supplementary.pdf ([0-9]* pages" \
	  paper/supplementary.log | tail -1
paper: paper/main.pdf

# The manuscript most journals in this area want at submission: double spaced, line
# numbered and de-identified. Built into paper/review/ so it never overwrites the
# typeset article, and from the same sources so the two cannot drift apart.
review: paper/main.pdf
	@mkdir -p paper/review
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error \
	  -outdir=review -usepretex='\def\reviewmode{}' main.tex
	@echo "review manuscript at paper/review/main.pdf"
	@grep -o "Output written on .*main.pdf ([0-9]* pages" paper/review/main.log \
	  | tail -1


# --- Quality ----------------------------------------------------------------------
test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check . --fix

# --- Cleaning ---------------------------------------------------------------------
clean:
	rm -rf $(STAMPS)
	rm -f figures/*.pdf figures/*.png
	rm -f results/metrics/*.json results/tables/*.tex results/macros.tex
	rm -f results/membership_*.csv
	rm -rf submission submission.tar.gz
	rm -f data/processed/*.parquet
	cd paper && rm -f main.pdf supplementary.pdf && rm -rf review
	cd paper && rm -f *.aux *.log *.out *.bbl *.blg *.fls *.fdb_latexmk \
	  *.synctex.gz *.toc
	@echo "clean done. The scrape cache under data/raw is kept."

distclean: clean
	rm -rf data/raw/html
	@echo "scrape cache removed. The next `make data` will re-scrape FBref."
