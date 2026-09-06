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

# Pin the timestamp LaTeX embeds so repeated builds differ only where content differs.
export SOURCE_DATE_EPOCH := 1735689600

CONFIG    := config.py src/features.py
PLOTTING  := src/plotting.py

RAW_MANIFESTS  := $(wildcard data/raw/*/manifest.json)
PROCESSED      := $(wildcard data/processed/*.parquet)

.PHONY: all data preprocess descriptive reduce cluster profiles supervised keepers \
        teams novel tables paper test lint format clean distclean help

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
	@echo "  tables      LaTeX tables and results/macros.tex"
	@echo "  paper       compile paper/main.pdf"
	@echo "  test        pytest"
	@echo "  all         everything through the PDF"
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

$(STAMPS)/novel: src/novel.py $(CONFIG) $(PLOTTING) $(STAMPS)/cluster $(STAMPS)/profiles
	$(PY) -m src.novel
	@touch $@
novel: $(STAMPS)/novel

# --- Stage 4: tables and macros ---------------------------------------------------
ANALYSIS_STAMPS := $(STAMPS)/descriptive $(STAMPS)/reduce $(STAMPS)/cluster \
                   $(STAMPS)/profiles $(STAMPS)/supervised $(STAMPS)/keepers \
                   $(STAMPS)/teams $(STAMPS)/novel

results/macros.tex: src/tables.py $(CONFIG) $(ANALYSIS_STAMPS)
	$(PY) -m src.tables
tables: results/macros.tex

# --- Stage 5: the paper -----------------------------------------------------------
TEX_SOURCES := paper/main.tex $(wildcard paper/sections/*.tex) paper/references.bib

paper/main.pdf: $(TEX_SOURCES) results/macros.tex
	@command -v latexmk >/dev/null 2>&1 || { \
	  echo "latexmk not found. Install MacTeX:  brew install --cask mactex-no-gui"; exit 1; }
	@.claude/hooks/check_prose.sh
	$(PY) -m pytest tests/test_results_exist.py -q
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	@echo "--- build warnings ---"
	@grep -E "(Undefined|Overfull|Underfull|LaTeX Warning)" paper/main.log || echo "none"
paper: paper/main.pdf

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
	rm -f data/processed/*.parquet
	cd paper && rm -f main.pdf main.aux main.log main.out main.bbl main.blg \
	  main.fls main.fdb_latexmk main.synctex.gz main.toc
	@echo "clean done. The scrape cache under data/raw is kept."

distclean: clean
	rm -rf data/raw/html
	@echo "scrape cache removed. The next `make data` will re-scrape FBref."
