"""Central configuration. Every seed, threshold and path used anywhere lives here."""

from pathlib import Path

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
HTML_CACHE = DATA_RAW / "html"
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics"
TABLES = RESULTS / "tables"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"

for _p in (DATA_RAW, DATA_PROCESSED, HTML_CACHE, RESULTS, METRICS, TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Data scope
# --------------------------------------------------------------------------------------
LEAGUE = "ENG-Premier League"
SEASON_PRIMARY = "2024-25"
SEASON_REPLICATION = "2025-26"
SEASONS = [SEASON_PRIMARY, SEASON_REPLICATION]

# soccerdata normalises season labels to a 4-digit key ("2024-25" -> "2425").
SEASON_KEYS = {"2024-25": "2425", "2025-26": "2526"}

PLAYER_STAT_TYPES = [
    "standard",
    "shooting",
    "passing",
    "passing_types",
    "goal_shot_creation",
    "defense",
    "possession",
    "playing_time",
    "misc",
    "keeper",
    "keeper_adv",
]
TEAM_STAT_TYPES = list(PLAYER_STAT_TYPES)

# --------------------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------------------
MIN_MINUTES = 450  # five full matches
MIN_MINUTES_SENSITIVITY = [270, 450, 900]
MAX_MISSING_FRACTION = 0.20  # drop feature columns missing in >20% of eligible players

POSITION_GROUPS = ["GK", "DF", "MF", "FW"]
OUTFIELD_GROUPS = ["DF", "MF", "FW"]

# --------------------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------------------
K_MIN, K_MAX = 2, 12
K_RANGE = list(range(K_MIN, K_MAX + 1))
PCA_VARIANCE_TARGET = 0.90
BOOTSTRAP_N = 200
GAP_STATISTIC_B = 50

# Consensus k-selection rule, declared in advance (see PLAN.md / paper methods).
# Choose the SMALLEST k whose silhouette is within SILHOUETTE_TOLERANCE of the best
# observed silhouette AND which ranks in the top TOP_N_RANK on at least MIN_CRITERIA of
# {silhouette, Calinski-Harabasz, gap}. Davies-Bouldin (lower is better) breaks ties.
K_RULE = {
    "silhouette_tolerance": 0.02,
    "top_n_rank": 3,
    "min_criteria": 2,
    "tiebreak": "davies_bouldin",
}

# --------------------------------------------------------------------------------------
# Dimensionality reduction
# --------------------------------------------------------------------------------------
UMAP_GRID = {"n_neighbors": [10, 15, 30, 50], "min_dist": [0.0, 0.1, 0.5]}
TSNE_PERPLEXITY = 30
KNN_PURITY_K = 10

# --------------------------------------------------------------------------------------
# Novel analyses selected for this study (see CLAUDE.md for skipped ones)
# --------------------------------------------------------------------------------------
NOVEL_ANALYSES = ["similarity_search", "cross_season_replication", "empirical_bayes_shrinkage"]
