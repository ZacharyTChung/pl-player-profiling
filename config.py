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
# Possession adjustment
# --------------------------------------------------------------------------------------
# Defensive actions can only happen while the opponent has the ball, so a player in a
# team that concedes possession accumulates them through exposure rather than through
# role or ability. Counts are rescaled to a common opponent-possession baseline:
#
#     padj = raw_per90 * PADJ_REFERENCE / (100 - team_possession_pct)
#
# A team with 65 percent of the ball faces 35 percent opponent possession and has its
# defensive counts scaled up by 50/35, while a team with 35 percent is scaled down by
# 50/65. The reference of 50 is the league average by construction, so the adjustment
# leaves the league-wide mean roughly unchanged and only redistributes between teams.
PADJ_REFERENCE_POSSESSION = 50.0

#: Counting statistics that occur while the opponent has the ball. Fouls committed are
#: included because they are overwhelmingly defensive events, and the choice is reported
#: as a sensitivity in the possession analysis rather than assumed.
PADJ_COUNTS = ["interceptions", "tackles_won", "fouls_committed"]

#: Team-level source column for each adjusted count, used to estimate the elasticity.
PADJ_TEAM_SOURCES = {
    "interceptions": "Performance__Int",
    "tackles_won": "Performance__TklW",
    "fouls_committed": "Performance__Fls",
}

#: The exponent applied to the exposure ratio.
#:
#: The textbook adjustment uses 1.0, which assumes defensive actions scale in direct
#: proportion to opponent possession. That assumption is testable, and in this league it
#: is false: regressing log team defensive volume on log opponent possession over the
#: forty team-seasons available gives elasticities near 0.3 to 0.5. Applying an exponent
#: of 1.0 therefore overcorrects by a factor of two to three and, as the tests in
#: tests/test_preprocess.py demonstrate, injects more team dependence than it removes.
#: "estimated" fits the exponent per feature from the team data; "unit" reproduces the
#: textbook behaviour and is retained so the comparison can be reported.
PADJ_ELASTICITY_MODE = "estimated"
PADJ_UNIT_ELASTICITY = 1.0

#: Guard against a degenerate divisor. No Premier League side has ever approached this.
PADJ_MIN_OPPONENT_POSSESSION = 5.0

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
