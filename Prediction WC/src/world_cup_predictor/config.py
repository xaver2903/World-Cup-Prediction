from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

RANDOM_SEED = 42
ROLLING_WINDOW = 5
N_SIMULATIONS = 10_000
TRAIN_CUTOFF_DATE = "2024-01-01"

REQUIRED_MATCH_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral_site",
    "home_elo_pre",
    "away_elo_pre",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_xg",
    "away_xg",
]

COMPETITION_IMPORTANCE = {
    "Friendly": 0.50,
    "Nations League": 0.75,
    "Qualifier": 0.85,
    "Continental Cup": 0.92,
    "World Cup": 1.00,
}

TARGET_MAP = {"home_win": 0, "draw": 1, "away_win": 2}
TARGET_REVERSE_MAP = {value: key for key, value in TARGET_MAP.items()}

FEATURE_COLUMNS = [
    "elo_diff",
    "form_points_diff",
    "goal_diff_form_diff",
    "goals_for_diff",
    "goals_against_diff",
    "shots_diff",
    "shots_on_target_diff",
    "xg_diff",
    "xga_diff",
    "win_rate_diff",
    "matches_played_diff",
    "neutral_site",
    "competition_importance",
]

CATEGORICAL_FEATURES = ["tournament_group"]

STAGE_COLUMNS = [
    "advance_from_group",
    "reach_round_of_16",
    "reach_quarterfinal",
    "reach_semifinal",
    "reach_final",
    "win_world_cup",
]
