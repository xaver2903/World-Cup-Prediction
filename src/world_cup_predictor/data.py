from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from world_cup_predictor.config import (
    COMPETITION_IMPORTANCE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    REQUIRED_MATCH_COLUMNS,
)


@dataclass
class ProjectData:
    matches: pd.DataFrame
    groups: pd.DataFrame
    ratings: pd.DataFrame


def load_world_cup_groups() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / "world_cup_2026_groups.csv")


def load_team_ratings() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / "team_ratings_2026.csv")


def _competition_catalog() -> list[dict]:
    return [
        {"label": "Friendly", "neutral_prob": 0.35, "home_boost": 55},
        {"label": "Qualifier", "neutral_prob": 0.08, "home_boost": 80},
        {"label": "Continental Cup", "neutral_prob": 0.65, "home_boost": 30},
        {"label": "Nations League", "neutral_prob": 0.25, "home_boost": 45},
        {"label": "World Cup", "neutral_prob": 1.00, "home_boost": 0},
    ]


def generate_synthetic_match_data(
    ratings: pd.DataFrame,
    n_matches: int = 6000,
    start_date: str = "2012-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    teams = ratings["team"].tolist()
    rating_lookup = ratings.set_index("team").to_dict(orient="index")
    competitions = _competition_catalog()
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    records = []
    for match_id in range(n_matches):
        home_team, away_team = rng.choice(teams, size=2, replace=False)
        competition = competitions[rng.integers(0, len(competitions))]
        neutral_site = int(rng.random() < competition["neutral_prob"])

        home_elo = rating_lookup[home_team]["base_elo"] + rng.normal(0, 18)
        away_elo = rating_lookup[away_team]["base_elo"] + rng.normal(0, 18)
        venue_boost = 0 if neutral_site else competition["home_boost"]
        elo_gap = (home_elo + venue_boost - away_elo) / 400
        reverse_gap = (away_elo - home_elo - venue_boost) / 400

        home_lambda = np.clip(
            np.exp(
                0.18
                + 0.55 * elo_gap
                + 0.55
                * np.log(
                    rating_lookup[home_team]["attack_rating"]
                    / rating_lookup[away_team]["defense_rating"]
                )
            ),
            0.2,
            3.8,
        )
        away_lambda = np.clip(
            np.exp(
                0.05
                + 0.55 * reverse_gap
                + 0.55
                * np.log(
                    rating_lookup[away_team]["attack_rating"]
                    / rating_lookup[home_team]["defense_rating"]
                )
            ),
            0.1,
            3.2,
        )

        home_score = int(rng.poisson(home_lambda))
        away_score = int(rng.poisson(away_lambda))

        home_shots = max(home_score + 4, int(rng.normal(home_lambda * 6.3, 2.6)))
        away_shots = max(away_score + 3, int(rng.normal(away_lambda * 6.0, 2.5)))
        home_sot = max(home_score, int(rng.normal(home_shots * 0.36, 1.1)))
        away_sot = max(away_score, int(rng.normal(away_shots * 0.35, 1.1)))

        records.append(
            {
                "match_id": match_id,
                "date": pd.Timestamp(rng.choice(date_range)).strftime("%Y-%m-%d"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": competition["label"],
                "neutral_site": neutral_site,
                "home_elo_pre": round(home_elo, 1),
                "away_elo_pre": round(away_elo, 1),
                "home_shots": home_shots,
                "away_shots": away_shots,
                "home_shots_on_target": min(home_shots, home_sot),
                "away_shots_on_target": min(away_shots, away_sot),
                "home_xg": round(max(0.1, rng.normal(home_lambda, 0.28)), 2),
                "away_xg": round(max(0.1, rng.normal(away_lambda, 0.28)), 2),
            }
        )

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)


def clean_match_data(matches: pd.DataFrame) -> pd.DataFrame:
    cleaned = matches.copy()
    missing = [column for column in REQUIRED_MATCH_COLUMNS if column not in cleaned.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned["date"] = pd.to_datetime(cleaned["date"])
    cleaned["neutral_site"] = cleaned["neutral_site"].astype(int)
    cleaned["competition_importance"] = cleaned["tournament"].map(COMPETITION_IMPORTANCE).fillna(0.70)
    cleaned["result_label"] = np.select(
        [cleaned["home_score"] > cleaned["away_score"], cleaned["home_score"] == cleaned["away_score"]],
        ["home_win", "draw"],
        default="away_win",
    )
    cleaned["goal_margin"] = cleaned["home_score"] - cleaned["away_score"]
    cleaned["total_goals"] = cleaned["home_score"] + cleaned["away_score"]
    cleaned["tournament_group"] = cleaned["tournament"].replace(
        {
            "Friendly": "friendly",
            "Qualifier": "qualifier",
            "Continental Cup": "continental",
            "Nations League": "nations_league",
            "World Cup": "world_cup",
        }
    )
    cleaned["match_id"] = np.arange(len(cleaned))
    return cleaned.sort_values("date").reset_index(drop=True)


def load_or_create_project_data(raw_matches_path: str | None = None) -> ProjectData:
    groups = load_world_cup_groups()
    ratings = load_team_ratings()

    if raw_matches_path:
        raw_matches = pd.read_csv(raw_matches_path)
    else:
        default_path = RAW_DATA_DIR / "historical_matches.csv"
        raw_matches = pd.read_csv(default_path) if default_path.exists() else generate_synthetic_match_data(ratings)

    cleaned = clean_match_data(raw_matches)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DATA_DIR / "clean_matches.csv", index=False)
    return ProjectData(matches=cleaned, groups=groups, ratings=ratings)
