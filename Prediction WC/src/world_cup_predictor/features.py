from __future__ import annotations

import numpy as np
import pandas as pd

from world_cup_predictor.config import FEATURE_COLUMNS, ROLLING_WINDOW


ROLLING_METRICS = {
    "points": "form_points",
    "goal_diff": "goal_diff_form",
    "goals_for": "goals_for",
    "goals_against": "goals_against",
    "shots_for": "shots",
    "shots_on_target_for": "shots_on_target",
    "xg_for": "xg",
    "xg_against": "xga",
    "win_flag": "win_rate",
}


def _build_team_long_table(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "is_home": 1,
            "neutral_site": matches["neutral_site"],
            "tournament_group": matches["tournament_group"],
            "competition_importance": matches["competition_importance"],
            "elo_pre": matches["home_elo_pre"],
            "goals_for": matches["home_score"],
            "goals_against": matches["away_score"],
            "shots_for": matches["home_shots"],
            "shots_on_target_for": matches["home_shots_on_target"],
            "xg_for": matches["home_xg"],
            "xg_against": matches["away_xg"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "date": matches["date"],
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "is_home": 0,
            "neutral_site": matches["neutral_site"],
            "tournament_group": matches["tournament_group"],
            "competition_importance": matches["competition_importance"],
            "elo_pre": matches["away_elo_pre"],
            "goals_for": matches["away_score"],
            "goals_against": matches["home_score"],
            "shots_for": matches["away_shots"],
            "shots_on_target_for": matches["away_shots_on_target"],
            "xg_for": matches["away_xg"],
            "xg_against": matches["home_xg"],
        }
    )
    long_df = pd.concat([home, away], ignore_index=True).sort_values(["team", "date", "match_id"])
    long_df["points"] = np.select(
        [long_df["goals_for"] > long_df["goals_against"], long_df["goals_for"] == long_df["goals_against"]],
        [3, 1],
        default=0,
    )
    long_df["win_flag"] = (long_df["goals_for"] > long_df["goals_against"]).astype(int)
    long_df["goal_diff"] = long_df["goals_for"] - long_df["goals_against"]
    return long_df.reset_index(drop=True)


def _add_rolling_team_features(long_df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    featured = long_df.copy()
    for metric, alias in ROLLING_METRICS.items():
        featured[f"{alias}_last{window}"] = (
            featured.groupby("team")[metric]
            .transform(lambda series: series.shift(1).rolling(window, min_periods=1).mean())
        )
    featured["elo_last"] = featured.groupby("team")["elo_pre"].shift(1)
    featured["matches_played_prior"] = featured.groupby("team").cumcount()
    return featured


def build_modeling_table(matches: pd.DataFrame, rolling_window: int = ROLLING_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_df = _build_team_long_table(matches)
    long_features = _add_rolling_team_features(long_df, window=rolling_window)

    home_features = (
        long_features[long_features["is_home"] == 1]
        .drop(columns=["team", "opponent", "date", "is_home"])
        .add_prefix("home_")
    )
    away_features = (
        long_features[long_features["is_home"] == 0]
        .drop(columns=["team", "opponent", "date", "is_home"])
        .add_prefix("away_")
    )

    model_df = matches.merge(home_features, left_on="match_id", right_on="home_match_id", how="left").merge(
        away_features, left_on="match_id", right_on="away_match_id", how="left"
    )

    defaults = {
        "form_points_last5": 1.25,
        "goal_diff_form_last5": 0.0,
        "goals_for_last5": matches["home_score"].mean(),
        "goals_against_last5": matches["away_score"].mean(),
        "shots_last5": matches["home_shots"].mean(),
        "shots_on_target_last5": matches["home_shots_on_target"].mean(),
        "xg_last5": matches["home_xg"].mean(),
        "xga_last5": matches["away_xg"].mean(),
        "win_rate_last5": 0.40,
        "elo_last": matches["home_elo_pre"].mean(),
        "matches_played_prior": 5.0,
    }

    for alias, default_value in defaults.items():
        model_df[f"home_{alias}"] = model_df[f"home_{alias}"].fillna(default_value)
        model_df[f"away_{alias}"] = model_df[f"away_{alias}"].fillna(default_value)

    model_df["elo_diff"] = model_df["home_elo_last"] - model_df["away_elo_last"]
    model_df["form_points_diff"] = model_df["home_form_points_last5"] - model_df["away_form_points_last5"]
    model_df["goal_diff_form_diff"] = model_df["home_goal_diff_form_last5"] - model_df["away_goal_diff_form_last5"]
    model_df["goals_for_diff"] = model_df["home_goals_for_last5"] - model_df["away_goals_for_last5"]
    model_df["goals_against_diff"] = model_df["away_goals_against_last5"] - model_df["home_goals_against_last5"]
    model_df["shots_diff"] = model_df["home_shots_last5"] - model_df["away_shots_last5"]
    model_df["shots_on_target_diff"] = model_df["home_shots_on_target_last5"] - model_df["away_shots_on_target_last5"]
    model_df["xg_diff"] = model_df["home_xg_last5"] - model_df["away_xg_last5"]
    model_df["xga_diff"] = model_df["away_xga_last5"] - model_df["home_xga_last5"]
    model_df["win_rate_diff"] = model_df["home_win_rate_last5"] - model_df["away_win_rate_last5"]
    model_df["matches_played_diff"] = model_df["home_matches_played_prior"] - model_df["away_matches_played_prior"]

    latest_snapshot = (
        long_features.sort_values(["team", "date", "match_id"])
        .groupby("team")
        .tail(1)[
            [
                "team",
                "elo_last",
                "form_points_last5",
                "goal_diff_form_last5",
                "goals_for_last5",
                "goals_against_last5",
                "shots_last5",
                "shots_on_target_last5",
                "xg_last5",
                "xga_last5",
                "win_rate_last5",
                "matches_played_prior",
            ]
        ]
        .reset_index(drop=True)
    )

    missing_feature_cols = [column for column in FEATURE_COLUMNS if column not in model_df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing modeling feature columns: {missing_feature_cols}")

    return model_df.sort_values("date").reset_index(drop=True), latest_snapshot


def build_future_match_features(fixtures: pd.DataFrame, team_snapshot: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    snapshot = team_snapshot.merge(ratings[["team", "base_elo"]], on="team", how="left")
    snapshot["elo_last"] = snapshot["elo_last"].fillna(snapshot["base_elo"])

    home_snapshot = snapshot.add_prefix("home_")
    away_snapshot = snapshot.add_prefix("away_")
    future = fixtures.merge(home_snapshot, left_on="home_team", right_on="home_team", how="left").merge(
        away_snapshot, left_on="away_team", right_on="away_team", how="left"
    )

    for column in [
        "form_points_last5",
        "goal_diff_form_last5",
        "goals_for_last5",
        "goals_against_last5",
        "shots_last5",
        "shots_on_target_last5",
        "xg_last5",
        "xga_last5",
        "win_rate_last5",
        "matches_played_prior",
    ]:
        future[f"home_{column}"] = future[f"home_{column}"].fillna(1.25 if "rate" not in column else 0.40)
        future[f"away_{column}"] = future[f"away_{column}"].fillna(1.25 if "rate" not in column else 0.40)

    future["elo_diff"] = future["home_elo_last"] - future["away_elo_last"]
    future["form_points_diff"] = future["home_form_points_last5"] - future["away_form_points_last5"]
    future["goal_diff_form_diff"] = future["home_goal_diff_form_last5"] - future["away_goal_diff_form_last5"]
    future["goals_for_diff"] = future["home_goals_for_last5"] - future["away_goals_for_last5"]
    future["goals_against_diff"] = future["away_goals_against_last5"] - future["home_goals_against_last5"]
    future["shots_diff"] = future["home_shots_last5"] - future["away_shots_last5"]
    future["shots_on_target_diff"] = future["home_shots_on_target_last5"] - future["away_shots_on_target_last5"]
    future["xg_diff"] = future["home_xg_last5"] - future["away_xg_last5"]
    future["xga_diff"] = future["away_xga_last5"] - future["home_xga_last5"]
    future["win_rate_diff"] = future["home_win_rate_last5"] - future["away_win_rate_last5"]
    future["matches_played_diff"] = future["home_matches_played_prior"] - future["away_matches_played_prior"]

    return future
