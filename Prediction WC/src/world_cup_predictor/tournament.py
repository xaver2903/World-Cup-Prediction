from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from world_cup_predictor.features import build_future_match_features
from world_cup_predictor.model import predict_match_probabilities


def build_group_stage_fixtures(groups: pd.DataFrame) -> pd.DataFrame:
    fixtures = []
    match_number = 1
    for group_name, group_df in groups.groupby("group", sort=True):
        teams = group_df["team"].tolist()
        for home_team, away_team in combinations(teams, 2):
            fixtures.append(
                {
                    "match_number": match_number,
                    "stage": "Group Stage",
                    "group": group_name,
                    "home_team": home_team,
                    "away_team": away_team,
                    "neutral_site": 1,
                    "competition_importance": 1.0,
                    "tournament_group": "world_cup",
                }
            )
            match_number += 1
    return pd.DataFrame(fixtures)


def add_expected_goals(fixtures: pd.DataFrame, ratings: pd.DataFrame, team_snapshot: pd.DataFrame) -> pd.DataFrame:
    rating_lookup = ratings.set_index("team").to_dict(orient="index")
    snapshot_lookup = team_snapshot.set_index("team").to_dict(orient="index")
    enriched = fixtures.copy()

    home_xg_values = []
    away_xg_values = []
    for row in enriched.itertuples(index=False):
        home_snapshot = snapshot_lookup.get(row.home_team, {})
        away_snapshot = snapshot_lookup.get(row.away_team, {})
        home_rating = rating_lookup[row.home_team]["base_elo"]
        away_rating = rating_lookup[row.away_team]["base_elo"]

        home_attack = home_snapshot.get("xg_last5", rating_lookup[row.home_team]["attack_rating"])
        away_attack = away_snapshot.get("xg_last5", rating_lookup[row.away_team]["attack_rating"])
        home_defense = home_snapshot.get("xga_last5", rating_lookup[row.home_team]["defense_rating"])
        away_defense = away_snapshot.get("xga_last5", rating_lookup[row.away_team]["defense_rating"])

        home_lambda = np.clip(
            np.exp(0.10 + (home_rating - away_rating) / 500) * (home_attack / max(away_defense, 0.5)),
            0.3,
            3.4,
        )
        away_lambda = np.clip(
            np.exp(0.02 + (away_rating - home_rating) / 500) * (away_attack / max(home_defense, 0.5)),
            0.2,
            3.0,
        )
        home_xg_values.append(round(home_lambda, 2))
        away_xg_values.append(round(away_lambda, 2))

    enriched["home_expected_goals"] = home_xg_values
    enriched["away_expected_goals"] = away_xg_values
    return enriched


def predict_group_stage_matches(groups: pd.DataFrame, model, team_snapshot: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    fixtures = build_group_stage_fixtures(groups)
    features = build_future_match_features(fixtures, team_snapshot=team_snapshot, ratings=ratings)
    predictions = predict_match_probabilities(model, features)
    return add_expected_goals(predictions, ratings=ratings, team_snapshot=team_snapshot)


def rank_group_table(results: pd.DataFrame) -> pd.DataFrame:
    standings = (
        results.groupby(["group", "team"], as_index=False)
        .agg(points=("points", "sum"), goals_for=("goals_for", "sum"), goals_against=("goals_against", "sum"))
        .assign(goal_difference=lambda df: df["goals_for"] - df["goals_against"])
        .sort_values(["group", "points", "goal_difference", "goals_for", "team"], ascending=[True, False, False, False, True])
    )
    standings["group_rank"] = standings.groupby("group").cumcount() + 1
    return standings


def _fixture_key(team_a: str, team_b: str) -> tuple[str, str]:
    return tuple(sorted([team_a, team_b]))


def precompute_pairwise_predictions(teams: list[str], model, team_snapshot: pd.DataFrame, ratings: pd.DataFrame) -> dict:
    pair_rows = []
    for team_a, team_b in combinations(sorted(teams), 2):
        pair_rows.append(
            {
                "stage": "Pairwise",
                "group": None,
                "home_team": team_a,
                "away_team": team_b,
                "neutral_site": 1,
                "competition_importance": 1.0,
                "tournament_group": "world_cup",
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_features = build_future_match_features(pair_df, team_snapshot=team_snapshot, ratings=ratings)
    pair_predictions = predict_match_probabilities(model, pair_features)
    pair_predictions = add_expected_goals(pair_predictions, ratings=ratings, team_snapshot=team_snapshot)

    lookup = {}
    for row in pair_predictions.itertuples(index=False):
        lookup[_fixture_key(row.home_team, row.away_team)] = {
            "team_a": row.home_team,
            "team_b": row.away_team,
            "home_win_prob": row.home_win_prob,
            "draw_prob": row.draw_prob,
            "away_win_prob": row.away_win_prob,
            "home_advancement_prob": row.home_advancement_prob,
            "away_advancement_prob": row.away_advancement_prob,
            "home_expected_goals": row.home_expected_goals,
            "away_expected_goals": row.away_expected_goals,
        }
    return lookup


def _oriented_probabilities(pair_lookup: dict, team_one: str, team_two: str) -> dict:
    record = pair_lookup[_fixture_key(team_one, team_two)]
    if record["team_a"] == team_one:
        return record
    return {
        "team_a": team_one,
        "team_b": team_two,
        "home_win_prob": record["away_win_prob"],
        "draw_prob": record["draw_prob"],
        "away_win_prob": record["home_win_prob"],
        "home_advancement_prob": record["away_advancement_prob"],
        "away_advancement_prob": record["home_advancement_prob"],
        "home_expected_goals": record["away_expected_goals"],
        "away_expected_goals": record["home_expected_goals"],
    }


def _sample_group_match(team_one: str, team_two: str, pair_lookup: dict, rng: np.random.Generator) -> dict:
    probs = _oriented_probabilities(pair_lookup, team_one, team_two)
    home_goals = int(rng.poisson(probs["home_expected_goals"]))
    away_goals = int(rng.poisson(probs["away_expected_goals"]))

    if home_goals > away_goals:
        home_points, away_points = 3, 0
    elif away_goals > home_goals:
        home_points, away_points = 0, 3
    else:
        home_points, away_points = 1, 1

    return {
        "team_one": team_one,
        "team_two": team_two,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_points": home_points,
        "away_points": away_points,
    }


def _group_results_to_long(group_name: str, simulated_matches: list[dict]) -> pd.DataFrame:
    rows = []
    for match in simulated_matches:
        rows.append(
            {
                "group": group_name,
                "team": match["team_one"],
                "points": match["home_points"],
                "goals_for": match["home_goals"],
                "goals_against": match["away_goals"],
            }
        )
        rows.append(
            {
                "group": group_name,
                "team": match["team_two"],
                "points": match["away_points"],
                "goals_for": match["away_goals"],
                "goals_against": match["home_goals"],
            }
        )
    return pd.DataFrame(rows)


def _rank_third_place(third_place_table: pd.DataFrame) -> pd.DataFrame:
    ranked = third_place_table.sort_values(
        ["points", "goal_difference", "goals_for", "team"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    ranked["third_rank"] = ranked.index + 1
    return ranked


def _seed_knockout_teams(standings: pd.DataFrame) -> pd.DataFrame:
    winners = standings[standings["group_rank"] == 1].sort_values(
        ["points", "goal_difference", "goals_for", "group"], ascending=[False, False, False, True]
    )
    runners_up = standings[standings["group_rank"] == 2].sort_values(
        ["points", "goal_difference", "goals_for", "group"], ascending=[False, False, False, True]
    )
    thirds = _rank_third_place(standings[standings["group_rank"] == 3]).head(8)
    seeded = pd.concat([winners, runners_up, thirds], ignore_index=True)
    seeded["seed"] = np.arange(1, len(seeded) + 1)
    return seeded


def build_round_of_32(seeded_teams: pd.DataFrame) -> list[tuple[str, str]]:
    ordered = seeded_teams.sort_values("seed").reset_index(drop=True)
    teams = ordered["team"].tolist()
    matches = [(teams[i], teams[-(i + 1)]) for i in range(len(teams) // 2)]

    adjusted = matches.copy()
    for index, (team_one, team_two) in enumerate(matches[:-1]):
        group_one = ordered.loc[ordered["team"] == team_one, "group"].iloc[0]
        group_two = ordered.loc[ordered["team"] == team_two, "group"].iloc[0]
        if group_one == group_two:
            next_team = adjusted[index + 1][1]
            adjusted[index + 1] = (adjusted[index + 1][0], team_two)
            adjusted[index] = (team_one, next_team)
    return adjusted


def _simulate_knockout_match(team_one: str, team_two: str, pair_lookup: dict, rng: np.random.Generator) -> str:
    if team_one == team_two:
        return team_one
    probs = _oriented_probabilities(pair_lookup, team_one, team_two)
    home_goals = int(rng.poisson(probs["home_expected_goals"]))
    away_goals = int(rng.poisson(probs["away_expected_goals"]))
    if home_goals > away_goals:
        return team_one
    if away_goals > home_goals:
        return team_two
    return team_one if rng.random() < probs["home_advancement_prob"] else team_two


def _advance_round(matchups: list[tuple[str, str]], pair_lookup: dict, rng: np.random.Generator) -> list[str]:
    return [_simulate_knockout_match(team_one, team_two, pair_lookup, rng) for team_one, team_two in matchups]


def _pair_round(teams: list[str]) -> list[tuple[str, str]]:
    return list(zip(teams[::2], teams[1::2]))


def simulate_tournament(groups: pd.DataFrame, pair_lookup: dict, n_simulations: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    teams = sorted(groups["team"].unique().tolist())
    tallies = pd.DataFrame({"team": teams})
    for column in [
        "advance_from_group",
        "reach_round_of_16",
        "reach_quarterfinal",
        "reach_semifinal",
        "reach_final",
        "win_world_cup",
    ]:
        tallies[column] = 0

    group_map = {group_name: group_df["team"].tolist() for group_name, group_df in groups.groupby("group", sort=True)}
    for _ in range(n_simulations):
        group_tables = []
        for group_name, team_list in group_map.items():
            simulated_matches = [_sample_group_match(team_one, team_two, pair_lookup, rng) for team_one, team_two in combinations(team_list, 2)]
            group_tables.append(_group_results_to_long(group_name, simulated_matches))

        standings = rank_group_table(pd.concat(group_tables, ignore_index=True))
        seeded = _seed_knockout_teams(standings)
        qualified_teams = set(seeded["team"])
        tallies.loc[tallies["team"].isin(qualified_teams), "advance_from_group"] += 1

        round_of_16_teams = _advance_round(build_round_of_32(seeded), pair_lookup, rng)
        tallies.loc[tallies["team"].isin(round_of_16_teams), "reach_round_of_16"] += 1

        quarterfinal_teams = _advance_round(_pair_round(round_of_16_teams), pair_lookup, rng)
        tallies.loc[tallies["team"].isin(quarterfinal_teams), "reach_quarterfinal"] += 1

        semifinal_teams = _advance_round(_pair_round(quarterfinal_teams), pair_lookup, rng)
        tallies.loc[tallies["team"].isin(semifinal_teams), "reach_semifinal"] += 1

        finalists = _advance_round(_pair_round(semifinal_teams), pair_lookup, rng)
        tallies.loc[tallies["team"].isin(finalists), "reach_final"] += 1

        champion = _simulate_knockout_match(finalists[0], finalists[1], pair_lookup, rng)
        tallies.loc[tallies["team"] == champion, "win_world_cup"] += 1

    probability_columns = [column for column in tallies.columns if column != "team"]
    tallies[probability_columns] = tallies[probability_columns] / n_simulations
    return tallies.sort_values("win_world_cup", ascending=False).reset_index(drop=True)
