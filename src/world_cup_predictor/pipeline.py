from __future__ import annotations

import pandas as pd

from world_cup_predictor.config import N_SIMULATIONS, OUTPUTS_DIR, PROCESSED_DATA_DIR, STAGE_COLUMNS, TRAIN_CUTOFF_DATE
from world_cup_predictor.data import load_or_create_project_data
from world_cup_predictor.features import build_modeling_table
from world_cup_predictor.model import prepare_training_frame, save_models, split_train_test, train_models
from world_cup_predictor.tournament import precompute_pairwise_predictions, predict_group_stage_matches, simulate_tournament
from world_cup_predictor.visualization import plot_champion_probabilities, plot_model_metrics, plot_stage_progression


def _write_markdown_outputs(evaluation_df: pd.DataFrame, group_predictions: pd.DataFrame, stage_probabilities: pd.DataFrame) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    top_team = stage_probabilities.iloc[0]
    best_model = evaluation_df.sort_values("log_loss").iloc[0]
    findings = f"""# Key Findings

- Best model by log loss: **{best_model['model']}**
- Most likely World Cup winner: **{top_team['team']}** at **{top_team['win_world_cup']:.2%}**
- Average predicted group-stage draw probability: **{group_predictions['draw_prob'].mean():.2%}**

## Why this project works well in a portfolio

It combines data cleaning, rolling feature engineering, supervised learning, probability-based evaluation, tournament simulation, and clear reporting in one project that is easy to explain.
"""
    resume_bullet = (
        "Built a Python ML pipeline that predicted FIFA World Cup match probabilities and simulated the 48-team "
        "2026 tournament 10,000 times, producing stage-by-stage advancement odds, champion probabilities, "
        "evaluation metrics, and portfolio-ready visualizations."
    )
    (OUTPUTS_DIR / "key_findings.md").write_text(findings)
    (OUTPUTS_DIR / "resume_bullet.md").write_text(resume_bullet)


def run_project(raw_matches_path: str | None = None) -> dict:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    project_data = load_or_create_project_data(raw_matches_path=raw_matches_path)
    model_df, team_snapshot = build_modeling_table(project_data.matches)
    model_df.to_csv(PROCESSED_DATA_DIR / "modeling_features.csv", index=False)

    training_df = prepare_training_frame(model_df)
    train_df, test_df = split_train_test(training_df, cutoff_date=TRAIN_CUTOFF_DATE)
    model_results, best_model = train_models(train_df, test_df)
    save_models(model_results)

    evaluation_df = pd.DataFrame([result.metrics for result in model_results]).sort_values("log_loss")
    evaluation_df.to_csv(OUTPUTS_DIR / "model_evaluation.csv", index=False)

    group_predictions = predict_group_stage_matches(
        groups=project_data.groups,
        model=best_model.pipeline,
        team_snapshot=team_snapshot,
        ratings=project_data.ratings,
    )
    group_predictions.to_csv(PROCESSED_DATA_DIR / "world_cup_group_predictions.csv", index=False)

    pair_lookup = precompute_pairwise_predictions(
        teams=project_data.groups["team"].unique().tolist(),
        model=best_model.pipeline,
        team_snapshot=team_snapshot,
        ratings=project_data.ratings,
    )
    stage_probabilities = simulate_tournament(project_data.groups, pair_lookup, N_SIMULATIONS)
    stage_probabilities = stage_probabilities[["team"] + STAGE_COLUMNS]
    stage_probabilities.to_csv(OUTPUTS_DIR / "team_stage_probabilities.csv", index=False)
    stage_probabilities.head(1).to_csv(OUTPUTS_DIR / "most_likely_winner.csv", index=False)

    plot_model_metrics(evaluation_df)
    plot_champion_probabilities(stage_probabilities)
    plot_stage_progression(stage_probabilities)
    _write_markdown_outputs(evaluation_df, group_predictions, stage_probabilities)

    return {
        "evaluation": evaluation_df,
        "group_predictions": group_predictions,
        "stage_probabilities": stage_probabilities,
        "selected_model": best_model.name,
    }
