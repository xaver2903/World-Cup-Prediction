from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from world_cup_predictor.config import FIGURES_DIR


def plot_model_metrics(metrics_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = metrics_df.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_f1", "log_loss", "multiclass_brier"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="metric", y="value", hue="model")
    plt.title("Model Evaluation Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_evaluation.png", dpi=150)
    plt.close()


def plot_champion_probabilities(stage_probs: pd.DataFrame, top_n: int = 15) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = stage_probs.nlargest(top_n, "win_world_cup").sort_values("win_world_cup", ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(top["team"], top["win_world_cup"], color="#1f77b4")
    plt.title("Top World Cup Champion Probabilities")
    plt.xlabel("Probability of Winning")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "champion_probabilities.png", dpi=150)
    plt.close()


def plot_stage_progression(stage_probs: pd.DataFrame, top_n: int = 12) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top = stage_probs.nlargest(top_n, "win_world_cup").copy()
    plot_df = top.melt(
        id_vars="team",
        value_vars=[
            "advance_from_group",
            "reach_round_of_16",
            "reach_quarterfinal",
            "reach_semifinal",
            "reach_final",
            "win_world_cup",
        ],
        var_name="stage",
        value_name="probability",
    )
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=plot_df, x="stage", y="probability", hue="team", marker="o")
    plt.title("Stage Progression Curves for Top Teams")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage_progression.png", dpi=150)
    plt.close()
