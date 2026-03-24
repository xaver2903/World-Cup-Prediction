# FIFA World Cup 2026 Match Prediction and Tournament Simulation

This project predicts FIFA World Cup match outcome probabilities and simulates the entire 2026 tournament to estimate each team's chance of advancing through every stage and winning the World Cup.

It is built to be:

- clean enough for a GitHub portfolio
- practical enough for a graduate student to explain in an interview
- reproducible and easy to run
- simple first, then stronger where it helps

## Project goals

1. Predict the probability of `home_win`, `draw`, and `away_win` for each match.
2. Simulate the full 2026 World Cup format:
   - 48 teams
   - 12 groups of 4
   - top 2 from each group advance
   - 8 best third-place teams also advance
   - Round of 32, Round of 16, quarterfinals, semifinals, final
3. Estimate each team's probability of:
   - advancing from the group
   - reaching the Round of 16
   - reaching the quarterfinals
   - reaching the semifinals
   - reaching the final
   - winning the World Cup

## Folder structure

```text
.
├── data
│   ├── raw
│   │   ├── match_data_template.csv
│   │   ├── team_ratings_2026.csv
│   │   └── world_cup_2026_groups.csv
│   └── processed
├── models
├── notebooks
│   └── world_cup_prediction.ipynb
├── outputs
│   └── figures
├── scripts
│   └── run_pipeline.py
├── src
│   └── world_cup_predictor
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       ├── pipeline.py
│       ├── tournament.py
│       └── visualization.py
├── tests
│   └── test_tournament.py
├── pyproject.toml
└── requirements.txt
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/run_pipeline.py
```

## Main outputs

Running the project creates:

- `data/processed/clean_matches.csv`
- `data/processed/modeling_features.csv`
- `data/processed/world_cup_group_predictions.csv`
- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `outputs/model_evaluation.csv`
- `outputs/team_stage_probabilities.csv`
- `outputs/most_likely_winner.csv`
- `outputs/key_findings.md`
- `outputs/resume_bullet.md`
- plots in `outputs/figures/`

## Dataset columns needed

These are the raw columns the pipeline expects:

- `date`
- `home_team`
- `away_team`
- `home_score`
- `away_score`
- `tournament`
- `neutral_site`
- `home_elo_pre`
- `away_elo_pre`
- `home_shots`
- `away_shots`
- `home_shots_on_target`
- `away_shots_on_target`
- `home_xg`
- `away_xg`

Optional but useful:

- `competition_importance`
- `home_possession`
- `away_possession`
- `home_rest_days`
- `away_rest_days`
- `home_fifa_rank`
- `away_fifa_rank`

## Modeling approach

- Baseline model: multinomial logistic regression
- Improved model: random forest classifier
- Probability target: `home_win`, `draw`, `away_win`
- Key features:
  - Elo difference
  - recent points per match
  - recent goal difference
  - recent goals for and against
  - recent shots and shots on target
  - recent xG and xGA
  - recent win rate
  - neutral-site indicator
  - competition importance

## 2026 simulation design

The simulator follows the 2026 format and uses a transparent knockout seeding rule:

1. Rank group winners by group-stage performance.
2. Rank runners-up by the same rule.
3. Rank the best eight third-place teams.
4. Seed the 32 knockout teams from 1 to 32.
5. Pair them as `1 vs 32`, `2 vs 31`, ..., `16 vs 17`.
6. If a Round of 32 match repeats a group-stage matchup, make a small swap with the next tie.

This is easy to explain and easy to update later if FIFA publishes an exact slot mapping.

## Assumptions

- If no real historical dataset is available, the code generates a realistic synthetic international match dataset so the project still runs end to end.
- The included 2026 team list is a practical template and can be swapped later for actual qualifiers.
- Group matches are simulated with expected-goal style score sampling.
- Knockout draws are resolved using the model's non-draw win probabilities as a proxy for extra time and penalties.

## Resume bullet

Built a Python machine learning pipeline that predicted FIFA World Cup match outcome probabilities and simulated the 48-team 2026 tournament 10,000 times, producing stage-by-stage advancement odds, champion probabilities, evaluation metrics, and recruiter-ready visualizations.
