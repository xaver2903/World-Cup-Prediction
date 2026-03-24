import pandas as pd

from world_cup_predictor.tournament import build_group_stage_fixtures


def test_group_fixture_count():
    groups = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A"],
            "team": ["Team 1", "Team 2", "Team 3", "Team 4"],
        }
    )
    fixtures = build_group_stage_fixtures(groups)
    assert len(fixtures) == 6
