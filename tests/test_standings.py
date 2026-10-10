import pandas as pd
import pytest

import standings
from standings import DATA_DIR


def _frames():
    # 3-round toy season. Round 2 is a sprint weekend.
    res = pd.DataFrame(
        [(1, "a", 25), (1, "b", 18), (2, "a", 25), (2, "b", 18), (3, "a", 25), (3, "b", 18)],
        columns=["round", "driver_id", "points"],
    ).assign(season=2099)
    spr = pd.DataFrame([(2, "a", 8), (2, "b", 7)], columns=["round", "driver_id", "points"]
                       ).assign(season=2099)
    cal = pd.DataFrame(
        {"season": 2099, "round": [1, 2, 3], "is_sprint": [False, True, False]}
    )
    return cal, res, spr


def test_cumulative_merges_sprint_into_round():
    cal, res, spr = _frames()
    cum = standings.cumulative_points(res, spr, 2099)
    assert cum.loc[2, "a"] == 58 and cum.loc[3, "a"] == 83
    assert cum.loc[3, "b"] == 61


def test_rounds_left_flags():
    cal, *_ = _frames()
    assert standings.rounds_left(cal, 2099, 0) == [False, True, False]
    assert standings.rounds_left(cal, 2099, 2) == [False]
    assert standings.rounds_left(cal, 2099, 3) == []


def test_actual_clinch_round_toy():
    cal, res, spr = _frames()
    # after R1: lead 7, left 25+33 -> no. after R2: lead 7+8... a=58,b=43 lead 15 vs 25 -> no
    # after R3: nobody left, a=83 > b=61 -> decided at round 3
    assert standings.actual_clinch_round(cal, res, spr, 2099) == 3


# ---- real-data validation (skipped until `python src/ingest.py` has been run) ----

OFFICIAL_FINAL = {  # season: (champion points) from official final standings
    2018: 408, 2019: 413, 2020: 347, 2021: 395.5,
    2022: 454, 2023: 575, 2024: 437, 2025: 423,
}

needs_data = pytest.mark.skipif(
    not (DATA_DIR / "results.parquet").exists(), reason="run src/ingest.py first"
)


@needs_data
@pytest.mark.parametrize("season,pts", OFFICIAL_FINAL.items())
def test_champion_points_match_official(season, pts):
    cal, res, spr = standings.load()
    cum = standings.cumulative_points(res, spr, season)
    assert cum.iloc[-1].max() == pts


@needs_data
def test_2026_after_round_16():
    cal, res, spr = standings.load()
    cum = standings.cumulative_points(res, spr, 2026)
    st = standings.standings_after(cum, 16)
    assert (st["antonelli"], st["russell"], st["hamilton"]) == (320, 236, 214)


@needs_data
def test_2023_clinch_round_is_qatar():
    # Verstappen sealed it in the Qatar sprint weekend. Imola was cancelled, so Qatar = round 17.
    cal, res, spr = standings.load()
    assert standings.actual_clinch_round(cal, res, spr, 2023) == 17
