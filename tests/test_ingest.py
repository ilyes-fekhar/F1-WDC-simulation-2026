import ingest


def _row(pos, drv):
    return {
        "position": str(pos), "positionText": str(pos), "points": str(max(26 - 2 * pos, 0)),
        "grid": str(pos), "laps": "57", "status": "Finished",
        "Driver": {"driverId": drv, "code": drv[:3].upper()},
        "Constructor": {"constructorId": "mercedes"},
    }


def _race(rnd, rows):
    return {"season": "2026", "round": str(rnd), "raceName": f"GP{rnd}", "date": "2026-03-08",
            "Circuit": {"circuitId": "x"}, "Results": rows}


def test_flatten_results_types():
    df = ingest.flatten_results([_race(1, [_row(1, "antonelli"), _row(2, "russell")])], "Results")
    assert list(df["driver_id"]) == ["antonelli", "russell"]
    assert df["position"].dtype.kind == "i" and df["points"].dtype.kind == "f"


def test_fetch_races_merges_race_split_across_pages(monkeypatch):
    # Race 2 straddles the page boundary: 1 row on page 1, 1 row on page 2.
    pages = {
        0: {"total": "150", "RaceTable": {"Races": [
            _race(1, [_row(1, "antonelli")]), _race(2, [_row(1, "russell")])]}},
        100: {"total": "150", "RaceTable": {"Races": [_race(2, [_row(2, "hamilton")])]}},
    }
    monkeypatch.setattr(ingest, "_get", lambda path, params=None, retries=5: pages[params["offset"]])
    races = ingest.fetch_races(2026, "results", "Results")
    assert [len(r["Results"]) for r in races] == [1, 2]


def test_calendar_sprint_flag():
    races = [
        {"season": "2026", "round": "1", "raceName": "A", "date": "d", "Circuit": {"circuitId": "x"}},
        {"season": "2026", "round": "2", "raceName": "B", "date": "d", "Circuit": {"circuitId": "y"},
         "Sprint": {"date": "d"}},
    ]
    df = ingest.flatten_calendar(races)
    assert list(df["is_sprint"]) == [False, True]
