import json

import numpy as np

import snapshot
from test_model import _multi_season

SEASON = 2092


def _data(latest):
    cal, res, spr = _multi_season(n=3, rounds=14)
    res = res[(res["season"] != SEASON) | (res["round"] <= latest)]
    return cal, res, spr


def test_snapshot_contents_are_coherent_and_serialisable():
    cal, res, spr = _data(11)
    s = snapshot.build_snapshot(cal, res, spr, SEASON, 10, sims=600)
    json.dumps(s)                                              # must be plain JSON
    assert s["after_round"] == 10 and len(s["rounds_left"]) == 4
    assert abs(sum(s["p_title"].values()) - 1) < 0.01
    by = [r["decided_by"] for r in s["clinch"]]
    assert by == sorted(by) and abs(by[-1] + s["undecided_tie"] - 1) < 1e-6
    assert abs(sum(d["p_win"] for d in s["next_race"]["drivers"]) - 1) < 0.03
    assert s["next_race"]["round"] == 11


def test_snapshot_does_not_use_future_rounds():
    cal, res, spr = _data(14)
    a = snapshot.build_snapshot(cal, res, spr, SEASON, 9, sims=400, seed=1)
    res_less = res[(res["season"] != SEASON) | (res["round"] <= 9)]
    b = snapshot.build_snapshot(cal, res_less, spr, SEASON, 9, sims=400, seed=1)
    assert a["p_title"] == b["p_title"] and a["ratings"] == b["ratings"]
    assert a["model"]["data_hash"] == b["model"]["data_hash"]


def test_finished_season_snapshot_has_no_simulation():
    cal, res, spr = _data(14)
    s = snapshot.build_snapshot(cal, res, spr, SEASON, 14, sims=200)
    assert s["clinch"] == [] and s["next_race"] is None and s["p_comeback"] == 0.0


def test_update_is_idempotent_and_flags_live_vs_backfilled():
    cal, res, spr = _data(11)
    hist, changed = snapshot.update_history(None, cal, res, spr, SEASON, sims=300, first_round=6)
    assert changed == [6, 7, 8, 9, 10, 11]
    flags = {s["after_round"]: s["backfilled"] for s in hist["snapshots"]}
    assert flags[11] is False and all(flags[r] for r in range(6, 11))
    assert len(hist["scores"]) == 5                            # forecasts for rounds 7..11 scored

    before = json.dumps(hist, sort_keys=True)
    hist2, changed2 = snapshot.update_history(json.loads(before), cal, res, spr, SEASON, sims=300)
    assert changed2 == [] and json.dumps(hist2, sort_keys=True) == before

    cal, res12, spr = _data(12)                                # a new race appears
    hist3, changed3 = snapshot.update_history(json.loads(before), cal, res12, spr, SEASON, sims=300)
    assert changed3 == [12]
    flags3 = {s["after_round"]: s["backfilled"] for s in hist3["snapshots"]}
    assert flags3[11] is False and flags3[12] is False         # earlier live snapshot is never rewritten
    assert len(hist3["scores"]) == 6
    assert np.isfinite([r["win_logloss_model"] for r in hist3["scores"]]).all()
