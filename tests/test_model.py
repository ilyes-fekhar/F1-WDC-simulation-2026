import numpy as np
import pandas as pd

import ratings
import simulate

TEAM_STRENGTH = {"t1": 1.5, "t2": 0.5, "t3": -0.5, "t4": -1.5}
DRIVERS = {"t1": ("a1", "a2"), "t2": ("b1", "b2"), "t3": ("c1", "c2"), "t4": ("d1", "d2")}
OFFSET = {"a1": 0.3, "a2": -0.3}                       # a1 is the better t1 driver
PTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]


def synthetic(rounds=14, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for rnd in range(1, rounds + 1):
        perf = {d: TEAM_STRENGTH[t] + OFFSET.get(d, 0.0) + rng.normal(0, 0.6)
                for t, ds in DRIVERS.items() for d in ds}
        for pos, d in enumerate(sorted(perf, key=perf.get, reverse=True), 1):
            team = next(t for t, ds in DRIVERS.items() if d in ds)
            rows.append(dict(season=2099, round=rnd, driver_id=d, constructor_id=team,
                             position=pos, points=float(PTS[pos - 1]) if pos <= 10 else 0.0,
                             status="Retired" if rng.random() < 0.05 else "Finished"))
    res = pd.DataFrame(rows)
    spr = res[res["round"].isin([2, 5])].assign(status="Finished")
    return res, spr


def test_blom_scores_monotonic_and_centered():
    s = ratings.blom_scores(20)
    assert (np.diff(s) < 0).all() and abs(s.sum()) < 1e-9


def test_fit_recovers_team_order_and_driver_offset():
    res, spr = synthetic()
    rt = ratings.fit(res, spr, 2099)
    teams_by_strength = rt.table.groupby("team")["strength"].mean().sort_values(ascending=False)
    assert list(teams_by_strength.index) == ["t1", "t2", "t3", "t4"]
    assert rt.table.at["a1", "offset"] > 0 > rt.table.at["a2", "offset"]
    assert 0.1 < rt.sigma_indep < 1.5 and rt.sigma_shared >= 0


def test_race_points_scores_in_order_and_zeroes_dnf():
    u = np.array([[3.0, 2.0, 1.0, 0.0]])
    dnf = np.array([[False, True, False, False]])
    pts = simulate.race_points(u, dnf, simulate.GP_TABLE)
    assert pts.tolist() == [[25.0, 0.0, 18.0, 15.0]]


def test_simulation_probabilities_are_coherent():
    res, spr = synthetic()
    rt = ratings.fit(res, spr, 2099)
    base = {d: 0.0 for ds in DRIVERS.values() for d in ds}
    base["a1"] = 300.0                                  # huge lead, 3 GPs left
    left = [False, True, False]
    sim = simulate.simulate(rt, base, left, n_sims=2000, seed=1)
    champ = np.array(sim["drivers"])[sim["champ"]]
    assert (champ == "a1").mean() > 0.99
    assert np.bincount(sim["clinch_idx"], minlength=len(left) + 1).sum() == 2000
    assert (sim["clinch_idx"] == 0).mean() > 0.99       # 300-pt lead is decided immediately


def test_drift_sd_adds_uncertainty():
    res, spr = synthetic()
    rt = ratings.fit(res, spr, 2099)
    base = {d: 0.0 for ds in DRIVERS.values() for d in ds}
    base["a1"], base["b1"] = 40.0, 30.0
    left = [False] * 4
    p = {}
    for drift in (0.0, 2.0):
        sim = simulate.simulate(rt, base, left, n_sims=4000, seed=2, drift_sd=drift)
        p[drift] = (np.array(sim["drivers"])[sim["champ"]] == "a1").mean()
    assert p[2.0] < p[0.0]


def test_drift_errors_small_when_strengths_are_stable():
    import drift
    frames_r, frames_s = [], []
    for i, season in enumerate(range(2090, 2095)):
        r, s = synthetic(rounds=20, seed=10 + i)
        frames_r.append(r.assign(season=season)); frames_s.append(s.assign(season=season))
    res, spr = pd.concat(frames_r), pd.concat(frames_s)
    df = drift.drift_errors(res, spr, range(2090, 2095), horizon=7)
    assert len(df) >= 15
    assert drift.summarize(df)["drift_sd"] < 0.35       # true drift is zero in the synthetic data


def _multi_season(n=3, rounds=14):
    frames_r, frames_s, cal = [], [], []
    for i in range(n):
        season = 2090 + i
        r, s = synthetic(rounds=rounds, seed=40 + i)
        frames_r.append(r.assign(season=season)); frames_s.append(s.assign(season=season))
        cal += [(season, rd, rd in (2, 5)) for rd in range(1, rounds + 1)]
    res = pd.concat(frames_r, ignore_index=True)
    res["grid"] = res["position"]
    return (pd.DataFrame(cal, columns=["season", "round", "is_sprint"]), res,
            pd.concat(frames_s, ignore_index=True))


def test_backtest_race_level_beats_uniform_on_synthetic_data():
    import backtest
    cal, res, spr = _multi_season()
    df = backtest.race_level(res, spr, [2090, 2091, 2092], min_rounds=5, n_sims=600)
    assert len(df) > 200
    sc = backtest.score_race_level(df).set_index("event")
    # "points" is degenerate with only 8 synthetic cars (nearly everyone scores), so test win/podium
    assert (sc.loc[["win", "podium"], "skill_vs_uniform_%"] > 0).all()
    assert not backtest.calibration(df, "podium").empty


def test_backtest_season_level_runs():
    import backtest
    cal, res, spr = _multi_season()
    out = backtest.season_level(cal, res, spr, [2090, 2091], horizons=(7,), n_sims=500)
    assert len(out) == 2 and out["P_leader_%"].between(0, 100).all()


def test_sharpen_makes_the_favourite_more_dominant():
    res, spr = synthetic()
    rt = ratings.fit(res, spr, 2099)
    base = {d: 0.0 for ds in DRIVERS.values() for d in ds}
    wins = {}
    for sh in (1.0, 2.0):
        sim = simulate.simulate(rt, base, [False], n_sims=4000, seed=5, sharpen=sh)
        wins[sh] = (sim["pts"][:, 0, :] == 25).mean(0).max()
    assert wins[2.0] > wins[1.0]


def test_tune_evaluate_and_paired_diff():
    import backtest, tune
    cal, res, spr = _multi_season()
    df = backtest.race_level(res, spr, [2090, 2091, 2092], n_sims=400, sharpens=(1.0, 1.5))
    assert set(df["sharpen"]) == {1.0, 1.5}
    ev = tune.evaluate(df, split=2090)
    assert set(ev["set"]) == {"train", "test"} and len(ev) == 4
    sc = backtest.score_race_level(df[df["sharpen"] == 1.0])
    assert {"brier_diff_vs_freq", "diff_se"} <= set(sc.columns) and (sc["diff_se"] > 0).all()


def test_tune_summarize_has_train_and_test_columns():
    import backtest, tune
    cal, res, spr = _multi_season()
    frames = []
    for hl in (4.0, 8.0):
        df = backtest.race_level(res, spr, [2090, 2091, 2092], n_sims=300, halflife=hl)
        ev = tune.evaluate(df, split=2090)
        ev.insert(0, "halflife", hl); ev.insert(1, "k_driver", 8.0); ev.insert(2, "dnf_prior", 10.0)
        frames.append(ev)
    out = tune.summarize(frames)
    assert len(out) == 2 and {"logloss_train", "logloss_test"} <= set(out.columns)
