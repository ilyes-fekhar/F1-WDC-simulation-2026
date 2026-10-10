"""Backtests for the rung A model.

Part 1, race level (the real test): rolling origin. For every race, fit on earlier rounds only,
predict P(win), P(podium), P(points) for each driver, and score against what happened.
Baselines: (uniform) every driver equally likely; (freq) the driver's own event frequency so far
this season, shrunk toward the uniform rate. Thousands of driver-races, so differences are meaningful.
If the model cannot beat "freq", the ratings machinery is not adding value and we simplify.

Part 2, season level: for each completed season, forecast at 10, 7 and 4 rounds to go and compare
with the actual champion and clinch round. About 24 forecasts: illustrative, not statistical proof.

Usage: python src/backtest.py [--part both|race|season] [--start 2018] [--sims 2000]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import clinch
import simulate
from explore import with_dnf
from ratings import fit
from standings import (actual_clinch_round, cumulative_points, fl_points, load, rounds_left,
                       standings_after)

EVENTS = {"win": 1, "podium": 3, "points": 10}     # event -> number of qualifying positions


def train_view(res: pd.DataFrame, spr: pd.DataFrame, season: int, upto: int):
    """Only data available once round `upto` of `season` has finished (no future leakage)."""
    r = res[(res["season"] == season) & (res["round"] <= upto)]
    s = spr[(spr["season"] < season) | ((spr["season"] == season) & (spr["round"] <= upto))]
    return r, s


def observed(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"win": df["position"] == 1, "podium": df["position"] <= 3,
                         "points": (df["position"] <= 10) & ~df["dnf"]}, index=df.index)


def race_level(res, spr, seasons, min_rounds=5, n_sims=2000, halflife=6.0, k_driver=2.0,
               dnf_prior=10.0, shrink=3.0, min_drivers=8, sharpens=(1.0,)) -> pd.DataFrame:
    """Long format: one row per (race, driver, sharpen). Fit once per race, simulate per sharpen."""
    allr = with_dnf(res)
    rows = []
    for s in seasons:
        sr = allr[allr["season"] == s]
        for rnd in sorted(sr["round"].unique()):
            if rnd <= min_rounds:
                continue
            r_tr, s_tr = train_view(res, spr, s, rnd - 1)
            rt = fit(r_tr, s_tr, s, halflife, k_driver, dnf_prior)
            act = sr[sr["round"] == rnd].set_index("driver_id")
            drivers = [d for d in rt.table.index if d in act.index]
            if len(drivers) < min_drivers:
                continue
            obs = observed(act.loc[drivers])
            past = sr[sr["round"] < rnd]
            past_obs = observed(past)
            past_ids = past["driver_id"].to_numpy()
            starts = past.groupby("driver_id").size()
            n = len(drivers)
            base_rows = []
            for d in drivers:
                row = {"season": s, "round": rnd, "driver_id": d}
                for ev, slots in EVENTS.items():
                    rate = min(slots / n, 1.0)
                    k = past_obs[ev].to_numpy()[past_ids == d].sum()
                    row[f"{ev}_freq"] = (k + shrink * rate) / (starts.get(d, 0) + shrink)
                    row[f"{ev}_uniform"] = rate
                    row[f"{ev}_obs"] = float(obs.at[d, ev])
                base_rows.append(row)
            zero = {d: 0.0 for d in rt.table.index}
            for sh in sharpens:
                sim = simulate.simulate(rt, zero, [False], n_sims, int(rnd), sharpen=sh)
                pts = sim["pts"][:, 0, :]
                idx = [sim["drivers"].index(d) for d in drivers]
                p = {"win": (pts[:, idx] == 25).mean(0), "podium": (pts[:, idx] >= 15).mean(0),
                     "points": (pts[:, idx] > 0).mean(0)}
                for j, row in enumerate(base_rows):
                    rows.append({**row, "sharpen": sh, **{f"{ev}_model": p[ev][j] for ev in EVENTS}})
    return pd.DataFrame(rows)


def score_race_level(df: pd.DataFrame) -> pd.DataFrame:
    """Score one sharpen setting. brier_diff_vs_freq < 0 means the model is better; diff_se is
    the standard error across races (drivers within a race are correlated, so races are the unit)."""
    out = []
    for ev in EVENTS:
        y = df[f"{ev}_obs"]
        rec = {"event": ev, "driver_races": len(df)}
        for k in ("model", "freq", "uniform"):
            p = df[f"{ev}_{k}"].clip(1e-3, 1 - 1e-3)
            rec[f"brier_{k}"] = ((p - y) ** 2).mean()
            rec[f"logloss_{k}"] = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        diff = (df[f"{ev}_model"] - y) ** 2 - (df[f"{ev}_freq"] - y) ** 2
        per_race = diff.groupby([df["season"], df["round"]]).mean()
        rec["brier_diff_vs_freq"] = per_race.mean()
        rec["diff_se"] = per_race.std() / np.sqrt(len(per_race))
        rec["skill_vs_freq_%"] = 100 * (1 - rec["brier_model"] / rec["brier_freq"])
        rec["skill_vs_uniform_%"] = 100 * (1 - rec["brier_model"] / rec["brier_uniform"])
        out.append(rec)
    return pd.DataFrame(out)


def calibration(df: pd.DataFrame, ev: str) -> pd.DataFrame:
    bins = [0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    g = df.groupby(pd.cut(df[f"{ev}_model"], bins, include_lowest=True), observed=True)
    return pd.DataFrame({"n": g.size(), "mean_predicted": g[f"{ev}_model"].mean(),
                         "observed_rate": g[f"{ev}_obs"].mean()})


def season_level(cal, res, spr, seasons, horizons=(10, 7, 4), n_sims=5000, drift=0.15,
                 sharpen=1.0) -> pd.DataFrame:
    rows = []
    for s in seasons:
        total = int(cal.loc[cal["season"] == s, "round"].max())
        cum = cumulative_points(res, spr, s)
        champ = cum.iloc[-1].idxmax()
        actual_cl = actual_clinch_round(cal, res, spr, s)
        for h in horizons:
            cut = total - h
            if cut < 6:
                continue
            r_tr, s_tr = train_view(res, spr, s, cut)
            rt = fit(r_tr, s_tr, s)
            base = standings_after(cum, cut)
            left = rounds_left(cal, s, cut)
            sim = simulate.simulate(rt, base, left, n_sims, 0, fl_points(s), drift, sharpen)
            already = clinch.is_decided(base, left, fl_points(s))   # decided before the forecast
            champs = np.array(sim["drivers"])[sim["champ"]]
            leader = max(base, key=base.get)
            ci = sim["clinch_idx"]
            cum_dec = [float(np.mean(ci <= k)) for k in range(len(left))]
            k_act = None if actual_cl is None else actual_cl - cut - 1
            rows.append({
                "season": s, "to_go": h, "leader": leader, "leader_won": leader == champ,
                "P_leader_%": 100 * np.mean(champs == leader),
                "P_champion_%": 100 * np.mean(champs == champ),
                "pred_median_clinch_R": cut if already else next(
                    (cut + 1 + k for k, c in enumerate(cum_dec) if c >= 0.5), None),
                "actual_clinch_R": actual_cl,
                "P_decided_by_actual_%": (100 * cum_dec[k_act]
                                          if k_act is not None and 0 <= k_act < len(left) else np.nan),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["both", "race", "season"], default="both")
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--drift", type=float, default=0.15)
    ap.add_argument("--sharpen", type=float, default=1.0)
    a = ap.parse_args()

    cal, res, spr = load()
    current = int(res["season"].max())
    all_seasons = [s for s in sorted(res["season"].unique()) if s >= a.start]
    done = [s for s in all_seasons if s < current]

    if a.part in ("both", "race"):
        df = race_level(res, spr, all_seasons, n_sims=a.sims, sharpens=(a.sharpen,))
        print(f"=== Race level: {len(df)} driver-races, seasons {all_seasons[0]}-{all_seasons[-1]} ===")
        print(score_race_level(df).round(4).to_string(index=False))
        for ev in ("win", "podium"):
            print(f"\nCalibration: {ev}")
            print(calibration(df, ev).round(3).to_string())
        print("\nBrier skill > 0 means better than that baseline. Check 'freq' first.")
    if a.part in ("both", "season"):
        sl = season_level(cal, res, spr, done, n_sims=max(a.sims, 3000), drift=a.drift,
                          sharpen=a.sharpen)
        print("\n=== Season level (forecast made N rounds before the end) ===")
        print(sl.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
