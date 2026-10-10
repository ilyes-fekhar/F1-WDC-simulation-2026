"""Rung A Monte Carlo simulation of the rest of the season.

Per simulated race: driver performance = skill + shared team shock + independent shock;
DNFs are Bernoulli per team; finishers are ranked and scored. Sprint weekends get a second,
independent draw with a scaled-down DNF rate.

Outputs: P(title) per driver, P(title decided) by round, P(comeback).
Known limit: no fastest-lap bonus point is simulated (fine for 2025+, needed for 2019-2024 backtests).

Usage: python src/simulate.py [--season 2026] [--sims 20000] [--seed 0]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import clinch
from ratings import Ratings, fit
from standings import cumulative_points, fl_points, load, rounds_left, standings_after

GP_TABLE = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)
SPRINT_TABLE = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)


def race_points(u: np.ndarray, dnf: np.ndarray, table: np.ndarray) -> np.ndarray:
    """u, dnf: (..., drivers). Rank finishers by u (higher = better); DNFs score 0."""
    u = np.where(dnf, -np.inf, u)
    ranks = np.argsort(np.argsort(-u, axis=-1), axis=-1)
    pts = np.where(ranks < len(table), table[np.minimum(ranks, len(table) - 1)], 0.0)
    return np.where(dnf, 0.0, pts)


def simulate(rt: Ratings, base: dict[str, float], left: list[bool], n_sims: int = 20000,
             seed: int = 0, fl: int = 0, drift_sd: float = 0.0, sharpen: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    drivers = list(base)                      # everyone with points; non-line-up drivers never score
    tab = rt.table
    # sharpen > 1 widens strength gaps relative to race noise (corrects under-confidence in favourites)
    skill = sharpen * np.array([tab.at[d, "skill"] if d in tab.index else 0.0 for d in drivers])
    p_dnf = np.array([tab.at[d, "p_dnf"] if d in tab.index else 1.0 for d in drivers])
    teams = sorted(tab["team"].unique())
    team_idx = np.array([teams.index(tab.at[d, "team"]) if d in tab.index else 0 for d in drivers])
    S, R, D, T = n_sims, len(left), len(drivers), len(teams)
    # one random shift per team per simulated season (upgrades, development race); 0 = off
    shift = rng.normal(0.0, drift_sd, (S, 1, T)) if drift_sd > 0 else np.zeros((S, 1, T))

    def draw(p: np.ndarray):
        shared = rng.normal(0.0, rt.sigma_shared, (S, R, T)) + shift
        u = skill + shared[:, :, team_idx] + rng.normal(0.0, rt.sigma_indep, (S, R, D))
        return u, rng.random((S, R, D)) < p

    pts = race_points(*draw(p_dnf), GP_TABLE)
    if any(left):
        sp = race_points(*draw(np.minimum(p_dnf * rt.sprint_dnf_ratio, 1.0)), SPRINT_TABLE)
        pts = pts + sp * np.array(left, dtype=float)[None, :, None]

    start = np.array([base[d] for d in drivers])
    cum = start + np.cumsum(pts, axis=1)                          # (S, R, D)
    champ = cum[:, -1, :].argmax(axis=1)

    top2 = np.partition(cum, -2, axis=2)[:, :, -2:]                # [..., 0]=second, [..., 1]=top
    rem_after = np.array([clinch.max_remaining(left[k + 1:], fl) for k in range(R)])
    decided = top2[:, :, 1] > top2[:, :, 0] + rem_after[None, :]
    clinch_idx = np.where(decided.any(axis=1), decided.argmax(axis=1), R)   # R = tie at the end
    return {"drivers": drivers, "champ": champ, "clinch_idx": clinch_idx, "pts": pts,
            "final": cum[:, -1, :]}


def report(sim: dict, base: dict[str, float], left: list[bool], first_round: int,
           actual_avg: pd.Series, n_sims: int) -> None:
    drivers = sim["drivers"]
    leader = max(base, key=base.get)
    champ = pd.Series(np.array(drivers)[sim["champ"]])
    title = champ.value_counts(normalize=True)
    print("\n--- P(title) ---")
    print((title.head(6) * 100).round(1).astype(str).add("%").to_string())
    print(f"\nP(comeback: anyone but {leader}) = {100 * (champ != leader).mean():.1f}%")

    ci, is_leader = sim["clinch_idx"], (champ == leader).to_numpy()
    rows = []
    for k in range(len(left) + 1):
        label = f"R{first_round + k}" if k < len(left) else "tie/undecided"
        rows.append({"round": label,
                     "decided_here_%": 100 * np.mean(ci == k),
                     "decided_by_here_%": 100 * np.mean(ci <= k),
                     f"{leader}_clinched_by_here_%": 100 * np.mean((ci <= k) & is_leader)})
    print("\n--- When is the title decided? ---")
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    nonsprint = [k for k, s in enumerate(left) if not s]
    sim_avg = pd.Series(sim["pts"][:, nonsprint, :].mean(axis=(0, 1)), index=drivers)
    cmp = pd.DataFrame({"sim_pts_per_GP": sim_avg, "actual_pts_per_GP": actual_avg}).dropna()
    print("\n--- Sanity: simulated vs actual 2026 points per GP (top 8 by actual) ---")
    print(cmp.sort_values("actual_pts_per_GP", ascending=False).head(8).round(1).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--halflife", type=float, default=6.0)
    ap.add_argument("--k-driver", type=float, default=2.0)
    ap.add_argument("--dnf-prior", type=float, default=10.0)
    ap.add_argument("--drift-sd", type=float, default=0.15)
    ap.add_argument("--sharpen", type=float, default=1.0)
    a = ap.parse_args()

    cal, res, spr = load()
    season = a.season or int(res["season"].max())
    rt = fit(res, spr, season, a.halflife, a.k_driver, a.dnf_prior)
    cum = cumulative_points(res, spr, season)
    base = standings_after(cum, rt.last_round)
    left = rounds_left(cal, season, rt.last_round)

    print(f"Fitted on {season}, rounds 1-{rt.last_round}. Noise: shared={rt.sigma_shared:.2f}, "
          f"indep={rt.sigma_indep:.2f}, sprint DNF ratio={rt.sprint_dnf_ratio:.2f}")
    print(rt.table.round(2).to_string())

    sim = simulate(rt, base, left, a.sims, a.seed, fl_points(season), a.drift_sd, a.sharpen)
    actual_avg = res[res["season"] == season].groupby("driver_id")["points"].mean()
    report(sim, base, left, rt.last_round + 1, actual_avg, a.sims)


if __name__ == "__main__":
    main()
