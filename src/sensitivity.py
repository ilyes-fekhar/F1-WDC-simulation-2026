"""How much do the headline numbers move when the untuned parameters move?

Usage: python src/sensitivity.py [--sims 5000] [--drift 0 0.15 0.3]
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from ratings import fit
from simulate import simulate
from standings import cumulative_points, fl_points, load, rounds_left, standings_after


def headline(sim: dict, base: dict[str, float], left: list[bool], first: int) -> dict:
    leader = max(base, key=base.get)
    champ = np.array(sim["drivers"])[sim["champ"]]
    ci = sim["clinch_idx"]
    cum = [float(np.mean(ci <= k)) for k in range(len(left))]
    out = {"P_leader_%": 100 * np.mean(champ == leader),
           "P_comeback_%": 100 * np.mean(champ != leader),
           "median_clinch_R": next((first + k for k, c in enumerate(cum) if c >= 0.5), None)}
    for k in (2, 3):
        if k < len(left):
            out[f"decided_by_R{first + k}_%"] = 100 * cum[k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--sims", type=int, default=5000)
    ap.add_argument("--drift", type=float, nargs="*", default=[0.0, 0.15, 0.30])
    a = ap.parse_args()

    cal, res, spr = load()
    season = a.season or int(res["season"].max())
    rows = []
    for halflife, k_driver, drift in itertools.product([3, 6, 12], [2, 8, 20], a.drift):
        rt = fit(res, spr, season, halflife, k_driver)
        cum = cumulative_points(res, spr, season)
        base = standings_after(cum, rt.last_round)
        left = rounds_left(cal, season, rt.last_round)
        sim = simulate(rt, base, left, a.sims, 0, fl_points(season), drift)
        ranked = sorted(base, key=base.get, reverse=True)
        gap = rt.table.at[ranked[0], "skill"] - rt.table.at[ranked[1], "skill"]
        rows.append({"halflife": halflife, "k_driver": k_driver, "drift_sd": drift,
                     "skill_gap_P1-P2": gap, **headline(sim, base, left, rt.last_round + 1)})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
