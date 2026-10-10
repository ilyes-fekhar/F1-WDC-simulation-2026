"""Pre-model diagnostics. Run before building the race model.

Questions answered:
  1. How concentrated is the season in one team, and is that trend moving?
  2. Who actually wins races, round by round?
  3. How big is the teammate gap (driver skill vs car)?
  4. How often do drivers DNF, and why? (decides how to model reliability)

Usage: python src/explore.py [--season 2026]
"""
from __future__ import annotations

import argparse

import pandas as pd

from standings import load

FINISHED = r"^(Finished|\+\d+ Laps?|Lapped)$"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)


def with_dnf(res: pd.DataFrame) -> pd.DataFrame:
    res = res.copy()
    res["dnf"] = ~res["status"].str.match(FINISHED)
    return res


def team_concentration(r: pd.DataFrame) -> None:
    pts = r.pivot_table(index="constructor_id", columns="round", values="points",
                        aggfunc="sum", fill_value=0.0)
    share = pts.div(pts.sum(axis=0), axis=1)
    top = pts.sum(axis=1).sort_values(ascending=False).index[:4]
    print("\n--- GP points share by round, top 4 teams (%) ---")
    print((share.loc[top] * 100).round(0).astype(int).to_string())
    rounds = sorted(pts.columns)
    cut = max(len(rounds) - 5, 1)
    early, late = pts[rounds[:cut]].sum(axis=1), pts[rounds[cut:]].sum(axis=1)
    out = pd.DataFrame({"share_early_%": early / early.sum() * 100,
                        "share_last5_%": late / late.sum() * 100}).loc[top].round(1)
    print(f"\n--- Share of GP points: first {cut} rounds vs last {len(rounds) - cut} ---")
    print(out.to_string())


def winners(r: pd.DataFrame) -> None:
    w = r[r["position"] == 1].sort_values("round")
    print("\n--- Race winners ---")
    print(w[["round", "race_name", "driver_id", "constructor_id", "grid"]].to_string(index=False))


def teammate_h2h(r: pd.DataFrame) -> None:
    m = r.merge(r, on=["round", "constructor_id"], suffixes=("", "_o"))
    m = m[m["driver_id"] < m["driver_id_o"]].copy()
    m["first_ahead"] = m["position"] < m["position_o"]
    m["gap"] = m["position_o"] - m["position"]            # >0: first driver finished ahead
    h = (m.groupby(["constructor_id", "driver_id", "driver_id_o"])
          .agg(rounds=("round", "count"), first_ahead=("first_ahead", "sum"),
               mean_pos_gap=("gap", "mean")).round(2))
    print("\n--- Teammate head-to-head (first driver listed; gap>0 = first ahead) ---")
    print(h.to_string())


def dnfs(res_all: pd.DataFrame, r: pd.DataFrame, season: int) -> None:
    rate = res_all.groupby("season")["dnf"].mean().mul(100).round(1)
    print("\n--- DNF rate by season (% of starts) ---")
    print(rate.to_string())
    print(f"\n--- {season}: DNFs by driver ---")
    print(r.groupby("driver_id")["dnf"].agg(["sum", "count"]).sort_values("sum", ascending=False)
          .head(10).to_string())
    print(f"\n--- {season}: DNFs by team ---")
    print(r.groupby("constructor_id")["dnf"].agg(["sum", "count"])
          .sort_values("sum", ascending=False).to_string())
    print(f"\n--- {season}: retirement causes (status strings) ---")
    print(r.loc[r["dnf"], "status"].value_counts().head(12).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    a = ap.parse_args()
    _, res, _ = load()
    res = with_dnf(res)
    season = a.season or int(res["season"].max())
    r = res[res["season"] == season]
    print(f"=== {season}: {r['round'].nunique()} rounds ===")
    team_concentration(r)
    winners(r)
    teammate_h2h(r)
    dnfs(res, r, season)


if __name__ == "__main__":
    main()