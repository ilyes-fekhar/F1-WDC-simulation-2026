"""Rebuild championship standings from cached results + sprints, and report clinch status.

Usage (from repo root):
    python src/standings.py              # current season status + historical clinch rounds
    python src/standings.py --season 2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import clinch

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FL_SEASONS = range(2019, 2025)   # fastest-lap bonus point existed 2019-2024


def fl_points(season: int) -> int:
    return 1 if season in FL_SEASONS else 0


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cal = pd.read_parquet(DATA_DIR / "calendar.parquet")
    res = pd.read_parquet(DATA_DIR / "results.parquet")
    spr = pd.read_parquet(DATA_DIR / "sprints.parquet")
    return cal, res, spr


def cumulative_points(res: pd.DataFrame, spr: pd.DataFrame, season: int) -> pd.DataFrame:
    """Index = round, columns = driver_id, values = cumulative points after that round.
    GP and sprint points are merged into the same round."""
    parts = [d.loc[d["season"] == season, ["round", "driver_id", "points"]] for d in (res, spr)]
    table = (
        pd.concat(parts)
        .pivot_table(index="round", columns="driver_id", values="points",
                     aggfunc="sum", fill_value=0.0)
        .sort_index()
    )
    return table.cumsum()


def standings_after(cum: pd.DataFrame, rnd: int) -> dict[str, float]:
    return cum.loc[cum.index <= rnd].iloc[-1].to_dict()


def rounds_left(cal: pd.DataFrame, season: int, after_round: int) -> list[bool]:
    c = cal[(cal["season"] == season) & (cal["round"] > after_round)].sort_values("round")
    return c["is_sprint"].astype(bool).tolist()


def actual_clinch_round(cal, res, spr, season: int) -> int | None:
    """First completed round after which the title was mathematically decided
    (strict '>', so a points tie is not counted as decided)."""
    cum = cumulative_points(res, spr, season)
    for rnd in cum.index:
        if clinch.is_decided(standings_after(cum, rnd), rounds_left(cal, season, rnd),
                             fl_points(season)):
            return int(rnd)
    return None


def season_report(cal, res, spr, season: int) -> None:
    cum = cumulative_points(res, spr, season)
    done = int(cum.index.max())
    total = int(cal.loc[cal["season"] == season, "round"].max())
    st = standings_after(cum, done)
    left = rounds_left(cal, season, done)
    ranked = sorted(st.items(), key=lambda kv: -kv[1])
    leader = ranked[0][0]
    fl = fl_points(season)

    print(f"\n=== {season}: after round {done} of {total} ===")
    for i, (d, p) in enumerate(ranked[:8], 1):
        print(f"{i:>2}. {d:<16}{p:>7.1f}   gap {ranked[0][1] - p:>6.1f}")
    print(f"\nRounds left: {len(left)}  (sprint flags: {left})")
    print(f"Max points still available: {clinch.max_remaining(left, fl)}")
    print(f"Still mathematically alive: {clinch.still_possible(st, left, fl)}")
    print(f"Decided already: {clinch.is_decided(st, left, fl)}")
    if left:
        idx = clinch.earliest_clinch(st, left, leader, fl)
        if idx is None:
            print("Earliest possible clinch: not possible this season")
        else:
            print(f"Earliest possible clinch ({leader} sweeps, rivals score 0): "
                  f"round {done + 1 + idx}")


def historical_table(cal, res, spr, seasons: list[int]) -> None:
    print("\n=== Actual clinch round by season (strict points rule) ===")
    for s in seasons:
        total = int(cal.loc[cal["season"] == s, "round"].max())
        r = actual_clinch_round(cal, res, spr, s)
        cum = cumulative_points(res, spr, s)
        champ = cum.iloc[-1].idxmax()
        print(f"{s}: champion {champ:<14} points {cum.iloc[-1].max():>6.1f}   "
              f"clinched after round {r} of {total}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    a = ap.parse_args()
    cal, res, spr = load()
    season = a.season or int(res["season"].max())
    season_report(cal, res, spr, season)
    historical_table(cal, res, spr, sorted(s for s in res["season"].unique() if s != season))


if __name__ == "__main__":
    main()
