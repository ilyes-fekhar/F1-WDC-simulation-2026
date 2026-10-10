"""Pull F1 data from Jolpica (Ergast-compatible API) and cache it as parquet.

Usage (from repo root):
    python src/ingest.py                      # 2018..current year, skips cached seasons
    python src/ingest.py --start 2021 --refresh 2026
Outputs: data/calendar.parquet, data/results.parquet, data/sprints.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.jolpi.ca/ergast/f1"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PAGE = 100          # max rows per request; the limit counts result rows, not races
PAUSE = 0.5         # polite delay between calls (Jolpica rate-limits bursts)
FIRST_SPRINT_SEASON = 2021

_session = requests.Session()


def _get(path: str, params: dict | None = None, retries: int = 5) -> dict:
    url = f"{BASE}/{path}.json"
    for attempt in range(retries):
        r = _session.get(url, params=params, timeout=30)
        if r.status_code == 429:                      # rate-limited: back off
            time.sleep(2 * 2 ** attempt)
            continue
        r.raise_for_status()
        time.sleep(PAUSE)
        return r.json()["MRData"]
    raise RuntimeError(f"Still rate-limited after {retries} tries: {url}")


def fetch_races(season: int, endpoint: str | None, list_key: str | None) -> list[dict]:
    """Return all races for a season, paging and merging races split across pages."""
    path = f"{season}/{endpoint}" if endpoint else str(season)
    races: dict[str, dict] = {}
    offset = 0
    while True:
        m = _get(path, {"limit": PAGE, "offset": offset})
        for race in m["RaceTable"]["Races"]:
            key = race["round"]
            if key not in races:
                races[key] = race
            elif list_key:                            # same race, next page of rows
                races[key][list_key].extend(race[list_key])
        offset += PAGE
        if offset >= int(m["total"]):
            break
    return list(races.values())


# ---------- flatten: nested JSON -> flat rows ----------

def flatten_calendar(races: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "season": int(r["season"]),
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "date": r["date"],
            "circuit_id": r["Circuit"]["circuitId"],
            "is_sprint": "Sprint" in r,               # verify on first run, see summary
        }
        for r in races
    ]
    return pd.DataFrame(rows)


def flatten_results(races: list[dict], list_key: str) -> pd.DataFrame:
    rows = []
    for race in races:
        for r in race[list_key]:
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "date": race["date"],
                    "driver_id": r["Driver"]["driverId"],
                    "code": r["Driver"].get("code"),
                    "constructor_id": r["Constructor"]["constructorId"],
                    "grid": int(r["grid"]),
                    "position": int(r["position"]),
                    "position_text": r["positionText"],   # 'R' = retired, 'D' = DSQ, ...
                    "points": float(r["points"]),
                    "status": r["status"],
                    "laps": int(r["laps"]),
                }
            )
    return pd.DataFrame(rows)


def flatten_qualifying(races: list[dict]) -> pd.DataFrame:
    rows = []
    for race in races:
        for q in race["QualifyingResults"]:
            rows.append(
                {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": q["Driver"]["driverId"],
                    "constructor_id": q["Constructor"]["constructorId"],
                    "quali_position": int(q["position"]),
                    "q1": q.get("Q1"), "q2": q.get("Q2"), "q3": q.get("Q3"),
                }
            )
    return pd.DataFrame(rows)


# ---------- cache-aware update ----------

def update(name: str, fetch_season, seasons: list[int], refresh: set[int]) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    have = set(old["season"].unique()) if not old.empty else set()
    todo = [s for s in seasons if s not in have or s in refresh]

    frames = [old[~old["season"].isin(todo)]] if not old.empty else []
    for s in todo:
        print(f"  {name}: fetching {s}")
        frames.append(fetch_season(s))
    frames = [f for f in frames if not f.empty]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df = df.sort_values(["season", "round"]).reset_index(drop=True)
        df.to_parquet(path, index=False)
    return df


def summarize(cal: pd.DataFrame, res: pd.DataFrame, spr: pd.DataFrame) -> None:
    print("\nSanity check (races, drivers' rows, sprint weekends per season):")
    out = pd.DataFrame(
        {
            "races_cal": cal.groupby("season")["round"].nunique(),
            "races_res": res.groupby("season")["round"].nunique(),
            "result_rows": res.groupby("season").size(),
            "sprints_cal": cal.groupby("season")["is_sprint"].sum(),
            "sprints_res": spr.groupby("season")["round"].nunique() if not spr.empty else 0,
        }
    ).fillna(0).astype(int)
    print(out.to_string())
    print("\nIf sprints_cal != sprints_res, the 'Sprint' flag in the calendar "
          "isn't reliable: derive is_sprint from sprints.parquet instead.")


def main() -> None:
    this_year = dt.date.today().year
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=this_year)
    ap.add_argument("--refresh", type=int, nargs="*", default=[this_year],
                    help="seasons to re-download even if cached (default: current year)")
    a = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    seasons = list(range(a.start, a.end + 1))
    sprint_seasons = [s for s in seasons if s >= FIRST_SPRINT_SEASON]
    refresh = set(a.refresh)

    cal = update("calendar", lambda s: flatten_calendar(fetch_races(s, None, None)),
                 seasons, refresh)
    res = update("results",
                 lambda s: flatten_results(fetch_races(s, "results", "Results"), "Results"),
                 seasons, refresh)
    spr = update("sprints",
                 lambda s: flatten_results(fetch_races(s, "sprint", "SprintResults"),
                                           "SprintResults"),
                 sprint_seasons, refresh)
    qual = update("qualifying",
                  lambda s: flatten_qualifying(fetch_races(s, "qualifying", "QualifyingResults")),
                  seasons, refresh)
    summarize(cal, res, spr)
    print("\nQualifying rows per season (expect roughly the same as result_rows):")
    print(qual.groupby("season").size().to_string())


if __name__ == "__main__":
    main()