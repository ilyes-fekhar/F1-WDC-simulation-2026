"""Estimate team-strength drift from history.

For each past season: fit ratings on rounds 1..(last - horizon), then compare each team's
fitted strength with what it actually scored over the final `horizon` rounds.
  forecast error variance = drift variance + ordinary race noise over those rounds
so the race-noise part (implied by the fitted sigmas) is subtracted to leave drift.

Usage: python src/drift.py [--horizon 7] [--halflife 6]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from explore import with_dnf
from ratings import fit, score_finishers
from standings import load


def drift_errors(res: pd.DataFrame, spr: pd.DataFrame, seasons, horizon: int = 7,
                 halflife: float = 6.0, min_scores: int = 8) -> pd.DataFrame:
    rows = []
    for s in seasons:
        full = with_dnf(res[res["season"] == s])
        if full.empty:
            continue
        last = int(full["round"].max())
        cut = last - horizon
        if cut < 8:
            continue
        rt = fit(res[(res["season"] == s) & (res["round"] <= cut)], spr, s, halflife)
        test = score_finishers(full)
        test = test[test["round"] > cut]
        actual = test.groupby("constructor_id")["score"].agg(["mean", "size"])
        n_rounds = test.groupby("constructor_id")["round"].nunique()
        for team, hat in rt.table.groupby("team")["strength"].first().items():
            if team in actual.index and actual.at[team, "size"] >= min_scores:
                noise_var = (rt.sigma_indep ** 2 / 2 + rt.sigma_shared ** 2) / n_rounds[team]
                rows.append({"season": s, "team": team, "err": actual.at[team, "mean"] - hat,
                             "noise_sd": float(np.sqrt(noise_var))})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    err_var = float((df["err"] ** 2).mean())
    noise_var = float((df["noise_sd"] ** 2).mean())
    return {"team_seasons": len(df), "bias": float(df["err"].mean()),
            "err_sd": err_var ** 0.5, "race_noise_sd": noise_var ** 0.5,
            "drift_sd": max(err_var - noise_var, 0.0) ** 0.5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=7)
    ap.add_argument("--halflife", type=float, default=6.0)
    a = ap.parse_args()
    _, res, spr = load()
    seasons = sorted(s for s in res["season"].unique() if s < res["season"].max())
    df = drift_errors(res, spr, seasons, a.horizon, a.halflife)

    print(f"Forecasting the last {a.horizon} rounds from the earlier ones (half-life {a.halflife}):")
    by = df.groupby("season").apply(lambda g: pd.Series(summarize(g)), include_groups=False)
    print(by[["team_seasons", "bias", "err_sd", "race_noise_sd", "drift_sd"]].round(2).to_string())
    print("\nOverall:", {k: round(v, 2) for k, v in summarize(df).items()})
    print("\n-> use the overall drift_sd as --drift-sd in simulate.py / sensitivity.py")


if __name__ == "__main__":
    main()
