"""Rung A ratings, fitted on one season of results.

Each race, finishers are converted to normal scores (best finisher = highest). Then:
  team strength  = recency-weighted mean of the team's per-race mean score
  driver offset  = shrunk, recency-weighted mean gap to teammate (only races where both finished)
  noise          = split into a shared team-race shock and an independent driver shock
  DNF rate       = per team, shrunk toward the field-wide rate

Defaults (halflife=6, k_driver=2, dnf_prior=10) were chosen by the train/test backtest in tune.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from explore import with_dnf

_ND = NormalDist()


def blom_scores(n: int) -> np.ndarray:
    """Expected normal order statistics for ranks 1..n; rank 1 (best) gets the highest score."""
    ranks = np.arange(1, n + 1)
    return np.array([_ND.inv_cdf((n - k + 0.625) / (n + 0.25)) for k in ranks])


def score_finishers(r: pd.DataFrame) -> pd.DataFrame:
    """Classified finishers with a per-race normal score (best finisher = highest)."""
    fin = r[~r["dnf"]].copy()
    fin["rank"] = fin.groupby("round")["position"].rank(method="first").astype(int)
    fin["n"] = fin.groupby("round")["rank"].transform("max")
    fin["score"] = [blom_scores(n)[k - 1] for k, n in zip(fin["rank"], fin["n"])]
    return fin


@dataclass
class Ratings:
    table: pd.DataFrame        # index driver_id: team, strength, offset, skill, p_dnf
    sigma_shared: float        # team-race shock (shared by teammates)
    sigma_indep: float         # independent driver-race shock
    sprint_dnf_ratio: float    # sprint DNF rate relative to GP DNF rate
    last_round: int


def sprint_dnf_ratio(res: pd.DataFrame, spr: pd.DataFrame) -> float:
    """Sprint DNF rate / GP DNF rate, measured on the same weekends."""
    keys = spr[["season", "round"]].drop_duplicates()
    gp = with_dnf(res).merge(keys, on=["season", "round"])
    gp_rate = gp["dnf"].mean()
    if not gp_rate > 0:                      # no reference DNFs (or no overlap): assume equal rates
        return 1.0
    return float(np.clip(with_dnf(spr)["dnf"].mean() / gp_rate, 0.1, 1.5))


def fit(res: pd.DataFrame, spr: pd.DataFrame, season: int, halflife: float = 6.0,
        k_driver: float = 2.0, dnf_prior: float = 10.0) -> Ratings:
    r = with_dnf(res[res["season"] == season])
    last = int(r["round"].max())
    r = r.assign(w=0.5 ** ((last - r["round"]) / halflife))

    # 1. normal scores among classified finishers
    fin = score_finishers(r)

    # 2. team strength
    tr = (fin.groupby(["round", "constructor_id"])
             .agg(team_score=("score", "mean"), n_fin=("score", "size"), w=("w", "first"))
             .reset_index())
    tr["ws"] = tr["team_score"] * tr["w"]
    strength = tr.groupby("constructor_id")["ws"].sum() / tr.groupby("constructor_id")["w"].sum()

    # 3. driver offsets (races where exactly two teammates finished)
    fin = fin.merge(tr[["round", "constructor_id", "team_score", "n_fin"]],
                    on=["round", "constructor_id"])
    pair = fin[fin["n_fin"] == 2].copy()
    pair["wd"] = (pair["score"] - pair["team_score"]) * pair["w"]
    offset = pair.groupby("driver_id")["wd"].sum() / (pair.groupby("driver_id")["w"].sum() + k_driver)

    # 4. noise scales from residuals
    fin["e"] = (fin["score"] - fin["constructor_id"].map(strength)
                - fin["driver_id"].map(offset).fillna(0.0))
    grp = fin[fin["n_fin"] == 2].groupby(["round", "constructor_id"])["e"]
    shared = grp.mean()
    half_diff = grp.apply(lambda x: (x.iloc[0] - x.iloc[1]) / 2)
    sig_i2 = 2.0 * float(np.mean(half_diff ** 2))                 # Var(half-diff) = sig_i^2 / 2
    sig_s2 = max(float(np.mean(shared ** 2)) - sig_i2 / 2.0, 0.0)  # Var(mean) = sig_s^2 + sig_i^2/2

    # 5. DNF rate per team, shrunk to the field rate
    d = r.groupby("constructor_id")["dnf"].agg(["sum", "count"])
    p_field = float(r["dnf"].mean())
    p_team = (d["sum"] + dnf_prior * p_field) / (d["count"] + dnf_prior)

    # 6. current line-up = whoever started the last completed round
    lineup = r[r["round"] == last][["driver_id", "constructor_id"]].drop_duplicates("driver_id")
    table = lineup.set_index("driver_id").rename(columns={"constructor_id": "team"})
    table["strength"] = table["team"].map(strength)
    table["offset"] = table.index.to_series().map(offset).fillna(0.0)
    table["skill"] = table["strength"] + table["offset"]
    table["p_dnf"] = table["team"].map(p_team)

    return Ratings(table=table.sort_values("skill", ascending=False),
                   sigma_shared=float(np.sqrt(sig_s2)), sigma_indep=float(np.sqrt(sig_i2)),
                   sprint_dnf_ratio=sprint_dnf_ratio(res, spr), last_round=last)
