"""Forecast snapshots: the unit of the live site.

A snapshot is the full forecast made right after a given round, using ONLY data up to that round
(so rebuilding an old one never leaks the future). history.json holds every snapshot plus a running
scorecard: each snapshot also forecasts the NEXT race, and that forecast is later scored against
what actually happened.

Snapshots are flagged  backfilled=False  when they were produced as the latest round (a genuine
before-the-race forecast, timestamped by the commit) and  backfilled=True  when reconstructed afterwards.

Pure functions only; update.py does the file and network work.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import subprocess

import numpy as np
import pandas as pd

import clinch
import simulate
from backtest import EVENTS, observed, train_view
from explore import with_dnf
from ratings import fit
from standings import cumulative_points, fl_points, rounds_left, standings_after

MODEL_VERSION = "rungA-1"
DEFAULTS = {"halflife": 6.0, "k_driver": 2.0, "dnf_prior": 10.0, "drift_sd": 0.15}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def data_fingerprint(res: pd.DataFrame, season: int, upto: int) -> str:
    r = res[(res["season"] == season) & (res["round"] <= upto)]
    r = r[["round", "driver_id", "position", "points", "status"]].sort_values(["round", "driver_id"])
    return hashlib.sha256(r.to_csv(index=False).encode()).hexdigest()[:12]


def _race_meta(cal: pd.DataFrame, season: int, rnd: int) -> dict:
    row = cal[(cal["season"] == season) & (cal["round"] == rnd)]
    if row.empty:
        return {"round": int(rnd), "name": None, "date": None, "sprint": False}
    r = row.iloc[0]
    name = str(r["race_name"]).split(" Grand Prix")[0] if "race_name" in row.columns else None
    return {"round": int(rnd), "name": name,
            "date": str(r["date"]) if "date" in row.columns else None,
            "sprint": bool(r["is_sprint"])}


def build_snapshot(cal, res, spr, season: int, upto: int, sims: int = 50000, seed: int | None = None,
                   params: dict | None = None, backfilled: bool = False, now=None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    seed = season * 100 + upto if seed is None else seed
    r_tr, s_tr = train_view(res, spr, season, upto)          # no future data
    rt = fit(r_tr, s_tr, season, p["halflife"], p["k_driver"], p["dnf_prior"])
    base = standings_after(cumulative_points(res, spr, season), upto)
    left = rounds_left(cal, season, upto)
    fl = fl_points(season)
    future = (cal[(cal["season"] == season) & (cal["round"] > upto)]
              .sort_values("round")["round"].astype(int).tolist())
    ranked = sorted(base.items(), key=lambda kv: -kv[1])
    leader = ranked[0][0]

    snap = {
        "season": int(season), "after_round": int(upto), "race": _race_meta(cal, season, upto),
        "generated_at": (now or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "backfilled": bool(backfilled),
        "model": {"version": MODEL_VERSION, **p, "sims": int(sims), "seed": int(seed),
                  "git": _git_sha(), "data_hash": data_fingerprint(res, season, upto)},
        "standings": [{"driver": d, "points": float(v)} for d, v in ranked[:10]],
        "leader": leader,
        "rounds_left": [_race_meta(cal, season, r) for r in future],
        "max_remaining": int(clinch.max_remaining(left, fl)),
        "decided_now": bool(clinch.is_decided(base, left, fl)),
        "alive": clinch.still_possible(base, left, fl),
        "ratings": [{"driver": d, "team": str(row["team"]), "skill": round(float(row["skill"]), 3),
                     "p_dnf": round(float(row["p_dnf"]), 3)} for d, row in rt.table.iterrows()],
    }

    if not left:                                              # season over: nothing left to simulate
        snap.update({"p_title": {leader: 1.0}, "p_leader": 1.0, "p_comeback": 0.0, "clinch": [],
                     "undecided_tie": 0.0, "median_clinch_round": None, "next_race": None})
        return snap

    sim = simulate.simulate(rt, base, left, sims, seed, fl, p["drift_sd"])
    champs = np.array(sim["drivers"])[sim["champ"]]
    shares = pd.Series(champs).value_counts(normalize=True)
    ci, R = sim["clinch_idx"], len(left)
    rows, cum_by = [], []
    for k in range(R):
        by = float(np.mean(ci <= k))
        cum_by.append(by)
        rows.append({"round": future[k], "name": _race_meta(cal, season, future[k])["name"],
                     "decided_here": round(float(np.mean(ci == k)), 5), "decided_by": round(by, 5),
                     "leader_clinched_by": round(float(np.mean((ci <= k) & (champs == leader))), 5)})
    snap.update({
        "p_title": {d: round(float(v), 5) for d, v in shares.items() if v >= 1e-4},
        "p_leader": round(float(np.mean(champs == leader)), 5),
        "p_comeback": round(float(np.mean(champs != leader)), 5),
        "clinch": rows, "undecided_tie": round(float(np.mean(ci == R)), 5),
        "median_clinch_round": next((future[k] for k, c in enumerate(cum_by) if c >= 0.5), None),
    })

    # forecast of the next race, to be scored once it has been run
    n1 = min(sims, 20000)
    s1 = simulate.simulate(rt, {d: 0.0 for d in rt.table.index}, [False], n1, seed + 1)
    pts = s1["pts"][:, 0, :]
    nxt = [{"driver": d, "p_win": round(float((pts[:, j] == 25).mean()), 4),
            "p_podium": round(float((pts[:, j] >= 15).mean()), 4),
            "p_points": round(float((pts[:, j] > 0).mean()), 4),
            "exp_points": round(float(pts[:, j].mean()), 2)} for j, d in enumerate(s1["drivers"])]
    nxt.sort(key=lambda r: -r["exp_points"])
    snap["next_race"] = {**_race_meta(cal, season, future[0]), "drivers": nxt}
    return snap


def compute_scores(snaps: list[dict], res: pd.DataFrame, season: int) -> list[dict]:
    """Score every snapshot's next-race forecast against the real result (model vs 'everyone equal')."""
    allr = with_dnf(res[res["season"] == season])
    rows = []
    for s in sorted(snaps, key=lambda s: s["after_round"]):
        nr = s.get("next_race")
        if not nr:
            continue
        act = allr[allr["round"] == nr["round"]].set_index("driver_id")
        if act.empty:
            continue
        fc = pd.DataFrame(nr["drivers"]).set_index("driver")
        ds = [d for d in fc.index if d in act.index]
        if len(ds) < 8:
            continue
        obs = observed(act.loc[ds])
        n = len(ds)
        rec = {"round": nr["round"], "name": nr["name"], "live": not s["backfilled"], "n_drivers": n}
        for ev, slots in EVENTS.items():
            y = obs[ev].astype(float).to_numpy()
            pm = fc.loc[ds, f"p_{ev}"].clip(1e-3, 1 - 1e-3).to_numpy()
            pu = np.full(n, min(slots / n, 1.0)).clip(1e-3, 1 - 1e-3)
            for tag, q in (("model", pm), ("uniform", pu)):
                rec[f"{ev}_brier_{tag}"] = round(float(((q - y) ** 2).mean()), 5)
                rec[f"{ev}_logloss_{tag}"] = round(float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()), 5)
        pred = list(fc.loc[ds, "p_podium"].sort_values(ascending=False).index[:3])
        actual = list(act.sort_values("position").index[:3])
        rec["podium_pred"], rec["podium_actual"] = pred, actual
        rec["podium_hits"] = len(set(pred) & set(actual))
        winners = act.index[act["position"] == 1]
        rec["favourite"] = str(fc["p_win"].idxmax())
        rec["p_favourite"] = float(fc["p_win"].max())
        rec["winner"] = str(winners[0]) if len(winners) else None
        rows.append(rec)
    return rows


def update_history(hist: dict | None, cal, res, spr, season: int, sims: int = 50000,
                   first_round: int = 6, force: bool = False, params: dict | None = None, now=None):
    """Add any missing snapshots (never rewrites existing ones unless force). Returns (hist, changed_rounds)."""
    latest = int(res.loc[res["season"] == season, "round"].max())
    if hist is None:
        hist = {"season": int(season), "model_version": MODEL_VERSION, "snapshots": [], "scores": []}
    if hist.get("model_version") != MODEL_VERSION and not force:
        raise RuntimeError("history was built with another model version; rerun with --force to rebuild")
    existing = {s["after_round"]: s for s in hist["snapshots"]}
    changed = []
    for r in range(first_round, latest + 1):
        if r in existing and not force:
            continue
        existing[r] = build_snapshot(cal, res, spr, season, r, sims, params=params,
                                     backfilled=(r < latest), now=now)
        changed.append(r)
    hist["model_version"] = MODEL_VERSION
    hist["snapshots"] = [existing[k] for k in sorted(existing)]
    hist["scores"] = compute_scores(hist["snapshots"], res, season)
    return hist, changed
