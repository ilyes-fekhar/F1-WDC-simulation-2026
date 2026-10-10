"""Tune model parameters on early seasons, check them on later ones.

Protocol: score every race with a rolling-origin backtest (fit on earlier rounds only), then
compare configurations on seasons <= --split (train) and > --split (test). A setting is only
credible if it also improves the TEST block. 'sharpen' was tested this way and rejected
(log loss got worse as it increased, on both blocks), so it now defaults to 1.0 only.

Usage:
    python src/tune.py --sims 1000 --halflives 3 6 12 --k-drivers 2 8 20
    python src/tune.py --sharpens 1 1.2 1.4          # re-test the sharpen knob
"""
from __future__ import annotations

import argparse
import itertools

import pandas as pd

import backtest
from standings import load


def evaluate(df: pd.DataFrame, split: int) -> pd.DataFrame:
    rows = []
    for sh, g in df.groupby("sharpen"):
        for label, part in (("train", g[g["season"] <= split]), ("test", g[g["season"] > split])):
            if part.empty:
                continue
            sc = backtest.score_race_level(part)
            top = part[part["win_model"] > 0.4]
            rows.append({
                "sharpen": sh, "set": label,
                "logloss_model": sc["logloss_model"].sum(),
                "logloss_freq": sc["logloss_freq"].sum(),
                "brier_model": sc["brier_model"].sum(),
                "brier_freq": sc["brier_freq"].sum(),
                "win>0.4_pred": top["win_model"].mean() if len(top) else float("nan"),
                "win>0.4_obs": top["win_obs"].mean() if len(top) else float("nan"),
                "n_top": len(top),
            })
    return pd.DataFrame(rows)


def summarize(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One row per configuration, train and test side by side, best train log loss first."""
    t = pd.concat(frames, ignore_index=True)
    keys = ["halflife", "k_driver", "dnf_prior", "sharpen"]
    wide = t.pivot_table(index=keys, columns="set", values=["logloss_model", "brier_model"])
    wide.columns = [f"{m.replace('_model', '')}_{s}" for m, s in wide.columns]
    return wide.reset_index().sort_values("logloss_train").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--split", type=int, default=2023)
    ap.add_argument("--halflives", type=float, nargs="*", default=[6.0])
    ap.add_argument("--k-drivers", type=float, nargs="*", default=[2.0])
    ap.add_argument("--dnf-priors", type=float, nargs="*", default=[10.0])
    ap.add_argument("--sharpens", type=float, nargs="*", default=[1.0])
    a = ap.parse_args()

    _, res, spr = load()
    seasons = sorted(res["season"].unique())
    frames = []
    for hl, kd, dp in itertools.product(a.halflives, a.k_drivers, a.dnf_priors):
        print(f"backtesting halflife={hl} k_driver={kd} dnf_prior={dp} ...", flush=True)
        df = backtest.race_level(res, spr, seasons, n_sims=a.sims, halflife=hl, k_driver=kd,
                                 dnf_prior=dp, sharpens=tuple(a.sharpens))
        ev = evaluate(df, a.split)
        ev.insert(0, "halflife", hl)
        ev.insert(1, "k_driver", kd)
        ev.insert(2, "dnf_prior", dp)
        frames.append(ev)

    base = pd.concat(frames).groupby("set")[["logloss_freq", "brier_freq"]].first()
    print(f"\nBaseline 'freq' (train <= {a.split}, test after):")
    print(base.round(4).to_string())
    out = summarize(frames)
    print("\nConfigurations (lower is better), sorted by TRAIN log loss:")
    print(out.round(4).to_string(index=False))
    best = out.iloc[0]
    print(f"\nBest on train: halflife={best['halflife']}, k_driver={best['k_driver']}, "
          f"dnf_prior={best['dnf_prior']}, sharpen={best['sharpen']}")
    print("Only adopt it if its test log loss is also below the current default's test log loss.")


if __name__ == "__main__":
    main()
