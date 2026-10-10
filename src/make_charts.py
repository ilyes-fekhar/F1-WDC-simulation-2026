"""Generate the publication charts into charts/ from your own data and the locked model settings.

  1. clinch_timeline.png     when the title is decided (the headline chart)
  2. lead_vs_remaining.png   why: the lead against the points still available
  3. backtest_calibration.png   does "30%" mean 30%? and model vs baselines
  4. backtest_seasons.png    past seasons: confidence in the leader vs what happened

Usage: python src/make_charts.py [--sims 50000] [--skip-backtest]
Re-run after every race; each chart carries the run date.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import backtest
import clinch
from ratings import fit
from simulate import simulate
from standings import cumulative_points, fl_points, load, rounds_left, standings_after

OUT = Path(__file__).resolve().parents[1] / "charts"
TEAL, TEAL_DARK, GREY, RED, INK, MUTED = "#00A19B", "#006F6B", "#B8BEC6", "#D64545", "#1F2933", "#52606D"


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color="#E6E9ED", linewidth=0.8)
    ax.set_axisbelow(True)


def header(fig, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.95, title, fontsize=22, fontweight="bold", color=INK, va="top")
    fig.text(0.06, 0.885, subtitle, fontsize=12.5, color=MUTED, va="top")


def footer(fig, text: str) -> None:
    fig.text(0.06, 0.025, text, fontsize=9.5, color="#7B8794")


COUNTRY = {"Brazilian": "Brazil", "Dutch": "Netherlands", "Italian": "Italy", "Spanish": "Spain",
           "British": "Britain", "Austrian": "Austria", "Belgian": "Belgium", "Hungarian": "Hungary",
           "Japanese": "Japan", "Chinese": "China", "Canadian": "Canada", "Australian": "Australia"}


def short_race(name: str) -> str:
    n = str(name).split(" Grand Prix")[0]
    return COUNTRY.get(n, n)


def surname(driver_id: str) -> str:
    return driver_id.split("_")[-1].title()


# ---------------------------------------------------------------- 1. clinch timeline

def plot_clinch(per, rounds, names, path, subtitle, callout, foot, cum=None) -> None:
    per = np.asarray(per, dtype=float)
    cum = np.cumsum(per) if cum is None else np.asarray(cum, dtype=float)
    R = len(per)
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    fig.subplots_adjust(top=0.78, bottom=0.24, left=0.07, right=0.97)
    colors = [TEAL_DARK if i == int(per.argmax()) else TEAL for i in range(R)]
    ax.bar(range(R), per, color=colors, width=0.6)
    for i, v in enumerate(per):
        ax.text(i, v + 1.2, f"{v:.0f}%" if v >= 1 else f"{v:.1f}%", ha="center",
                fontsize=13, fontweight="bold", color=INK)
    ax.set_xticks(range(R))
    ax.set_xticklabels([f"R{r}\n{names.get(r, '')}" for r in rounds], fontsize=11)
    ax.set_ylim(0, max(per.max() * 1.25, 10))
    ax.set_ylabel("Chance the title is decided in this race (%)", color=MUTED)
    style(ax)
    ax.text(-0.5, -0.17, "Decided by the end of this race:", transform=ax.get_xaxis_transform(),
            fontsize=10.5, color=MUTED, ha="left")
    for i, c in enumerate(cum):
        ax.text(i, -0.25, f"{c:.0f}%", transform=ax.get_xaxis_transform(), ha="center",
                fontsize=12, fontweight="bold", color=INK)
    ax.text(0.98, 0.95, callout, transform=ax.transAxes, ha="right", va="top", fontsize=13,
            color=INK, bbox=dict(boxstyle="round,pad=0.6", fc="#F3F5F7", ec=GREY))
    header(fig, "When will Antonelli's 2026 title be decided?", subtitle)
    footer(fig, foot)
    fig.savefig(path)
    plt.close(fig)


def chart_clinch(fc: dict, path: Path) -> None:
    ci, left, first = fc["sim"]["clinch_idx"], fc["left"], fc["first"]
    R = len(left)
    per = [100 * float(np.mean(ci == k)) for k in range(R)]
    rounds = [first + k for k in range(R)]
    ranked = sorted(fc["base"], key=fc["base"].get, reverse=True)
    gap = fc["base"][ranked[0]] - fc["base"][ranked[1]]
    plot_clinch(
        per, rounds, fc["names"], path,
        f"{fc['n']:,} simulated seasons · standings after round {fc['rt'].last_round} · "
        f"{surname(ranked[0])} leads {surname(ranked[1])} by {gap:.0f} points",
        f"{surname(fc['leader'])} champion: {fc['p_leader']:.1f}%\nComeback: {100 - fc['p_leader']:.1f}%",
        f"Rating-based Monte Carlo, backtested on every race since 2018 · model run {dt.date.today():%d %b %Y} "
        "· probabilities, not certainties",
    )


# ---------------------------------------------------------------- 2. lead vs points available

def chart_lead(fc: dict, path: Path) -> None:
    sim, base, left, first, fl = fc["sim"], fc["base"], fc["left"], fc["first"], fc["fl"]
    drivers = sim["drivers"]
    L = drivers.index(fc["leader"])
    cum = np.array([base[d] for d in drivers]) + np.cumsum(sim["pts"], axis=1)       # (S, R, D)
    lead = cum[:, :, L] - np.delete(cum, L, axis=2).max(axis=2)                        # vs closest rival
    lead0 = base[fc["leader"]] - max(v for d, v in base.items() if d != fc["leader"])
    lo, mid, hi = np.percentile(lead, [5, 50, 95], axis=0)
    x = np.arange(len(left) + 1)
    pad = lambda a: np.concatenate([[lead0], a])
    remaining = [clinch.max_remaining(left[k:], fl) for k in range(len(left) + 1)]

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.08, right=0.97)
    ax.fill_between(x, pad(lo), pad(hi), color=TEAL, alpha=0.22, label="Lead: 90% of simulations")
    ax.plot(x, pad(mid), color=TEAL_DARK, linewidth=3, label="Lead: median simulation")
    ax.step(x, remaining, where="post", color=RED, linewidth=3, label="Points still available to a rival")
    ax.set_xticks(x)
    ax.set_xticklabels([f"after\nR{first - 1 + k}" for k in range(len(left) + 1)], fontsize=11)
    ax.set_ylabel("Points", color=MUTED)
    style(ax)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.text(0.01, 0.04, "The title is decided once the lead is above the red line.",
            transform=ax.transAxes, fontsize=12, color=INK)
    header(fig, f"{surname(fc['leader'])}'s lead vs the points still available",
           f"Lead over the closest rival in each simulated season · {fc['n']:,} simulations")
    footer(fig, f"Model run {dt.date.today():%d %b %Y} · rating-based Monte Carlo")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- 3. calibration

def chart_calibration(df, path: Path) -> None:
    sc = backtest.score_race_level(df)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.75), dpi=150,
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.07, right=0.97, wspace=0.28)
    for ev, color, marker in (("podium", TEAL, "o"), ("win", INK, "s")):
        c = backtest.calibration(df, ev).dropna()
        a1.scatter(c["mean_predicted"], c["observed_rate"], marker=marker, color=color, alpha=0.85,
                   s=40 + 260 * np.sqrt(c["n"] / c["n"].max()), label=f"{ev} (size = races)")
    a1.plot([0, 1], [0, 1], ls="--", color=GREY)
    a1.set_xlabel("Predicted probability", color=MUTED)
    a1.set_ylabel("How often it actually happened", color=MUTED)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.legend(frameon=False, fontsize=10, loc="upper left")
    a1.set_title("Close to the diagonal = trustworthy", fontsize=12, color=INK, loc="left")
    style(a1)
    a1.grid(axis="x", color="#E6E9ED", linewidth=0.8)

    totals = {k: float(sc[f"logloss_{k}"].sum()) for k in ("uniform", "freq", "model")}
    labels = ["Everyone\nequally likely", "Driver's own\nrecord so far", "This model"]
    bars = a2.bar(labels, [totals["uniform"], totals["freq"], totals["model"]],
                  color=[GREY, GREY, TEAL], width=0.6)
    for b, v in zip(bars, totals.values()):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center",
                fontsize=12, fontweight="bold", color=INK)
    a2.set_ylim(0, max(totals.values()) * 1.15)
    a2.set_title("Log loss, summed over win / podium / points", fontsize=12, color=INK, loc="left")
    a2.set_ylabel("Lower = better", color=MUTED)
    style(a2)
    header(fig, "Does the model deserve trust?",
           f"Backtest of every race since 2018 · {len(df):,} driver-races · each prediction uses only earlier data")
    footer(fig, f"Model run {dt.date.today():%d %b %Y}")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- 4. season-level dots

def chart_seasons(sl, path: Path, horizon: int = 7) -> None:
    d = sl[sl["to_go"] == horizon].sort_values("season")
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    fig.subplots_adjust(top=0.78, bottom=0.17, left=0.08, right=0.97)
    for _, r in d.iterrows():
        ok = bool(r["leader_won"])
        ax.scatter(r["season"], r["P_leader_%"], s=260, color=TEAL if ok else RED,
                   marker="o" if ok else "X", zorder=3)
        ax.text(r["season"], r["P_leader_%"] - 7, f"{r['P_leader_%']:.0f}%", ha="center",
                fontsize=11, color=INK)
    ax.axhline(50, color=GREY, ls="--")
    ax.set_ylim(0, 108)
    ax.set_xticks(list(d["season"]))
    ax.set_ylabel("Model's confidence that the leader wins (%)", color=MUTED)
    style(ax)
    ax.text(0.01, 0.04, "Circle = leader won · X = leader lost", transform=ax.transAxes,
            fontsize=11.5, color=INK)
    allw = sl["leader_won"].mean() * 100
    header(fig, f"Backtest: the leader's chances, {horizon} rounds before the end",
           f"Across all horizons the model averaged {sl['P_leader_%'].mean():.0f}% confidence; "
           f"the leader actually won {allw:.0f}% of the time ({int(sl['leader_won'].sum())}/{len(sl)})")
    footer(fig, f"Model run {dt.date.today():%d %b %Y} · 8 seasons: illustrative, not proof")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- driver

def forecast(cal, res, spr, season: int, sims: int, seed: int, drift: float) -> dict:
    rt = fit(res, spr, season)
    base = standings_after(cumulative_points(res, spr, season), rt.last_round)
    left = rounds_left(cal, season, rt.last_round)
    fl = fl_points(season)
    sim = simulate(rt, base, left, sims, seed, fl, drift)
    leader = max(base, key=base.get)
    champs = np.array(sim["drivers"])[sim["champ"]]
    names = {}
    if "race_name" in cal.columns:
        c = cal[cal["season"] == season]
        names = {int(r): short_race(n) for r, n in zip(c["round"], c["race_name"])}
    return {"rt": rt, "base": base, "left": left, "fl": fl, "sim": sim, "leader": leader,
            "first": rt.last_round + 1, "names": names, "n": sims,
            "p_leader": 100 * float(np.mean(champs == leader))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=float, default=0.15)
    ap.add_argument("--skip-backtest", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    cal, res, spr = load()
    season = int(res["season"].max())
    fc = forecast(cal, res, spr, season, a.sims, a.seed, a.drift)
    chart_clinch(fc, OUT / "clinch_timeline.png")
    chart_lead(fc, OUT / "lead_vs_remaining.png")
    if not a.skip_backtest:
        seasons = sorted(res["season"].unique())
        chart_calibration(backtest.race_level(res, spr, seasons, n_sims=2000), OUT / "backtest_calibration.png")
        sl = backtest.season_level(cal, res, spr, [s for s in seasons if s < season], drift=a.drift)
        chart_seasons(sl, OUT / "backtest_seasons.png")
    print("Charts saved in", OUT)


if __name__ == "__main__":
    main()
