"""Refresh the data, then append any missing forecast snapshots to docs/data/.

Safe to run as often as you like: if no new race has appeared it changes nothing (so the
scheduled job produces no commit and no redeploy).

Usage: python src/update.py [--sims 50000] [--no-ingest] [--force] [--first-round 6]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import snapshot
from standings import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data"


def dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=50000)
    ap.add_argument("--first-round", type=int, default=6)
    ap.add_argument("--no-ingest", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="only recompute the scorecard; snapshots untouched")
    ap.add_argument("--force", action="store_true", help="rebuild every snapshot (loses 'live' flags)")
    a = ap.parse_args()

    if not a.no_ingest:
        subprocess.run([sys.executable, str(ROOT / "src" / "ingest.py")], check=True)
    cal, res, spr = load()
    season = int(res["season"].max())
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"history-{season}.json"
    hist = json.loads(path.read_text()) if path.exists() else None

    if a.rescore:
        if hist is None:
            sys.exit("nothing to rescore: no history file yet")
        hist["scores"] = snapshot.compute_scores(hist["snapshots"], res, season)
        path.write_text(dump(hist))
        print(f"Rescored {len(hist['scores'])} races.")
        return

    hist, changed = snapshot.update_history(hist, cal, res, spr, season, a.sims,
                                            first_round=a.first_round, force=a.force)
    if not changed:
        print("No new race: nothing to do.")
        return

    path.write_text(dump(hist))
    (OUT / "latest.json").write_text(dump(hist["snapshots"][-1]))
    seasons = sorted(int(p.stem.split("-")[1]) for p in OUT.glob("history-*.json"))
    (OUT / "index.json").write_text(dump({"seasons": seasons, "latest_season": season}))
    print(f"Added snapshots for rounds {changed}; latest after round {hist['snapshots'][-1]['after_round']}.")


if __name__ == "__main__":
    main()
