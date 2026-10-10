# F1 Title Predictor: when does Antonelli clinch the 2026 championship?

## 1. What this project is trying to answer

Three questions about the 2026 Formula 1 Drivers' Championship, answered with numbers instead of gut feeling:

1. **Who wins the title?** (a probability per driver)
2. **When is the title mathematically decided?** (a probability for each remaining round)
3. **Is a comeback realistic?** (the chance that anyone other than the current leader wins)

Situation used throughout, after round 16 of 23: Antonelli 320, Russell 236, Hamilton 214, Leclerc 191, Verstappen 188, Norris 188. Seven rounds remain (Singapore with a sprint, USA, Mexico, Brazil, Las Vegas, Qatar, Abu Dhabi), so at most **183 points** are still available to any driver.

## 2. The core idea in one paragraph

The points arithmetic is **exact**: it tells us who can still win and the earliest possible clinch. The only uncertainty is *how the remaining races will go*. So the project (a) builds a model of how each driver finishes a race, (b) **simulates the rest of the season tens of thousands of times** (Monte Carlo), and (c) counts how often each outcome happens. Then it **checks the model against history** so the probabilities can be trusted (or not).

## 3. Pipeline at a glance

```
Jolpica API ──ingest.py──> data/*.parquet ──standings.py──> points after each round
                                  │                              │
                                  ├──explore.py (diagnostics)    ├──clinch.py (exact arithmetic)
                                  │                              │
                                  └──ratings.py (fit) ──> simulate.py (Monte Carlo) ──> P(title), clinch-round table, P(comeback)

Validation:  backtest.py (score past races/seasons)  ->  tune.py (choose parameters on early seasons, test on later ones)
Robustness:  drift.py (how much do team strengths move?)  ->  sensitivity.py (how much do answers move with parameters?)
```

## 4. File-by-file reference

### Data and arithmetic

**`src/ingest.py`: download and cache the data**
- Pulls from the Jolpica API (an Ergast-compatible F1 database): `calendar`, `results` (race), `sprints`, `qualifying`, 2018 to now. Saves one parquet file per dataset in `data/`.
- `fetch_races`: pages through the API. The API's limit counts *result rows* (max 100), not races, so a race can be split across pages; the function merges them.
- `_get`: one request with retry and back-off if the API says "429 too many requests".
- `flatten_*`: turn nested JSON into flat tables (one row per driver per race).
- `update`: cache logic. Already-downloaded seasons are skipped; the current season is always re-downloaded (`--refresh`).
- Qualifying is downloaded but **not used by the model yet**.

**`src/clinch.py`: the exact points arithmetic (no statistics, no randomness)**
- `max_remaining(rounds_left, fl)`: most points one driver can still score. A GP win is 25, a sprint win is 8. `fl` is the fastest-lap bonus point (1 in 2019 to 2024, 0 otherwise).
- `is_decided(standings, rounds_left, fl)`: true if the leader's points are **strictly greater** than second place plus everything second place could still score.
- `still_possible`: drivers who can still mathematically reach the leader.
- `first_decided_round` and `earliest_clinch`: apply per-round points in order and report the first round after which the title is decided. `earliest_clinch` is the best case for the leader (wins everything, rivals score nothing).
- Known simplification: a points tie is treated as *not* decided (the wins-countback tiebreak is ignored).

**`src/standings.py`: rebuild the championship table from raw results**
- `cumulative_points`: points after each round, merging GP and sprint points into the same round.
- `standings_after`, `rounds_left` (list of True/False sprint flags), `fl_points` (the 2019 to 2024 bonus rule).
- `actual_clinch_round`: when a past title was actually decided. Prints for every season when run directly.
- Run directly (`python src/standings.py`) it prints the 2026 situation and the historical clinch table.

**`src/explore.py`: diagnostics before any modelling**
- Team share of points per round, race winners, teammate head-to-heads, DNF rates and causes.
- `with_dnf` flags a driver as DNF if the status is not "Finished", "+N Laps" or "Lapped" (this also counts "Did not start").
- Findings that shaped the model: 2026 DNF rate is 21% (74 of 352 starts, Aston Martin and Cadillac far worse than Mercedes), 2026 statuses are just "Retired" (so no mechanical-versus-crash split is possible), and Mercedes won 8 of the first 10 races but 3 of the last 6.

### The model

**`src/ratings.py`: turn one season of results into numbers the simulator can use**
- `blom_scores(n)`: converts a finishing order into "normal scores": roughly +1.9 for the winner of a 20-car race, 0 for mid-field, -1.9 for last. Removes the problem that P1 to P2 and P9 to P10 are not equally meaningful gaps.
- `score_finishers`: applies those scores to each race's **classified finishers only** (DNFs say nothing about pace).
- `fit` computes, for the current season:
  - **team strength**: the team's average score across races, with recent races weighted more (see `halflife`).
  - **driver offset**: how much better or worse than their teammate a driver is, shrunk toward zero (see `k_driver`).
  - **skill** = strength + offset.
  - **sigma_indep**: random race-to-race variation of an individual driver.
  - **sigma_shared**: random variation shared by both teammates in one race (car suited to the track, setup luck).
  - **p_dnf**: each team's DNF probability, shrunk toward the field average (see `dnf_prior`).
  - **sprint_dnf_ratio**: sprints have fewer DNFs than GPs; this is the measured ratio.
  - the **line-up**: whoever started the last completed round.

**`src/simulate.py`: the Monte Carlo season simulator**
For each simulated season and each remaining race:
1. Performance = `sharpen x skill` + shared team shock + individual shock + a season-level team shift (`drift_sd`).
2. Each driver DNFs with probability `p_dnf`.
3. Finishers are ranked by performance; points 25-18-15-12-10-8-6-4-2-1 (sprint: 8-7-6-5-4-3-2-1). DNFs score 0.
4. Sprint weekends get a second, independent draw.
5. Points are added to the current standings; the champion is whoever has most at the end.
6. After every round the code checks `is_decided` to record when the title was settled.
Outputs: P(title) per driver, the "decided by round X" table, P(comeback), and a sanity table comparing simulated and actual points per GP.
- `race_points`: ranks drivers and assigns points.
- Not simulated: the fastest-lap bonus (at most 1 point per race; the clinch arithmetic does include it for 2019 to 2024).

### Validation and robustness

**`src/backtest.py`: grade the model against history**
- `train_view`: only data available before the race being predicted (no peeking at the future).
- `race_level`: for every race since 2018, fit on earlier rounds, predict each driver's chance of **win / podium / points**, compare with what happened. About 2,900 driver-races, so results are meaningful. Compared with two baselines: **uniform** (everyone equally likely) and **freq** (the driver's own record this season so far).
- `score_race_level`: Brier score, log loss, and a paired difference against `freq` with a standard error computed across races.
- `calibration`: when the model says "30%", does it happen about 30% of the time?
- `season_level`: forecasts each past season at 10, 7 and 4 rounds to go and compares with the real champion and clinch round. Only 24 forecasts from 8 seasons, so treat as illustration.

**`src/tune.py`: choose parameters honestly**
Scores each parameter setting on early seasons (train, up to 2023) and later seasons (test, after 2023). A setting is only believed if it also helps on the test block. `evaluate` and `summarize` build the comparison tables.

**`src/drift.py`: do team strengths change within a season?**
For every past season: rate teams using all but the last 7 rounds, then see how wrong that was for the last 7. Subtract ordinary race noise; the remainder is "drift". Result: **drift_sd of about 0.15** (small compared with the roughly 0.4 gap between Mercedes and Ferrari), and bias of about 0 (the ratings are not systematically off).

**`src/sensitivity.py`: how much do the headline answers move if parameters move?**
A grid over `halflife`, `k_driver` and `drift_sd`, reporting P(leader), P(comeback), median clinch round, and "decided by round X".

**`src/make_charts.py`: the publication charts**
Runs the locked-settings forecast and the backtests, then saves four PNGs to `charts/`: `clinch_timeline.png` (when the title is decided, the headline chart), `lead_vs_remaining.png` (the lead against the points still available: why the clinch happens when it does), `backtest_calibration.png` (does "30%" mean 30%, and model vs baselines) and `backtest_seasons.png` (past seasons: confidence in the leader vs what happened). Each chart carries the run date. Use `--skip-backtest` to regenerate only the two forecast charts quickly after a race.

### Live site and automation

**`src/snapshot.py`: the forecast memory.** `build_snapshot` produces the complete forecast as of a given round using only data up to that round (no future leakage): standings, P(title), the clinch-round table, ratings, plus a forecast of the *next* race. `compute_scores` later grades each next-race forecast against the real result (model vs "everyone equally likely"). `update_history` adds missing snapshots and never rewrites old ones. A snapshot is marked `backfilled=false` only when it was produced as the latest round (a genuine pre-race forecast, timestamped by the commit); anything reconstructed afterwards is `backfilled=true`.

**`src/update.py`: the one command that does everything.** Refreshes data, adds any new snapshots, writes `docs/data/history-<season>.json`, `latest.json` and `index.json`. If no new race has appeared it changes nothing.

**`.github/workflows/update.yml`: automation.** Every 3 hours GitHub runs `update.py`; when a new race is found it commits the new JSON and redeploys `docs/` to GitHub Pages. Needs Pages enabled with source "GitHub Actions" (repository Settings, Pages).

**`docs/index.html`:** a minimal landing page that reads `latest.json`. The full interactive page (before/after comparison, history, scorecard) is the next build.

### Tests (`tests/`)
- `test_clinch.py`: arithmetic checks, including the exact 2026 scenarios and the 2023 fastest-lap case.
- `test_ingest.py`: JSON flattening and merging of races split across pages (using a fake API).
- `test_standings.py`: points rebuild; with real data it checks every champion's total against the official final (2018 to 2025) and the 2023 clinch round.
- `test_model.py`: ratings recover known strengths from synthetic seasons, the simulator behaves coherently, drift and sharpen behave as designed, the backtest and tune functions run.
- `test_charts.py`: the chart code writes a valid image.
- `test_snapshot.py`: snapshots are coherent and JSON-safe, never use future data, and updates are idempotent with correct live/backfilled flags.
Run all with `python -m pytest -q`.

## 5. Parameter glossary

| Parameter | Where | Meaning | Value used |
|---|---|---|---|
| `halflife` | `ratings.fit` | A race's weight halves every this-many rounds into the past. Smaller = reacts faster to recent form. Applies to team strength and driver offsets. | **6** |
| `k_driver` | `ratings.fit` | "Pseudo-races" of prior pulling each driver's teammate offset toward zero. Bigger = trust driver differences less. | **2** |
| `dnf_prior` | `ratings.fit` | "Pseudo-starts" at the field-wide DNF rate mixed into each team's own DNF record. | **10** |
| `sharpen` | `simulate`, `backtest` | Multiplies skill relative to race noise (bigger = favourites win more). Tested and **rejected**, so left at 1.0. | **1.0** |
| `drift_sd` | `simulate` | Season-level random shift in each team's strength, measuring "the car might get better or worse". Estimated by `drift.py`. | **0.15** |
| `n_sims` / `--sims` | `simulate`, `backtest` | Number of simulated seasons or races. More = smoother percentages. | 20,000 default; use 50,000+ for final numbers |
| `seed` | `simulate` | Makes random draws repeatable. | 0 |
| `fl` | `clinch`, `simulate` | Fastest-lap bonus per GP (1 for 2019 to 2024, 0 otherwise). | automatic by season |
| `shrink` | `backtest` | Strength of the pull toward uniform in the `freq` baseline. | 3 |
| `min_rounds` | `backtest` | Don't predict until this many rounds of the season exist. | 5 |
| `split` | `tune` | Last season of the training block. | 2023 |

## 6. Metrics, in plain words

- **Brier score**: average squared error of a probability. Predicting 70% for something that happens counts as error 0.09; lower is better.
- **Log loss**: punishes confident wrong predictions very heavily; lower is better.
- **Calibration**: among all cases where the model said about 30%, what share actually happened? A good model's answer is about 30%.
- **Baseline**: a deliberately simple predictor the model must beat to justify its complexity.
- **Train / test split**: parameters are chosen on 2018 to 2023, then judged on 2024 to 2026, which they never influenced.
- **Rolling origin**: every prediction uses only earlier data, exactly as a real forecast would.
- **Monte Carlo**: replay the unknown future many times with random draws, then read probabilities off the frequencies.

## 7. What we found

**Arithmetic.** Earliest possible clinch is round 18 (USA): Antonelli sweeps the Singapore sprint, Singapore GP and USA GP while Russell scores at most 17 points. If Antonelli wins everything and Russell is second each time, round 19 (Mexico). A comeback needs Antonelli to average about fifth place while Russell wins every race.

**A bug we caught.** The 2023 title was first reported decided one round too early. Cause: 2019 to 2024 had a fastest-lap bonus point per GP, which the first version ignored (180 points were available, not 174). Fixed with the `fl` parameter and a regression test.

**Backtest, race level (2,877 driver-races).** The model beats "everyone equally likely" by about 25 to 34% on win, podium and points. Against the strong `freq` baseline, per-event Brier scores were close; summed over the three events the model is **better in both blocks** (log loss 0.8924 vs 0.9206 on train, 0.9371 vs 0.9742 on test, at the original untuned settings).

**A hypothesis that failed.** The model seemed under-confident about dominant favourites in 2018 to 2023, so we tried `sharpen`. Log loss got steadily worse as it increased, on both train and test, and the pattern reversed in 2024 to 2026. Rejected.

**Parameter tuning (9 configurations).** Every configuration beat the `freq` baseline on the test block (best 0.9246 vs 0.9742). Less teammate shrinkage was better at every half-life (`k_driver` 2 beat 8, which beat 20), so driver differences carry real signal. Half-life barely mattered (differences of about 0.001 to 0.005). Chosen: `halflife=6, k_driver=2` (best on train; test log loss 0.9257 vs 0.9325 for the old default).

**Season level.** The leader won 20 of 24 forecasts (average predicted probability 88%). All four misses were in 2021 and 2025, where the model had itself given the leader only 54 to 72%, so it correctly identified the open fights.

**2026 headline (from the sensitivity run at `halflife=6, k_driver=2, drift 0.15`):** Antonelli about 99.6%, comeback about 0.4%, median clinch round 20 (Brazil), about 27% decided by Mexico and about 69% by Brazil. Across every setting tried: 99.3% to 99.7% and comeback 0.26% to 0.68%.

## 8. Honest limitations

- Parameter uncertainty is only partly covered (`drift_sd`); the ratings are treated as known otherwise.
- Teammates' DNFs are independent in the simulation; real teammate collisions or shared failures are not modelled.
- DNFs are one process per team (2026 statuses have no cause, so crashes and mechanical failures cannot be separated).
- Qualifying and grid positions are downloaded but unused, and circuit-specific effects are absent.
- The wins-countback tiebreak is ignored; fastest-lap points are not simulated.
- The season-level backtest is 8 independent seasons: illustrative, not statistical proof.
- Regulation changes (2026 included) weaken what earlier seasons say about car strength.

## 9. How to run everything

```bash
pip install -r requirements.txt
python -m pytest -q                                      # all tests
python src/ingest.py                                     # download/update data (current season refreshes)
python src/standings.py                                  # 2026 status + historical clinch rounds
python src/explore.py                                    # diagnostics
python src/simulate.py --sims 50000                      # the headline 2026 forecast
python src/sensitivity.py --sims 5000 --drift 0 0.15     # robustness to parameters
python src/backtest.py --sims 2000                       # history check
python src/drift.py                                      # team-strength drift estimate
python src/make_charts.py                                # the four publication charts -> charts/
python src/update.py                                     # refresh data + add new forecast snapshots (what CI runs)
python src/tune.py --sims 1000 --halflives 3 6 12 --k-drivers 0.5 1 2 8    # re-tune parameters
```
Re-run `ingest.py` and `simulate.py` after every race; date-stamp any chart you publish.

## 10. The 60-second explanation

"The points arithmetic is exact, so I only had to model how the remaining races go. I rate each team and driver from this season's results, with recent races counting more, estimate how often each team retires, and simulate the rest of the season 50,000 times. That gives the chance each driver wins the title and the round when it gets decided. To check I wasn't fooling myself I replayed every race since 2018 using only the data available beforehand, and compared against simple baselines. I tuned two settings on 2018 to 2023 and tested them on 2024 to 2026. I also tried one idea that made things worse, and dropped it."

**Questions you should expect**
- *Why simulate instead of just using ML?* The rules are known exactly; only race outcomes are uncertain, and the simulation turns outcome probabilities into title and clinch-date probabilities.
- *How do you know it works?* Backtested against baselines on held-out seasons; limits are listed above.
- *Why not use more data (telemetry, practice)?* With 16 races and 22 drivers, more inputs cannot be told apart in a backtest and would overfit.
- *What would change your mind?* A comeback-relevant effect I do not model: teammate crashes, a big car upgrade or a regulation change.
