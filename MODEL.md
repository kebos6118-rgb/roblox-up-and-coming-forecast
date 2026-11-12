# Roblox Up-and-Coming forecaster v1 — MODEL.md

## Design
Monte Carlo rank-diffusion. Each candidate game's chart rank is simulated
hour by hour from the last history read through the scored window (warm-up
diffusion from last read to `window.from`, then the 72 counted hours). The
forecast is the average place occupancy over 2,048 draws x 72 hours — that
is exactly the object the scorer averages over hourly reads.

## Features (per game, from rank history only)
- **Rank velocity**: EWMA of per-hour rank changes (half-life 18h), shrunk
  x0.6, decaying inside the sim (half-life 30h). Rank momentum is persistent
  (corr ~0.22 per interval); raw `playing` counts are NOT used — they are
  mostly diurnal noise (corr with next rank change ~0).
- **Volatility**: per-rank band sd measured on the practice history (0.19
  ranks 1-3, 0.37 4-10, 0.66 11-20, 1.08 21-40, 1.41 41+), scaled x1.6.
- **Exit hazard**: 0.004/h, higher for old/worse-ranked games; a game with
  `details.created` is force-exited at publish + 28 days (chart eligibility
  limit). `details` is used only when present; nulls never crash.
- **Candidates**: the last chart (97 games) plus games in the top 40 in the
  last 48h that dropped off. Cross-chart signals (Top Trending / Top Playing
  Now) measured as NOT leading indicators and are ignored.
- **Calibration**: 0.25 of each place left UNNAMED (a miss then costs
  ~ln(0.25/90) = -5.9 instead of the -9.21 floor); named sums ~0.75/place,
  top 10 = per-place argmax, greedy distinct.

## What did NOT help (tested, rejected)
- Mean reversion to a long-run rank: hurts on all three cuts (MR 0.008 ->
  -41.9 mean; MR 0 -> -38.4). The chart is growth bursts, not equilibrium.
- Strong persistent momentum (shrink 1.0 + slow decay) with MR=0: -46.8.
- New-game upward drift (measured -0.32 ranks/h in day 1, mostly
  survivorship): no gain (-37.73 vs -37.72), left off (RB_NEWBOOST=0).

## Backtest (exact RULES.txt scorer; cuts of the kit history)
| cut | history thru | score reads | baseline | model v1 | delta |
|-----|--------------|-------------|----------|----------|-------|
| 20  | Nov 4 01:21Z | 14 (Nov 4-7) | -53.03 | -39.61 | +13.42 |
| 24  | Nov 5 06:48Z | 10 (Nov 5-7) | -56.90 | -38.31 | +18.59 |
| 26  | Nov 5 21:59Z |  8 (Nov 5-7) | -52.13 | -35.24 | +16.89 |
| mean (32 reads) | | | -54.02 | -37.72 | **+16.30** |

Model typical-p per place: 0.019 / 0.022 / 0.029 (baseline 0.005 / 0.003 /
0.005); top-10 hits 4.9 / 5.7 / 5.3 (baseline 4.2 / 4.4 / 5.6);
agreement 31 / 40 / 46 (baseline 32 / 28 / 42). MC noise across seeds:
+/-0.03 log points (2048 draws is plenty).

## Verification (full kit input)
- Two runs byte-identical; runtime ~2.4s (limit 300s); forecast.json
  19,890 bytes (limit 1,048,576); 36-45 rows/place (limit 100); per-place
  sums ~0.746 (<= 1); top10 exactly 10 distinct ids.
- Edge test with null votes/genres/details, ok:false and empty marks, and
  malformed rows (missing rank/id): runs clean, valid forecast.
- Caveat: backtest cuts score 47-88h horizons with 8-14 reads, shorter and
  noisier than the real 72h final window; the +16 mean edge is large, but
  the final window also adds house-recorder hourly reads the practice
  history lacks, so the true edge is uncertain.
