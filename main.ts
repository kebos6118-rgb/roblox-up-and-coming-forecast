// roblox-up-and-coming-forecast-01 entry (Node 22, standard library only).
// A rank-diffusion Monte Carlo forecaster: each game's chart rank is simulated
// forward hour by hour over the scored window, and the forecast is the average
// place occupancy across draws and hours.
//
// Why this shape: the chart sorts by user growth, and the raw `playing` counts
// are mostly diurnal noise (corr with next rank change ≈ 0), while RANK itself
// carries the signal — rank velocity is persistent (corr ≈ 0.22 per interval),
// rank volatility grows with rank, exits slide out through ranks 9-10, and new
// top-10 entries come almost entirely from ranks 11-20. So the state is the
// rank: per-game velocity (exponentially weighted, shrunk, decaying), per-band
// volatility (x1.6 — the chart moves more than the raw per-read volatility
// suggests), and an exit hazard for games sliding off the chart. Backtests on
// three history cuts showed mean reversion toward a long-run rank HURTS (the
// chart is growth bursts, not equilibrium), so there is none (MR = 0). One
// static forecast is scored at every hourly read, so averaging occupancy over
// the whole window is exactly the scored object. Unnamed mass (0.25/place) is
// the insurance for entrants the simulation cannot name (brand-new games,
// which almost never reach the top 10 in 72h).
//
// Deterministic: mulberry32 with a fixed seed; no clock, no unseeded random.
// Tuning knobs are env-overridable for development; the grader leaves them
// unset, so the defaults below are the entry.
//
//   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const env = (k: string, d: number): number => {
  const v = process.env[k];
  return v === undefined || v === '' ? d : Number(v);
};
const SEED = Math.round(env('RB_SEED', 20261113));
// Velocity: half-life (h) of the EWMA over past rank changes.
const HL = env('RB_HL', 18);
// Shrinkage applied to the EWMA velocity (0..1).
const SHRINK = env('RB_SHRINK', 0.6);
// Momentum decay: velocity half-life (h) inside the simulation.
const TAU = env('RB_TAU', 30);
// Mean reversion toward the game's long-run rank, per hour.
// Default 0: backtests on three cuts all showed mean reversion HURTS — the
// chart is driven by growth bursts, not reversion to an equilibrium rank.
const MR = env('RB_MR', 0);
// Global multiplier on per-band rank volatility.
// Default 1.6: backtest peak (1.0 -> -41.9 mean, 1.3 -> -38.0, 1.6 -> -37.7,
// 2.0 -> -38.3 over three cuts).
const SVOL = env('RB_SVOL', 1.6);
// Exit hazard per hour for a game (scaled by rank/age inside).
const EXIT = env('RB_EXIT', 0.004);
// Unnamed mass left at every place.
const UNNAMED = env('RB_UNNAMED', 0.25);
// Monte Carlo draws.
const DRAWS = Math.round(env('RB_DRAWS', 2048));
// Extra upward drift (ranks/hour) for games first seen recently: new games
// climb faster than old ones (measured mean -0.32 ranks/h in their first day,
// vs ~0 after; partly survivorship, so the default is 0 unless tuned).
const NEWBOOST = env('RB_NEWBOOST', 0);
// Min named p kept per row (smaller mass goes unnamed).
const PMIN = 0.0005;

interface Row {
  rank: number;
  universe_id: number;
  playing: number;
  name?: string;
}
interface Mark {
  at: number;
  rows: Row[];
  details: Record<string, { created?: string } | null> | null;
}

/** mulberry32: small deterministic PRNG. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Box-Muller normal sampler on a fixed PRNG stream. */
function makeRandn(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = rng() * 2 - 1;
      v = rng() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const mul = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * mul;
    return u * mul;
  };
}

/** Per-band sd of rank change per hour, measured on the practice history. */
function sigmaOf(rank: number): number {
  if (rank <= 3) return 0.19;
  if (rank <= 10) return 0.37;
  if (rank <= 20) return 0.66;
  if (rank <= 40) return 1.08;
  return 1.41;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
  window: { from: string; until: string; slots: number };
};
const PLACES = task.places;
const marks: Mark[] = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '')
  .map((l) => JSON.parse(l) as { at: string; ok?: boolean; up_and_coming: Row[]; details?: Record<string, { created?: string } | null> | null })
  .filter((m) => m.ok !== false && Array.isArray(m.up_and_coming) && Number.isFinite(Date.parse(m.at)))
  .map((m) => ({
    at: Date.parse(m.at),
    rows: m.up_and_coming.filter((r) => Number.isFinite(r?.rank) && Number.isFinite(r?.universe_id)),
    details: m.details ?? null,
  }));
if (marks.length === 0) throw new Error('no usable marks in history');
const lastMark = marks[marks.length - 1]!;
const lastTime = lastMark.at;
const winFrom = Date.parse(task.window.from);
const winUntil = Date.parse(task.window.until);
const totalH = Math.max(1, Math.round((winUntil - lastTime) / 3600000));
const warmH = Math.max(0, Math.round((winFrom - lastTime) / 3600000));
const countH = totalH - warmH; // hours inside the scored window

// ---- Per-game tracks --------------------------------------------------------
interface Track {
  id: number;
  points: Array<{ t: number; rank: number }>;
  lastRank: number;
  lastT: number;
  firstT: number;
  createdMs: number | null;
  vel: number; // ranks/hour, EWMA, shrunk
  eq: number; // long-run rank (EWMA over all points)
  sig: number;
}
const tracks = new Map<number, Track>();
for (const m of marks) {
  for (const r of m.rows) {
    let tr = tracks.get(r.universe_id);
    if (tr === undefined) {
      tr = {
        id: r.universe_id,
        points: [],
        lastRank: r.rank,
        lastT: m.at,
        firstT: m.at,
        createdMs: null,
        vel: 0,
        eq: r.rank,
        sig: sigmaOf(r.rank),
      };
      tracks.set(r.universe_id, tr);
    }
    tr.points.push({ t: m.at, rank: r.rank });
    tr.lastRank = r.rank;
    tr.lastT = m.at;
  }
  // details.created, when the recorder supplies it, is the true publish date.
  if (m.details) {
    for (const [k, d] of Object.entries(m.details)) {
      const tr = tracks.get(Number(k));
      if (tr && d && d.created) tr.createdMs = Date.parse(d.created);
    }
  }
}
// Velocity: EWMA of per-hour rank changes (most recent first), then shrink.
// Long-run rank: EWMA over all points with a ~3-day half-life.
const lamV = Math.pow(0.5, 1 / HL);
const lamE = Math.pow(0.5, 1 / 72);
for (const tr of tracks.values()) {
  let w = 0;
  let v = 0;
  for (let i = tr.points.length - 1; i >= 1; i--) {
    const dt = (tr.points[i]!.t - tr.points[i - 1]!.t) / 3600000;
    if (dt <= 0) continue;
    const dv = (tr.points[i]!.rank - tr.points[i - 1]!.rank) / dt;
    const a = Math.pow(lamV, (lastTime - tr.points[i]!.t) / 3600000);
    w += a;
    v += a * dv;
  }
  tr.vel = w > 0 ? (v / w) * SHRINK : 0;
  let ew = 0;
  let e = 0;
  for (const p of tr.points) {
    const a = Math.pow(lamE, (lastTime - p.t) / 3600000);
    ew += a;
    e += a * p.rank;
  }
  tr.eq = ew > 0 ? e / ew : tr.lastRank;
  tr.sig = sigmaOf(tr.lastRank) * SVOL;
}

// Candidates: the last chart, plus games in the top 40 within the last 48h
// that have since dropped off (they can climb back).
const lastIds = new Set(lastMark.rows.map((r) => r.universe_id));
const cands: Track[] = [];
for (const tr of tracks.values()) {
  if (lastIds.has(tr.id)) {
    cands.push(tr);
    continue;
  }
  const recent = tr.points.filter((p) => p.t >= lastTime - 48 * 3600000);
  if (recent.some((p) => p.rank <= 40)) cands.push(tr);
}
const C = cands.length;
if (C < PLACES) throw new Error(`only ${C} candidate games, fewer than ${PLACES} places`);
// Exit hazard: older games and worse ranks leave faster. Age is measured from
// publish (details.created) when known, else from first sight in the history.
const ageH = cands.map((tr) => {
  const start = tr.createdMs ?? tr.firstT;
  return Math.max(0, (winFrom - start) / 3600000);
});
// A game published ~28 days before the window leaves during it; force exit
// at its 28th day when created is known.
const exitAtH = cands.map((tr) =>
  tr.createdMs === null ? Infinity : (tr.createdMs + 28 * 24 * 3600000 - lastTime) / 3600000,
);
const lamTau = Math.pow(0.5, 1 / TAU);

// ---- Monte Carlo ------------------------------------------------------------
const rng = mulberry32(SEED);
const randn = makeRandn(rng);
const counts = new Float64Array(PLACES * C);
const r = new Float64Array(C);
const v = new Float64Array(C);
const alive = new Uint8Array(C);
const topR = new Float64Array(PLACES);
const topI = new Int32Array(PLACES);
for (let d = 0; d < DRAWS; d++) {
  for (let i = 0; i < C; i++) {
    r[i] = cands[i]!.lastRank;
    v[i] = cands[i]!.vel;
    alive[i] = 1;
  }
  for (let h = 0; h < totalH; h++) {
    const decay = Math.pow(lamTau, 1);
    for (let i = 0; i < C; i++) {
      if (!alive[i]) continue;
      if (h >= exitAtH[i]!) {
        alive[i] = 0;
        continue;
      }
      // Exit hazard: base rate, higher for old/worse-ranked games.
      const hz = EXIT * (0.5 + Math.min(2, ageH[i]! / 240) + Math.max(0, (r[i] - 20) / 40));
      if (rng() < hz) {
        alive[i] = 0;
        continue;
      }
      v[i] = v[i] * decay;
      const drift = v[i] + MR * (cands[i]!.eq - r[i]) - NEWBOOST * Math.max(0, 1 - (ageH[i]! + h - warmH) / 72);
      r[i] = r[i] + drift + cands[i]!.sig * randn();
      if (r[i] < 0.5) r[i] = 0.5;
      if (r[i] > 200) r[i] = 200;
    }
    if (h < warmH) continue;
    // Top-10 selection by simulated rank (ties: lower id first).
    let nTop = 0;
    for (let i = 0; i < C; i++) {
      if (!alive[i]) continue;
      const x = r[i]!;
      let pos = nTop;
      while (pos > 0) {
        const px = topR[pos - 1]!;
        const pi = topI[pos - 1]!;
        if (x < px || (x === px && cands[i]!.id < cands[pi]!.id)) pos--;
        else break;
      }
      if (pos < PLACES) {
        const lim = Math.min(nTop + 1, PLACES);
        for (let s = lim - 1; s > pos; s--) {
          topR[s] = topR[s - 1]!;
          topI[s] = topI[s - 1]!;
        }
        topR[pos] = x;
        topI[pos] = i;
        if (nTop < PLACES) nTop++;
      }
    }
    for (let k = 0; k < nTop; k++) counts[k * C + topI[k]!]! += 1;
  }
}
const denom = DRAWS * countH;

// ---- Assemble the forecast ---------------------------------------------------
interface ProbRow {
  universe_id: number;
  place: number;
  p: number;
}
const floor6 = (p: number): number => Math.floor(p * 1e6) / 1e6;
const probabilities: ProbRow[] = [];
const namedSums: number[] = [];
for (let k = 0; k < PLACES; k++) {
  const order = cands
    .map((c, i) => ({ id: c.id, f: counts[k * C + i]! / denom }))
    .sort((a, b) => (a.f !== b.f ? b.f - a.f : a.id - b.id))
    .filter((e) => e.f > 0)
    .slice(0, 100);
  let sum = 0;
  for (const e of order) sum += e.f;
  const scale = sum > 0 ? (1 - UNNAMED) / sum : 0;
  let named = 0;
  for (const e of order) {
    const p = floor6(e.f * scale);
    if (p < PMIN) continue;
    probabilities.push({ universe_id: e.id, place: k + 1, p });
    named += p;
  }
  namedSums.push(named);
}
probabilities.sort((a, b) =>
  a.place !== b.place ? a.place - b.place : a.p !== b.p ? b.p - a.p : a.universe_id - b.universe_id,
);
// top10: argmax game per place, greedy distinct in place order.
const top10: number[] = [];
const used = new Set<number>();
for (let k = 0; k < PLACES; k++) {
  let best = -1;
  let bestF = -1;
  for (let i = 0; i < C; i++) {
    if (used.has(cands[i]!.id)) continue;
    const f = counts[k * C + i]!;
    if (f > bestF || (f === bestF && best >= 0 && cands[i]!.id < cands[best]!.id)) {
      best = i;
      bestF = f;
    }
  }
  if (best >= 0) {
    top10.push(cands[best]!.id);
    used.add(cands[best]!.id);
  }
}
if (top10.length < PLACES) throw new Error(`top10 has only ${top10.length} games`);
mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10 })}\n`);

// ---- Sanity print (stdout only, never into the file) --------------------------
console.log(`candidates=${C} draws=${DRAWS} sim_hours=${totalH} (warm ${warmH}, counted ${countH})`);
console.log(`named sums per place=${JSON.stringify(namedSums.map((s) => +s.toFixed(4)))}`);
console.log(`top10=${JSON.stringify(top10)}`);
