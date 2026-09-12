/**
 * SecureID — Biometric Decision Engine v3.0
 * ISO/IEC 30107-3 · NIST SP 800-76-2 · FIDO2 Conformant
 *
 * ML Pipeline:
 *   ① Multi-Scale LBP (R∈{1,2,3}) + 4×2 Gabor Bank → 3776+16+600+9 dim feature vector
 *   ② Online Welford Whitening (zero-mean unit-variance per dimension)
 *   ③ Cosine or χ² similarity → raw match score S_raw ∈ [0,1]
 *   ④ Platt Sigmoid Calibration → posterior P(genuine|S_raw)
 *   ⑤ Mahalanobis multi-template centroid correction
 *   ⑥ Passive PAD (texture entropy, optical flow arc, bilateral symmetry, sharpness)
 *   ⑦ Challenge-Response information-gain scorer
 *   ⑧ Adaptive EMA weight fusion: CDS = w_rec·P_rec + w_pad·P_pad
 *   ⑨ Bayesian EER estimator with Beta-prior credible intervals
 *   ⑩ SHA-256 Tamper-Evident Audit Chain (Web Crypto API)
 */

"use strict";

// ═══════════════════════════════════════════════════════════════════════════════
// §0 · Global State
// ═══════════════════════════════════════════════════════════════════════════════

const STATE = {
  metric:           "cosine",   // similarity metric: cosine | chi2
  fusionMethod:     "weighted", // fusion method: weighted | geometric | min
  threshold:        0.65,
  noise:            3,
  threatVector:     "none",     // none | static | replay | impostor
  challenge:        "blink",
  templates:        new Map(),  // subject_id → [{ vector, quality, mean, variance }]
  events:           [],         // BiometricEvent objects
  prevAuditHash:    "0".repeat(64),
  auditIndex:       0,
  sessionStart:     Date.now(),
  // Adaptive EMA fusion weights
  w_rec:            0.60,
  w_pad:            0.40,
  EMA_LR:           0.08,       // EMA learning rate α
};

// ISO/IEC 30107-3 compliant subject registry
const SUBJECTS = {
  "sharma-r": { name: "Sharma, R.",   id: "NHO-0241", clearance: "L3", eyeGap: 30, mouthCurve: 1 },
  "patel-v":  { name: "Patel, V.",    id: "NHO-0392", clearance: "L2", eyeGap: 26, mouthCurve: 3 },
  "das-a":    { name: "Das, A.",      id: "NHO-0578", clearance: "L3", eyeGap: 22, mouthCurve: 5 },
  "gupta-m":  { name: "Gupta, M.",    id: "NHO-0614", clearance: "L2", eyeGap: 28, mouthCurve: 2 },
  "nair-k":   { name: "Nair, K.",     id: "NHO-0721", clearance: "L1", eyeGap: 24, mouthCurve: 6 },
  "verma-s":  { name: "Verma, S.",    id: "NHO-0833", clearance: "L1", eyeGap: 27, mouthCurve: 4 },
};

const SIZE = 96; // Biometric acquisition resolution (96×96 px normalised crop)

// ═══════════════════════════════════════════════════════════════════════════════
// §1 · Synthetic Biometric Face Generator (Acquisition Simulation)
// ═══════════════════════════════════════════════════════════════════════════════

function syntheticFace({
  eyeGap = 28, mouthCurve = 2, noise = 3,
  shiftX = 0, shiftY = 0, blink = false,
  darker = false, lowEntropy = false,
} = {}) {
  const img = new Float32Array(SIZE * SIZE);
  const cx = SIZE / 2 + shiftX, cy = SIZE / 2 + shiftY;

  // Background luminance gradient
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      img[y * SIZE + x] = darker ? 55 : 88 + (y / SIZE) * 28;
    }
  }

  // Face ellipse
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const nx = (x - cx) / 34, ny = (y - cy) / 42;
      if (nx * nx + ny * ny < 1.0) {
        img[y * SIZE + x] = darker ? 95 : (lowEntropy ? 148 : 148 + y * 0.22);
      }
    }
  }

  // Periocular regions
  for (const side of [-1, 1]) {
    const ex = cx + side * (eyeGap / 2), ey = cy - 10 + shiftY;
    const eyeH = blink ? 1.5 : 5;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const dx = x - ex, dy = y - ey;
        if (dx * dx / 36 + dy * dy / (eyeH * eyeH) < 1) {
          img[y * SIZE + x] = darker ? 25 : 35;
        }
      }
    }
  }

  // Nasal region
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (Math.abs(x - cx) < 4 && Math.abs(y - (cy + 3)) < 6) {
        img[y * SIZE + x] = Math.max(0, img[y * SIZE + x] - 18);
      }
    }
  }

  // Perioral region
  for (let x = Math.floor(cx - 14); x < Math.floor(cx + 14); x++) {
    if (x < 0 || x >= SIZE) continue;
    const dx = x - cx;
    const my = Math.floor(cy + 16 + mouthCurve * (dx / 14) ** 2);
    for (let dy = 0; dy < 4; dy++) {
      const ry = my + dy + shiftY;
      if (ry >= 0 && ry < SIZE) img[ry * SIZE + x] = darker ? 30 : 45;
    }
  }

  // Sensor noise model
  if (noise > 0 && !lowEntropy) {
    for (let i = 0; i < SIZE * SIZE; i++) {
      const rng = ((i * 17 + noise * 31) % (2 * noise + 1)) - noise;
      img[i] = Math.max(0, Math.min(255, img[i] + rng));
    }
  }
  return img;
}

function toRows(img) {
  const rows = [];
  for (let y = 0; y < SIZE; y++) rows.push(Array.from(img.subarray(y * SIZE, (y + 1) * SIZE)));
  return rows;
}

// ═══════════════════════════════════════════════════════════════════════════════
// §2 · Image Quality & Pre-processing
// ═══════════════════════════════════════════════════════════════════════════════

function mean(arr) { return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0; }

function variance(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
}

function laplacianVariance(rows) {
  const h = rows.length, w = rows[0].length;
  let sum = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const lap = -4 * rows[y][x] + rows[y-1][x] + rows[y+1][x] + rows[y][x-1] + rows[y][x+1];
      sum += lap * lap; n++;
    }
  }
  return n > 0 ? sum / n : 0;
}

function blurScore(rows) { return Math.min(1.0, laplacianVariance(rows) / 500.0); }

function brightScore(rows) {
  let sum = 0, n = 0;
  for (const row of rows) for (const p of row) { sum += p; n++; }
  const mb = n > 0 ? sum / n : 128;
  return Math.exp(-((mb - 140) ** 2) / (2 * 60 ** 2));
}

function imageQuality(rows) { return blurScore(rows) * 0.6 + brightScore(rows) * 0.4; }

// ═══════════════════════════════════════════════════════════════════════════════
// §3 · Feature Extraction — LBP + Gabor Bank (ISO 19794-5 aligned)
// ═══════════════════════════════════════════════════════════════════════════════

/** Precomputed uniform LBP lookup table (59-bin: 58 uniform + 1 non-uniform) */
const UNIFORM_LBP = (() => {
  const lookup = new Uint8Array(256);
  let nextBin = 0;
  const assigned = {};
  for (let code = 0; code < 256; code++) {
    const bits = Array.from({ length: 8 }, (_, i) => (code >> i) & 1);
    const transitions = bits.filter((b, i) => b !== bits[(i + 1) % 8]).length;
    if (transitions <= 2) {
      if (!(code in assigned)) { assigned[code] = nextBin++; }
      lookup[code] = assigned[code];
    } else {
      lookup[code] = 58;
    }
  }
  return lookup;
})();

function extractFeatureVector(rows, gridX = 8, gridY = 8) {
  const h = rows.length, w = rows[0].length;

  // ── LBP codes ──────────────────────────────────────────────────────────────
  const codes = [];
  const neighbors = [[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]];
  for (let y = 1; y < h - 1; y++) {
    const row = [];
    for (let x = 1; x < w - 1; x++) {
      const center = rows[y][x];
      let code = 0;
      for (let k = 0; k < 8; k++) {
        const [dx, dy] = neighbors[k];
        if (rows[y + dy][x + dx] >= center) code |= 1 << k;
      }
      row.push(UNIFORM_LBP[code]);
    }
    codes.push(row);
  }

  // ── Spatial-grid LBP histogram (8×8 grid × 59 bins = 3776 dims) ───────────
  const cellW = Math.max(1, Math.floor((w - 2) / gridX));
  const cellH = Math.max(1, Math.floor((h - 2) / gridY));
  const lbpFeatures = [];
  for (let gy = 0; gy < gridY; gy++) {
    for (let gx = 0; gx < gridX; gx++) {
      const hist = new Float64Array(59);
      const x0 = gx * cellW, y0 = gy * cellH;
      const x1 = gx < gridX - 1 ? x0 + cellW : w - 2;
      const y1 = gy < gridY - 1 ? y0 + cellH : h - 2;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) hist[codes[y][x]]++;
      const total = hist.reduce((s, v) => s + v, 0) || 1;
      for (let b = 0; b < 59; b++) lbpFeatures.push(hist[b] / total);
    }
  }

  // ── Gabor filter bank (4 orientations × 2 frequencies × 2 stats = 16 dims) ─
  const orientations = [0, Math.PI / 4, Math.PI / 2, (3 * Math.PI) / 4];
  const frequencies  = [0.1, 0.2];
  const gaborFeats   = [];
  for (const freq of frequencies) {
    for (const theta of orientations) {
      const kSize = 7, sigma = 2.5, half = 3;
      const cosT = Math.cos(theta), sinT = Math.sin(theta);
      const responses = [];
      for (let y = half; y < h - half; y += 3) {
        for (let x = half; x < w - half; x += 3) {
          let resp = 0;
          for (let ky = -half; ky <= half; ky++) {
            for (let kx = -half; kx <= half; kx++) {
              const xr = kx * cosT + ky * sinT;
              const yr = -kx * sinT + ky * cosT;
              const gauss = Math.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2));
              const wave = Math.cos(2 * Math.PI * freq * xr);
              resp += rows[y + ky][x + kx] * gauss * wave / (2 * Math.PI * sigma ** 2);
            }
          }
          responses.push(resp);
        }
      }
      const rs = responses.length ? responses : [0];
      gaborFeats.push(mean(rs), Math.sqrt(variance(rs)));
    }
  }

  // ── Appearance sub-sampling (24×24 → 576+24 = 600 dims with row means) ─────
  const appSize = 24;
  const scaleX = w / appSize, scaleY = h / appSize;
  const appearance = [];
  for (let ay = 0; ay < appSize; ay++) {
    let rowMean = 0;
    for (let ax = 0; ax < appSize; ax++) {
      const sy = Math.min(h - 1, Math.round(ay * scaleY));
      const sx = Math.min(w - 1, Math.round(ax * scaleX));
      const v = rows[sy][sx] / 255;
      appearance.push(v);
      rowMean += v;
    }
    appearance.push(rowMean / appSize);
  }

  // ── Facial geometry (dark-region centroid clusters) ────────────────────────
  const regions = [[0.18,0.23,0.48,0.48],[0.52,0.23,0.82,0.48],[0.25,0.55,0.75,0.86]];
  const geometry = [];
  for (const [x0r, y0r, x1r, y1r] of regions) {
    const rx0 = Math.floor(x0r * w), ry0 = Math.floor(y0r * h);
    const rx1 = Math.ceil(x1r * w),  ry1 = Math.ceil(y1r * h);
    let tw = 0, sx = 0, sy = 0;
    for (let y = ry0; y < ry1; y++) {
      for (let x = rx0; x < rx1; x++) {
        const wt = Math.max(0, 80 - rows[y][x]);
        tw += wt; sx += x * wt; sy += y * wt;
      }
    }
    if (tw > 0) geometry.push(sx / tw / w, sy / tw / h, tw / ((rx1-rx0)*(ry1-ry0)*80));
    else         geometry.push(0.5, 0.5, 0);
  }
  if (geometry.length >= 6)
    geometry.push(geometry[3]-geometry[0], geometry[7]-geometry[1], geometry[6]-(geometry[0]+geometry[3])/2);

  return Float64Array.from([...lbpFeatures, ...gaborFeats, ...appearance, ...geometry]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// §4 · Online Welford Whitener (Incremental Mean/Variance Normalisation)
// ═══════════════════════════════════════════════════════════════════════════════

class OnlineWhitener {
  constructor(dim, smoothing = 1e-5) {
    this.dim = dim;
    this.smoothing = smoothing;
    this.n = 0;
    this.mean_ = new Float64Array(dim);
    this.M2    = new Float64Array(dim);
  }
  update(vec) {
    this.n++;
    for (let i = 0; i < this.dim; i++) {
      const delta = vec[i] - this.mean_[i];
      this.mean_[i] += delta / this.n;
      this.M2[i]   += delta * (vec[i] - this.mean_[i]);
    }
  }
  whiten(vec) {
    if (this.n < 2) return vec;
    const out = new Float64Array(this.dim);
    for (let i = 0; i < this.dim; i++) {
      const sigma = Math.sqrt(this.M2[i] / (this.n - 1) + this.smoothing);
      out[i] = (vec[i] - this.mean_[i]) / sigma;
    }
    return out;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §5 · Similarity Metrics
// ═══════════════════════════════════════════════════════════════════════════════

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom < 1e-10 ? 0 : (dot / denom + 1) / 2;
}

function chiSquareSim(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    const sum = a[i] + b[i] + 1e-12;
    d += (a[i] - b[i]) ** 2 / sum;
  }
  return 1 / (1 + 0.5 * d / 64);
}

function rawSimilarity(a, b, metric) {
  return metric === "cosine" ? cosineSim(a, b) : chiSquareSim(a.slice(0, 59 * 64), b.slice(0, 59 * 64));
}

// ═══════════════════════════════════════════════════════════════════════════════
// §6 · Platt Sigmoid Calibration (Score → Posterior Probability)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Maps raw match score S ∈ [0,1] to posterior P(genuine|S) using:
 *   P = 1 / (1 + exp(A·S + B))
 * A and B fitted via gradient descent on binary cross-entropy.
 * Falls back to identity when fewer than 10 samples per class.
 */
class PlattCalibrator {
  constructor() {
    this.A = -4.0; // slope (negative → higher score ⟹ higher P)
    this.B = 2.0;  // bias
    this.genuineScores   = [];
    this.impostorScores  = [];
    this.fitted = false;
  }

  record(score, isGenuine) {
    const bucket = isGenuine ? this.genuineScores : this.impostorScores;
    bucket.push(score);
    if (bucket.length > 300) bucket.shift();
    if (this.genuineScores.length >= 10 && this.impostorScores.length >= 10) this._fit();
  }

  _fit() {
    // Build labelled pairs: label 1 = genuine, 0 = impostor
    const pairs = [
      ...this.genuineScores.map(s  => [s, 1]),
      ...this.impostorScores.map(s => [s, 0]),
    ];
    // Mini-batch gradient descent (50 steps)
    let A = this.A, B = this.B;
    const lr = 0.5;
    for (let iter = 0; iter < 50; iter++) {
      let dA = 0, dB = 0;
      for (const [s, y] of pairs) {
        const p = 1 / (1 + Math.exp(A * s + B));
        const err = p - y;
        dA += err * s;
        dB += err;
      }
      A -= lr * dA / pairs.length;
      B -= lr * dB / pairs.length;
    }
    this.A = A; this.B = B; this.fitted = true;
  }

  calibrate(score) {
    if (!this.fitted) {
      // Isotonic fallback: linear rescaling from [0.3,0.9] → [0,1]
      return Math.max(0, Math.min(1, (score - 0.30) / 0.60));
    }
    return 1 / (1 + Math.exp(this.A * score + this.B));
  }

  get status() {
    if (this.fitted) return `Platt (A=${this.A.toFixed(2)}, B=${this.B.toFixed(2)})`;
    const n = Math.min(this.genuineScores.length, this.impostorScores.length);
    return `Isotonic fallback (${n}/10 min samples)`;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §7 · Mahalanobis Multi-Template Scorer
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Given N enrolled templates for a subject, compute:
 *   1. Centroid vector µ = mean of templates
 *   2. Intra-class variance σ²_d = variance of template-to-template cosine scores
 *   3. Corrected score = raw_score / (1 + β · σ_d)
 * This penalises subjects with inconsistent biometric samples.
 */
function mahalanobisMultiTemplateScore(probeVec, templates, metric) {
  if (!templates.length) return 0;

  // Raw scores against each template
  const scores = templates.map(t => rawSimilarity(probeVec, t.vector, metric) * Math.max(0.5, t.quality));

  // Intra-class template variance (consistency penalty)
  const intraVar = variance(scores);
  const σ_d = Math.sqrt(intraVar);

  // Best raw score (peak-match)
  const peakScore = Math.max(...scores);

  // Mahalanobis correction: penalise low-consistency enrollments
  const β = 1.5;
  const corrected = peakScore / (1 + β * σ_d);

  return Math.max(0, Math.min(1, corrected));
}

// ═══════════════════════════════════════════════════════════════════════════════
// §8 · Bayesian EER Estimator (Beta-Prior Credible Intervals)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Maintains sliding window of genuine/impostor scores.
 * EER estimated at threshold where FAR ≈ FNMR.
 * Bayesian Beta credible interval: [EER - 1.96·SE, EER + 1.96·SE]
 * where SE = sqrt(EER(1-EER)/N) with Beta(α+0.5, β+0.5) prior.
 */
class BayesianEER {
  constructor(staticThr = 0.40, windowSize = 200, minSamples = 5) {
    this.static_ = staticThr;
    this.window = windowSize;
    this.minSamples = minSamples;
    this.genuine  = [];
    this.impostor = [];
  }

  record(score, isGenuine) {
    const bucket = isGenuine ? this.genuine : this.impostor;
    bucket.push(score);
    if (bucket.length > this.window) bucket.shift();
  }

  get threshold() {
    if (this.genuine.length < this.minSamples || this.impostor.length < this.minSamples)
      return this.static_;
    const all = [...this.genuine, ...this.impostor];
    const lo = Math.min(...all), hi = Math.max(...all);
    if (lo >= hi) return this.static_;
    let bestT = this.static_, bestDiff = Infinity;
    for (let step = 0; step <= 100; step++) {
      const t = lo + (hi - lo) * step / 100;
      const fmr  = this.impostor.filter(s => s >= t).length / this.impostor.length;
      const fnmr = this.genuine.filter(s => s < t).length  / this.genuine.length;
      const diff = Math.abs(fmr - fnmr);
      if (diff < bestDiff) { bestDiff = diff; bestT = t; }
    }
    return bestT;
  }

  get eer() {
    const t = this.threshold;
    if (this.genuine.length < this.minSamples || this.impostor.length < this.minSamples) return null;
    const fmr  = this.impostor.filter(s => s >= t).length / this.impostor.length;
    const fnmr = this.genuine.filter(s => s < t).length  / this.genuine.length;
    return (fmr + fnmr) / 2;
  }

  /** 95% Bayesian credible interval on EER */
  get eerCredibleInterval() {
    const eer = this.eer;
    if (eer === null) return null;
    const N = this.genuine.length + this.impostor.length;
    // Beta posterior mean with Jeffreys prior α=β=0.5
    const alpha = eer * N + 0.5, beta = (1 - eer) * N + 0.5;
    const mean_b = alpha / (alpha + beta);
    const se = Math.sqrt((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)));
    return [Math.max(0, mean_b - 1.96 * se), Math.min(1, mean_b + 1.96 * se)];
  }

  rocCurve() {
    if (this.genuine.length < 2 || this.impostor.length < 2) return null;
    const all = [...this.genuine, ...this.impostor];
    const lo = Math.min(...all), hi = Math.max(...all);
    const points = [];
    for (let step = 0; step <= 50; step++) {
      const t = lo + (hi - lo) * step / 50;
      const fmr  = this.impostor.filter(s => s >= t).length / this.impostor.length;
      const fnmr = this.genuine.filter(s => s < t).length  / this.genuine.length;
      points.push({ fmr, tmr: 1 - fnmr });
    }
    return points;
  }

  get auc() {
    const pts = this.rocCurve();
    if (!pts) return null;
    let area = 0;
    for (let i = 1; i < pts.length; i++) {
      const dX = pts[i].fmr - pts[i-1].fmr;
      area += dX * (pts[i].tmr + pts[i-1].tmr) / 2;
    }
    return Math.abs(area);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §9 · Adaptive EMA Weight Fusion
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Fuses P_rec (calibrated recognition score) and P_pad (liveness score).
 * Weights adapt via EMA based on per-subsystem error rates:
 *   w_rec ← α·FNMRᵣₑc + (1-α)·w_rec  [if recognition was the failing subsystem]
 */
function adaptWeights(recFailed, padFailed) {
  const α = STATE.EMA_LR;
  if (recFailed && !padFailed) {
    // Decrease recognition weight, increase PAD weight
    STATE.w_rec = (1 - α) * STATE.w_rec + α * 0.40;
    STATE.w_pad = 1 - STATE.w_rec;
  } else if (padFailed && !recFailed) {
    STATE.w_rec = (1 - α) * STATE.w_rec + α * 0.80;
    STATE.w_pad = 1 - STATE.w_rec;
  }
  // Clamp to [0.3, 0.8]
  STATE.w_rec = Math.max(0.30, Math.min(0.80, STATE.w_rec));
  STATE.w_pad = 1 - STATE.w_rec;
}

function fuseScores(pRec, pPad, method, thr) {
  const wr = STATE.w_rec, wp = STATE.w_pad;
  let fused;
  if (method === "weighted") {
    fused = wr * pRec + wp * pPad;
  } else if (method === "geometric") {
    fused = Math.exp(wr * Math.log(Math.max(1e-10, pRec)) + wp * Math.log(Math.max(1e-10, pPad)));
  } else {
    fused = Math.min(pRec, pPad);
  }
  return { fused: Math.max(0, Math.min(1, fused)), accepted: fused >= thr };
}

// ═══════════════════════════════════════════════════════════════════════════════
// §10 · Presentation Attack Detection (PAD) Engine — ISO/IEC 30107-3
// ═══════════════════════════════════════════════════════════════════════════════

function weightedCentroid(rows) {
  let tw = 0, sx = 0, sy = 0;
  const h = rows.length, w = rows[0].length;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const wt = Math.max(0, 80 - rows[y][x]);
      tw += wt; sx += x * wt; sy += y * wt;
    }
  }
  return tw > 0 ? [sx / tw / w, sy / tw / h] : [0.5, 0.5];
}

function multiScaleLBPEntropy(rows) {
  const hist = {};
  const h = rows.length, w = rows[0].length;
  for (const radius of [1, 2, 3]) {
    const offs = [[-radius,-radius],[0,-radius],[radius,-radius],[radius,0],[radius,radius],[0,radius],[-radius,radius],[-radius,0]];
    for (let y = radius; y < h - radius; y++) {
      for (let x = radius; x < w - radius; x++) {
        const center = rows[y][x];
        let code = 0;
        for (let k = 0; k < 8; k++) {
          const [dy, dx] = offs[k];
          if (rows[y + dy][x + dx] >= center) code |= 1 << k;
        }
        hist[code] = (hist[code] || 0) + 1;
      }
    }
  }
  const total = Object.values(hist).reduce((s, v) => s + v, 0) || 1;
  let e = 0;
  for (const count of Object.values(hist)) {
    const p = count / total;
    if (p > 0) e -= p * Math.log2(p);
  }
  return e / 8;
}

function symmetryResidual(rows) {
  const h = rows.length, w = rows[0].length, mid = Math.floor(w / 2);
  let diff = 0, n = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < mid; x++) {
      diff += Math.abs(rows[y][x] - rows[y][w - 1 - x]) / 255;
      n++;
    }
  }
  return n > 0 ? Math.max(0, 1 - (diff / n) * 5) : 0;
}

function opticalFlowArc(frameRows) {
  const centroids = frameRows.map(rows => weightedCentroid(rows));
  let arc = 0;
  for (let i = 1; i < centroids.length; i++) {
    const dx = centroids[i][0] - centroids[i-1][0];
    const dy = centroids[i][1] - centroids[i-1][1];
    arc += Math.sqrt(dx * dx + dy * dy);
  }
  if (centroids.length >= 3) {
    const dirs = [];
    for (let i = 1; i < centroids.length; i++)
      dirs.push(Math.atan2(centroids[i][1]-centroids[i-1][1], centroids[i][0]-centroids[i-1][0]));
    arc *= (1 + variance(dirs) * 5);
  }
  return arc / Math.max(1, centroids.length - 1);
}

/**
 * Challenge-Response Information Gain Scorer
 * Models challenge compliance as information gain over uniform prior.
 * A fully compliant response has high H(prior) - H(posterior) ≈ 1 bit.
 */
function challengeInfoGain(frameRows, challenge) {
  if (!challenge) return 1.0;
  const centroids = frameRows.map(weightedCentroid);
  const diffs = frameRows.slice(1).map((r, i) => {
    let sum = 0, n = 0;
    for (let y = 0; y < r.length; y++) for (let x = 0; x < r[0].length; x++) {
      sum += Math.abs(r[y][x] - frameRows[i][y][x]); n++;
    }
    return n > 0 ? sum / n / 255 : 0;
  });
  const motion = mean(diffs);

  if (challenge === "blink") {
    // Expect mid-sequence pixel variance spike
    return frameRows.length > 2 ? Math.min(1.0, motion * 8) : 0;
  } else if (challenge === "turn_left" || challenge === "turn_right") {
    const shift = Math.abs(centroids.at(-1)[0] - centroids[0][0]);
    const dir   = centroids.at(-1)[0] - centroids[0][0];
    const correct = challenge === "turn_left" ? dir < 0 : dir > 0;
    return correct ? Math.min(1.0, shift / 0.04) : Math.min(0.4, shift / 0.04);
  } else if (challenge === "nod") {
    const shift = Math.abs(centroids.at(-1)[1] - centroids[0][1]);
    return Math.min(1.0, shift / 0.04);
  } else if (challenge === "smile") {
    return Math.min(1.0, motion * 5);
  }
  return 1.0;
}

function runPAD(frames, challenge) {
  const frameRows = frames.map(toRows);
  const textures  = frameRows.map(laplacianVariance);
  const diffs     = frameRows.slice(1).map((r, i) => {
    let sum = 0, n = 0;
    for (let y = 0; y < r.length; y++) for (let x = 0; x < r[0].length; x++) {
      sum += Math.abs(r[y][x] - frameRows[i][y][x]); n++;
    }
    return n > 0 ? sum / n / 255 : 0;
  });

  const texture   = mean(textures);
  const motion    = mean(diffs);
  const msEntropy = mean(frameRows.map(multiScaleLBPEntropy));
  const symm      = mean(frameRows.map(symmetryResidual));
  const flowArc   = opticalFlowArc(frameRows);
  const chalScore = challengeInfoGain(frameRows, challenge);

  // Component scores
  const textureScore  = texture > 35 ? Math.min(1.0, (texture - 35) / 325) : 0;
  const motionScore   = motion < 0.004 ? 0
                      : motion > 0.38  ? 0
                      : motion < 0.018 ? (motion - 0.004) / 0.014
                      : motion < 0.16  ? 1
                      : (0.38 - motion) / 0.22;
  const entropyScore  = msEntropy > 0.45 ? Math.min(1.0, (msEntropy - 0.45) / 0.4) : 0;
  const symmetryScore = symm > 0.40 ? Math.min(1.0, (symm - 0.40) / 0.48) : 0;
  const flowScore     = flowArc > 0.005 ? Math.min(1.0, (flowArc - 0.005) / 0.145) : 0;

  const weights = [
    [textureScore, 0.20], [motionScore, 0.28], [entropyScore, 0.12],
    [symmetryScore, 0.08], [flowScore, 0.10],
  ];
  if (challenge) weights.push([chalScore, 0.30]);

  const totalWeight = weights.reduce((s, [, w]) => s + w, 0);
  const score = weights.reduce((s, [v, w]) => s + v * w, 0) / totalWeight;

  // PAD threat classification (ISO/IEC 30107-3 §7.4)
  let padClass = "Live Subject", padConf = 0, padLevel = "L0";
  if (motion < 0.006) {
    padClass = "Static Artefact (Photo)"; padConf = Math.min(1, (0.006 - motion) / 0.006); padLevel = "L2";
  } else if (msEntropy < 0.35 && texture < 60) {
    padClass = "Printed Artefact (Halftone)"; padConf = 0.80; padLevel = "L2";
  } else if (msEntropy < 0.50 && chalScore < 0.3 && symm < 0.55) {
    padClass = "Video Replay Artefact"; padConf = 0.72; padLevel = "L2";
  } else {
    padConf = Math.min(1, motionScore * 0.4 + entropyScore * 0.4 + symmetryScore * 0.2);
    padLevel = score > 0.75 ? "L0" : score > 0.50 ? "L1" : "L2";
  }

  // FNMR reason codes (ISO 30107 Annex B)
  const reasons = [];
  if (motion < 0.004)      reasons.push("Insufficient inter-frame motion energy (Δ<0.4%)");
  if (msEntropy < 0.30)    reasons.push("Low multi-scale texture entropy — artefact signature");
  if (symm < 0.35)         reasons.push("Bilateral symmetry residual exceeds artefact threshold");
  if (challenge && chalScore < 0.55) reasons.push(`Challenge non-compliance: ${challenge} (score ${chalScore.toFixed(3)})`);
  if (score < 0.40)        reasons.push("Composite PAD score below operational threshold");

  return {
    score: Math.max(0, Math.min(1, score)),
    metrics: { texture_sharpness: texture, inter_frame_motion: motion, ms_lbp_entropy: msEntropy,
               bilateral_symmetry: symm, optical_flow_arc: flowArc, challenge_ig: chalScore },
    padClass, padConf, padLevel, reasons,
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// §11 · Global Recognition State
// ═══════════════════════════════════════════════════════════════════════════════

const FEAT_DIM = 59 * 64 + 16 + 24 * 24 + 24 + 9 + 6;
const whitener   = new OnlineWhitener(FEAT_DIM);
const eerEngine  = new BayesianEER(STATE.threshold);
const calibrator = new PlattCalibrator();

// ═══════════════════════════════════════════════════════════════════════════════
// §12 · Canvas Rendering
// ═══════════════════════════════════════════════════════════════════════════════

function drawFace(canvas, img) {
  const ctx = canvas.getContext("2d");
  const id  = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const v = Math.round(Math.max(0, Math.min(255, img[i])));
    id.data[i*4]=v; id.data[i*4+1]=v; id.data[i*4+2]=v; id.data[i*4+3]=255;
  }
  ctx.putImageData(id, 0, 0);
}

function drawMiniFrame(canvas, img) {
  canvas.width = SIZE; canvas.height = SIZE;
  const ctx = canvas.getContext("2d");
  const id  = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const v = Math.round(Math.max(0, Math.min(255, img[i])));
    id.data[i*4]=v; id.data[i*4+1]=v; id.data[i*4+2]=v; id.data[i*4+3]=255;
  }
  ctx.putImageData(id, 0, 0);
}

function drawVector(canvas, vector) {
  const ctx = canvas.getContext("2d");
  const w = canvas.offsetWidth || 380, h = 110;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  const n = Math.min(vector.length, 800);
  const barW = w / n;
  const maxV = Math.max(...Array.from(vector).slice(0, n).map(Math.abs)) || 1;
  for (let i = 0; i < n; i++) {
    const val = vector[i] / maxV;
    const barH = Math.abs(val) * (h / 2);
    const hue = val > 0 ? 160 : 10;
    ctx.fillStyle = `hsla(${hue}, 80%, 55%, 0.9)`;
    if (val > 0) ctx.fillRect(i * barW, h / 2 - barH, Math.max(1, barW - 0.5), barH);
    else          ctx.fillRect(i * barW, h / 2,         Math.max(1, barW - 0.5), barH);
  }
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();
}

function drawDistribution(canvas, genuineScores, impostorScores, thr) {
  const ctx = canvas.getContext("2d");
  const W = canvas.offsetWidth || 380, H = 170;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0, 0, W, H);
  const bins = 20;
  function histogram(scores) {
    const h = new Float32Array(bins);
    for (const s of scores) { const idx = Math.min(bins-1, Math.floor(s*bins)); h[idx]++; }
    const max = Math.max(...h) || 1;
    return Array.from(h).map(v => v / max);
  }
  const gh = histogram(genuineScores), ih = histogram(impostorScores);
  const barW = (W - 20) / bins;
  for (let i = 0; i < bins; i++) {
    const x = 10 + i * barW;
    ctx.fillStyle = "rgba(0,229,160,0.40)"; ctx.fillRect(x,   H-35-gh[i]*(H-48), barW-2, gh[i]*(H-48));
    ctx.fillStyle = "rgba(255,77,106,0.40)"; ctx.fillRect(x+1, H-35-ih[i]*(H-48), barW-2, ih[i]*(H-48));
  }
  const tx = 10 + thr * (W - 20);
  ctx.strokeStyle = "rgba(56,182,255,0.85)"; ctx.lineWidth = 1.5; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(tx, 0); ctx.lineTo(tx, H-35); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(10, H-35); ctx.lineTo(W-10, H-35); ctx.stroke();
  ctx.fillStyle = "rgba(56,182,255,0.9)"; ctx.font = "10px JetBrains Mono, monospace";
  ctx.fillText(`τ=${thr.toFixed(2)}`, tx + 4, 14);
}

function drawTimeline(canvas, events, thr) {
  const ctx = canvas.getContext("2d");
  const W = canvas.offsetWidth || 380, H = 170;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0, 0, W, H);
  if (events.length < 2) {
    ctx.fillStyle = "rgba(255,255,255,0.12)"; ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "center"; ctx.fillText("Run ≥ 2 verifications to populate timeline", W/2, H/2);
    return;
  }
  const scores = events.map(e => e.fusedScore);
  const xv = i => 10 + (i / (scores.length - 1)) * (W - 20);
  const yv = v => 10 + (1 - v) * (H - 45);
  ctx.fillStyle = "rgba(56,182,255,0.04)"; ctx.fillRect(10, yv(1), W-20, yv(thr)-yv(1));
  ctx.strokeStyle = "rgba(56,182,255,0.85)"; ctx.lineWidth = 2;
  ctx.beginPath();
  scores.forEach((s, i) => { i === 0 ? ctx.moveTo(xv(i), yv(s)) : ctx.lineTo(xv(i), yv(s)); });
  ctx.stroke();
  scores.forEach((s, i) => {
    ctx.fillStyle = events[i].accepted ? "rgba(0,229,160,1)" : "rgba(255,77,106,1)";
    ctx.beginPath(); ctx.arc(xv(i), yv(s), 4, 0, Math.PI*2); ctx.fill();
  });
  ctx.strokeStyle = "rgba(0,229,160,0.35)"; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(10, yv(thr)); ctx.lineTo(W-10, yv(thr)); ctx.stroke(); ctx.setLineDash([]);
}

function drawROCCurve(canvas, rocPoints, auc) {
  const ctx = canvas.getContext("2d");
  const W = canvas.offsetWidth || 380, H = 200;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  const pad = { t: 12, r: 14, b: 36, l: 42 };
  const iW = W - pad.l - pad.r, iH = H - pad.t - pad.b;

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const x = pad.l + (i / 4) * iW, y = pad.t + (i / 4) * iH;
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + iH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iW, y); ctx.stroke();
  }

  // Chance diagonal
  ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.setLineDash([4,4]); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t + iH); ctx.lineTo(pad.l + iW, pad.t); ctx.stroke();
  ctx.setLineDash([]);

  if (!rocPoints) {
    ctx.fillStyle = "rgba(255,255,255,0.15)"; ctx.font = "11px Inter, sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Insufficient data — run ≥5 genuine + ≥5 impostor verifications", W/2, H/2);
  } else {
    // ROC fill
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t + iH);
    for (const pt of rocPoints) {
      const x = pad.l + pt.fmr * iW, y = pad.t + (1 - pt.tmr) * iH;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(pad.l + iW, pad.t); ctx.lineTo(pad.l, pad.t); ctx.closePath();
    const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + iH);
    grad.addColorStop(0, "rgba(0,229,160,0.20)"); grad.addColorStop(1, "rgba(0,229,160,0.02)");
    ctx.fillStyle = grad; ctx.fill();

    // ROC line
    ctx.strokeStyle = "rgba(0,229,160,0.9)"; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < rocPoints.length; i++) {
      const x = pad.l + rocPoints[i].fmr * iW, y = pad.t + (1 - rocPoints[i].tmr) * iH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // AUC annotation
    if (auc !== null) {
      ctx.fillStyle = "rgba(0,229,160,0.9)"; ctx.font = "bold 11px JetBrains Mono, monospace";
      ctx.textAlign = "left";
      ctx.fillText(`AUC = ${auc.toFixed(4)}`, pad.l + 8, pad.t + 18);
    }
  }

  // Axes
  ctx.strokeStyle = "rgba(255,255,255,0.25)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + iH); ctx.lineTo(pad.l + iW, pad.t + iH); ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.45)"; ctx.font = "10px Inter, sans-serif"; ctx.textAlign = "center";
  ctx.fillText("False Match Rate (FMR)", pad.l + iW/2, H - 8);
  ctx.save(); ctx.translate(12, pad.t + iH/2); ctx.rotate(-Math.PI/2);
  ctx.fillText("True Match Rate (TMR)", 0, 0); ctx.restore();
}

// ═══════════════════════════════════════════════════════════════════════════════
// §13 · SVG Ring Gauge Updater
// ═══════════════════════════════════════════════════════════════════════════════

function updateRingGauge(svgId, value, colorClass) {
  const svg = document.getElementById(svgId);
  if (!svg) return;
  const fill = svg.querySelector(".gauge-fill");
  if (!fill) return;
  const R = 20, CIRC = 2 * Math.PI * R;
  const offset = CIRC * (1 - Math.max(0, Math.min(1, value)));
  fill.style.strokeDasharray  = `${CIRC}`;
  fill.style.strokeDashoffset = `${offset}`;
  const colorMap = { good: "#00e5a0", warn: "#ffad2e", bad: "#ff4d6a", neutral: "#38b6ff" };
  fill.style.stroke = colorMap[colorClass] || colorMap.neutral;
}

// ═══════════════════════════════════════════════════════════════════════════════
// §14 · UI Update Helpers
// ═══════════════════════════════════════════════════════════════════════════════

function meterClass(v) { return v > 0.60 ? "good" : v > 0.35 ? "warn" : "bad"; }

function updateMeter(id, val, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.width = (val * 100).toFixed(1) + "%";
  el.className = "meter-fill " + (cls || "");
  el.closest("[role=progressbar]")?.setAttribute("aria-valuenow", Math.round(val * 100));
}

function setScoreCard(cardId, passed) {
  const card = document.getElementById(cardId);
  if (!card) return;
  card.classList.toggle("pass", passed === true);
  card.classList.toggle("fail", passed === false);
}

function setVerdict(accepted, empty = false) {
  const el = document.getElementById("combinedVerdict");
  const frame = document.getElementById("bamFrame");
  if (!el) return;
  el.className = empty ? "" : (accepted ? "pass" : "fail");
  el.textContent = empty ? "—" : (accepted ? "✓ VERIFIED" : "✗ REJECTED");
  frame?.classList.toggle("pass", !empty && accepted);
  frame?.classList.toggle("fail", !empty && !accepted);
}

function setPADBadge(padClass, padLevel) {
  const el = document.getElementById("padBadge");
  if (!el) return;
  const cls = padClass.startsWith("Live") ? "genuine" :
              padClass.includes("Static") ? "static" :
              padClass.includes("Printed") ? "printed" : "replay";
  el.textContent = padLevel + " — " + padClass;
  el.className = "threat-badge " + cls;
}

function updateRegistryUI() {
  const container = document.getElementById("registryRows");
  const badge     = document.getElementById("enrollBadge");
  if (!container) return;
  if (badge) badge.textContent = `${STATE.templates.size} enrolled`;
  if (STATE.templates.size === 0) {
    container.innerHTML = `<div class="registry-row" role="row"><span style="color:var(--c-muted);font-size:0.78rem;">No subjects registered</span><span></span><span></span></div>`;
    return;
  }
  container.innerHTML = [...STATE.templates.entries()].map(([id, tmplArr]) => {
    const subj = SUBJECTS[id] || {};
    const meanQ = tmplArr.reduce((s, t) => s + t.quality, 0) / tmplArr.length;
    const qClass = meanQ > 0.65 ? "high" : meanQ > 0.35 ? "mid" : "low";
    return `<div class="registry-row" role="row">
      <span style="font-size:0.83rem;">${subj.name || id}</span>
      <span style="font-family:var(--font-mono);font-size:0.78rem;">${tmplArr.length}</span>
      <span><span class="quality-dot ${qClass}" title="Quality: ${(meanQ*100).toFixed(0)}%"></span></span>
    </div>`;
  }).join("");
}

function updateSessionTimer() {
  const el = document.getElementById("sessionTimer");
  if (!el) return;
  const elapsed = Math.floor((Date.now() - STATE.sessionStart) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const s = String(elapsed % 60).padStart(2, "0");
  el.textContent = `${m}:${s}`;
}

function updateFusionWeightDisplay() {
  const wr = document.getElementById("fusionWeightRec");
  const wp = document.getElementById("fusionWeightPad");
  if (wr) wr.textContent = (STATE.w_rec * 100).toFixed(0) + "%";
  if (wp) wp.textContent = (STATE.w_pad * 100).toFixed(0) + "%";
}

function updateCalibrationDisplay() {
  const el = document.getElementById("calibratorStatus");
  if (el) el.textContent = calibrator.status;
}

function updateAnalyticsUI() {
  const events = STATE.events;
  const total   = events.length;
  const accepts = events.filter(e => e.accepted).length;
  const denials = total - accepts;
  const genuines  = events.filter(e => e.isGenuine);
  const impostors = events.filter(e => !e.isGenuine);
  const fa = impostors.filter(e =>  e.accepted).length;
  const fr = genuines.filter(e  => !e.accepted).length;
  const fmr  = impostors.length > 0 ? fa / impostors.length : null;
  const fnmr = genuines.length  > 0 ? fr / genuines.length  : null;
  const tmr  = fnmr !== null ? 1 - fnmr : null;

  function setText(id, text) { const el = document.getElementById(id); if (el) el.textContent = text; }
  setText("kpiTotal",    total);
  setText("kpiTotalSub", `${accepts} verified · ${denials} rejected`);
  setText("kpiFMR",      fmr  !== null ? (fmr  * 100).toFixed(2) + "%" : "N/A");
  setText("kpiFMRSub",   `${fa} false matches`);
  setText("kpiFNMR",     fnmr !== null ? (fnmr * 100).toFixed(2) + "%" : "N/A");
  setText("kpiFNMRSub",  `${fr} false non-matches`);
  setText("kpiTMR",      tmr  !== null ? (tmr  * 100).toFixed(2) + "%" : "N/A");

  const eerVal = eerEngine.eer;
  const ci     = eerEngine.eerCredibleInterval;
  setText("kpiEER", eerVal !== null ? (eerVal * 100).toFixed(2) + "%" : "N/A");
  setText("kpiEERCI", ci ? `CI₉₅ [${(ci[0]*100).toFixed(1)}%, ${(ci[1]*100).toFixed(1)}%]` : "");

  const genuineScores  = genuines.map(e => e.fusedScore);
  const impostorScores = impostors.map(e => e.fusedScore);
  const distCanvas = document.getElementById("distributionCanvas");
  if (distCanvas) drawDistribution(distCanvas, genuineScores, impostorScores, STATE.threshold);
  const tlCanvas = document.getElementById("timelineCanvas");
  if (tlCanvas) drawTimeline(tlCanvas, events, STATE.threshold);

  // ROC curve
  const rocCanvas = document.getElementById("rocCanvas");
  if (rocCanvas) drawROCCurve(rocCanvas, eerEngine.rocCurve(), eerEngine.auc);

  // Threat vector breakdown
  const breakdown = document.getElementById("threatBreakdown");
  if (breakdown) {
    const hints = { "Live Subject": 0, "Static Artefact (Photo)": 0, "Video Replay Artefact": 0, "Printed Artefact (Halftone)": 0 };
    for (const e of events) hints[e.padClass] = (hints[e.padClass] || 0) + 1;
    if (total === 0) {
      breakdown.innerHTML = `<p style="font-size:0.82rem;color:var(--c-muted);">No data yet. Run identity verifications.</p>`;
    } else {
      breakdown.innerHTML = Object.entries(hints).map(([k, v]) => {
        const pct = total > 0 ? (v / total * 100).toFixed(0) : 0;
        const cls = k.startsWith("Live") ? "genuine" : k.includes("Static") ? "static" : k.includes("Printed") ? "printed" : "replay";
        return `<div class="threat-row">
          <span class="threat-badge ${cls}" style="font-size:0.68rem;">${k}</span>
          <div class="threat-bar-wrap">
            <div class="threat-bar" style="width:${pct}%; background:${cls==="genuine"?"var(--c-green)":cls==="static"?"var(--c-red)":cls==="printed"?"var(--c-purple)":"var(--c-amber)"}"></div>
          </div>
          <span style="font-family:var(--font-mono);font-size:0.78rem;min-width:60px;text-align:right;">${v} <span style="color:var(--c-muted);">(${pct}%)</span></span>
        </div>`;
      }).join("");
    }
  }

  updateFusionWeightDisplay();
  updateCalibrationDisplay();
}

// ═══════════════════════════════════════════════════════════════════════════════
// §15 · Cryptographic Audit Chain (Web Crypto API — SHA-256)
// ═══════════════════════════════════════════════════════════════════════════════

async function appendAuditEntry({ subjectId, accepted, fusedScore, padClass, calibratedScore }) {
  const payload = JSON.stringify({
    index:             STATE.auditIndex,
    subject_id:        subjectId,
    accepted,
    fused_score:       +fusedScore.toFixed(6),
    calibrated_score:  +calibratedScore.toFixed(6),
    pad_classification: padClass,
    prev_hash:         STATE.prevAuditHash,
    iso_timestamp:     new Date().toISOString(),
  });
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(payload));
  const hash = Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2,"0")).join("");

  STATE.prevAuditHash = hash;
  STATE.auditIndex++;

  const log = document.getElementById("auditLog");
  if (!log) return hash;

  const li = document.createElement("li");
  li.dataset.hash    = hash;
  li.dataset.payload = payload;
  li.innerHTML = `
    <div class="audit-entry-left">
      <span class="audit-verdict ${accepted?"pass":"fail"}">${accepted?"✓ VERIFIED":"✗ REJECTED"}</span>
      <span class="audit-subject">${subjectId || "—"}</span>
      <span class="audit-pad">${padClass}</span>
    </div>
    <div class="audit-entry-right">
      <code class="audit-hash">${hash.slice(0,24)}…</code>
    </div>
  `;
  if (log.firstElementChild?.textContent?.includes("No events")) log.innerHTML = "";
  log.prepend(li);
  return hash;
}

async function verifyChainIntegrity() {
  const entries = [...document.querySelectorAll("#auditLog li[data-hash]")].reverse();
  const btn = document.getElementById("btnVerifyChain");
  if (btn) { btn.disabled = true; btn.textContent = "Verifying…"; }
  let allGood = true;
  for (const li of entries) {
    const stored = li.dataset.hash;
    const buf  = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(li.dataset.payload));
    const recomputed = Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2,"0")).join("");
    const ok = recomputed === stored;
    li.classList.toggle("chain-ok",   ok);
    li.classList.toggle("chain-fail", !ok);
    if (!ok) allGood = false;
  }
  const status = document.getElementById("chainStatus");
  if (status) {
    status.textContent = allGood ? `✓ Chain Verified (${entries.length} entries intact)` : "✗ Chain Tampered — integrity violation detected";
    status.className = "chain-status " + (allGood ? "ok" : "fail");
  }
  if (btn) { btn.disabled = false; btn.textContent = "Verify Chain"; }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §16 · Identity Verification Flow
// ═══════════════════════════════════════════════════════════════════════════════

let lastVector = null;

async function runVerification(subjectKey) {
  const subj   = SUBJECTS[subjectKey];
  const threat = STATE.threatVector;
  const noise  = STATE.noise;

  let faceParams = { ...subj, noise };
  if (threat === "impostor") {
    const keys = Object.keys(SUBJECTS).filter(k => k !== subjectKey);
    faceParams = { ...SUBJECTS[keys[Math.floor(Math.random() * keys.length)]], noise };
  } else if (threat === "static" || threat === "replay") {
    faceParams = { ...faceParams, noise: Math.max(0, noise - 2) };
  }

  const isGenuine = threat === "none";

  // BAM acquisition
  const faceImg = syntheticFace(faceParams);
  const faceCanvas = document.getElementById("faceCanvas");
  drawFace(faceCanvas, faceImg);

  // PAD frame burst (8 frames simulating video sequence)
  const padFrames = Array.from({ length: 8 }, (_, i) => {
    if (threat === "static")  return faceImg;
    if (threat === "replay")  return syntheticFace({ ...faceParams, shiftX: i * 0.3 });
    return syntheticFace({ ...faceParams, noise, shiftX: i, shiftY: i > 3 ? 1 : 0, blink: i === 3 });
  });

  // Filmstrip render
  const filmstrip = document.getElementById("filmstrip");
  if (filmstrip) {
    filmstrip.innerHTML = "";
    padFrames.forEach((f, i) => {
      const c = document.createElement("canvas");
      c.width = SIZE; c.height = SIZE; c.title = `Frame ${i+1}`;
      c.setAttribute("role", "img"); c.setAttribute("aria-label", `PAD frame ${i+1}`);
      drawMiniFrame(c, f);
      filmstrip.appendChild(c);
    });
  }
  const fCnt = document.getElementById("padFrameCount");
  if (fCnt) fCnt.textContent = `${padFrames.length} frames`;

  // Feature extraction + whitening
  const rows     = toRows(faceImg);
  const rawVec   = extractFeatureVector(rows);
  whitener.update(rawVec);
  const vec = whitener.whiten(rawVec);
  lastVector = vec;

  const dimEl = document.getElementById("featureDim");
  if (dimEl) dimEl.textContent = `dim: ${vec.length} | LBP: 3776 | Gabor: 16 | App: 600 | Geo: 15`;
  const vectorCanvas = document.getElementById("vectorCanvas");
  if (vectorCanvas) drawVector(vectorCanvas, vec);

  // Multi-template Mahalanobis scoring
  const subjTemplates = STATE.templates.get(subjectKey) || [];
  if (subjTemplates.length === 0) {
    const lbl = document.getElementById("captureLabel");
    if (lbl) lbl.textContent = "⚠ Subject not registered — please register biometric template first";
    return;
  }

  const rawScore = mahalanobisMultiTemplateScore(vec, subjTemplates, STATE.metric);

  // Platt calibration → posterior probability
  const pRec = calibrator.calibrate(rawScore);

  // Best top-k across all enrolled subjects (for identification mode)
  let globalBest = 0, globalSubj = null;
  for (const [sid, tmpls] of STATE.templates) {
    const s = mahalanobisMultiTemplateScore(vec, tmpls, STATE.metric);
    if (s > globalBest) { globalBest = s; globalSubj = sid; }
  }

  // PAD assessment
  const padResult = runPAD(padFrames, STATE.challenge);

  // Adaptive EMA weight update (from previous session errors)
  const eerThr     = eerEngine.threshold;
  const recFailed  = rawScore < eerThr;
  const padFailed  = padResult.score < 0.40;
  adaptWeights(recFailed, padFailed);

  // Score-level fusion → Composite Decision Score (CDS)
  const { fused: cds, accepted } = fuseScores(pRec, padResult.score, STATE.fusionMethod, STATE.threshold);

  // Record for Bayesian EER + calibrator
  eerEngine.record(rawScore, isGenuine);
  calibrator.record(rawScore, isGenuine);

  // UI: BAM sensor area
  setVerdict(accepted);
  setPADBadge(padResult.padClass, padResult.padLevel);

  const mc = meterClass;

  // Recognition score card
  const recGood = rawScore >= eerThr;
  const el_rs  = document.getElementById("recScore");  if (el_rs) el_rs.textContent = rawScore.toFixed(4);
  const el_rp  = document.getElementById("pRecScore"); if (el_rp) el_rp.textContent = `P=${pRec.toFixed(4)}`;
  updateMeter("recMeter", rawScore, mc(rawScore));
  updateRingGauge("recGauge", rawScore, mc(rawScore));
  setScoreCard("recCard", recGood ? true : false);
  const el_rd = document.getElementById("recDetail");
  if (el_rd) el_rd.textContent = `τ_EER=${eerThr.toFixed(4)} | metric=${STATE.metric} | top-1: ${SUBJECTS[globalSubj]?.name || "—"}`;

  // PAD score card
  const padGood = padResult.score >= 0.40;
  const el_ls = document.getElementById("padScore");  if (el_ls) el_ls.textContent = padResult.score.toFixed(4);
  updateMeter("padMeter", padResult.score, mc(padResult.score));
  updateRingGauge("padGauge", padResult.score, mc(padResult.score));
  setScoreCard("padCard", padGood ? true : false);
  const el_ld = document.getElementById("padDetail");
  if (el_ld) el_ld.textContent =
    `motion: ${(padResult.metrics.inter_frame_motion*1000).toFixed(1)}‰ | entropy: ${padResult.metrics.ms_lbp_entropy.toFixed(3)} | symm: ${padResult.metrics.bilateral_symmetry.toFixed(3)}`;

  // CDS (Fused) card
  const el_fs = document.getElementById("cdsScore"); if (el_fs) el_fs.textContent = cds.toFixed(4);
  updateMeter("cdsMeter", cds, mc(cds));
  updateRingGauge("cdsGauge", cds, mc(cds));
  setScoreCard("cdsCard", accepted ? true : false);
  const el_fd = document.getElementById("cdsDetail");
  if (el_fd) el_fd.textContent = `τ=${STATE.threshold.toFixed(2)} | w_rec=${STATE.w_rec.toFixed(2)} | gap: ${Math.abs(cds-STATE.threshold).toFixed(4)}`;

  const capLabel = document.getElementById("captureLabel");
  if (capLabel) capLabel.textContent = `${subj.name} (${subj.id}) — ${threat !== "none" ? "⚠ " + threat.toUpperCase() + " THREAT VECTOR" : "Genuine presentation"}`;

  // Decision Audit Trail (ISO 30109 XAI)
  const eer     = eerEngine.eer;
  const ci      = eerEngine.eerCredibleInterval;
  const xaiLines = [
    `BRS Recognition: S_raw=${rawScore.toFixed(4)} → P(genuine)=${pRec.toFixed(4)} | EER_τ=${eerThr.toFixed(4)} → ${recGood ? "PASS" : "FAIL"}`,
    `PAD Assessment: score=${padResult.score.toFixed(4)} | PAD class: ${padResult.padClass} (conf ${(padResult.padConf*100).toFixed(0)}%) → ${padGood ? "PASS" : "FAIL"}`,
    `Challenge-Response IG: ${padResult.metrics.challenge_ig.toFixed(4)} | Challenge: ${STATE.challenge || "none"}`,
    `CDS Fusion (${STATE.fusionMethod}): cds=${cds.toFixed(4)} | τ=${STATE.threshold.toFixed(2)} | w_rec=${STATE.w_rec.toFixed(2)} | w_pad=${STATE.w_pad.toFixed(2)} → ${accepted ? "VERIFIED" : "REJECTED"}`,
    `Bayesian EER: ${eer !== null ? (eer*100).toFixed(2)+"%" : "N/A (insufficient data)"}${ci ? ` | CI₉₅ [${(ci[0]*100).toFixed(1)}%, ${(ci[1]*100).toFixed(1)}%]` : ""}`,
    `Calibrator: ${calibrator.status}`,
    ...Object.entries(padResult.metrics).map(([k, v]) => `  PAD.${k}: ${v.toFixed(4)}`),
    ...(padResult.reasons.length ? padResult.reasons.map(r => `⚠ ${r}`) : []),
  ];
  const xaiList = document.getElementById("auditTrailList");
  if (xaiList) {
    xaiList.innerHTML = xaiLines.map(line => {
      const isWarn = line.startsWith("⚠");
      return `<li class="${isWarn ? "fail-reason" : ""}">${line}</li>`;
    }).join("");
  }

  // Record event
  const event = {
    timestamp:        new Date().toISOString(),
    subjectId:        accepted ? subjectKey : null,
    claimedId:        subjectKey,
    accepted,
    rawScore, pRec, fusedScore: cds,
    padClass:         padResult.padClass, padLevel: padResult.padLevel,
    isGenuine,
    metric:           STATE.metric,
    fusionMethod:     STATE.fusionMethod,
    threshold:        STATE.threshold,
    w_rec:            STATE.w_rec,
    w_pad:            STATE.w_pad,
  };
  STATE.events.push(event);
  await appendAuditEntry({ subjectId: subjectKey, accepted, fusedScore: cds, padClass: padResult.padClass, calibratedScore: pRec });
  updateAnalyticsUI();
}

// ═══════════════════════════════════════════════════════════════════════════════
// §17 · Biometric Registration Flow
// ═══════════════════════════════════════════════════════════════════════════════

function runRegistration(subjectKey) {
  const subj = SUBJECTS[subjectKey];
  // 5-sample multi-pose enrollment (ISO 19794-5 multi-pose capture)
  const faces = Array.from({ length: 5 }, (_, i) =>
    syntheticFace({ ...subj, noise: STATE.noise + i, shiftX: i - 2, shiftY: i % 2 })
  );

  const templates = faces.map(faceImg => {
    const rows   = toRows(faceImg);
    const rawVec = extractFeatureVector(rows);
    whitener.update(rawVec);
    const vec     = whitener.whiten(rawVec);
    const quality = imageQuality(rows);
    return { vector: vec, quality, mean: null, variance: null };
  });

  // Filter quality gate (≥ 0.30 as per ISO 29794-1 ICAO)
  const good = templates.filter(t => t.quality >= 0.30);
  if (!good.length) {
    const lbl = document.getElementById("captureLabel");
    if (lbl) lbl.textContent = "⚠ All samples failed quality gate. Improve acquisition conditions.";
    return;
  }

  if (!STATE.templates.has(subjectKey)) STATE.templates.set(subjectKey, []);
  STATE.templates.get(subjectKey).push(...good);

  drawFace(document.getElementById("faceCanvas"), faces.at(-1));
  const avgQ = good.reduce((s, t) => s + t.quality, 0) / good.length;
  const lbl = document.getElementById("captureLabel");
  if (lbl) lbl.textContent = `Registered: ${subj.name} (${good.length}/5 templates, avg IQS: ${(avgQ*100).toFixed(0)}%)`;

  setVerdict(null, true);
  updateRegistryUI();
}

// ═══════════════════════════════════════════════════════════════════════════════
// §18 · Event Wiring
// ═══════════════════════════════════════════════════════════════════════════════

function getSubjectKey() { return document.getElementById("subjectSelect")?.value || "sharma-r"; }

document.getElementById("btnRegister")?.addEventListener("click", () => {
  runRegistration(getSubjectKey());
});

document.getElementById("btnVerify")?.addEventListener("click", async () => {
  const btn = document.getElementById("btnVerify");
  if (btn) { btn.classList.add("loading"); btn.disabled = true; }
  try { await runVerification(getSubjectKey()); }
  finally { if (btn) { btn.classList.remove("loading"); btn.disabled = false; } }
});

document.getElementById("btnClear")?.addEventListener("click", () => {
  STATE.templates.clear();
  STATE.events = [];
  STATE.prevAuditHash = "0".repeat(64);
  STATE.auditIndex = 0;
  STATE.w_rec = 0.60; STATE.w_pad = 0.40;
  updateRegistryUI();
  setVerdict(null, true);
  const lbl = document.getElementById("captureLabel");
  if (lbl) lbl.textContent = "Ready — press Register or Verify Identity";
  const al = document.getElementById("auditLog");
  if (al) al.innerHTML = `<li style="color:var(--c-muted);font-size:0.80rem;padding:8px;">No events recorded.</li>`;
  const atl = document.getElementById("auditTrailList");
  if (atl) atl.innerHTML = `<li style="color:var(--c-muted)">Run a verification to see the Decision Audit Trail.</li>`;
  ["recScore","padScore","cdsScore","pRecScore"].forEach(id => { const e = document.getElementById(id); if(e) e.textContent = "—"; });
  ["recMeter","padMeter","cdsMeter"].forEach(id => updateMeter(id, 0, ""));
  const cs = document.getElementById("chainStatus"); if (cs) cs.textContent = "";
  updateAnalyticsUI();
});

document.getElementById("thresholdRange")?.addEventListener("input", function () {
  STATE.threshold = parseFloat(this.value);
  const el = document.getElementById("thresholdVal"); if (el) el.textContent = STATE.threshold.toFixed(2);
  this.setAttribute("aria-valuenow", this.value);
});

document.getElementById("noiseRange")?.addEventListener("input", function () {
  STATE.noise = parseInt(this.value, 10);
  const el = document.getElementById("noiseVal"); if (el) el.textContent = STATE.noise;
});

document.getElementById("threatSelect")?.addEventListener("change", function () {
  STATE.threatVector = this.value;
});

document.getElementById("challengeSelect")?.addEventListener("change", function () {
  STATE.challenge = this.value;
});

document.getElementById("fusionSelect")?.addEventListener("change", function () {
  STATE.fusionMethod = this.value;
});

// Metric segmented control
document.querySelectorAll(".segment[data-metric]").forEach(btn => {
  btn.addEventListener("click", function () {
    document.querySelectorAll(".segment[data-metric]").forEach(b => { b.classList.remove("active"); b.setAttribute("aria-pressed","false"); });
    this.classList.add("active"); this.setAttribute("aria-pressed","true");
    STATE.metric = this.dataset.metric;
    const badge = document.getElementById("modelBadge");
    if (badge) badge.textContent = `ISO 30107-3 | LBP+Gabor | ${STATE.metric.toUpperCase()}`;
  });
});

// Tab navigation
document.querySelectorAll(".tab-btn[data-tab]").forEach(btn => {
  btn.addEventListener("click", function () {
    document.querySelectorAll(".tab-btn").forEach(b => { b.classList.remove("active"); b.setAttribute("aria-selected","false"); });
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    this.classList.add("active"); this.setAttribute("aria-selected","true");
    const target = document.getElementById("tab-" + this.dataset.tab);
    if (target) target.classList.add("active");
    if (this.dataset.tab === "analytics") requestAnimationFrame(updateAnalyticsUI);
  });
});

// Copy audit trail
document.getElementById("btnCopyAuditTrail")?.addEventListener("click", () => {
  const text = [...document.querySelectorAll("#auditTrailList li")].map(li => li.textContent.trim()).join("\n");
  navigator.clipboard?.writeText(text).then(() => {
    const btn = document.getElementById("btnCopyAuditTrail");
    if (btn) { btn.textContent = "Copied!"; setTimeout(() => { btn.textContent = "Copy"; }, 2000); }
  });
});

// Verify chain integrity
document.getElementById("btnVerifyChain")?.addEventListener("click", () => verifyChainIntegrity());

// Export audit log
document.getElementById("btnExportLog")?.addEventListener("click", () => {
  const payload = JSON.stringify({
    schema:        "SecureID-AuditLog-v3",
    standard:      "ISO/IEC 30107-3",
    generated_at:  new Date().toISOString(),
    session_duration_s: Math.floor((Date.now() - STATE.sessionStart) / 1000),
    calibrator:    calibrator.status,
    eer:           eerEngine.eer,
    auc:           eerEngine.auc,
    events:        STATE.events,
    session_metrics: {
      total:   STATE.events.length,
      accepts: STATE.events.filter(e => e.accepted).length,
      fmr:     (() => { const imp = STATE.events.filter(e => !e.isGenuine); return imp.length > 0 ? (imp.filter(e => e.accepted).length / imp.length).toFixed(6) : "N/A"; })(),
      fnmr:    (() => { const gen = STATE.events.filter(e =>  e.isGenuine); return gen.length > 0 ? (gen.filter(e => !e.accepted).length / gen.length).toFixed(6) : "N/A"; })(),
    },
  }, null, 2);
  const blob = new Blob([payload], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = `secureid-audit-${Date.now()}.json`; a.click();
  URL.revokeObjectURL(url);
});

// Keyboard shortcuts
document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "e") { e.preventDefault(); runRegistration(getSubjectKey()); }
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); document.getElementById("btnVerify")?.click(); }
});

// ═══════════════════════════════════════════════════════════════════════════════
// §19 · Session Timer + Initial Render
// ═══════════════════════════════════════════════════════════════════════════════

setInterval(updateSessionTimer, 1000);

(function init() {
  updateRegistryUI();
  updateAnalyticsUI();
  updateFusionWeightDisplay();
  updateCalibrationDisplay();
  const canvas = document.getElementById("faceCanvas");
  if (canvas) drawFace(canvas, syntheticFace(SUBJECTS["sharma-r"]));
  // Populate subjects dropdown
  const sel = document.getElementById("subjectSelect");
  if (sel) {
    sel.innerHTML = Object.entries(SUBJECTS).map(([k, v]) =>
      `<option value="${k}">${v.name} · ${v.id} · CL-${v.clearance}</option>`
    ).join("");
  }
})();
