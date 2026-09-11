/**
 * NHAI Datalake 3.0 — Browser Biometric Simulator v2.0
 *
 * Full JavaScript port of the Python ML pipeline:
 *   - Multi-scale LBP + Gabor feature extraction (pure JS)
 *   - Cosine similarity with online PCA whitening
 *   - Adaptive EER-based threshold
 *   - Optical flow arc + face symmetry liveness
 *   - Score-level fusion (weighted sum / geometric / min)
 *   - Session analytics: FAR, FRR, TAR, attack distribution
 *   - Canvas histogram + timeline rendering
 *   - SHA-256 cryptographic audit chain (via Web Crypto API)
 *   - XAI decision trace
 */

"use strict";

// ─── 0. Global State ─────────────────────────────────────────────────────────

const STATE = {
  metric:    "cosine",
  threshold: 0.65,
  noise:     3,
  attack:    "none",
  challenge: "blink",
  templates: new Map(),   // subject_id → [{ vector, quality }]
  events:    [],          // AuthEvent objects
  prevAuditHash: "0".repeat(64),
  auditIndex: 0,
};

const SUBJECTS = {
  "sharma-r":  { name: "Sharma, R.",  id: "NHO-0241", eyeGap: 30, mouthCurve: 1 },
  "patel-v":   { name: "Patel, V.",   id: "NHO-0392", eyeGap: 26, mouthCurve: 3 },
  "das-a":     { name: "Das, A.",     id: "NHO-0578", eyeGap: 22, mouthCurve: 5 },
  "gupta-m":   { name: "Gupta, M.",  id: "NHO-0614", eyeGap: 28, mouthCurve: 2 },
  "nair-k":    { name: "Nair, K.",   id: "NHO-0721", eyeGap: 24, mouthCurve: 6 },
};

const SIZE = 96;  // Face crop size

// ─── 1. Synthetic Face Generator ─────────────────────────────────────────────

/**
 * Generates a SIZE×SIZE grayscale face image as a Float32Array[SIZE*SIZE].
 * Each value is [0, 255].
 */
function syntheticFace({
  eyeGap = 28, mouthCurve = 2, noise = 3,
  shiftX = 0, shiftY = 0, blink = false,
  darker = false, lowEntropy = false,
} = {}) {
  const img = new Float32Array(SIZE * SIZE);

  // Background gradient
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      img[y * SIZE + x] = darker ? 55 : 88 + (y / SIZE) * 28;
    }
  }

  // Face oval
  const cx = SIZE / 2 + shiftX, cy = SIZE / 2 + shiftY;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const nx = (x - cx) / 34, ny = (y - cy) / 42;
      if (nx * nx + ny * ny < 1.0) {
        img[y * SIZE + x] = darker ? 95 : (lowEntropy ? 148 : 148 + y * 0.22);
      }
    }
  }

  // Eyes
  for (const side of [-1, 1]) {
    const ex = cx + side * (eyeGap / 2), ey = cy - 10 + shiftY;
    const eyeHeight = blink ? 1.5 : 5;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const dx = x - ex, dy = y - ey;
        if (dx * dx / 36 + dy * dy / (eyeHeight * eyeHeight) < 1) {
          img[y * SIZE + x] = darker ? 25 : 35;
        }
      }
    }
  }

  // Nose
  const nx2 = cx, ny2 = cy + 3;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (Math.abs(x - nx2) < 4 && Math.abs(y - ny2) < 6) {
        img[y * SIZE + x] = Math.max(0, img[y * SIZE + x] - 18);
      }
    }
  }

  // Mouth
  for (let x = Math.floor(cx - 14); x < Math.floor(cx + 14); x++) {
    if (x < 0 || x >= SIZE) continue;
    const dx = x - cx;
    const my = Math.floor(cy + 16 + mouthCurve * (dx / 14) ** 2);
    for (let dy = 0; dy < 4; dy++) {
      const ry = my + dy + shiftY;
      if (ry >= 0 && ry < SIZE) img[ry * SIZE + x] = darker ? 30 : 45;
    }
  }

  // Noise
  if (noise > 0 && !lowEntropy) {
    for (let i = 0; i < SIZE * SIZE; i++) {
      const rng = ((i * 17 + noise * 31) % (2 * noise + 1)) - noise;
      img[i] = Math.max(0, Math.min(255, img[i] + rng));
    }
  }

  return img;
}

/** Convert a Float32Array face to a 2D array of rows (for feature functions) */
function toRows(img) {
  const rows = [];
  for (let y = 0; y < SIZE; y++) {
    rows.push(Array.from(img.subarray(y * SIZE, (y + 1) * SIZE)));
  }
  return rows;
}

// ─── 2. Image Operations ─────────────────────────────────────────────────────

function mean(arr) {
  if (!arr.length) return 0;
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

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
      sum += lap * lap;
      n++;
    }
  }
  return n > 0 ? sum / n : 0;
}

function blurScore(rows) {
  const lv = laplacianVariance(rows);
  return Math.min(1.0, lv / 500.0);
}

function brightScore(rows) {
  let sum = 0, n = 0;
  for (const row of rows) for (const p of row) { sum += p; n++; }
  const mean_b = n > 0 ? sum / n : 128;
  const sigma = 60;
  return Math.exp(-((mean_b - 140) ** 2) / (2 * sigma ** 2));
}

function imageQuality(rows) {
  const blur = blurScore(rows);
  const brightness = brightScore(rows);
  return blur * 0.6 + brightness * 0.4;
}

// ─── 3. LBP + Gabor Feature Extraction ──────────────────────────────────────

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

function extractLBPGabor(rows, gridX = 8, gridY = 8) {
  const h = rows.length, w = rows[0].length;

  // LBP codes
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

  // Spatial grid LBP histogram
  const cellW = Math.max(1, Math.floor((w - 2) / gridX));
  const cellH = Math.max(1, Math.floor((h - 2) / gridY));
  const lbpFeatures = [];

  for (let gy = 0; gy < gridY; gy++) {
    for (let gx = 0; gx < gridX; gx++) {
      const hist = new Float64Array(59);
      const x0 = gx * cellW, y0 = gy * cellH;
      const x1 = gx < gridX - 1 ? x0 + cellW : w - 2;
      const y1 = gy < gridY - 1 ? y0 + cellH : h - 2;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          hist[codes[y][x]]++;
        }
      }
      const total = hist.reduce((s, v) => s + v, 0) || 1;
      for (let b = 0; b < 59; b++) lbpFeatures.push(hist[b] / total);
    }
  }

  // Gabor features (4 orientations × 2 frequencies)
  const orientations = [0, Math.PI / 4, Math.PI / 2, (3 * Math.PI) / 4];
  const frequencies = [0.1, 0.2];
  const gaborFeats = [];

  for (const freq of frequencies) {
    for (const theta of orientations) {
      const kSize = 7, sigma = 2.5;
      const cosT = Math.cos(theta), sinT = Math.sin(theta);
      const half = Math.floor(kSize / 2);

      let responses = [];
      for (let y = half; y < h - half; y += 3) {
        for (let x = half; x < w - half; x += 3) {
          let resp = 0;
          for (let ky = -half; ky <= half; ky++) {
            for (let kx = -half; kx <= half; kx++) {
              const xr = kx * cosT + ky * sinT;
              const yr = -kx * sinT + ky * cosT;
              const gauss = Math.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2));
              const wave = Math.cos(2 * Math.PI * freq * xr);
              const kernel = gauss * wave / (2 * Math.PI * sigma ** 2);
              resp += rows[y + ky][x + kx] * kernel;
            }
          }
          responses.push(resp);
        }
      }
      if (!responses.length) responses = [0];
      gaborFeats.push(mean(responses));
      gaborFeats.push(Math.sqrt(variance(responses)));
    }
  }

  // Appearance features (24×24 downsampled mean grid)
  const appSize = 24;
  const scaleX = w / appSize, scaleY = h / appSize;
  const appearance = [];
  for (let ay = 0; ay < appSize; ay++) {
    let rowMean = 0;
    for (let ax = 0; ax < appSize; ax++) {
      const sy = Math.min(h - 1, Math.round(ay * scaleY));
      const sx = Math.min(w - 1, Math.round(ax * scaleX));
      appearance.push(rows[sy][sx] / 255);
      rowMean += rows[sy][sx] / 255;
    }
    appearance.push(rowMean / appSize);
  }

  // Geometry features (dark-region centroids)
  const regions = [
    [0.18, 0.23, 0.48, 0.48],
    [0.52, 0.23, 0.82, 0.48],
    [0.25, 0.55, 0.75, 0.86],
  ];
  const geometry = [];
  for (const [x0r, y0r, x1r, y1r] of regions) {
    const rx0 = Math.floor(x0r * w), ry0 = Math.floor(y0r * h);
    const rx1 = Math.ceil(x1r * w), ry1 = Math.ceil(y1r * h);
    let tw = 0, sx = 0, sy = 0;
    for (let y = ry0; y < ry1; y++) {
      for (let x = rx0; x < rx1; x++) {
        const wt = Math.max(0, 80 - rows[y][x]);
        tw += wt; sx += x * wt; sy += y * wt;
      }
    }
    if (tw > 0) {
      geometry.push(sx / tw / w, sy / tw / h, tw / ((rx1 - rx0) * (ry1 - ry0) * 80));
    } else {
      geometry.push(0.5, 0.5, 0);
    }
  }
  // Eye distance, mouth offset
  if (geometry.length >= 6) {
    geometry.push(geometry[3] - geometry[0], geometry[7] - geometry[1], geometry[6] - (geometry[0] + geometry[3]) / 2);
  }

  return Float64Array.from([...lbpFeatures, ...gaborFeats, ...appearance, ...geometry]);
}

// ─── 4. Online PCA Whitener ───────────────────────────────────────────────────

class OnlinePCAWhitener {
  constructor(dim, smoothing = 1e-5) {
    this.dim = dim;
    this.smoothing = smoothing;
    this.n = 0;
    this.mean_ = new Float64Array(dim);
    this.M2 = new Float64Array(dim);
  }

  update(vec) {
    this.n++;
    for (let i = 0; i < this.dim; i++) {
      const delta = vec[i] - this.mean_[i];
      this.mean_[i] += delta / this.n;
      const delta2 = vec[i] - this.mean_[i];
      this.M2[i] += delta * delta2;
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

// ─── 5. Similarity Metrics ────────────────────────────────────────────────────

function cosineSimilarity(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  if (denom < 1e-10) return 0;
  return (dot / denom + 1) / 2; // Map [-1,1] → [0,1]
}

function chiSquareDist(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    const sum = a[i] + b[i] + 1e-12;
    d += (a[i] - b[i]) ** 2 / sum;
  }
  return 0.5 * d;
}

function similarity(a, b, metric) {
  if (metric === "cosine") return cosineSimilarity(a, b);
  const dist = chiSquareDist(a.slice(0, 59 * 64), b.slice(0, 59 * 64));
  return 1 / (1 + dist / 64);
}

// ─── 6. Adaptive EER Threshold ────────────────────────────────────────────────

class AdaptiveThreshold {
  constructor(staticThr = 0.40, windowSize = 200, minSamples = 5) {
    this.static_ = staticThr;
    this.window = windowSize;
    this.minSamples = minSamples;
    this.genuine = [];
    this.impostor = [];
  }

  record(score, isGenuine) {
    const bucket = isGenuine ? this.genuine : this.impostor;
    bucket.push(score);
    if (bucket.length > this.window) bucket.shift();
  }

  get threshold() {
    if (this.genuine.length < this.minSamples || this.impostor.length < this.minSamples) {
      return this.static_;
    }
    const all = [...this.genuine, ...this.impostor];
    const lo = Math.min(...all), hi = Math.max(...all);
    if (lo >= hi) return this.static_;

    let bestT = this.static_, bestDiff = Infinity;
    for (let step = 0; step <= 50; step++) {
      const t = lo + (hi - lo) * step / 50;
      const far = this.impostor.filter(s => s >= t).length / this.impostor.length;
      const frr = this.genuine.filter(s => s < t).length / this.genuine.length;
      const diff = Math.abs(far - frr);
      if (diff < bestDiff) { bestDiff = diff; bestT = t; }
    }
    return bestT;
  }

  get eer() {
    const t = this.threshold;
    if (this.genuine.length < this.minSamples || this.impostor.length < this.minSamples) return null;
    const far = this.impostor.filter(s => s >= t).length / this.impostor.length;
    const frr = this.genuine.filter(s => s < t).length / this.genuine.length;
    return (far + frr) / 2;
  }
}

// ─── 7. Score Fusion ──────────────────────────────────────────────────────────

function fuse(recSim, liveSim, method, thr) {
  let fused;
  if (method === "cosine") {
    fused = 0.60 * recSim + 0.40 * liveSim;
  } else if (method === "weighted") {
    // geometric (weighted product)
    fused = Math.exp(0.60 * Math.log(Math.max(1e-10, recSim)) + 0.40 * Math.log(Math.max(1e-10, liveSim)));
  } else {
    // min
    fused = Math.min(recSim, liveSim);
  }
  return { fused, accepted: fused >= thr };
}

// ─── 8. Liveness Engine ───────────────────────────────────────────────────────

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
    const offsets = [[-radius,-radius],[0,-radius],[radius,-radius],[radius,0],[radius,radius],[0,radius],[-radius,radius],[-radius,0]];
    for (let y = radius; y < h - radius; y++) {
      for (let x = radius; x < w - radius; x++) {
        const center = rows[y][x];
        let code = 0;
        for (let k = 0; k < 8; k++) {
          const [dy, dx] = offsets[k];
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

function symmetryScore(rows) {
  const h = rows.length, w = rows[0].length;
  const mid = Math.floor(w / 2);
  let diff = 0, n = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < mid; x++) {
      diff += Math.abs(rows[y][x] - rows[y][w - 1 - x]) / 255;
      n++;
    }
  }
  const asymmetry = n > 0 ? diff / n : 0;
  return Math.max(0, 1 - asymmetry * 5);
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
    for (let i = 1; i < centroids.length; i++) {
      dirs.push(Math.atan2(centroids[i][1] - centroids[i-1][1], centroids[i][0] - centroids[i-1][0]));
    }
    const dirVar = variance(dirs);
    arc *= (1 + dirVar * 5);
  }
  return arc / Math.max(1, centroids.length - 1);
}

function assessLiveness(frames, challenge) {
  const frameRows = frames.map(toRows);

  const textures = frameRows.map(laplacianVariance);
  const diffs = frameRows.slice(1).map((r, i) => {
    let sum = 0, n = 0;
    for (let y = 0; y < r.length; y++) {
      for (let x = 0; x < r[0].length; x++) {
        sum += Math.abs(r[y][x] - frameRows[i][y][x]);
        n++;
      }
    }
    return n > 0 ? sum / n / 255 : 0;
  });

  const texture = mean(textures);
  const motion = mean(diffs);
  const msEntropy = mean(frameRows.map(multiScaleLBPEntropy));
  const symm = mean(frameRows.map(symmetryScore));
  const flowArc = opticalFlowArc(frameRows);

  // Challenge score
  let challengeScore = 1.0;
  if (challenge === "blink") {
    challengeScore = frames.length > 2 ? Math.min(1.0, motion * 8) : 0;
  } else if (challenge === "turn_left" || challenge === "turn_right") {
    const centroids = frameRows.map(weightedCentroid);
    const shift = Math.abs(centroids.at(-1)[0] - centroids[0][0]);
    challengeScore = Math.min(1.0, shift / 0.05);
  } else if (challenge === "nod") {
    const centroids = frameRows.map(weightedCentroid);
    const shift = Math.abs(centroids.at(-1)[1] - centroids[0][1]);
    challengeScore = Math.min(1.0, shift / 0.04);
  } else if (challenge === "smile") {
    challengeScore = Math.min(1.0, motion * 5);
  }

  // Scores
  const textureScore = texture > 35 ? Math.min(1.0, (texture - 35) / 325) : 0;
  const motionScore = motion < 0.004 ? 0 : motion > 0.38 ? 0 : motion < 0.018 ? (motion - 0.004) / 0.014 : motion < 0.16 ? 1 : (0.38 - motion) / 0.22;
  const entropyScore = msEntropy > 0.45 ? Math.min(1.0, (msEntropy - 0.45) / 0.4) : 0;
  const symmetryS = symm > 0.40 ? Math.min(1.0, (symm - 0.40) / 0.48) : 0;
  const flowScore = flowArc > 0.005 ? Math.min(1.0, (flowArc - 0.005) / 0.145) : 0;

  const weights = [
    [textureScore, 0.20], [motionScore, 0.28], [entropyScore, 0.12],
    [symmetryS, 0.08], [flowScore, 0.10],
  ];
  if (challenge) weights.push([challengeScore, 0.30]);

  const totalWeight = weights.reduce((s, [, w]) => s + w, 0);
  const score = weights.reduce((s, [v, w]) => s + v * w, 0) / totalWeight;

  // Attack hint
  let attackHint = "genuine", attackConf = 0;
  if (motion < 0.006) {
    attackHint = "static"; attackConf = Math.min(1, (0.006 - motion) / 0.006);
  } else if (msEntropy < 0.35 && texture < 60) {
    attackHint = "printed"; attackConf = 0.75;
  } else if (msEntropy < 0.50 && challengeScore < 0.3 && symm < 0.55) {
    attackHint = "replay"; attackConf = 0.65;
  } else {
    attackConf = Math.min(1, motionScore * 0.4 + entropyScore * 0.4 + symmetryS * 0.2);
  }

  const reasons = [];
  if (motion < 0.004) reasons.push("insufficient frame-to-frame motion");
  if (msEntropy < 0.30) reasons.push("low multi-scale texture entropy — possible printed photo");
  if (symm < 0.35) reasons.push("low facial symmetry — possible spoofed image");
  if (challenge && challengeScore < 0.55) reasons.push(`challenge not satisfied: ${challenge}`);
  if (score < 0.40) reasons.push("liveness score below threshold");

  return {
    score: Math.max(0, Math.min(1, score)),
    metrics: { texture, motion, ms_lbp_entropy: msEntropy, symmetry: symm, optical_flow_arc: flowArc },
    attackHint,
    attackConf,
    reasons,
  };
}

// ─── 9. Per-recognizer State ──────────────────────────────────────────────────

const whitener = new OnlinePCAWhitener(59 * 64 + 16 + 24 * 24 + 24 + 9 + 6);
const adaptiveThr = new AdaptiveThreshold(STATE.threshold);

// ─── 10. Canvas Rendering ─────────────────────────────────────────────────────

function drawFace(canvas, faceImg) {
  const ctx = canvas.getContext("2d");
  const idata = ctx.createImageData(SIZE, SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    const v = Math.round(Math.max(0, Math.min(255, faceImg[i])));
    idata.data[i * 4]     = v;
    idata.data[i * 4 + 1] = v;
    idata.data[i * 4 + 2] = v;
    idata.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(idata, 0, 0);
}

function drawMiniFrame(canvas, faceImg) {
  const size = faceImg.length === SIZE * SIZE ? SIZE : Math.sqrt(faceImg.length);
  const ctx = canvas.getContext("2d");
  canvas.width = size; canvas.height = size;
  const idata = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i++) {
    const v = Math.round(Math.max(0, Math.min(255, faceImg[i])));
    idata.data[i * 4] = v;
    idata.data[i * 4 + 1] = v;
    idata.data[i * 4 + 2] = v;
    idata.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(idata, 0, 0);
}

function drawVector(canvas, vector) {
  const ctx = canvas.getContext("2d");
  const w = canvas.offsetWidth || 360, h = 100;
  canvas.width = w; canvas.height = h;
  ctx.clearRect(0, 0, w, h);

  const n = Math.min(vector.length, 800);
  const barW = w / n;
  const maxV = Math.max(...Array.from(vector).slice(0, n).map(Math.abs)) || 1;

  for (let i = 0; i < n; i++) {
    const val = vector[i] / maxV;
    const barH = Math.abs(val) * (h / 2);
    const hue = val > 0 ? 160 : 0;  // green positive, red negative
    const sat = 70 + Math.abs(val) * 30;
    ctx.fillStyle = `hsla(${hue}, ${sat}%, 55%, 0.85)`;
    if (val > 0) ctx.fillRect(i * barW, h / 2 - barH, Math.max(1, barW - 0.5), barH);
    else         ctx.fillRect(i * barW, h / 2,         Math.max(1, barW - 0.5), barH);
  }

  // Centre line
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
}

function drawDistribution(canvas, genuineScores, impostorScores, thr) {
  const ctx = canvas.getContext("2d");
  const W = canvas.offsetWidth || 360, H = 160;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  const bins = 20;
  function histogram(scores) {
    const h = new Float32Array(bins);
    for (const s of scores) {
      const idx = Math.min(bins - 1, Math.floor(s * bins));
      h[idx]++;
    }
    const max = Math.max(...h) || 1;
    return Array.from(h).map(v => v / max);
  }

  const gh = histogram(genuineScores);
  const ih = histogram(impostorScores);
  const barW = (W - 20) / bins;

  // Bars
  for (let i = 0; i < bins; i++) {
    const x = 10 + i * barW;
    ctx.fillStyle = "rgba(0,229,160,0.35)";
    ctx.fillRect(x, H - 30 - gh[i] * (H - 40), barW - 2, gh[i] * (H - 40));
    ctx.fillStyle = "rgba(255,77,106,0.35)";
    ctx.fillRect(x + 1, H - 30 - ih[i] * (H - 40), barW - 2, ih[i] * (H - 40));
  }

  // Threshold line
  const tx = 10 + thr * (W - 20);
  ctx.strokeStyle = "rgba(56,182,255,0.7)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(tx, 0); ctx.lineTo(tx, H - 30); ctx.stroke();
  ctx.setLineDash([]);

  // Axis
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(10, H - 30); ctx.lineTo(W - 10, H - 30); ctx.stroke();
}

function drawTimeline(canvas, events, thr) {
  const ctx = canvas.getContext("2d");
  const W = canvas.offsetWidth || 360, H = 160;
  canvas.width = W; canvas.height = H;
  ctx.clearRect(0, 0, W, H);

  if (events.length < 2) return;

  const scores = events.map(e => e.fusedScore);
  const x = i => 10 + (i / (scores.length - 1)) * (W - 20);
  const y = v => 10 + (1 - v) * (H - 40);

  // Threshold band
  ctx.fillStyle = "rgba(56,182,255,0.05)";
  ctx.fillRect(10, y(1), W - 20, y(thr) - y(1));

  // Score line
  ctx.strokeStyle = "rgba(56,182,255,0.8)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  scores.forEach((s, i) => { i === 0 ? ctx.moveTo(x(i), y(s)) : ctx.lineTo(x(i), y(s)); });
  ctx.stroke();

  // Dots
  scores.forEach((s, i) => {
    ctx.fillStyle = events[i].accepted ? "rgba(0,229,160,1)" : "rgba(255,77,106,1)";
    ctx.beginPath(); ctx.arc(x(i), y(s), 3.5, 0, 2 * Math.PI); ctx.fill();
  });

  // Threshold line
  ctx.strokeStyle = "rgba(0,229,160,0.3)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(10, y(thr)); ctx.lineTo(W - 10, y(thr)); ctx.stroke();
  ctx.setLineDash([]);
}

// ─── 11. UI Updates ───────────────────────────────────────────────────────────

function updateMeter(id, val, className) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.width = (val * 100).toFixed(1) + "%";
  el.className = "meter-fill " + (className || "");
  el.closest("[role=progressbar]")?.setAttribute("aria-valuenow", Math.round(val * 100));
}

function updateScoreCard(cardId, score, passed) {
  const card = document.getElementById(cardId);
  if (!card) return;
  card.classList.toggle("pass", passed);
  card.classList.toggle("fail", score !== null && !passed);
}

function setVerdict(accepted, empty = false) {
  const el = document.getElementById("combinedVerdict");
  const frame = document.getElementById("cameraFrame");
  if (!el || !frame) return;
  el.className = empty ? "" : (accepted ? "pass" : "fail");
  el.textContent = empty ? "—" : (accepted ? "✓ GRANTED" : "✗ DENIED");
  frame.classList.toggle("pass", !empty && accepted);
  frame.classList.toggle("fail", !empty && !accepted);
}

function setAttackBadge(hint) {
  const el = document.getElementById("attackBadge");
  if (!el) return;
  el.textContent = hint || "—";
  el.className = "attack-badge " + (hint || "");
}

function updateRegistryUI() {
  const container = document.getElementById("registryRows");
  const badge = document.getElementById("enrollBadge");
  if (!container) return;

  const total = [...STATE.templates.values()].reduce((s, arr) => s + arr.length, 0);
  if (badge) badge.textContent = `${STATE.templates.size} enrolled`;

  if (STATE.templates.size === 0) {
    container.innerHTML = `<div class="registry-row" role="row"><span style="color:var(--c-muted); font-size:0.78rem;">No subjects enrolled</span><span></span><span></span></div>`;
    return;
  }

  container.innerHTML = [...STATE.templates.entries()].map(([id, tmplArr]) => {
    const subj = SUBJECTS[id] || {};
    const meanQ = tmplArr.reduce((s, t) => s + t.quality, 0) / tmplArr.length;
    const qClass = meanQ > 0.65 ? "high" : meanQ > 0.35 ? "mid" : "low";
    return `<div class="registry-row" role="row">
      <span style="font-size:0.84rem;">${subj.name || id}</span>
      <span style="font-family:var(--font-mono); font-size:0.78rem;">${tmplArr.length}</span>
      <span><span class="quality-dot ${qClass}" title="Quality: ${(meanQ * 100).toFixed(0)}%"></span></span>
    </div>`;
  }).join("");
}

function updateAnalyticsUI() {
  const events = STATE.events;
  const total = events.length;
  const accepts = events.filter(e => e.accepted).length;
  const denials = total - accepts;

  const genuines = events.filter(e => e.isGenuine);
  const impostors = events.filter(e => !e.isGenuine);
  const fa = impostors.filter(e => e.accepted).length;
  const fr = genuines.filter(e => !e.accepted).length;

  const far = impostors.length > 0 ? fa / impostors.length : null;
  const frr = genuines.length > 0 ? fr / genuines.length : null;
  const tar = frr !== null ? 1 - frr : null;

  document.getElementById("kpiTotal").textContent = total;
  document.getElementById("kpiTotalSub").textContent = `${accepts} accepts · ${denials} denials`;
  document.getElementById("kpiFAR").textContent = far !== null ? (far * 100).toFixed(1) + "%" : "N/A";
  document.getElementById("kpiFARSub").textContent = `${fa} false accepts`;
  document.getElementById("kpiFRR").textContent = frr !== null ? (frr * 100).toFixed(1) + "%" : "N/A";
  document.getElementById("kpiFRRSub").textContent = `${fr} false rejects`;
  document.getElementById("kpiTAR").textContent = tar !== null ? (tar * 100).toFixed(1) + "%" : "N/A";

  // Score distributions
  const genuineScores = genuines.map(e => e.fusedScore);
  const impostorScores = impostors.map(e => e.fusedScore);
  const distCanvas = document.getElementById("distributionCanvas");
  if (distCanvas) drawDistribution(distCanvas, genuineScores, impostorScores, STATE.threshold);

  const tlCanvas = document.getElementById("timelineCanvas");
  if (tlCanvas) drawTimeline(tlCanvas, events, STATE.threshold);

  // Attack breakdown
  const breakdown = document.getElementById("attackBreakdown");
  if (breakdown) {
    const hints = { static: 0, replay: 0, printed: 0, genuine: 0 };
    for (const e of events) hints[e.attackHint] = (hints[e.attackHint] || 0) + 1;
    if (total === 0) {
      breakdown.innerHTML = `<p style="font-size:0.82rem; color:var(--c-muted);">No data yet. Run authentications.</p>`;
    } else {
      breakdown.innerHTML = Object.entries(hints).map(([k, v]) => {
        const pct = total > 0 ? (v / total * 100).toFixed(0) : 0;
        return `<div style="display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:1px solid var(--c-border);">
          <span class="attack-badge ${k}" style="font-size:0.70rem;">${k}</span>
          <span style="font-family:var(--font-mono); font-size:0.80rem; color:var(--c-ink);">${v} <span style="color:var(--c-muted);">(${pct}%)</span></span>
        </div>`;
      }).join("");
    }
  }
}

// ─── 12. Audit Chain ─────────────────────────────────────────────────────────

async function appendAuditEntry({ subjectId, accepted, fusedScore, attackHint }) {
  const payload = JSON.stringify({
    index: STATE.auditIndex,
    subject_id: subjectId,
    accepted,
    fused_score: fusedScore,
    attack_hint: attackHint,
    prev_hash: STATE.prevAuditHash,
  });
  const msgBuffer = new TextEncoder().encode(payload);
  const hashBuffer = await crypto.subtle.digest("SHA-256", msgBuffer);
  const hashHex = Array.from(new Uint8Array(hashBuffer)).map(b => b.toString(16).padStart(2, "0")).join("");

  STATE.prevAuditHash = hashHex;
  STATE.auditIndex++;

  const log = document.getElementById("auditLog");
  if (!log) return hashHex;

  const li = document.createElement("li");
  li.setAttribute("role", "listitem");
  li.innerHTML = `
    <span>${accepted ? "✓" : "✗"} <strong style="color:${accepted ? "var(--c-green)" : "var(--c-red)"};">${subjectId || "—"}</strong> — ${attackHint || "genuine"}</span>
    <strong style="font-size:0.68rem; color:var(--c-muted);">${hashHex.slice(0, 16)}…</strong>
  `;
  if (log.firstElementChild?.textContent?.includes("No events")) log.innerHTML = "";
  log.prepend(li);

  return hashHex;
}

// ─── 13. Authentication Flow ──────────────────────────────────────────────────

let lastVector = null;

async function runAuthentication(subjectKey) {
  const subj = SUBJECTS[subjectKey];
  const attack = STATE.attack;
  const noise = STATE.noise;

  // Determine face params based on attack
  let faceParams = { ...subj, noise };
  if (attack === "impostor") {
    const keys = Object.keys(SUBJECTS).filter(k => k !== subjectKey);
    const impostor = SUBJECTS[keys[Math.floor(Math.random() * keys.length)]];
    faceParams = { ...impostor, noise };
  } else if (attack === "static" || attack === "replay") {
    faceParams = { ...faceParams, noise: Math.max(0, noise - 2) };
  }

  const isGenuine = attack === "none";

  // Generate face
  const faceImg = syntheticFace(faceParams);
  const faceCanvas = document.getElementById("faceCanvas");
  drawFace(faceCanvas, faceImg);

  // Generate liveness frames
  const liveness_frames = Array.from({ length: 5 }, (_, i) => {
    if (attack === "static") return faceImg;
    if (attack === "replay") return syntheticFace({ ...faceParams, noise: Math.max(0, noise - 1), shiftX: i * 0.3 });
    return syntheticFace({ ...faceParams, noise, shiftX: i, shiftY: i > 2 ? 1 : 0, blink: i === 2 });
  });

  // Draw filmstrip
  const filmstrip = document.getElementById("filmstrip");
  filmstrip.innerHTML = "";
  liveness_frames.forEach((f, i) => {
    const c = document.createElement("canvas");
    c.width = SIZE; c.height = SIZE;
    c.title = `Frame ${i + 1}`;
    c.setAttribute("role", "img");
    c.setAttribute("aria-label", `Liveness frame ${i + 1}`);
    drawMiniFrame(c, f);
    filmstrip.appendChild(c);
  });
  document.getElementById("livenessFrameCount").textContent = `${liveness_frames.length} frames`;

  // Recognition
  const rows = toRows(faceImg);
  const rawVec = extractLBPGabor(rows);
  whitener.update(rawVec);
  const vec = whitener.whiten(rawVec);
  lastVector = vec;

  document.getElementById("featureDim").textContent = `dim: ${vec.length}`;
  const vectorCanvas = document.getElementById("vectorCanvas");
  if (vectorCanvas) drawVector(vectorCanvas, vec);

  let recSim = 0, recAccepted = false, topCandidate = null;
  const tmplList = [...STATE.templates.values()].flat();
  const subjTemplates = STATE.templates.get(subjectKey) || [];

  if (subjTemplates.length === 0) {
    document.getElementById("captureLabel").textContent = "⚠ Subject not enrolled — please enroll first";
    return;
  }

  // Score against enrolled subject
  let bestSim = 0;
  for (const tmpl of subjTemplates) {
    const s = similarity(vec, tmpl.vector, STATE.metric) * Math.max(0.5, tmpl.quality);
    if (s > bestSim) bestSim = s;
  }
  recSim = bestSim;

  // Active threshold
  const activeThr = adaptiveThr.threshold;
  recAccepted = recSim >= activeThr;

  // Also check top candidate across all subjects
  let globalBest = 0, globalSubj = null;
  for (const [sid, tmpls] of STATE.templates) {
    for (const t of tmpls) {
      const s = similarity(vec, t.vector, STATE.metric);
      if (s > globalBest) { globalBest = s; globalSubj = sid; }
    }
  }
  topCandidate = globalSubj;

  // Liveness
  const liveness = assessLiveness(liveness_frames, STATE.challenge);

  // Fusion
  const { fused, accepted } = fuse(recSim, liveness.score, STATE.metric, STATE.threshold);

  // Update adaptive threshold
  adaptiveThr.record(recSim, isGenuine);
  if (accepted && isGenuine) adaptiveThr.record(fused, true);
  else if (!accepted && !isGenuine) adaptiveThr.record(fused, false);

  // Update score cards
  const recGood = recSim >= activeThr;
  const liveGood = liveness.score >= 0.40;
  const meterClass = v => v > 0.60 ? "good" : v > 0.35 ? "warn" : "bad";

  document.getElementById("recScore").textContent = recSim.toFixed(3);
  updateMeter("recMeter", recSim, meterClass(recSim));
  updateScoreCard("recCard", recSim, recGood);
  document.getElementById("recDetail").textContent = `thr: ${activeThr.toFixed(3)} | metric: ${STATE.metric} | top: ${SUBJECTS[topCandidate]?.name || "—"}`;

  document.getElementById("liveScore").textContent = liveness.score.toFixed(3);
  updateMeter("liveMeter", liveness.score, meterClass(liveness.score));
  updateScoreCard("liveCard", liveness.score, liveGood);
  document.getElementById("liveDetail").textContent =
    `motion: ${(liveness.metrics.motion * 1000).toFixed(1)}‰ | entropy: ${liveness.metrics.ms_lbp_entropy.toFixed(3)} | symm: ${liveness.metrics.symmetry.toFixed(3)}`;

  document.getElementById("fusedScore").textContent = fused.toFixed(3);
  updateMeter("fusedMeter", fused, meterClass(fused));
  updateScoreCard("fusedCard", fused, accepted);
  document.getElementById("fusedDetail").textContent = `thr: ${STATE.threshold.toFixed(2)} | ${accepted ? "ACCEPT" : "DENY"} | gap: ${Math.abs(fused - STATE.threshold).toFixed(3)}`;

  setVerdict(accepted);
  setAttackBadge(liveness.attackHint);

  const captureLabel = document.getElementById("captureLabel");
  captureLabel.textContent = `${subj.name} (${subj.id}) — ${attack !== "none" ? "⚠ " + attack.toUpperCase() + " ATTACK" : "genuine"}`;

  // XAI trace
  const xaiLines = [
    `Recognition: ${recSim.toFixed(4)} vs thr ${activeThr.toFixed(4)} → ${recGood ? "PASS" : "FAIL"}`,
    `Liveness: ${liveness.score.toFixed(4)} vs thr 0.40 → ${liveGood ? "PASS" : "FAIL"}`,
    `Attack hint: ${liveness.attackHint} (conf ${(liveness.attackConf * 100).toFixed(0)}%)`,
    `Fusion (${STATE.metric}): ${fused.toFixed(4)} vs thr ${STATE.threshold.toFixed(2)} → ${accepted ? "ACCEPT" : "DENY"}`,
    `EER threshold: ${adaptiveThr.eer !== null ? adaptiveThr.eer.toFixed(4) : "N/A (insufficient data)"}`,
    ...Object.entries(liveness.metrics).map(([k, v]) => `  ${k}: ${v.toFixed(4)}`),
    ...(liveness.reasons.length ? liveness.reasons.map(r => `⚠ ${r}`) : []),
  ];
  const xaiList = document.getElementById("xaiList");
  xaiList.innerHTML = xaiLines.map((line, i) => {
    const isWarn = line.startsWith("⚠");
    return `<li class="${isWarn ? "fail-reason" : ""}">${line}</li>`;
  }).join("");

  // Record event
  const authEvent = {
    timestamp: new Date().toISOString(),
    subjectId: accepted ? subjectKey : null,
    claimedId: subjectKey,
    accepted,
    recSim, livenessScore: liveness.score, fusedScore: fused,
    attackHint: liveness.attackHint,
    isGenuine,
    metric: STATE.metric,
    threshold: STATE.threshold,
  };
  STATE.events.push(authEvent);
  await appendAuditEntry({ subjectId: subjectKey, accepted, fusedScore: fused, attackHint: liveness.attackHint });

  updateAnalyticsUI();
}

function runEnrollment(subjectKey) {
  const subj = SUBJECTS[subjectKey];
  const faces = Array.from({ length: 3 }, (_, i) =>
    syntheticFace({ ...subj, noise: STATE.noise + i, shiftX: i })
  );

  const templates = faces.map(faceImg => {
    const rows = toRows(faceImg);
    const rawVec = extractLBPGabor(rows);
    whitener.update(rawVec);
    const vec = whitener.whiten(rawVec);
    const quality = imageQuality(rows);
    return { vector: vec, quality };
  });

  if (!STATE.templates.has(subjectKey)) STATE.templates.set(subjectKey, []);
  STATE.templates.get(subjectKey).push(...templates);

  // Show last enrolled face
  drawFace(document.getElementById("faceCanvas"), faces.at(-1));
  document.getElementById("captureLabel").textContent =
    `Enrolled: ${subj.name} (${templates.length} templates, avg quality: ${(templates.reduce((s,t) => s+t.quality,0)/templates.length*100).toFixed(0)}%)`;

  setVerdict(null, true);
  updateRegistryUI();
}

// ─── 14. Event Wiring ─────────────────────────────────────────────────────────

function getSubjectKey() { return document.getElementById("subjectSelect")?.value || "sharma-r"; }

document.getElementById("btnEnroll").addEventListener("click", () => {
  runEnrollment(getSubjectKey());
});

document.getElementById("btnAuthenticate").addEventListener("click", async () => {
  const btn = document.getElementById("btnAuthenticate");
  btn.classList.add("loading");
  btn.disabled = true;
  try {
    await runAuthentication(getSubjectKey());
  } finally {
    btn.classList.remove("loading");
    btn.disabled = false;
  }
});

document.getElementById("btnClear").addEventListener("click", () => {
  STATE.templates.clear();
  STATE.events = [];
  STATE.prevAuditHash = "0".repeat(64);
  STATE.auditIndex = 0;
  updateRegistryUI();
  setVerdict(null, true);
  document.getElementById("captureLabel").textContent = "Ready — press Enroll or Authenticate";
  document.getElementById("auditLog").innerHTML = `<li style="color:var(--c-muted); font-size:0.80rem; padding:8px;">No events recorded.</li>`;
  document.getElementById("xaiList").innerHTML = `<li style="color:var(--c-muted)">Run an authentication to see the decision trace.</li>`;
  ["recScore","liveScore","fusedScore"].forEach(id => { const el = document.getElementById(id); if (el) el.textContent = "—"; });
  ["recMeter","liveMeter","fusedMeter"].forEach(id => updateMeter(id, 0, ""));
  updateAnalyticsUI();
});

document.getElementById("thresholdRange").addEventListener("input", function () {
  STATE.threshold = parseFloat(this.value);
  document.getElementById("thresholdVal").textContent = STATE.threshold.toFixed(2);
  this.setAttribute("aria-valuenow", this.value);
});

document.getElementById("noiseRange").addEventListener("input", function () {
  STATE.noise = parseInt(this.value, 10);
  document.getElementById("noiseVal").textContent = STATE.noise;
});

document.getElementById("attackSelect").addEventListener("change", function () {
  STATE.attack = this.value;
});

document.getElementById("challengeSelect").addEventListener("change", function () {
  STATE.challenge = this.value;
});

// Metric segmented control
document.querySelectorAll(".segment").forEach(btn => {
  btn.addEventListener("click", function () {
    document.querySelectorAll(".segment").forEach(b => {
      b.classList.remove("active");
      b.setAttribute("aria-pressed", "false");
    });
    this.classList.add("active");
    this.setAttribute("aria-pressed", "true");
    STATE.metric = this.dataset.metric;
    const badge = document.getElementById("modelBadge");
    if (badge) badge.textContent = `Model: LBPH + Gabor (${STATE.metric})`;
  });
});

// Tab navigation
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", function () {
    document.querySelectorAll(".tab-btn").forEach(b => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

    this.classList.add("active");
    this.setAttribute("aria-selected", "true");
    const target = document.getElementById("tab-" + this.dataset.tab);
    if (target) target.classList.add("active");

    if (this.dataset.tab === "analytics") {
      requestAnimationFrame(updateAnalyticsUI);
    }
  });
});

// Copy XAI
document.getElementById("btnCopyXAI").addEventListener("click", () => {
  const text = [...document.querySelectorAll("#xaiList li")].map(li => li.textContent.trim()).join("\n");
  navigator.clipboard?.writeText(text).then(() => {
    const btn = document.getElementById("btnCopyXAI");
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = "Copy"; }, 2000);
  });
});

// Export audit log
document.getElementById("btnExportLog").addEventListener("click", () => {
  const payload = JSON.stringify({
    events: STATE.events,
    generated_at: new Date().toISOString(),
    session_metrics: {
      total: STATE.events.length,
      accepts: STATE.events.filter(e => e.accepted).length,
      far: (() => {
        const imp = STATE.events.filter(e => !e.isGenuine);
        return imp.length > 0 ? (imp.filter(e => e.accepted).length / imp.length).toFixed(4) : "N/A";
      })(),
    },
  }, null, 2);
  const blob = new Blob([payload], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `nhai-audit-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// ─── 15. Keyboard shortcuts ───────────────────────────────────────────────────

document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "e") { e.preventDefault(); runEnrollment(getSubjectKey()); }
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); document.getElementById("btnAuthenticate").click(); }
});

// ─── 16. Initial render ───────────────────────────────────────────────────────

(function init() {
  updateRegistryUI();
  updateAnalyticsUI();

  // Draw a placeholder face on load
  const faceImg = syntheticFace(SUBJECTS["sharma-r"]);
  const canvas = document.getElementById("faceCanvas");
  if (canvas) drawFace(canvas, faceImg);
})();
