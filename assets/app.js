/**
 * Datalake 3.0 Offline Biometrics Demo — app.js
 * Industry-grade rewrite: encapsulated state, XSS-safe DOM, fixed liveness
 * logic, ResizeObserver canvas, rolling audit log, ARIA live regions.
 */

// ─── Configuration ────────────────────────────────────────────────────────────

const CONFIG = Object.freeze({
  SIZE: 96,
  RECOGNITION_THRESHOLD: 0.78,
  LIVENESS_THRESHOLD: 0.62,
  CHALLENGE_PASS_THRESHOLD: 0.55,
  BASE_NOISE: 3,
  LBP_GRID: 8,
  LBP_BINS: 59,
  LIVENESS_WEIGHTS: Object.freeze({
    texture: 0.24,
    entropy: 0.16,
    motion: 0.34,
    exposureVariance: 0.12,
    challenge: 0.30,
    normalizer: 1.16,
  }),
  DISTANCE_WEIGHTS: Object.freeze({ lbp: 0.52, appearance: 0.95, geometry: 3.2 }),
  AUDIT_MAX_ENTRIES: 10,
});

// ─── Subject profiles ─────────────────────────────────────────────────────────

const SUBJECTS = Object.freeze({
  "operator-a": Object.freeze({ label: "Operator A", eyeGap: 30, mouthCurve:  0, noseOffset:  0, brow:  0 }),
  "operator-b": Object.freeze({ label: "Operator B", eyeGap: 20, mouthCurve:  8, noseOffset:  3, brow:  2 }),
  "operator-c": Object.freeze({ label: "Operator C", eyeGap: 36, mouthCurve: -5, noseOffset: -4, brow: -2 }),
});

// ─── Math utilities ───────────────────────────────────────────────────────────

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function mean(values) {
  return values.length
    ? values.reduce(function(sum, v) { return sum + v; }, 0) / values.length
    : 0;
}

function variance(values) {
  var avg = mean(values);
  return values.length
    ? mean(values.map(function(v) { return (v - avg) * (v - avg); }))
    : 0;
}

// ─── Image utilities ──────────────────────────────────────────────────────────

function createMatrix(fill) {
  fill = fill === undefined ? 235 : fill;
  var SIZE = CONFIG.SIZE;
  var out = [];
  for (var y = 0; y < SIZE; y++) {
    var row = [];
    for (var x = 0; x < SIZE; x++) row.push(fill);
    out.push(row);
  }
  return out;
}

/**
 * Generates a synthetic grayscale face for a given subject.
 * @param {string} subjectId
 * @param {object} [options]
 * @returns {Array} SIZE×SIZE pixel matrix
 */
function syntheticFace(subjectId, options) {
  options = options || {};
  var profile = SUBJECTS[subjectId];
  if (!profile) {
    console.error('[syntheticFace] Unknown subjectId: "' + subjectId + '"');
    return createMatrix(128);
  }

  var SIZE     = CONFIG.SIZE;
  var lighting = options.lighting !== undefined ? options.lighting : 0;
  var noise    = options.noise    !== undefined ? options.noise    : 0;
  var blink    = Boolean(options.blink);
  var image    = createMatrix(235 + lighting);
  var cx = Math.round(SIZE / 2 + (options.shiftX || 0));
  var cy = Math.round(SIZE / 2 + (options.shiftY || 0));

  // Face oval
  for (var y = 0; y < SIZE; y++) {
    for (var x = 0; x < SIZE; x++) {
      var nx = (x - cx) / 32;
      var ny = (y - cy) / 40;
      if (nx * nx + ny * ny <= 1) image[y][x] = 171 + lighting;
    }
  }

  // Eyes
  var eyeY = cy - 13 + profile.brow;
  var eyeXList = [cx - profile.eyeGap / 2, cx + profile.eyeGap / 2];
  for (var ei = 0; ei < eyeXList.length; ei++) {
    var eyeX = eyeXList[ei];
    for (var ey = Math.floor(eyeY - 4); ey <= eyeY + 4; ey++) {
      for (var ex = Math.floor(eyeX - 7); ex <= eyeX + 7; ex++) {
        if (ey < 0 || ey >= SIZE || ex < 0 || ex >= SIZE) continue;
        if (blink) {
          if (Math.abs(ey - eyeY) <= 1) image[ey][ex] = 58;
        } else {
          var enx = (ex - eyeX) / 6;
          var eny = (ey - eyeY) / 3;
          if (enx * enx + eny * eny <= 1) image[ey][ex] = 42;
        }
      }
    }
  }

  // Nose
  var noseX = cx + profile.noseOffset;
  for (var offset = 0; offset < 12; offset++) {
    var ny2 = cy - 5 + offset;
    var nx2 = noseX + Math.floor(offset / 3);
    if (ny2 >= 0 && ny2 < SIZE && nx2 >= 0 && nx2 < SIZE) image[ny2][nx2] = 116;
  }

  // Mouth
  var mouthY = cy + 22;
  for (var dx = -17; dx <= 17; dx++) {
    var my = mouthY + Math.floor((dx * dx) / 72) + profile.mouthCurve;
    var mx = cx + dx;
    if (my >= 0 && my < SIZE && mx >= 0 && mx < SIZE) image[my][mx] = 82;
  }

  // Noise
  if (noise) {
    for (var ny3 = 0; ny3 < SIZE; ny3++) {
      for (var nx3 = 0; nx3 < SIZE; nx3++) {
        var delta = ((nx3 * 17 + ny3 * 31 + noise * 13) % (2 * noise + 1)) - noise;
        image[ny3][nx3] = clamp(image[ny3][nx3] + delta, 0, 255);
      }
    }
  }

  return image.map(function(row) { return row.map(function(p) { return clamp(p, 0, 255); }); });
}

function resizeImage(image, width, height) {
  var srcH   = image.length;
  var srcW   = image[0].length;
  var xScale = (srcW - 1) / Math.max(1, width  - 1);
  var yScale = (srcH - 1) / Math.max(1, height - 1);
  var out    = [];
  for (var y = 0; y < height; y++) {
    var srcY = y * yScale;
    var y0   = Math.floor(srcY);
    var y1   = Math.min(y0 + 1, srcH - 1);
    var wy   = srcY - y0;
    var row  = [];
    for (var x = 0; x < width; x++) {
      var srcX   = x * xScale;
      var x0     = Math.floor(srcX);
      var x1     = Math.min(x0 + 1, srcW - 1);
      var wx     = srcX - x0;
      var top    = image[y0][x0] * (1 - wx) + image[y0][x1] * wx;
      var bottom = image[y1][x0] * (1 - wx) + image[y1][x1] * wx;
      row.push(Math.round(top * (1 - wy) + bottom * wy));
    }
    out.push(row);
  }
  return out;
}

function equalize(image) {
  var hist = [];
  for (var i = 0; i < 256; i++) hist.push(0);
  image.forEach(function(row) { row.forEach(function(p) { hist[p]++; }); });
  var total   = image.length * image[0].length;
  var running = 0;
  var cdf     = hist.map(function(count) { running += count; return running; });
  var cdfMin  = 0;
  for (var j = 0; j < cdf.length; j++) { if (cdf[j] > 0) { cdfMin = cdf[j]; break; } }
  var denominator = total - cdfMin;
  if (denominator <= 0) return image.map(function(row) { return row.slice(); });
  var lut = cdf.map(function(v) { return clamp(Math.round(((v - cdfMin) * 255) / denominator), 0, 255); });
  return image.map(function(row) { return row.map(function(p) { return lut[p]; }); });
}

function normalizeImage(image) {
  var SIZE = CONFIG.SIZE;
  return equalize(resizeImage(image, SIZE, SIZE));
}

// ─── Feature extraction ───────────────────────────────────────────────────────

function uniformBin(code) {
  var transitions = 0;
  var prev = code & 1;
  for (var i = 1; i < 8; i++) {
    var bit = (code >> i) & 1;
    if (bit !== prev) transitions++;
    prev = bit;
  }
  if (prev !== (code & 1)) transitions++;
  return transitions <= 2 ? code % CONFIG.LBP_BINS : CONFIG.LBP_BINS;
}

function lbpFeatures(image) {
  var SIZE      = CONFIG.SIZE;
  var LBP_GRID  = CONFIG.LBP_GRID;
  var LBP_BINS  = CONFIG.LBP_BINS;
  var cell      = Math.floor((SIZE - 2) / LBP_GRID);
  var hists     = [];
  var NEIGHBORS = [[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0]];

  var codes = [];
  for (var cy = 0; cy < SIZE - 2; cy++) {
    var codeRow = [];
    for (var cx = 0; cx < SIZE - 2; cx++) codeRow.push(0);
    codes.push(codeRow);
  }

  for (var y = 1; y < SIZE - 1; y++) {
    for (var x = 1; x < SIZE - 1; x++) {
      var code = 0;
      for (var ni = 0; ni < NEIGHBORS.length; ni++) {
        var dx = NEIGHBORS[ni][0];
        var dy = NEIGHBORS[ni][1];
        if (image[y + dy][x + dx] >= image[y][x]) code |= 1 << ni;
      }
      codes[y - 1][x - 1] = uniformBin(code);
    }
  }

  for (var gy = 0; gy < LBP_GRID; gy++) {
    for (var gx = 0; gx < LBP_GRID; gx++) {
      var hist   = [];
      for (var hi = 0; hi <= LBP_BINS; hi++) hist.push(0);
      var startX = gx * cell;
      var startY = gy * cell;
      var endX   = gx < LBP_GRID - 1 ? (gx + 1) * cell : SIZE - 2;
      var endY   = gy < LBP_GRID - 1 ? (gy + 1) * cell : SIZE - 2;
      for (var hy = startY; hy < endY; hy++) {
        for (var hx = startX; hx < endX; hx++) hist[codes[hy][hx]]++;
      }
      var total = hist.reduce(function(s, v) { return s + v; }, 0) || 1;
      hist.forEach(function(v) { hists.push(v / total); });
    }
  }
  return hists;
}

function regionCentroid(image, region) {
  var SIZE   = CONFIG.SIZE;
  var left   = Math.round(region[0] * SIZE);
  var top    = Math.round(region[1] * SIZE);
  var right  = Math.round(region[2] * SIZE);
  var bottom = Math.round(region[3] * SIZE);
  var total = 0, xSum = 0, ySum = 0;
  for (var y = top; y < bottom; y++) {
    for (var x = left; x < right; x++) {
      var weight = Math.max(0, 80 - image[y][x]);
      total += weight;
      xSum  += x * weight;
      ySum  += y * weight;
    }
  }
  if (!total) return [(region[0] + region[2]) / 2, (region[1] + region[3]) / 2, 0];
  return [
    xSum / total / (SIZE - 1),
    ySum / total / (SIZE - 1),
    total / ((right - left) * (bottom - top) * 80),
  ];
}

function appearanceFeatures(image) {
  var small    = resizeImage(image, 24, 24);
  var features = [];
  small.forEach(function(row) { row.forEach(function(p) { features.push(p / 255); }); });
  small.forEach(function(row) { features.push(mean(row) / 255); });
  for (var x = 0; x < 24; x++) {
    features.push(mean(small.map(function(row) { return row[x]; })) / 255);
  }
  return features;
}

function geometryFeatures(image) {
  var REGIONS = [
    [0.18, 0.23, 0.48, 0.48],
    [0.52, 0.23, 0.82, 0.48],
    [0.25, 0.55, 0.75, 0.86],
    [0.38, 0.35, 0.62, 0.65],
  ];
  var points   = REGIONS.map(function(r) { return regionCentroid(image, r); });
  var features = [];
  points.forEach(function(p) { p.forEach(function(v) { features.push(v); }); });
  var leftEye    = points[0];
  var rightEye   = points[1];
  var mouth      = points[2];
  var nose       = points[3];
  var eyeCenterX = (leftEye[0] + rightEye[0]) / 2;
  var eyeCenterY = (leftEye[1] + rightEye[1]) / 2;
  features.push(rightEye[0] - leftEye[0], mouth[1] - eyeCenterY, nose[0] - eyeCenterX);
  return features;
}

function extractTemplate(image) {
  var normalized = normalizeImage(image);
  return lbpFeatures(normalized).concat(appearanceFeatures(normalized)).concat(geometryFeatures(normalized));
}

// ─── Similarity & scoring ─────────────────────────────────────────────────────

function chiSquare(left, right) {
  var sum = 0;
  for (var i = 0; i < left.length; i++) {
    sum += (left[i] - right[i]) * (left[i] - right[i]) / (left[i] + right[i] + 1e-12);
  }
  return 0.5 * sum;
}

function rms(left, right) {
  return Math.sqrt(mean(left.map(function(v, i) { return (v - right[i]) * (v - right[i]); })));
}

function featureDistance(left, right) {
  var LBP_GRID  = CONFIG.LBP_GRID;
  var LBP_BINS  = CONFIG.LBP_BINS;
  var W         = CONFIG.DISTANCE_WEIGHTS;
  var lbpEnd    = (LBP_BINS + 1) * LBP_GRID * LBP_GRID;
  var appEnd    = lbpEnd + 24 * 24 + 48;
  var lbp = chiSquare(left.slice(0, lbpEnd), right.slice(0, lbpEnd)) / 64;
  var app = rms(left.slice(lbpEnd, appEnd),  right.slice(lbpEnd, appEnd));
  var geo = rms(left.slice(appEnd),           right.slice(appEnd));
  return W.lbp * lbp + W.appearance * app + W.geometry * geo;
}

// ─── Liveness assessment ──────────────────────────────────────────────────────

function laplacianVariance(image) {
  var SIZE   = CONFIG.SIZE;
  var values = [];
  for (var y = 1; y < SIZE - 1; y++) {
    for (var x = 1; x < SIZE - 1; x++) {
      values.push(-4 * image[y][x] + image[y-1][x] + image[y+1][x] + image[y][x-1] + image[y][x+1]);
    }
  }
  return variance(values);
}

function imageEntropy(image) {
  var hist = [];
  for (var i = 0; i < 256; i++) hist.push(0);
  image.forEach(function(row) { row.forEach(function(p) { hist[p]++; }); });
  var total = image.length * image[0].length;
  return hist.reduce(function(score, count) {
    if (!count) return score;
    var p = count / total;
    return score - p * Math.log2(p);
  }, 0) / 8;
}

function meanAbsDiff(left, right) {
  var SIZE   = CONFIG.SIZE;
  var values = [];
  for (var y = 0; y < SIZE; y++) {
    for (var x = 0; x < SIZE; x++) values.push(Math.abs(left[y][x] - right[y][x]) / 255);
  }
  return mean(values);
}

function cropImage(image, x0, y0, x1, y1) {
  var SIZE   = CONFIG.SIZE;
  var left   = Math.round(x0 * SIZE);
  var top    = Math.round(y0 * SIZE);
  var right  = Math.round(x1 * SIZE);
  var bottom = Math.round(y1 * SIZE);
  return image.slice(top, bottom).map(function(row) { return row.slice(left, right); });
}

function eyeOpenness(image) {
  var pixels = cropImage(image, 0.22, 0.27, 0.45, 0.45).concat(cropImage(image, 0.55, 0.27, 0.78, 0.45));
  var flat   = [];
  pixels.forEach(function(row) { row.forEach(function(p) { flat.push(p / 255); }); });
  var darkRatio = flat.filter(function(p) { return p < 0.32; }).length / flat.length;
  return darkRatio + 0.35 * variance(flat);
}

function weightedCentroid(image) {
  var SIZE  = CONFIG.SIZE;
  var total = 0, xSum = 0, ySum = 0;
  for (var y = 0; y < SIZE; y++) {
    for (var x = 0; x < SIZE; x++) {
      var weight = Math.max(0, 255 - image[y][x]);
      total += weight;
      xSum  += x * weight;
      ySum  += y * weight;
    }
  }
  if (!total) return [0.5, 0.5];
  return [xSum / total / (SIZE - 1), ySum / total / (SIZE - 1)];
}

function windowScore(value, tooLow, idealLow, idealHigh, tooHigh) {
  if (value <= tooLow || value >= tooHigh) return 0;
  if (value >= idealLow && value <= idealHigh) return 1;
  if (value < idealLow) return (value - tooLow) / (idealLow - tooLow);
  return (tooHigh - value) / (tooHigh - idealHigh);
}

function rangeScore(value, low, high) {
  if (value <= low)  return 0;
  if (value >= high) return 1;
  return (value - low) / (high - low);
}

function directionScore(delta) {
  if (delta <= 0.015) return 0;
  if (delta >= 0.075) return 1;
  return (delta - 0.015) / 0.06;
}

function blinkScore(frames) {
  var openness    = frames.map(eyeOpenness);
  var peak        = Math.max.apply(null, openness);
  var valley      = Math.min.apply(null, openness);
  var valleyIndex = openness.indexOf(valley);
  var leftPeak    = Math.max.apply(null, openness.slice(0, valleyIndex));
  var rightPeak   = Math.max.apply(null, openness.slice(valleyIndex + 1));
  var recovery =
    valleyIndex > 0 &&
    valleyIndex < openness.length - 1 &&
    leftPeak  - valley > 0.012 &&
    rightPeak - valley > 0.012;
  return recovery ? clamp((peak - valley) / 0.035, 0, 1) : 0;
}

/**
 * Assess liveness from a sequence of raw frames.
 * BUG FIX: `passed` is evaluated independently from reasons array,
 * preventing double-counting of "liveness below threshold".
 */
function assessLiveness(rawFrames, challenge) {
  var LW      = CONFIG.LIVENESS_WEIGHTS;
  var frames  = rawFrames.map(normalizeImage);
  var diffs   = frames.slice(1).map(function(frame, i) { return meanAbsDiff(frames[i], frame); });
  var brightness  = frames.map(function(f) { return mean(f.map(function(row) { return mean(row.map(function(p) { return p / 255; })); })); });
  var centroids   = frames.map(weightedCentroid);
  var xVals       = centroids.map(function(c) { return c[0]; });
  var yVals       = centroids.map(function(c) { return c[1]; });
  var xShift      = Math.max.apply(null, xVals) - Math.min.apply(null, xVals);
  var yShift      = Math.max.apply(null, yVals) - Math.min.apply(null, yVals);
  var texture     = mean(frames.map(laplacianVariance));
  var entropyScore = mean(frames.map(imageEntropy));
  var motion      = mean(diffs);
  var exposureVariance = variance(brightness);
  var blink       = blinkScore(frames);

  var challengeValue = 0;
  if (challenge === "blink")     challengeValue = blink;
  if (challenge === "turn_left") challengeValue = directionScore(centroids[centroids.length - 1][0] - centroids[0][0]);
  if (challenge === "static")    challengeValue = motion < 0.004 ? 0 : 0.15;

  var weighted =
    rangeScore(texture,          35,      360)   * LW.texture +
    rangeScore(entropyScore,     0.45,    0.92)  * LW.entropy +
    windowScore(motion,          0.004,   0.018,  0.16, 0.38) * LW.motion +
    windowScore(exposureVariance, 0.00001, 0.0003, 0.035, 0.12) * LW.exposureVariance +
    challengeValue * LW.challenge;

  var score = clamp(weighted / LW.normalizer, 0, 1);

  // FIX: independent evaluation — reasons don't affect `passed`
  var challengePassed = challengeValue >= CONFIG.CHALLENGE_PASS_THRESHOLD;
  var passed          = score >= CONFIG.LIVENESS_THRESHOLD && challengePassed;

  var reasons = [];
  if (motion < 0.004)    reasons.push("insufficient motion");
  if (texture < 35)      reasons.push("low texture detail");
  if (!challengePassed)  reasons.push("challenge failed");
  if (score < CONFIG.LIVENESS_THRESHOLD) reasons.push("liveness below threshold");

  return {
    score: score,
    challengeValue: challengeValue,
    passed: passed,
    metrics: { texture: texture, entropy: entropyScore, motion: motion, exposureVariance: exposureVariance, xShift: xShift, yShift: yShift, blink: blink },
    reasons: reasons,
  };
}

// ─── Frame generation ─────────────────────────────────────────────────────────

/**
 * Build synthetic frames — decoupled from DOM, params passed explicitly.
 */
function buildFrames(subjectId, challenge, opts) {
  opts = opts || {};
  var lighting  = opts.lighting !== undefined ? opts.lighting : 0;
  var jitter    = opts.jitter   !== undefined ? opts.jitter   : 4;
  var BASE_NOISE = CONFIG.BASE_NOISE;
  var shifts = challenge === "turn_left"
    ? [0, jitter + 1, jitter + 3, jitter + 5, jitter + 7]
    : [0, jitter / 2, jitter, jitter / 2, 0];

  return shifts.map(function(shift, index) {
    return syntheticFace(subjectId, {
      shiftX:   challenge === "static" ? 0 : shift,
      shiftY:   challenge === "static" ? 0 : Math.sin(index) * 0.8,
      blink:    challenge === "blink" && index === 2,
      lighting: lighting,
      noise:    challenge === "static" ? 0 : BASE_NOISE,
    });
  });
}

// ─── Identification ───────────────────────────────────────────────────────────

function identify(templates, image) {
  var vector = extractTemplate(image);
  var ranked = templates.map(function(t) {
    return { subjectId: t.subjectId, score: 1 / (1 + featureDistance(vector, t.vector)) };
  }).sort(function(a, b) { return b.score - a.score; });

  var bestBySubject = {};
  ranked.forEach(function(c) {
    if (!bestBySubject[c.subjectId]) bestBySubject[c.subjectId] = c;
  });

  var candidates = Object.keys(bestBySubject)
    .map(function(k) { return bestBySubject[k]; })
    .sort(function(a, b) { return b.score - a.score; });

  return { vector: vector, candidates: candidates };
}

// ─── DOM helpers (XSS-safe) ───────────────────────────────────────────────────

function el(tag, opts) {
  opts = opts || {};
  var node = document.createElement(tag);
  if (opts.text)      node.textContent = opts.text;
  if (opts.className) node.className   = opts.className;
  if (opts.attrs) {
    Object.keys(opts.attrs).forEach(function(k) { node.setAttribute(k, opts.attrs[k]); });
  }
  return node;
}

// ─── Rendering ────────────────────────────────────────────────────────────────

function drawMatrix(canvas, image, palette) {
  palette = palette || "face";
  var SIZE      = CONFIG.SIZE;
  var ctx       = canvas.getContext("2d");
  var imageData = ctx.createImageData(SIZE, SIZE);
  var data      = imageData.data;
  for (var y = 0; y < SIZE; y++) {
    for (var x = 0; x < SIZE; x++) {
      var pixel = image[y][x];
      var idx   = (y * SIZE + x) * 4;
      if (palette === "face") {
        data[idx]     = clamp(pixel - 8, 0, 255);
        data[idx + 1] = clamp(pixel + 5, 0, 255);
        data[idx + 2] = clamp(pixel - 1, 0, 255);
      } else {
        data[idx]     = clamp(30  + pixel * 0.40, 0, 255);
        data[idx + 1] = clamp(90  + pixel * 0.45, 0, 255);
        data[idx + 2] = clamp(116 + pixel * 0.35, 0, 255);
      }
      data[idx + 3] = 255;
    }
  }
  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(imageData, 0, 0);
}

function drawFilmstrip(filmstripEl, frames) {
  var SIZE = CONFIG.SIZE;
  filmstripEl.innerHTML = "";
  frames.forEach(function(frame, index) {
    var canvas = document.createElement("canvas");
    canvas.width  = SIZE;
    canvas.height = SIZE;
    canvas.setAttribute("aria-label", "Frame " + (index + 1) + " of " + frames.length);
    canvas.setAttribute("role", "img");
    drawMatrix(canvas, frame, "trace");
    filmstripEl.appendChild(canvas);
  });
}

function drawVector(canvas, vector) {
  var ctx    = canvas.getContext("2d");
  var width  = canvas.offsetWidth  || canvas.width;
  var height = canvas.offsetHeight || canvas.height;
  canvas.width  = width;
  canvas.height = height;

  var style  = getComputedStyle(document.documentElement);
  var bgColor = style.getPropertyValue("--color-surface-2").trim() || "#f8faf8";

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = bgColor;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = style.getPropertyValue("--color-border").trim() || "#d8e0d7";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, height - 18);
  ctx.lineTo(width, height - 18);
  ctx.stroke();

  var sample   = vector.filter(function(_, i) { return i % 41 === 0; }).slice(0, 90);
  var barWidth = width / sample.length;
  sample.forEach(function(value, index) {
    var barHeight = clamp(value * 460, 2, height - 24);
    ctx.fillStyle = index % 5 === 0 ? "#246a8f" : "#1d8f67";
    ctx.fillRect(index * barWidth, height - 18 - barHeight, Math.max(2, barWidth - 1), barHeight);
  });
}

function setMeter(meterEl, score, isGood) {
  meterEl.style.width      = Math.round(clamp(score, 0, 1) * 100) + "%";
  meterEl.style.background = isGood ? "var(--color-green)" : "var(--color-red)";
}

function renderRegistry(registryListEl, templates) {
  registryListEl.innerHTML = "";
  var counts = templates.reduce(function(acc, t) {
    acc[t.subjectId] = (acc[t.subjectId] || 0) + 1;
    return acc;
  }, {});
  Object.keys(SUBJECTS).forEach(function(subjectId) {
    var row   = el("div", { className: "registry-row" });
    var name  = el("span", { text: SUBJECTS[subjectId].label });
    var count = el("strong", { text: String(counts[subjectId] || 0) });
    row.appendChild(name);
    row.appendChild(count);
    registryListEl.appendChild(row);
  });
}

/**
 * Prepend a new audit entry. Rolling window of AUDIT_MAX_ENTRIES.
 */
function logDecision(auditLogEl, items) {
  while (auditLogEl.children.length >= CONFIG.AUDIT_MAX_ENTRIES) {
    auditLogEl.removeChild(auditLogEl.lastChild);
  }

  var separator = el("li", { className: "audit-separator" });
  var ts = el("span", { text: new Date().toLocaleTimeString(), className: "audit-ts" });
  separator.appendChild(ts);
  auditLogEl.insertBefore(separator, auditLogEl.firstChild);

  var fragment = document.createDocumentFragment();
  items.forEach(function(pair) {
    var li     = el("li");
    var lspan  = el("span", { text: pair[0] });
    var strong = el("strong", { text: String(pair[1]) });
    li.appendChild(lspan);
    li.appendChild(strong);
    fragment.appendChild(li);
  });
  auditLogEl.insertBefore(fragment, separator);
}

// ─── Engine (closure-based, fully encapsulated) ───────────────────────────────

function createBiometricEngine() {
  var state = {
    templates:       [],
    activeChallenge: "blink",
    lastFrames:      [],
    lastVector:      [],
  };

  var els = null;

  function queryElements() {
    return {
      claimedIdentity:  document.querySelector("#claimedIdentity"),
      probeSubject:     document.querySelector("#probeSubject"),
      lighting:         document.querySelector("#lighting"),
      jitter:           document.querySelector("#jitter"),
      faceCanvas:       document.querySelector("#faceCanvas"),
      vectorCanvas:     document.querySelector("#vectorCanvas"),
      filmstrip:        document.querySelector("#filmstrip"),
      registryList:     document.querySelector("#registryList"),
      captureLabel:     document.querySelector("#captureLabel"),
      combinedVerdict:  document.querySelector("#combinedVerdict"),
      recognitionScore: document.querySelector("#recognitionScore"),
      livenessScore:    document.querySelector("#livenessScore"),
      challengeScore:   document.querySelector("#challengeScore"),
      recognitionMeter: document.querySelector("#recognitionMeter"),
      livenessMeter:    document.querySelector("#livenessMeter"),
      challengeMeter:   document.querySelector("#challengeMeter"),
      resultBox:        document.querySelector("#resultBox"),
      auditLog:         document.querySelector("#auditLog"),
    };
  }

  function attachResizeObserver() {
    var ro = new ResizeObserver(function() {
      if (state.lastVector.length) drawVector(els.vectorCanvas, state.lastVector);
    });
    ro.observe(els.vectorCanvas.parentElement);
  }

  function enroll(subjectId) {
    var count = state.templates.filter(function(t) { return t.subjectId === subjectId; }).length;
    var sample = syntheticFace(subjectId, {
      shiftX:   (count % 3) - 1,
      shiftY:   count % 2,
      noise:    2 + (count % 3),
      lighting: Number(els.lighting.value) / 3,
    });
    state.templates.push({
      subjectId: subjectId,
      vector:    extractTemplate(sample),
      createdAt: new Date().toISOString(),
    });
    renderRegistry(els.registryList, state.templates);
  }

  function runScenario(params) {
    var claimedId = params.claimedId;
    var probeId   = params.probeId;
    var challenge = params.challenge;
    var spoof     = params.spoof || false;
    var RECOGNITION_THRESHOLD  = CONFIG.RECOGNITION_THRESHOLD;
    var CHALLENGE_PASS_THRESHOLD = CONFIG.CHALLENGE_PASS_THRESHOLD;

    // Update challenge state & aria-pressed — no dropdown value mutation
    state.activeChallenge = challenge;
    document.querySelectorAll(".segment").forEach(function(btn) {
      var isActive = btn.dataset.challenge === challenge;
      btn.classList.toggle("active", isActive);
      btn.setAttribute("aria-pressed", String(isActive));
    });

    var lighting    = Number(els.lighting.value);
    var jitter      = Number(els.jitter.value);
    var frames      = buildFrames(probeId, spoof ? "static" : challenge, { lighting: lighting, jitter: jitter });
    var probe       = frames[frames.length - 1];
    var recognition = identify(state.templates, probe);
    var best        = recognition.candidates[0] || { subjectId: null, score: 0 };
    var verified    = best.subjectId === claimedId && best.score >= RECOGNITION_THRESHOLD;
    var liveness    = assessLiveness(frames, challenge);
    var accepted    = verified && liveness.passed;

    state.lastFrames = frames;
    state.lastVector = recognition.vector;

    // Render
    drawMatrix(els.faceCanvas, probe);
    drawFilmstrip(els.filmstrip, frames);
    drawVector(els.vectorCanvas, recognition.vector);

    // Meta
    var subjectProfile = SUBJECTS[probeId];
    els.captureLabel.textContent =
      (subjectProfile ? subjectProfile.label : probeId) +
      " \u00b7 " + challenge.replace(/_/g, " ");

    // Scores
    els.recognitionScore.textContent = best.score.toFixed(3);
    els.livenessScore.textContent    = liveness.score.toFixed(3);
    els.challengeScore.textContent   = liveness.challengeValue.toFixed(3);
    setMeter(els.recognitionMeter, best.score,              verified);
    setMeter(els.livenessMeter,    liveness.score,          liveness.passed);
    setMeter(els.challengeMeter,   liveness.challengeValue, liveness.challengeValue >= CHALLENGE_PASS_THRESHOLD);

    // Result box — XSS-safe
    var decisionLabel = el("span", { text: "Decision" });
    var decisionValue = el("strong", { text: accepted ? "Access approved" : "Access denied" });
    els.resultBox.className = "result-box " + (accepted ? "pass" : "fail");
    els.resultBox.replaceChildren(decisionLabel, decisionValue);

    // Verdict badge (aria-live)
    els.combinedVerdict.className   = accepted ? "pass" : "fail";
    els.combinedVerdict.textContent = accepted ? "PASS" : "FAIL";

    // Audit
    var reasons = [];
    if (!verified) reasons.push("identity mismatch");
    reasons.push.apply(reasons, liveness.reasons);

    var claimedLabel = SUBJECTS[claimedId] ? SUBJECTS[claimedId].label : claimedId;
    var matchLabel   = best.subjectId
      ? (SUBJECTS[best.subjectId] ? SUBJECTS[best.subjectId].label : best.subjectId)
      : "None";

    logDecision(els.auditLog, [
      ["Claimed identity",  claimedLabel],
      ["Top match",         matchLabel],
      ["Recognition score", best.score.toFixed(4)],
      ["Liveness score",    liveness.score.toFixed(4)],
      ["Recognition thr.",  RECOGNITION_THRESHOLD.toFixed(2)],
      ["Liveness thr.",     CONFIG.LIVENESS_THRESHOLD.toFixed(2)],
      ["Verdict",           accepted ? "\u2713 All checks passed" : "\u2717 " + reasons.join(", ")],
    ]);
  }

  function bootstrap() {
    els = queryElements();
    attachResizeObserver();

    Object.keys(SUBJECTS).forEach(function(id) { enroll(id); enroll(id); });

    // Segment buttons — single delegated listener
    document.querySelector(".segmented").addEventListener("click", function(e) {
      var btn = e.target.closest(".segment");
      if (!btn) return;
      var challenge = btn.dataset.challenge;
      runScenario({ claimedId: els.claimedIdentity.value, probeId: els.probeSubject.value, challenge: challenge, spoof: challenge === "static" });
    });

    document.querySelector("#runVerified").addEventListener("click", function() {
      runScenario({ claimedId: "operator-a", probeId: "operator-a", challenge: "blink" });
    });
    document.querySelector("#runSpoof").addEventListener("click", function() {
      runScenario({ claimedId: els.claimedIdentity.value, probeId: els.claimedIdentity.value, challenge: "blink", spoof: true });
    });
    document.querySelector("#runMismatch").addEventListener("click", function() {
      runScenario({ claimedId: "operator-a", probeId: "operator-b", challenge: "turn_left" });
    });
    document.querySelector("#enrollCurrent").addEventListener("click", function() {
      enroll(els.probeSubject.value);
      runScenario({ claimedId: els.claimedIdentity.value, probeId: els.probeSubject.value, challenge: state.activeChallenge, spoof: state.activeChallenge === "static" });
    });

    [els.claimedIdentity, els.probeSubject, els.lighting, els.jitter].forEach(function(control) {
      control.addEventListener("input", function() {
        runScenario({ claimedId: els.claimedIdentity.value, probeId: els.probeSubject.value, challenge: state.activeChallenge, spoof: state.activeChallenge === "static" });
      });
    });

    runScenario({ claimedId: "operator-a", probeId: "operator-a", challenge: "blink" });
  }

  return { bootstrap: bootstrap, enroll: enroll, runScenario: runScenario };
}

// ─── Entry point ──────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", function() {
  var engine = createBiometricEngine();
  engine.bootstrap();
});
