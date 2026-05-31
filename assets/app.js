const SIZE = 96;
const RECOGNITION_THRESHOLD = 0.78;
const LIVENESS_THRESHOLD = 0.62;

const subjects = {
  "operator-a": {
    label: "Operator A",
    eyeGap: 30,
    mouthCurve: 0,
    noseOffset: 0,
    brow: 0,
  },
  "operator-b": {
    label: "Operator B",
    eyeGap: 20,
    mouthCurve: 8,
    noseOffset: 3,
    brow: 2,
  },
  "operator-c": {
    label: "Operator C",
    eyeGap: 36,
    mouthCurve: -5,
    noseOffset: -4,
    brow: -2,
  },
};

const state = {
  templates: [],
  activeChallenge: "blink",
  lastFrames: [],
  lastVector: [],
};

const els = {
  claimedIdentity: document.querySelector("#claimedIdentity"),
  probeSubject: document.querySelector("#probeSubject"),
  lighting: document.querySelector("#lighting"),
  jitter: document.querySelector("#jitter"),
  faceCanvas: document.querySelector("#faceCanvas"),
  vectorCanvas: document.querySelector("#vectorCanvas"),
  filmstrip: document.querySelector("#filmstrip"),
  registryList: document.querySelector("#registryList"),
  captureLabel: document.querySelector("#captureLabel"),
  combinedVerdict: document.querySelector("#combinedVerdict"),
  recognitionScore: document.querySelector("#recognitionScore"),
  livenessScore: document.querySelector("#livenessScore"),
  challengeScore: document.querySelector("#challengeScore"),
  recognitionMeter: document.querySelector("#recognitionMeter"),
  livenessMeter: document.querySelector("#livenessMeter"),
  challengeMeter: document.querySelector("#challengeMeter"),
  resultBox: document.querySelector("#resultBox"),
  auditLog: document.querySelector("#auditLog"),
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function mean(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function variance(values) {
  const avg = mean(values);
  return values.length
    ? mean(values.map((value) => (value - avg) * (value - avg)))
    : 0;
}

function createMatrix(fill = 235) {
  return Array.from({ length: SIZE }, () => Array.from({ length: SIZE }, () => fill));
}

function syntheticFace(subjectId, options = {}) {
  const profile = subjects[subjectId];
  const image = createMatrix(235 + (options.lighting || 0));
  const cx = Math.round(SIZE / 2 + (options.shiftX || 0));
  const cy = Math.round(SIZE / 2 + (options.shiftY || 0));
  const blink = Boolean(options.blink);
  const noise = options.noise || 0;

  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) {
      const nx = (x - cx) / 32;
      const ny = (y - cy) / 40;
      if (nx * nx + ny * ny <= 1) {
        image[y][x] = 171 + (options.lighting || 0);
      }
    }
  }

  const eyeY = cy - 13 + profile.brow;
  [cx - profile.eyeGap / 2, cx + profile.eyeGap / 2].forEach((eyeX) => {
    for (let y = Math.floor(eyeY - 4); y <= eyeY + 4; y += 1) {
      for (let x = Math.floor(eyeX - 7); x <= eyeX + 7; x += 1) {
        if (y < 0 || y >= SIZE || x < 0 || x >= SIZE) continue;
        if (blink) {
          if (Math.abs(y - eyeY) <= 1) image[y][x] = 58;
        } else {
          const nx = (x - eyeX) / 6;
          const ny = (y - eyeY) / 3;
          if (nx * nx + ny * ny <= 1) image[y][x] = 42;
        }
      }
    }
  });

  const noseX = cx + profile.noseOffset;
  for (let offset = 0; offset < 12; offset += 1) {
    const y = cy - 5 + offset;
    const x = noseX + Math.floor(offset / 3);
    if (y >= 0 && y < SIZE && x >= 0 && x < SIZE) image[y][x] = 116;
  }

  const mouthY = cy + 22;
  for (let dx = -17; dx <= 17; dx += 1) {
    const y = mouthY + Math.floor((dx * dx) / 72) + profile.mouthCurve;
    const x = cx + dx;
    if (y >= 0 && y < SIZE && x >= 0 && x < SIZE) image[y][x] = 82;
  }

  if (noise) {
    for (let y = 0; y < SIZE; y += 1) {
      for (let x = 0; x < SIZE; x += 1) {
        const delta = ((x * 17 + y * 31 + noise * 13) % (2 * noise + 1)) - noise;
        image[y][x] = clamp(image[y][x] + delta, 0, 255);
      }
    }
  }

  return image.map((row) => row.map((pixel) => clamp(pixel, 0, 255)));
}

function resize(image, width, height) {
  const srcH = image.length;
  const srcW = image[0].length;
  const xScale = (srcW - 1) / Math.max(1, width - 1);
  const yScale = (srcH - 1) / Math.max(1, height - 1);
  const out = [];

  for (let y = 0; y < height; y += 1) {
    const srcY = y * yScale;
    const y0 = Math.floor(srcY);
    const y1 = Math.min(y0 + 1, srcH - 1);
    const wy = srcY - y0;
    const row = [];
    for (let x = 0; x < width; x += 1) {
      const srcX = x * xScale;
      const x0 = Math.floor(srcX);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const wx = srcX - x0;
      const top = image[y0][x0] * (1 - wx) + image[y0][x1] * wx;
      const bottom = image[y1][x0] * (1 - wx) + image[y1][x1] * wx;
      row.push(Math.round(top * (1 - wy) + bottom * wy));
    }
    out.push(row);
  }

  return out;
}

function equalize(image) {
  const hist = Array.from({ length: 256 }, () => 0);
  image.flat().forEach((pixel) => {
    hist[pixel] += 1;
  });

  const total = image.length * image[0].length;
  let running = 0;
  const cdf = hist.map((count) => {
    running += count;
    return running;
  });
  const cdfMin = cdf.find((value) => value > 0) || 0;
  const denominator = total - cdfMin;
  if (denominator <= 0) return image.map((row) => row.slice());

  const lut = cdf.map((value) => clamp(Math.round(((value - cdfMin) * 255) / denominator), 0, 255));
  return image.map((row) => row.map((pixel) => lut[pixel]));
}

function normalize(image) {
  return equalize(resize(image, SIZE, SIZE));
}

function uniformBin(code) {
  let transitions = 0;
  let prev = code & 1;
  for (let i = 1; i < 8; i += 1) {
    const bit = (code >> i) & 1;
    if (bit !== prev) transitions += 1;
    prev = bit;
  }
  if (prev !== (code & 1)) transitions += 1;
  return transitions <= 2 ? code % 58 : 58;
}

function lbpFeatures(image) {
  const grid = 8;
  const cell = Math.floor((SIZE - 2) / grid);
  const hists = [];
  const codes = Array.from({ length: SIZE - 2 }, () => Array.from({ length: SIZE - 2 }, () => 0));
  const neighbors = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
  ];

  for (let y = 1; y < SIZE - 1; y += 1) {
    for (let x = 1; x < SIZE - 1; x += 1) {
      let code = 0;
      neighbors.forEach(([dx, dy], index) => {
        if (image[y + dy][x + dx] >= image[y][x]) code |= 1 << index;
      });
      codes[y - 1][x - 1] = uniformBin(code);
    }
  }

  for (let gy = 0; gy < grid; gy += 1) {
    for (let gx = 0; gx < grid; gx += 1) {
      const hist = Array.from({ length: 59 }, () => 0);
      const startX = gx * cell;
      const startY = gy * cell;
      const endX = gx < grid - 1 ? (gx + 1) * cell : SIZE - 2;
      const endY = gy < grid - 1 ? (gy + 1) * cell : SIZE - 2;
      for (let y = startY; y < endY; y += 1) {
        for (let x = startX; x < endX; x += 1) hist[codes[y][x]] += 1;
      }
      const total = hist.reduce((sum, value) => sum + value, 0) || 1;
      hists.push(...hist.map((value) => value / total));
    }
  }

  return hists;
}

function regionCentroid(image, [x0, y0, x1, y1]) {
  const left = Math.round(x0 * SIZE);
  const top = Math.round(y0 * SIZE);
  const right = Math.round(x1 * SIZE);
  const bottom = Math.round(y1 * SIZE);
  let total = 0;
  let xSum = 0;
  let ySum = 0;

  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      const weight = Math.max(0, 80 - image[y][x]);
      total += weight;
      xSum += x * weight;
      ySum += y * weight;
    }
  }

  if (!total) return [(x0 + x1) / 2, (y0 + y1) / 2, 0];
  return [
    xSum / total / (SIZE - 1),
    ySum / total / (SIZE - 1),
    total / ((right - left) * (bottom - top) * 80),
  ];
}

function appearanceFeatures(image) {
  const small = resize(image, 24, 24);
  const features = small.flat().map((pixel) => pixel / 255);
  small.forEach((row) => features.push(mean(row) / 255));
  for (let x = 0; x < 24; x += 1) {
    features.push(mean(small.map((row) => row[x])) / 255);
  }
  return features;
}

function geometryFeatures(image) {
  const regions = [
    [0.18, 0.23, 0.48, 0.48],
    [0.52, 0.23, 0.82, 0.48],
    [0.25, 0.55, 0.75, 0.86],
    [0.38, 0.35, 0.62, 0.65],
  ];
  const points = regions.map((region) => regionCentroid(image, region));
  const features = points.flat();
  const [leftEye, rightEye, mouth, nose] = points;
  const eyeCenterX = (leftEye[0] + rightEye[0]) / 2;
  const eyeCenterY = (leftEye[1] + rightEye[1]) / 2;
  features.push(rightEye[0] - leftEye[0], mouth[1] - eyeCenterY, nose[0] - eyeCenterX);
  return features;
}

function extractTemplate(image) {
  const normalized = normalize(image);
  return [...lbpFeatures(normalized), ...appearanceFeatures(normalized), ...geometryFeatures(normalized)];
}

function chiSquare(left, right) {
  let sum = 0;
  for (let i = 0; i < left.length; i += 1) {
    sum += ((left[i] - right[i]) ** 2) / (left[i] + right[i] + 1e-12);
  }
  return 0.5 * sum;
}

function rms(left, right) {
  return Math.sqrt(mean(left.map((value, index) => (value - right[index]) ** 2)));
}

function distance(left, right) {
  const lbpEnd = 59 * 8 * 8;
  const appearanceEnd = lbpEnd + 24 * 24 + 48;
  const lbp = chiSquare(left.slice(0, lbpEnd), right.slice(0, lbpEnd)) / 64;
  const app = rms(left.slice(lbpEnd, appearanceEnd), right.slice(lbpEnd, appearanceEnd));
  const geo = rms(left.slice(appearanceEnd), right.slice(appearanceEnd));
  return 0.52 * lbp + 0.95 * app + 3.2 * geo;
}

function identify(image) {
  const vector = extractTemplate(image);
  const ranked = state.templates
    .map((template) => ({
      subjectId: template.subjectId,
      score: 1 / (1 + distance(vector, template.vector)),
    }))
    .sort((a, b) => b.score - a.score);

  const bestBySubject = new Map();
  ranked.forEach((candidate) => {
    if (!bestBySubject.has(candidate.subjectId)) bestBySubject.set(candidate.subjectId, candidate);
  });

  return {
    vector,
    candidates: Array.from(bestBySubject.values()).sort((a, b) => b.score - a.score),
  };
}

function laplacianVariance(image) {
  const values = [];
  for (let y = 1; y < SIZE - 1; y += 1) {
    for (let x = 1; x < SIZE - 1; x += 1) {
      values.push(
        -4 * image[y][x] +
          image[y - 1][x] +
          image[y + 1][x] +
          image[y][x - 1] +
          image[y][x + 1],
      );
    }
  }
  return variance(values);
}

function entropy(image) {
  const hist = Array.from({ length: 256 }, () => 0);
  image.flat().forEach((pixel) => {
    hist[pixel] += 1;
  });
  const total = image.length * image[0].length;
  return (
    hist.reduce((score, count) => {
      if (!count) return score;
      const p = count / total;
      return score - p * Math.log2(p);
    }, 0) / 8
  );
}

function meanAbsDiff(left, right) {
  const values = [];
  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) values.push(Math.abs(left[y][x] - right[y][x]) / 255);
  }
  return mean(values);
}

function crop(image, x0, y0, x1, y1) {
  const left = Math.round(x0 * SIZE);
  const top = Math.round(y0 * SIZE);
  const right = Math.round(x1 * SIZE);
  const bottom = Math.round(y1 * SIZE);
  return image.slice(top, bottom).map((row) => row.slice(left, right));
}

function eyeOpenness(image) {
  const pixels = [
    ...crop(image, 0.22, 0.27, 0.45, 0.45).flat(),
    ...crop(image, 0.55, 0.27, 0.78, 0.45).flat(),
  ].map((pixel) => pixel / 255);
  const darkRatio = pixels.filter((pixel) => pixel < 0.32).length / pixels.length;
  return darkRatio + 0.35 * variance(pixels);
}

function weightedCentroid(image) {
  let total = 0;
  let xSum = 0;
  let ySum = 0;
  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) {
      const weight = Math.max(0, 255 - image[y][x]);
      total += weight;
      xSum += x * weight;
      ySum += y * weight;
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
  if (value <= low) return 0;
  if (value >= high) return 1;
  return (value - low) / (high - low);
}

function directionScore(delta) {
  if (delta <= 0.015) return 0;
  if (delta >= 0.075) return 1;
  return (delta - 0.015) / 0.06;
}

function blinkScore(frames) {
  const openness = frames.map(eyeOpenness);
  const peak = Math.max(...openness);
  const valley = Math.min(...openness);
  const valleyIndex = openness.indexOf(valley);
  const leftPeak = Math.max(...openness.slice(0, valleyIndex));
  const rightPeak = Math.max(...openness.slice(valleyIndex + 1));
  const recovery =
    valleyIndex > 0 &&
    valleyIndex < openness.length - 1 &&
    leftPeak - valley > 0.012 &&
    rightPeak - valley > 0.012;
  return recovery ? clamp((peak - valley) / 0.035, 0, 1) : 0;
}

function assessLiveness(rawFrames, challenge) {
  const frames = rawFrames.map(normalize);
  const diffs = frames.slice(1).map((frame, index) => meanAbsDiff(frames[index], frame));
  const brightness = frames.map((frame) => mean(frame.flat().map((pixel) => pixel / 255)));
  const centroids = frames.map(weightedCentroid);
  const xShift = Math.max(...centroids.map(([x]) => x)) - Math.min(...centroids.map(([x]) => x));
  const yShift = Math.max(...centroids.map(([, y]) => y)) - Math.min(...centroids.map(([, y]) => y));
  const texture = mean(frames.map(laplacianVariance));
  const entropyScore = mean(frames.map(entropy));
  const motion = mean(diffs);
  const exposureVariance = variance(brightness);
  const blink = blinkScore(frames);

  let challengeValue = 0;
  if (challenge === "blink") challengeValue = blink;
  if (challenge === "turn_left") challengeValue = directionScore(centroids.at(-1)[0] - centroids[0][0]);
  if (challenge === "static") challengeValue = motion < 0.004 ? 0 : 0.15;

  const weighted =
    rangeScore(texture, 35, 360) * 0.24 +
    rangeScore(entropyScore, 0.45, 0.92) * 0.16 +
    windowScore(motion, 0.004, 0.018, 0.16, 0.38) * 0.34 +
    windowScore(exposureVariance, 0.00001, 0.0003, 0.035, 0.12) * 0.12 +
    challengeValue * 0.3;
  const score = clamp(weighted / 1.16, 0, 1);
  const reasons = [];
  if (motion < 0.004) reasons.push("insufficient motion");
  if (texture < 35) reasons.push("low texture detail");
  if (challengeValue < 0.55) reasons.push("challenge failed");
  if (score < LIVENESS_THRESHOLD) reasons.push("liveness below threshold");

  return {
    score,
    challengeValue,
    passed: score >= LIVENESS_THRESHOLD && reasons.length === 0,
    metrics: { texture, entropy: entropyScore, motion, exposureVariance, xShift, yShift, blink },
    reasons,
  };
}

function buildFrames(subjectId, mode) {
  const lighting = Number(els.lighting.value);
  const jitter = Number(els.jitter.value);
  const baseNoise = 3;
  const shifts =
    mode === "turn_left"
      ? [0, jitter + 1, jitter + 3, jitter + 5, jitter + 7]
      : [0, jitter / 2, jitter, jitter / 2, 0];

  return shifts.map((shift, index) =>
    syntheticFace(subjectId, {
      shiftX: mode === "static" ? 0 : shift,
      shiftY: mode === "static" ? 0 : Math.sin(index) * 0.8,
      blink: mode === "blink" && index === 2,
      lighting,
      noise: mode === "static" ? 0 : baseNoise,
    }),
  );
}

function drawMatrix(canvas, image, palette = "face") {
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(SIZE, SIZE);
  const data = imageData.data;
  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) {
      const pixel = image[y][x];
      const index = (y * SIZE + x) * 4;
      if (palette === "face") {
        data[index] = clamp(pixel - 8, 0, 255);
        data[index + 1] = clamp(pixel + 5, 0, 255);
        data[index + 2] = clamp(pixel - 1, 0, 255);
      } else {
        data[index] = clamp(30 + pixel * 0.4, 0, 255);
        data[index + 1] = clamp(90 + pixel * 0.45, 0, 255);
        data[index + 2] = clamp(116 + pixel * 0.35, 0, 255);
      }
      data[index + 3] = 255;
    }
  }
  ctx.imageSmoothingEnabled = false;
  ctx.putImageData(imageData, 0, 0);
}

function drawFilmstrip(frames) {
  els.filmstrip.innerHTML = "";
  frames.forEach((frame, index) => {
    const canvas = document.createElement("canvas");
    canvas.width = SIZE;
    canvas.height = SIZE;
    canvas.setAttribute("aria-label", `Frame ${index + 1}`);
    drawMatrix(canvas, frame, "trace");
    els.filmstrip.appendChild(canvas);
  });
}

function drawVector(vector) {
  const canvas = els.vectorCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8faf8";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d8e0d7";
  ctx.beginPath();
  ctx.moveTo(0, height - 18);
  ctx.lineTo(width, height - 18);
  ctx.stroke();

  const sample = vector.filter((_, index) => index % 41 === 0).slice(0, 90);
  const barWidth = width / sample.length;
  sample.forEach((value, index) => {
    const barHeight = clamp(value * 460, 2, height - 24);
    ctx.fillStyle = index % 5 === 0 ? "#246a8f" : "#1d8f67";
    ctx.fillRect(index * barWidth, height - 18 - barHeight, Math.max(2, barWidth - 1), barHeight);
  });
}

function setMeter(element, score, good = true) {
  element.style.width = `${Math.round(clamp(score, 0, 1) * 100)}%`;
  element.style.background = good ? "var(--green)" : "var(--red)";
}

function renderRegistry() {
  const counts = state.templates.reduce((acc, template) => {
    acc[template.subjectId] = (acc[template.subjectId] || 0) + 1;
    return acc;
  }, {});
  els.registryList.innerHTML = Object.keys(subjects)
    .map(
      (subjectId) => `
        <div class="registry-row">
          <span>${subjects[subjectId].label}</span>
          <strong>${counts[subjectId] || 0}</strong>
        </div>
      `,
    )
    .join("");
}

function logDecision(items) {
  els.auditLog.innerHTML = items
    .map(([label, value]) => `<li><span>${label}</span><strong>${value}</strong></li>`)
    .join("");
}

function runScenario({ claimedId, probeId, challenge, spoof = false }) {
  els.claimedIdentity.value = claimedId;
  els.probeSubject.value = probeId;
  state.activeChallenge = challenge;
  document.querySelectorAll(".segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.challenge === challenge);
  });

  const frames = buildFrames(probeId, spoof ? "static" : challenge);
  const probe = frames.at(-1);
  const recognition = identify(probe);
  const best = recognition.candidates[0] || { subjectId: null, score: 0 };
  const verified = best.subjectId === claimedId && best.score >= RECOGNITION_THRESHOLD;
  const liveness = assessLiveness(frames, challenge);
  const accepted = verified && liveness.passed;

  state.lastFrames = frames;
  state.lastVector = recognition.vector;

  drawMatrix(els.faceCanvas, probe);
  drawFilmstrip(frames);
  drawVector(recognition.vector);

  els.captureLabel.textContent = `${subjects[probeId].label} · ${challenge.replace("_", " ")}`;
  els.recognitionScore.textContent = best.score.toFixed(3);
  els.livenessScore.textContent = liveness.score.toFixed(3);
  els.challengeScore.textContent = liveness.challengeValue.toFixed(3);
  setMeter(els.recognitionMeter, best.score, verified);
  setMeter(els.livenessMeter, liveness.score, liveness.passed);
  setMeter(els.challengeMeter, liveness.challengeValue, liveness.challengeValue >= 0.55);

  els.resultBox.className = `result-box ${accepted ? "pass" : "fail"}`;
  els.resultBox.innerHTML = `
    <span>Decision</span>
    <strong>${accepted ? "Access approved" : "Access denied"}</strong>
  `;
  els.combinedVerdict.className = accepted ? "pass" : "fail";
  els.combinedVerdict.textContent = accepted ? "PASS" : "FAIL";

  const reasons = [];
  if (!verified) reasons.push("identity mismatch");
  reasons.push(...liveness.reasons);

  logDecision([
    ["Claimed identity", subjects[claimedId].label],
    ["Top match", best.subjectId ? subjects[best.subjectId].label : "None"],
    ["Recognition threshold", RECOGNITION_THRESHOLD.toFixed(2)],
    ["Liveness threshold", LIVENESS_THRESHOLD.toFixed(2)],
    ["Reason", reasons.length ? reasons.join(", ") : "all checks passed"],
  ]);
}

function enroll(subjectId) {
  const count = state.templates.filter((template) => template.subjectId === subjectId).length;
  const sample = syntheticFace(subjectId, {
    shiftX: (count % 3) - 1,
    shiftY: count % 2,
    noise: 2 + (count % 3),
    lighting: Number(els.lighting.value) / 3,
  });
  state.templates.push({
    subjectId,
    vector: extractTemplate(sample),
    createdAt: new Date().toISOString(),
  });
  renderRegistry();
}

function bootstrap() {
  Object.keys(subjects).forEach((subjectId) => {
    enroll(subjectId);
    enroll(subjectId);
  });

  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      state.activeChallenge = button.dataset.challenge;
      document.querySelectorAll(".segment").forEach((segment) => segment.classList.remove("active"));
      button.classList.add("active");
      runScenario({
        claimedId: els.claimedIdentity.value,
        probeId: els.probeSubject.value,
        challenge: state.activeChallenge,
        spoof: state.activeChallenge === "static",
      });
    });
  });

  document.querySelector("#runVerified").addEventListener("click", () => {
    runScenario({ claimedId: "operator-a", probeId: "operator-a", challenge: "blink" });
  });
  document.querySelector("#runSpoof").addEventListener("click", () => {
    runScenario({ claimedId: els.claimedIdentity.value, probeId: els.claimedIdentity.value, challenge: "blink", spoof: true });
  });
  document.querySelector("#runMismatch").addEventListener("click", () => {
    runScenario({ claimedId: "operator-a", probeId: "operator-b", challenge: "turn_left" });
  });
  document.querySelector("#enrollCurrent").addEventListener("click", () => {
    enroll(els.probeSubject.value);
    runScenario({
      claimedId: els.claimedIdentity.value,
      probeId: els.probeSubject.value,
      challenge: state.activeChallenge,
      spoof: state.activeChallenge === "static",
    });
  });

  [els.claimedIdentity, els.probeSubject, els.lighting, els.jitter].forEach((control) => {
    control.addEventListener("input", () => {
      runScenario({
        claimedId: els.claimedIdentity.value,
        probeId: els.probeSubject.value,
        challenge: state.activeChallenge,
        spoof: state.activeChallenge === "static",
      });
    });
  });

  runScenario({ claimedId: "operator-a", probeId: "operator-a", challenge: "blink" });
}

bootstrap();
