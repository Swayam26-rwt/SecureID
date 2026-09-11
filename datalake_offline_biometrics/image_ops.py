"""Small image-processing helpers used by the offline biometric pipeline.

Extended in v2.0.0 with Gabor filter banks, Local Phase Quantization (LPQ),
CLAHE-style adaptive histogram equalization, and image quality metrics.
All algorithms use only Python standard library — zero external dependencies.
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Sequence

GrayImage = Sequence[Sequence[int | float]]
MutableGrayImage = list[list[int]]


# ── Validation helpers ────────────────────────────────────────────────────────

def validate_gray_image(image: GrayImage, *, min_size: int = 8) -> None:
    if not image or not image[0]:
        raise ValueError("image must be a non-empty 2D grayscale matrix")

    width = len(image[0])
    if len(image) < min_size or width < min_size:
        raise ValueError(f"image must be at least {min_size}x{min_size} pixels")

    for row in image:
        if len(row) != width:
            raise ValueError("image rows must all have the same width")


def clamp_u8(value: int | float) -> int:
    return max(0, min(255, int(round(value))))


def to_u8_matrix(image: GrayImage) -> MutableGrayImage:
    validate_gray_image(image)
    return [[clamp_u8(pixel) for pixel in row] for row in image]


def shape(image: GrayImage) -> tuple[int, int]:
    validate_gray_image(image)
    return len(image), len(image[0])


# ── Resize / equalization ─────────────────────────────────────────────────────

def resize_gray(image: GrayImage, width: int, height: int) -> MutableGrayImage:
    """Resize a grayscale image with bilinear interpolation."""

    src = to_u8_matrix(image)
    src_h, src_w = len(src), len(src[0])
    if width <= 0 or height <= 0:
        raise ValueError("target width and height must be positive")

    if src_w == width and src_h == height:
        return [row[:] for row in src]

    out: MutableGrayImage = []
    x_scale = (src_w - 1) / max(1, width - 1)
    y_scale = (src_h - 1) / max(1, height - 1)

    for y in range(height):
        src_y = y * y_scale
        y0 = int(math.floor(src_y))
        y1 = min(y0 + 1, src_h - 1)
        wy = src_y - y0
        row: list[int] = []
        for x in range(width):
            src_x = x * x_scale
            x0 = int(math.floor(src_x))
            x1 = min(x0 + 1, src_w - 1)
            wx = src_x - x0

            top = src[y0][x0] * (1 - wx) + src[y0][x1] * wx
            bottom = src[y1][x0] * (1 - wx) + src[y1][x1] * wx
            row.append(clamp_u8(top * (1 - wy) + bottom * wy))
        out.append(row)
    return out


def histogram_equalize(image: GrayImage) -> MutableGrayImage:
    """Apply global histogram equalization to stabilize lighting changes."""

    src = to_u8_matrix(image)
    hist = [0] * 256
    for row in src:
        for pixel in row:
            hist[pixel] += 1

    total = len(src) * len(src[0])
    cdf: list[int] = []
    running = 0
    for count in hist:
        running += count
        cdf.append(running)

    cdf_min = next((value for value in cdf if value > 0), 0)
    denominator = total - cdf_min
    if denominator <= 0:
        return [row[:] for row in src]

    lut = [clamp_u8((value - cdf_min) * 255 / denominator) for value in cdf]
    return [[lut[pixel] for pixel in row] for row in src]


def clahe_equalize(
    image: GrayImage,
    *,
    tile_rows: int = 8,
    tile_cols: int = 8,
    clip_limit: float = 3.0,
) -> MutableGrayImage:
    """CLAHE-style Contrast Limited Adaptive Histogram Equalization.

    Divides the image into tiles, applies clip-limited equalization per tile,
    and bilinearly interpolates across tile boundaries for smooth results.
    This provides significantly better illumination robustness than global HE.

    Args:
        image: Input grayscale image.
        tile_rows: Number of tile rows (default 8).
        tile_cols: Number of tile columns (default 8).
        clip_limit: Maximum histogram bin fraction before clipping (default 3.0).
                    Lower values = less contrast enhancement.
    """
    src = to_u8_matrix(image)
    h, w = len(src), len(src[0])
    tile_h = max(1, h // tile_rows)
    tile_w = max(1, w // tile_cols)

    # Build per-tile CLUTs (Contrast-Limited LUT)
    luts: list[list[list[int]]] = []
    for tr in range(tile_rows):
        luts_row: list[list[int]] = []
        for tc in range(tile_cols):
            y0 = tr * tile_h
            y1 = min(y0 + tile_h, h)
            x0 = tc * tile_w
            x1 = min(x0 + tile_w, w)

            hist = [0] * 256
            pixel_count = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    hist[src[y][x]] += 1
                    pixel_count += 1

            # Clip and redistribute
            clip_val = max(1, int(clip_limit * pixel_count / 256))
            excess = 0
            for i in range(256):
                if hist[i] > clip_val:
                    excess += hist[i] - clip_val
                    hist[i] = clip_val
            redist = excess // 256
            for i in range(256):
                hist[i] += redist

            # Build LUT from CDF
            cdf = 0
            tile_lut: list[int] = []
            for count in hist:
                cdf += count
                tile_lut.append(clamp_u8((cdf - hist[0]) * 255 // max(1, pixel_count - hist[0])))
            luts_row.append(tile_lut)
        luts.append(luts_row)

    # Bilinear interpolation across tile boundaries
    out: MutableGrayImage = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            # Tile coordinates (fractional)
            ty = (y / tile_h) - 0.5
            tx = (x / tile_w) - 0.5
            r0 = int(math.floor(ty))
            c0 = int(math.floor(tx))
            r1 = r0 + 1
            c1 = c0 + 1
            wy = ty - r0
            wx = tx - c0

            def tile_map(tr: int, tc: int, pix: int) -> int:
                tr = max(0, min(tile_rows - 1, tr))
                tc = max(0, min(tile_cols - 1, tc))
                return luts[tr][tc][pix]

            p = src[y][x]
            val = (
                tile_map(r0, c0, p) * (1 - wx) * (1 - wy)
                + tile_map(r0, c1, p) * wx * (1 - wy)
                + tile_map(r1, c0, p) * (1 - wx) * wy
                + tile_map(r1, c1, p) * wx * wy
            )
            out[y][x] = clamp_u8(val)
    return out


def normalize_face(image: GrayImage, *, width: int, height: int) -> MutableGrayImage:
    """Normalize face crop: resize → CLAHE equalization."""
    return clahe_equalize(resize_gray(image, width, height))


# ── Basic statistics ──────────────────────────────────────────────────────────

def flatten(image: GrayImage) -> list[float]:
    validate_gray_image(image)
    return [float(pixel) for row in image for pixel in row]


def mean(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def variance(values: Iterable[float]) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    avg = mean(values_list)
    return sum((value - avg) ** 2 for value in values_list) / len(values_list)


def entropy(image: GrayImage) -> float:
    src = to_u8_matrix(image)
    hist = [0] * 256
    total = len(src) * len(src[0])
    for row in src:
        for pixel in row:
            hist[pixel] += 1
    score = 0.0
    for count in hist:
        if count:
            probability = count / total
            score -= probability * math.log2(probability)
    return score / 8.0


def laplacian_variance(image: GrayImage) -> float:
    src = to_u8_matrix(image)
    h, w = len(src), len(src[0])
    if h < 3 or w < 3:
        return 0.0

    responses: list[float] = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            value = (
                -4 * src[y][x]
                + src[y - 1][x]
                + src[y + 1][x]
                + src[y][x - 1]
                + src[y][x + 1]
            )
            responses.append(float(value))
    return variance(responses)


def mean_absolute_difference(left: GrayImage, right: GrayImage) -> float:
    left_matrix = to_u8_matrix(left)
    right_matrix = resize_gray(right, len(left_matrix[0]), len(left_matrix))
    diffs: list[float] = []
    for y, row in enumerate(left_matrix):
        for x, pixel in enumerate(row):
            diffs.append(abs(pixel - right_matrix[y][x]) / 255.0)
    return mean(diffs)


def weighted_centroid(image: GrayImage) -> tuple[float, float]:
    src = to_u8_matrix(image)
    h, w = len(src), len(src[0])
    total_weight = 0.0
    x_sum = 0.0
    y_sum = 0.0

    for y, row in enumerate(src):
        for x, pixel in enumerate(row):
            weight = max(0.0, 255.0 - float(pixel))
            total_weight += weight
            x_sum += x * weight
            y_sum += y * weight

    if total_weight == 0:
        return 0.5, 0.5

    return x_sum / total_weight / max(1, w - 1), y_sum / total_weight / max(1, h - 1)


def crop_fraction(
    image: GrayImage,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> MutableGrayImage:
    src = to_u8_matrix(image)
    h, w = len(src), len(src[0])
    left = max(0, min(w - 1, int(round(x0 * w))))
    top = max(0, min(h - 1, int(round(y0 * h))))
    right = max(left + 1, min(w, int(round(x1 * w))))
    bottom = max(top + 1, min(h, int(round(y1 * h))))
    return [row[left:right] for row in src[top:bottom]]


# ── Gabor filter bank ─────────────────────────────────────────────────────────

def _gabor_kernel(
    size: int,
    theta: float,
    frequency: float,
    sigma: float,
) -> list[list[float]]:
    """Compute a 2D Gabor filter kernel.

    Args:
        size: Kernel size (should be odd).
        theta: Orientation in radians.
        frequency: Spatial frequency of the sinusoidal component.
        sigma: Standard deviation of the Gaussian envelope.
    """
    half = size // 2
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    kernel: list[list[float]] = []
    for dy in range(-half, half + 1):
        row: list[float] = []
        for dx in range(-half, half + 1):
            x_rot = dx * cos_t + dy * sin_t
            y_rot = -dx * sin_t + dy * cos_t
            gaussian = math.exp(-(x_rot ** 2 + y_rot ** 2) / (2 * sigma ** 2))
            sinusoid = math.cos(2 * math.pi * frequency * x_rot)
            row.append(gaussian * sinusoid)
        kernel.append(row)
    return kernel


def gabor_features(
    image: GrayImage,
    *,
    orientations: int = 4,
    frequencies: tuple[float, ...] = (0.1, 0.2),
    kernel_size: int = 9,
    sigma: float = 2.0,
) -> list[float]:
    """Extract Gabor filter bank features from a grayscale face image.

    Applies a bank of Gabor filters at multiple orientations and frequencies.
    For each filter response, computes the mean and variance of magnitudes,
    yielding a compact yet discriminative texture descriptor.

    Args:
        image: Input grayscale image.
        orientations: Number of evenly-spaced orientations (default 4).
        frequencies: Spatial frequencies to use (default (0.1, 0.2)).
        kernel_size: Gabor kernel size in pixels (default 9).
        sigma: Gaussian envelope standard deviation (default 2.0).

    Returns:
        Feature vector of length 2 * orientations * len(frequencies).
    """
    src = to_u8_matrix(image)
    h, w = len(src), len(src[0])
    half = kernel_size // 2
    features: list[float] = []

    thetas = [math.pi * k / orientations for k in range(orientations)]
    for freq in frequencies:
        for theta in thetas:
            kernel = _gabor_kernel(kernel_size, theta, freq, sigma)
            responses: list[float] = []
            for y in range(half, h - half):
                for x in range(half, w - half):
                    value = 0.0
                    for ky in range(kernel_size):
                        for kx in range(kernel_size):
                            value += src[y - half + ky][x - half + kx] * kernel[ky][kx]
                    responses.append(abs(value))
            if responses:
                mu = sum(responses) / len(responses)
                sigma2 = sum((r - mu) ** 2 for r in responses) / len(responses)
                features.append(mu / 255.0)
                features.append(math.sqrt(sigma2) / 255.0)
            else:
                features.extend([0.0, 0.0])

    return features


# ── Local Phase Quantization (LPQ) ────────────────────────────────────────────

def lpq_features(image: GrayImage, *, win_size: int = 7) -> list[float]:
    """Local Phase Quantization descriptor for blur-invariant face recognition.

    LPQ computes short-time Fourier transform phase information in local
    windows, producing a histogram that is largely insensitive to centrally-
    symmetric blur (lens defocus, motion blur). This complements LBP which
    is sensitive to blur.

    Args:
        image: Input grayscale image (should be normalized).
        win_size: Window size for local phase computation (default 7, must be odd).

    Returns:
        256-bin normalized histogram as a feature vector.
    """
    src = to_u8_matrix(image)
    h, w = len(src), len(src[0])
    half = win_size // 2

    # Frequency points at which to evaluate short-time Fourier transform
    freq_points = [
        (1.0 / win_size, 0.0),
        (0.0, 1.0 / win_size),
        (1.0 / win_size, 1.0 / win_size),
        (1.0 / win_size, -1.0 / win_size),
    ]

    hist = [0] * 256

    for y in range(half, h - half):
        for x in range(half, w - half):
            # Extract local window
            window = [
                src[y + dy][x + dx]
                for dy in range(-half, half + 1)
                for dx in range(-half, half + 1)
            ]
            code = 0
            for bit_idx, (u, v) in enumerate(freq_points):
                # Compute DFT at frequency (u, v) for this window
                real_part = 0.0
                imag_part = 0.0
                for pix_idx, pixel in enumerate(window):
                    py = pix_idx // win_size - half
                    px = pix_idx % win_size - half
                    angle = -2 * math.pi * (u * px + v * py)
                    real_part += pixel * math.cos(angle)
                    imag_part += pixel * math.sin(angle)

                # Phase quantization: real and imaginary signs → 2 bits each
                if real_part >= 0:
                    code |= 1 << (2 * bit_idx)
                if imag_part >= 0:
                    code |= 1 << (2 * bit_idx + 1)

            hist[code & 0xFF] += 1

    total = sum(hist) or 1
    return [count / total for count in hist]


# ── Image quality metrics ─────────────────────────────────────────────────────

def blur_score(image: GrayImage) -> float:
    """Estimate image sharpness using Laplacian variance (higher = sharper).

    Returns a normalized score in [0, 1] where 1.0 is a well-focused image.
    Useful for rejecting blurry enrollment samples.
    """
    lv = laplacian_variance(image)
    # Empirically calibrated: lv < 20 is very blurry, lv > 500 is sharp
    return max(0.0, min(1.0, (lv - 20.0) / 480.0))


def brightness_score(image: GrayImage) -> float:
    """Estimate brightness quality: 1.0 for well-lit, lower for over/underexposure.

    Optimal face images have mean brightness around 120-160 out of 255.
    """
    src = to_u8_matrix(image)
    flat = [float(p) for row in src for p in row]
    mu = sum(flat) / max(1, len(flat))
    # Gaussian-like window centred at 140
    score = math.exp(-((mu - 140.0) ** 2) / (2 * 60.0 ** 2))
    return score


def occlusion_ratio(image: GrayImage) -> float:
    """Estimate facial occlusion by detecting large uniform or dark regions.

    Returns fraction of pixels that appear to be occluded [0, 1].
    High values suggest glasses, masks, or partial coverage.
    """
    src = to_u8_matrix(image)
    h, w = len(src), len(src[0])
    total = h * w
    suspect = 0
    # A pixel is "suspect" if it's in a very low-variance neighbourhood
    half = 2
    for y in range(half, h - half):
        for x in range(half, w - half):
            nbhd = [
                src[y + dy][x + dx]
                for dy in range(-half, half + 1)
                for dx in range(-half, half + 1)
            ]
            rng = max(nbhd) - min(nbhd)
            if rng < 8:  # Extremely flat region
                suspect += 1
    return min(1.0, suspect / max(1, total))


def image_quality_score(image: GrayImage) -> dict[str, float]:
    """Aggregate quality report for a face image.

    Returns:
        dict with keys: 'blur', 'brightness', 'occlusion', 'overall'
        All scores in [0, 1]; higher is better for 'blur' and 'brightness',
        lower is better for 'occlusion'. 'overall' combines all three.
    """
    blur = blur_score(image)
    brightness = brightness_score(image)
    occ = occlusion_ratio(image)
    overall = blur * 0.45 + brightness * 0.35 + (1.0 - occ) * 0.20
    return {
        "blur": round(blur, 4),
        "brightness": round(brightness, 4),
        "occlusion": round(occ, 4),
        "overall": round(overall, 4),
    }


# ── Synthetic face generator ──────────────────────────────────────────────────

def synthetic_face(
    *,
    eye_gap: int = 28,
    mouth_curve: int = 0,
    shift_x: int = 0,
    shift_y: int = 0,
    blink: bool = False,
    noise: int | float = 0,
    lighting: int | float = 0,
    size: int = 96,
    nose_offset: int = 0,
    brow_offset: int = 0,
) -> MutableGrayImage:
    """Generate a synthetic 2D grayscale face matrix for testing and benchmarking.

    Extended in v2.0.0 with nose_offset, brow_offset, and lighting parameters.
    """
    bg = clamp_u8(235 + lighting)
    image = [[bg for _ in range(size)] for _ in range(size)]
    cx = size // 2 + shift_x
    cy = size // 2 + shift_y

    face_val = clamp_u8(172 + lighting)
    for y in range(size):
        for x in range(size):
            nx = (x - cx) / 32
            ny = (y - cy) / 40
            if nx * nx + ny * ny <= 1.0:
                image[y][x] = face_val

    eye_y = cy - 13 + brow_offset
    for eye_x in (cx - eye_gap // 2, cx + eye_gap // 2):
        for y in range(eye_y - 3, eye_y + 4):
            for x in range(eye_x - 6, eye_x + 7):
                if 0 <= y < size and 0 <= x < size:
                    if blink:
                        if abs(y - eye_y) <= 1:
                            image[y][x] = 65
                    else:
                        nx = (x - eye_x) / 6
                        ny = (y - eye_y) / 3
                        if nx * nx + ny * ny <= 1.0:
                            image[y][x] = 42

    nose_x = cx + nose_offset
    for offset in range(10):
        y = cy - 4 + offset
        x = nose_x + offset // 3
        if 0 <= y < size and 0 <= x < size:
            image[y][x] = 118

    mouth_y = cy + 18
    for offset in range(-14, 15):
        curve = int(round((offset * offset) / 45.0 * (1 if mouth_curve >= 0 else -1)))
        y = mouth_y + (curve if mouth_curve != 0 else 0)
        x = cx + offset
        for dy in (-1, 0, 1):
            if 0 <= y + dy < size and 0 <= x < size:
                image[y + dy][x] = 58

    if noise:
        noise_val = int(round(noise if noise > 1 else noise * 100))
        for y in range(size):
            for x in range(size):
                delta = ((x * 17 + y * 31 + noise_val) % (noise_val * 2 + 1)) - noise_val
                image[y][x] = clamp_u8(image[y][x] + delta)

    return image
