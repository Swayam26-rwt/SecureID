"""Small image-processing helpers used by the offline biometric pipeline."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

GrayImage = Sequence[Sequence[int | float]]
MutableGrayImage = list[list[int]]


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


def normalize_face(image: GrayImage, *, width: int, height: int) -> MutableGrayImage:
    return histogram_equalize(resize_gray(image, width, height))


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


def synthetic_face(
    *,
    eye_gap: int = 28,
    mouth_curve: int = 0,
    shift_x: int = 0,
    shift_y: int = 0,
    blink: bool = False,
    noise: int | float = 0,
    size: int = 96,
) -> MutableGrayImage:
    """Generate a synthetic 2D grayscale face matrix for testing and benchmarking."""
    image = [[235 for _ in range(size)] for _ in range(size)]
    cx = size // 2 + shift_x
    cy = size // 2 + shift_y

    for y in range(size):
        for x in range(size):
            nx = (x - cx) / 32
            ny = (y - cy) / 40
            if nx * nx + ny * ny <= 1.0:
                image[y][x] = 172

    eye_y = cy - 13
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

    nose_x = cx + mouth_curve // 2
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
