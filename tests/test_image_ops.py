from __future__ import annotations

import unittest

from datalake_offline_biometrics.image_ops import (
    clamp_u8,
    crop_fraction,
    histogram_equalize,
    laplacian_variance,
    mean_absolute_difference,
    normalize_face,
    resize_gray,
    shape,
    synthetic_face,
    to_u8_matrix,
    validate_gray_image,
    weighted_centroid,
)


class TestImageOps(unittest.TestCase):
    def test_clamp_u8(self) -> None:
        self.assertEqual(clamp_u8(-50), 0)
        self.assertEqual(clamp_u8(0), 0)
        self.assertEqual(clamp_u8(128.4), 128)
        self.assertEqual(clamp_u8(128.6), 129)
        self.assertEqual(clamp_u8(255), 255)
        self.assertEqual(clamp_u8(300), 255)

    def test_validate_gray_image_valid(self) -> None:
        valid_img = [[100 for _ in range(16)] for _ in range(16)]
        validate_gray_image(valid_img, min_size=8)
        self.assertEqual(shape(valid_img), (16, 16))

    def test_validate_gray_image_invalid(self) -> None:
        with self.assertRaises(ValueError):
            validate_gray_image([], min_size=8)
        with self.assertRaises(ValueError):
            validate_gray_image([[10, 20], [10]], min_size=1)
        with self.assertRaises(ValueError):
            validate_gray_image([[10, 20], [30, 40]], min_size=8)

    def test_resize_gray(self) -> None:
        img = [[(x + y) * 8 for x in range(16)] for y in range(16)]
        resized = resize_gray(img, 32, 24)
        self.assertEqual(len(resized), 24)
        self.assertEqual(len(resized[0]), 32)
        # Check identical dimensions returns copy
        same = resize_gray(img, 16, 16)
        self.assertEqual(same, img)

    def test_histogram_equalize(self) -> None:
        # Dark low-contrast image
        dark_img = [[50 for _ in range(16)] for _ in range(16)]
        eq = histogram_equalize(dark_img)
        self.assertEqual(len(eq), 16)
        self.assertEqual(len(eq[0]), 16)

    def test_laplacian_variance(self) -> None:
        # Smooth flat image has ~0 variance
        flat = [[128 for _ in range(16)] for _ in range(16)]
        self.assertAlmostEqual(laplacian_variance(flat), 0.0)
        # Texture pattern has higher variance
        textured = [[(x % 2) * 255 for x in range(16)] for y in range(16)]
        self.assertGreater(laplacian_variance(textured), 10.0)

    def test_mean_absolute_difference(self) -> None:
        img_a = [[100 for _ in range(16)] for _ in range(16)]
        img_b = [[100 for _ in range(16)] for _ in range(16)]
        self.assertEqual(mean_absolute_difference(img_a, img_b), 0.0)
        img_c = [[200 for _ in range(16)] for _ in range(16)]
        self.assertAlmostEqual(mean_absolute_difference(img_a, img_c), 100.0 / 255.0, places=3)

    def test_weighted_centroid(self) -> None:
        # Symmetrical image centroid is near 0.5, 0.5
        img = [[255 for _ in range(16)] for _ in range(16)]
        img[8][8] = 0  # dark spot in center
        cx, cy = weighted_centroid(img)
        self.assertAlmostEqual(cx, 0.5, places=1)
        self.assertAlmostEqual(cy, 0.5, places=1)

    def test_crop_fraction(self) -> None:
        img = [[x for x in range(20)] for _ in range(20)]
        cropped = crop_fraction(img, x0=0.25, y0=0.25, x1=0.75, y1=0.75)
        self.assertEqual(len(cropped), 10)
        self.assertEqual(len(cropped[0]), 10)

    def test_synthetic_face(self) -> None:
        face = synthetic_face(size=64, noise=0)
        self.assertEqual(len(face), 64)
        self.assertEqual(len(face[0]), 64)
        # Verify valid pixel boundaries
        for row in face:
            for pixel in row:
                self.assertTrue(0 <= pixel <= 255)


if __name__ == "__main__":
    unittest.main()
