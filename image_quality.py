from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

LOW_LIGHT_MEAN = 105.0
LOW_CONTRAST_STDDEV = 42.0
SOFT_IMAGE_LAPLACIAN_VARIANCE = 115.0


@dataclass(frozen=True)
class HardImageSignals:
    brightness: float
    contrast: float
    sharpness: float
    is_dark: bool
    is_low_contrast: bool
    is_soft: bool

    @property
    def is_hard(self) -> bool:
        return self.is_dark or self.is_low_contrast or self.is_soft


def assess_image_quality(image: np.ndarray) -> HardImageSignals:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, stddev = cv2.meanStdDev(gray)
    brightness = float(mean[0, 0])
    contrast = float(stddev[0, 0])
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return HardImageSignals(
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        is_dark=brightness < LOW_LIGHT_MEAN,
        is_low_contrast=contrast < LOW_CONTRAST_STDDEV,
        is_soft=sharpness < SOFT_IMAGE_LAPLACIAN_VARIANCE,
    )


def build_hard_image_variants(image: np.ndarray) -> list[np.ndarray]:
    """Build a small set of detection-only recovery views for hard images.

    These are only used when the image already looks degraded, so we bias toward
    recovering weak facial structure rather than preserving natural appearance.
    """
    signals = assess_image_quality(image)
    variants: list[np.ndarray] = []

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clip_limit = 3.0 if signals.is_low_contrast else 2.5
    lightness = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lightness)
    variants.append(cv2.cvtColor(cv2.merge((lightness, a_channel, b_channel)), cv2.COLOR_LAB2BGR))

    if signals.is_dark:
        gamma = float(np.clip(0.48 + signals.brightness / 500.0, 0.48, 0.68))
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
        lifted = cv2.LUT(image, lut)
        lifted_lab = cv2.cvtColor(lifted, cv2.COLOR_BGR2LAB)
        lifted_l, lifted_a, lifted_b = cv2.split(lifted_lab)
        lifted_l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lifted_l)
        variants.append(cv2.cvtColor(cv2.merge((lifted_l, lifted_a, lifted_b)), cv2.COLOR_LAB2BGR))

    if signals.is_soft or signals.is_low_contrast:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
        smooth = cv2.GaussianBlur(denoised, (0, 0), 1.2)
        restored = cv2.addWeighted(denoised, 1.60, smooth, -0.60, 0)
        variants.append(restored)

    return variants
