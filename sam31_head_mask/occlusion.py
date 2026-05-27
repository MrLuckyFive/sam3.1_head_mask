"""Pixel-level occlusion rendering (blur / mosaic / fill) over binary masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class OcclusionCfg:
    """Knobs controlling how detected mask regions are rendered."""

    mode: str = "blur"            # "blur" | "mosaic" | "fill"
    blur_ksize: int = 51          # Gaussian kernel size (odd)
    mosaic_block: int = 24        # pixel block edge for the mosaic mode
    fill_color: Tuple[int, int, int] = (0, 0, 0)  # BGR fill for "fill"
    mask_expand: int = 10         # dilation radius (pixels) before drawing
    debug_overlay: bool = False   # draw mask contour for visual QA


def apply_occlusion(
    frame_bgr: np.ndarray, mask: np.ndarray, cfg: OcclusionCfg
) -> np.ndarray:
    """Render `cfg.mode` occlusion over the pixels where `mask` is truthy.

    Both `frame_bgr` and the returned image are H×W×3 uint8 in BGR.
    """
    if mask is None or not mask.any():
        return frame_bgr

    if cfg.mask_expand > 0:
        k = cfg.mask_expand * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0

    if cfg.mode == "blur":
        k = max(3, cfg.blur_ksize | 1)
        blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
        out = np.where(mask[..., None], blurred, frame_bgr)
    elif cfg.mode == "mosaic":
        b = max(2, cfg.mosaic_block)
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(
            frame_bgr, (max(1, w // b), max(1, h // b)),
            interpolation=cv2.INTER_LINEAR,
        )
        mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        out = np.where(mask[..., None], mosaic, frame_bgr)
    elif cfg.mode == "fill":
        out = frame_bgr.copy()
        out[mask] = cfg.fill_color
    else:
        raise ValueError(f"unknown occlusion mode: {cfg.mode!r}")

    if cfg.debug_overlay:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, (0, 255, 0), 2)

    return out
