"""Unit tests for the renderer. These do NOT touch SAM 3.1; they verify the
masked-pixel transformations are well-defined and reversible-ish."""

import numpy as np
import pytest

from sam31_head_mask.occlusion import OcclusionCfg, apply_occlusion


def _checkerboard(h: int = 64, w: int = 64) -> np.ndarray:
    base = np.indices((h, w)).sum(axis=0) % 2
    rgb = np.stack([base * 200, base * 100, (1 - base) * 200], axis=-1)
    return rgb.astype(np.uint8)


def test_empty_mask_is_noop():
    frame = _checkerboard()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    out = apply_occlusion(frame, mask, OcclusionCfg(mode="blur"))
    np.testing.assert_array_equal(out, frame)


def test_full_mask_blur_changes_pixels():
    frame = _checkerboard()
    mask = np.ones(frame.shape[:2], dtype=bool)
    out = apply_occlusion(frame, mask, OcclusionCfg(mode="blur", blur_ksize=15))
    assert out.shape == frame.shape and out.dtype == frame.dtype
    assert not np.array_equal(out, frame)


def test_partial_mask_keeps_outside_pixels():
    frame = _checkerboard()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[20:40, 20:40] = True
    out = apply_occlusion(frame, mask, OcclusionCfg(mode="mosaic", mosaic_block=4, mask_expand=0))
    # Pixels outside the mask should be unchanged.
    outside = np.ones(frame.shape[:2], dtype=bool)
    outside[20:40, 20:40] = False
    np.testing.assert_array_equal(out[outside], frame[outside])


def test_fill_mode_writes_color():
    frame = _checkerboard()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    mask[10:30, 10:30] = True
    out = apply_occlusion(
        frame, mask, OcclusionCfg(mode="fill", fill_color=(0, 0, 0), mask_expand=0)
    )
    np.testing.assert_array_equal(out[10:30, 10:30], 0)


def test_unknown_mode_raises():
    frame = _checkerboard()
    mask = np.ones(frame.shape[:2], dtype=bool)
    with pytest.raises(ValueError):
        apply_occlusion(frame, mask, OcclusionCfg(mode="nope"))
