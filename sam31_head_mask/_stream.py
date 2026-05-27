"""Internal helper: pulse a SAM 3.1 video session and aggregate per-frame masks.

Centralises the
``start_session → add_prompt → propagate_in_video → close_session`` choreography
so video / hdf5 pipelines can share it.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


def _aggregate_mask(
    response_outputs: dict,
    target_h: int,
    target_w: int,
    restrict_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, int]:
    """Union all object masks on a frame into a single binary mask of (H, W)."""
    agg = np.zeros((target_h, target_w), dtype=bool)
    n_obj = 0
    binary_masks = response_outputs.get("out_binary_masks")
    if binary_masks is None or len(binary_masks) == 0:
        return agg, 0
    for m in binary_masks:
        if m is None:
            continue
        m = np.asarray(m).astype(bool)
        if m.shape != (target_h, target_w):
            m = cv2.resize(
                m.astype(np.uint8), (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if restrict_mask is not None:
            m = m & restrict_mask
        if not m.any():
            continue
        agg |= m
        n_obj += 1
    return agg, n_obj


@contextmanager
def session(predictor, resource_path: str, prompt: str, frame_index: int = 0):
    """Context manager that yields a session_id and tears it down on exit."""
    resp = predictor.handle_request(
        request=dict(type="start_session", resource_path=resource_path)
    )
    session_id = resp["session_id"]
    try:
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=prompt,
            )
        )
        yield session_id
    finally:
        try:
            predictor.handle_request(
                request=dict(type="close_session", session_id=session_id)
            )
        except Exception:  # noqa: BLE001
            pass


def propagate_masks(
    predictor,
    session_id: str,
    target_h: int,
    target_w: int,
    *,
    restrict_mask: Optional[np.ndarray] = None,
) -> Iterable[Tuple[int, np.ndarray, int]]:
    """Generator yielding ``(frame_index, union_mask_HxW_bool, n_objects)`` for
    each frame the predictor emits.
    """
    for response in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        frame_idx = response["frame_index"]
        agg, n_obj = _aggregate_mask(
            response["outputs"], target_h, target_w, restrict_mask
        )
        yield frame_idx, agg, n_obj
