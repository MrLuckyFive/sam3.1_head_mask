"""mp4 → mp4 head/face occlusion pipeline."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from ._stream import propagate_masks, session
from .occlusion import OcclusionCfg, apply_occlusion


@dataclass
class ProcessResult:
    frames_written: int
    avg_objects: float
    hit_ratio: float       # fraction of frames with at least one mask
    elapsed_sec: float
    fps_eff: float


def process_video(
    predictor,
    video_path: str,
    out_path: str,
    *,
    prompt: str = "human head",
    occlusion: Optional[OcclusionCfg] = None,
    restrict_region: Optional[Tuple[int, int, int, int]] = None,
    fourcc: str = "mp4v",
    progress_every: int = 100,
    log: Callable[[str], None] = print,
) -> ProcessResult:
    """Run SAM 3.1 over ``video_path`` and write the masked video to ``out_path``.

    The whole video is streamed once: each frame is read from the input,
    overlaid with the union of SAM 3.1 masks emitted for that frame index, and
    written to the output. Frames not yet seen by the predictor are written
    untouched.
    """
    occlusion = occlusion or OcclusionCfg()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    log(f"[video] open: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log(f"[video] {width}x{height}@{fps:.2f} total={total}")

    restrict_mask = None
    if restrict_region is not None:
        x1, y1, x2, y2 = restrict_region
        restrict_mask = np.zeros((height, width), dtype=bool)
        restrict_mask[y1:y2, x1:x2] = True

    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open VideoWriter: {out_path}")

    cur_idx = 0
    n_objs_per_frame: list[int] = []
    t0 = time.time()

    try:
        with session(predictor, video_path, prompt) as session_id:
            for frame_idx, union_mask, n_obj in propagate_masks(
                predictor, session_id, height, width,
                restrict_mask=restrict_mask,
            ):
                while cur_idx <= frame_idx:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    if cur_idx < frame_idx:
                        writer.write(frame_bgr)
                        cur_idx += 1
                        continue
                    out_frame = apply_occlusion(frame_bgr, union_mask, occlusion)
                    writer.write(out_frame)
                    n_objs_per_frame.append(n_obj)
                    cur_idx += 1
                if progress_every and (frame_idx + 1) % progress_every == 0:
                    elapsed = time.time() - t0
                    avg = float(np.mean(n_objs_per_frame)) if n_objs_per_frame else 0.0
                    log(
                        f"  frame {frame_idx+1}/{total} avg_obj={avg:.2f} "
                        f"fps={(frame_idx+1)/max(elapsed, 1e-6):.2f}"
                    )

            # Drain any unprocessed trailing frames untouched (predictor usually
            # covers all of them, but cv2 frame_count is not always exact).
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                writer.write(frame_bgr)
                cur_idx += 1
    finally:
        writer.release()
        cap.release()

    elapsed = time.time() - t0
    avg = float(np.mean(n_objs_per_frame)) if n_objs_per_frame else 0.0
    hit = (
        float(np.mean([1.0 if x > 0 else 0.0 for x in n_objs_per_frame]))
        if n_objs_per_frame
        else 0.0
    )
    return ProcessResult(
        frames_written=cur_idx,
        avg_objects=avg,
        hit_ratio=hit,
        elapsed_sec=elapsed,
        fps_eff=cur_idx / max(elapsed, 1e-6),
    )
