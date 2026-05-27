"""hdf5 → hdf5 head/face occlusion pipeline.

Input file layout (compatible with the Astribot-style dataset)::

    images_dict/
      head/   rgb (uint8 1-D, concat JPEG bytes)   rgb_size (int32, N)   rgb_timestamp (float64, N)
      left/   rgb …                                rgb_size …            rgb_timestamp …
      right/  rgb …                                rgb_size …            rgb_timestamp …
    <all other groups / datasets are copied through verbatim>

For each configured camera the pipeline

  1. extracts the raw JPEG bytes for every frame into a temp directory
     (no re-encoding — the SAM 3.1 detector sees the exact bytes that the
     robot wrote);
  2. opens a SAM 3.1 video session against that JPEG folder;
  3. for every frame: decodes the original JPEG → applies blur/mosaic/fill
     on the union of detected masks → re-encodes as JPEG (default quality 95);
  4. concatenates the new JPEG byte stream and rewrites
     ``images_dict/<cam>/rgb`` and ``rgb_size`` in the output hdf5.

Other groups/datasets and `rgb_timestamp` are copied through unchanged.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import h5py
import numpy as np

from ._stream import propagate_masks, session
from .occlusion import OcclusionCfg, apply_occlusion


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class H5CameraSpec:
    """How to handle one camera inside an hdf5 file.

    Attributes
    ----------
    name:
        Sub-group name under ``images_dict/`` (e.g. ``head``, ``left``, ``right``).
    prompt:
        Text prompt passed to SAM 3.1 for this camera. ``""`` skips inference
        and copies the camera through unchanged (still useful if you want to
        document why a camera is skipped).
    jpeg_quality:
        Re-encoding quality for the redacted frames (1..100).
    """

    name: str
    prompt: str = "human head"
    jpeg_quality: int = 95


@dataclass
class H5Stats:
    """Per-camera processing statistics."""

    camera: str
    n_frames: int = 0
    n_frames_with_mask: int = 0
    avg_objects: float = 0.0
    original_bytes: int = 0
    rewritten_bytes: int = 0
    elapsed_sec: float = 0.0

    @property
    def hit_ratio(self) -> float:
        return self.n_frames_with_mask / self.n_frames if self.n_frames else 0.0


@dataclass
class H5Result:
    src_path: str
    dst_path: str
    cameras: List[H5Stats] = field(default_factory=list)
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_jpegs(
    rgb: np.ndarray, sizes: np.ndarray, out_dir: Path,
) -> List[Path]:
    """Write each frame's raw JPEG bytes to ``out_dir/00000.jpg`` … in order.

    SAM 3.1 expects either an mp4 file or a directory of JPEGs whose stems are
    integers (it sorts by ``int(stem)``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    offsets = np.concatenate([[0], np.cumsum(sizes.astype(np.int64))])
    for i, n in enumerate(sizes):
        n = int(n)
        if n <= 0:
            raise ValueError(f"frame {i} has non-positive rgb_size={n}")
        path = out_dir / f"{i:05d}.jpg"
        path.write_bytes(bytes(rgb[offsets[i]:offsets[i] + n]))
        paths.append(path)
    return paths


def _decode_one(rgb: np.ndarray, offsets: np.ndarray, sizes: np.ndarray, idx: int):
    n = int(sizes[idx])
    buf = rgb[offsets[idx]:offsets[idx] + n]
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2.imdecode failed on frame {idx}")
    return img


def _copy_other_datasets(
    src: h5py.Group, dst: h5py.Group, *, skip_paths: Sequence[str]
) -> None:
    """Deep-copy datasets from ``src`` to ``dst`` skipping any names rooted at
    ``skip_paths`` (full posix-style paths from `src.name`).
    """
    skip = set(p.rstrip("/") for p in skip_paths)

    def _walk(s: h5py.Group, d: h5py.Group):
        for k in s.keys():
            child = s[k]
            cur = f"{s.name}/{k}".replace("//", "/")
            if cur in skip:
                continue
            if isinstance(child, h5py.Group):
                sub = d.create_group(k)
                for ak, av in child.attrs.items():
                    sub.attrs[ak] = av
                _walk(child, sub)
            else:
                ds = d.create_dataset(
                    k, data=child[...], dtype=child.dtype,
                    compression=child.compression,
                    compression_opts=child.compression_opts,
                    shuffle=child.shuffle,
                    chunks=child.chunks if child.chunks else None,
                )
                for ak, av in child.attrs.items():
                    ds.attrs[ak] = av

    for ak, av in src.attrs.items():
        dst.attrs[ak] = av
    _walk(src, dst)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_hdf5(
    predictor,
    src_path: str,
    dst_path: str,
    *,
    cameras: Sequence[H5CameraSpec] = (
        H5CameraSpec("head"),
        H5CameraSpec("left"),
        H5CameraSpec("right"),
    ),
    occlusion: Optional[OcclusionCfg] = None,
    work_dir: Optional[str] = None,
    keep_work_dir: bool = False,
    progress_every: int = 100,
    log: Callable[[str], None] = print,
) -> H5Result:
    """Render head/face occlusion into a new hdf5.

    Parameters
    ----------
    predictor:
        Result of ``sam31_head_mask.build_predictor(...)``.
    src_path:
        Path to the input ``.hdf5`` file (read-only).
    dst_path:
        Path to the output ``.hdf5`` file (overwritten if it exists). Will be
        materialised atomically: a ``<dst_path>.tmp`` is written and then
        renamed.
    cameras:
        Per-camera specs. Cameras missing from the file are silently skipped.
    occlusion:
        Rendering knobs. Defaults to a Gaussian blur with ``mask_expand=10``.
    work_dir:
        Where temporary JPEG dirs are placed. Defaults to a fresh tempdir.
    keep_work_dir:
        Don't delete the temp dir on exit (useful for debugging mask quality).
    """
    occlusion = occlusion or OcclusionCfg()

    src_path = os.fspath(src_path)
    dst_path = os.fspath(dst_path)
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)) or ".", exist_ok=True)

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="sam31_h5_")
    else:
        os.makedirs(work_dir, exist_ok=True)
    work_root = Path(work_dir)

    tmp_dst = dst_path + ".tmp"
    if os.path.exists(tmp_dst):
        os.remove(tmp_dst)

    result = H5Result(src_path=src_path, dst_path=dst_path)
    t0 = time.time()

    try:
        log(f"[h5] open src: {src_path}")
        with h5py.File(src_path, "r") as fin, h5py.File(tmp_dst, "w") as fout:
            # 1. Copy everything OTHER than the images we will rewrite.
            cam_paths_to_skip: List[str] = []
            if "images_dict" in fin:
                for spec in cameras:
                    if spec.name in fin["images_dict"]:
                        cam_paths_to_skip.append(f"/images_dict/{spec.name}/rgb")
                        cam_paths_to_skip.append(f"/images_dict/{spec.name}/rgb_size")
                        # rgb_timestamp is COPIED (frame indices unchanged)
            _copy_other_datasets(fin, fout, skip_paths=cam_paths_to_skip)

            # 2. Per-camera occlusion.
            if "images_dict" not in fin:
                log("[h5] WARNING: file has no 'images_dict' group; nothing to redact.")
                return result

            for spec in cameras:
                if spec.name not in fin["images_dict"]:
                    log(f"[h5]   skip camera '{spec.name}' (not present in file)")
                    continue
                stats = _process_one_camera(
                    predictor=predictor,
                    fin=fin,
                    fout=fout,
                    spec=spec,
                    occlusion=occlusion,
                    work_root=work_root,
                    progress_every=progress_every,
                    log=log,
                )
                result.cameras.append(stats)

        os.replace(tmp_dst, dst_path)
        result.elapsed_sec = time.time() - t0
        log(f"[h5] done -> {dst_path} in {result.elapsed_sec:.1f}s")
        return result
    except Exception:
        # Clean partial output on failure.
        if os.path.exists(tmp_dst):
            try:
                os.remove(tmp_dst)
            except OSError:
                pass
        raise
    finally:
        if not keep_work_dir:
            shutil.rmtree(work_root, ignore_errors=True)


def _process_one_camera(
    *,
    predictor,
    fin: h5py.File,
    fout: h5py.File,
    spec: H5CameraSpec,
    occlusion: OcclusionCfg,
    work_root: Path,
    progress_every: int,
    log: Callable[[str], None],
) -> H5Stats:
    g_in = fin["images_dict"][spec.name]
    rgb_in: np.ndarray = g_in["rgb"][...]            # 1-D uint8
    sizes_in: np.ndarray = g_in["rgb_size"][...]      # 1-D int
    timestamps = g_in["rgb_timestamp"][...]
    n_frames = int(len(sizes_in))
    offsets = np.concatenate([[0], np.cumsum(sizes_in.astype(np.int64))])

    # Probe shape from frame 0.
    probe = cv2.imdecode(
        rgb_in[offsets[0]:offsets[0] + int(sizes_in[0])], cv2.IMREAD_COLOR
    )
    if probe is None:
        raise RuntimeError(f"[h5][{spec.name}] cv2.imdecode failed on frame 0")
    H, W = probe.shape[:2]
    log(f"[h5][{spec.name}] {n_frames} frames @ {W}x{H}, prompt={spec.prompt!r}")

    stats = H5Stats(
        camera=spec.name, n_frames=n_frames, original_bytes=int(rgb_in.nbytes)
    )
    t_cam = time.time()

    if not spec.prompt:
        # No-op: copy raw bytes/sizes through (rgb_timestamp is already copied
        # by `_copy_other_datasets`).
        g_out = fout.require_group(f"images_dict/{spec.name}")
        g_out.create_dataset("rgb", data=rgb_in, dtype=rgb_in.dtype)
        g_out.create_dataset("rgb_size", data=sizes_in, dtype=sizes_in.dtype)
        stats.rewritten_bytes = int(rgb_in.nbytes)
        stats.elapsed_sec = time.time() - t_cam
        log(f"[h5][{spec.name}] passthrough (empty prompt)")
        return stats

    # 1. Materialise JPEGs to a temp directory.
    cam_jpg_dir = work_root / spec.name
    if cam_jpg_dir.exists():
        shutil.rmtree(cam_jpg_dir)
    _extract_jpegs(rgb_in, sizes_in, cam_jpg_dir)

    # 2. Stream SAM 3.1 over that folder.
    new_buffers: List[bytes] = [b""] * n_frames
    n_obj_seen: List[int] = []
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(spec.jpeg_quality)]

    with session(predictor, str(cam_jpg_dir), spec.prompt) as session_id:
        for frame_idx, union_mask, n_obj in propagate_masks(
            predictor, session_id, H, W
        ):
            if frame_idx >= n_frames:
                continue
            n_obj_seen.append(n_obj)
            if n_obj > 0:
                stats.n_frames_with_mask += 1
            # Decode original frame from the raw HDF5 buffer (lossless source).
            orig = _decode_one(rgb_in, offsets, sizes_in, frame_idx)
            redacted = (
                apply_occlusion(orig, union_mask, occlusion)
                if union_mask.any()
                else orig
            )
            ok, encoded = cv2.imencode(".jpg", redacted, encode_param)
            if not ok:
                raise RuntimeError(
                    f"[h5][{spec.name}] cv2.imencode failed on frame {frame_idx}"
                )
            new_buffers[frame_idx] = encoded.tobytes()
            if progress_every and (frame_idx + 1) % progress_every == 0:
                elapsed = time.time() - t_cam
                log(
                    f"[h5][{spec.name}]   {frame_idx+1}/{n_frames} "
                    f"avg_obj={np.mean(n_obj_seen):.2f} "
                    f"fps={(frame_idx+1)/max(elapsed,1e-6):.2f}"
                )

    # Frames the predictor did not emit (shouldn't happen with multiplex) →
    # fall back to passthrough of original bytes.
    n_passthrough = 0
    for i in range(n_frames):
        if not new_buffers[i]:
            n_passthrough += 1
            new_buffers[i] = bytes(rgb_in[offsets[i]:offsets[i] + int(sizes_in[i])])
    if n_passthrough:
        log(f"[h5][{spec.name}]   {n_passthrough} frames passed through (no SAM output)")

    # 3. Concatenate and write back.
    new_sizes = np.array([len(b) for b in new_buffers], dtype=sizes_in.dtype)
    new_rgb = np.concatenate(
        [np.frombuffer(b, dtype=np.uint8) for b in new_buffers]
    )

    g_out = fout.require_group(f"images_dict/{spec.name}")
    g_out.create_dataset("rgb", data=new_rgb, dtype=np.uint8)
    g_out.create_dataset("rgb_size", data=new_sizes, dtype=sizes_in.dtype)
    # rgb_timestamp is left untouched: `_copy_other_datasets` already copied it.

    stats.avg_objects = float(np.mean(n_obj_seen)) if n_obj_seen else 0.0
    stats.rewritten_bytes = int(new_rgb.nbytes)
    stats.elapsed_sec = time.time() - t_cam
    log(
        f"[h5][{spec.name}] done in {stats.elapsed_sec:.1f}s: "
        f"hit={stats.hit_ratio*100:.1f}% avg_obj={stats.avg_objects:.2f} "
        f"bytes {stats.original_bytes/1e6:.1f}MB -> {stats.rewritten_bytes/1e6:.1f}MB"
    )
    return stats
