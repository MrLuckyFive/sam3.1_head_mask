"""CLI: run SAM 3.1 head/face occlusion on a single hdf5 file.

The default behaviour processes the three cameras ``head``, ``left``, ``right``
with the prompt ``head`` and Gaussian blur. Cameras absent in the file are
silently skipped.

Use ``--cameras head:head left: right:human face`` to override:
each item is ``<camera_name>:<text_prompt>``; an empty prompt means
"copy this camera through without redaction".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from sam31_head_mask import (  # noqa: E402
    H5CameraSpec,
    OcclusionCfg,
    build_predictor,
    process_hdf5,
)


def _parse_camera_spec(item: str) -> H5CameraSpec:
    if ":" in item:
        name, prompt = item.split(":", 1)
    else:
        name, prompt = item, "human head"
    return H5CameraSpec(name=name.strip(), prompt=prompt.strip())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Input .hdf5 path.")
    p.add_argument("--dst", required=True, help="Output .hdf5 path.")
    p.add_argument("--checkpoint", required=True, help="Path to sam3.1_multiplex.pt")
    p.add_argument("--bpe_path", default=None)
    p.add_argument(
        "--cameras", nargs="+",
        default=["head:human head", "left:human head", "right:human head"],
        help='List of "<camera_name>:<prompt>". Empty prompt = copy through.',
    )
    p.add_argument("--mode", default="blur", choices=["blur", "mosaic", "fill"])
    p.add_argument("--blur_ksize", type=int, default=51)
    p.add_argument("--mosaic_block", type=int, default=24)
    p.add_argument("--mask_expand", type=int, default=10)
    p.add_argument("--jpeg_quality", type=int, default=95)
    p.add_argument("--prob_thresh", type=float, default=0.5)
    p.add_argument("--max_objects", type=int, default=8)
    p.add_argument("--work_dir", default=None,
                   help="Where to drop temporary JPEG dirs. Default: tempfile.")
    p.add_argument("--keep_work_dir", action="store_true")
    p.add_argument("--debug_overlay", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    cameras = [_parse_camera_spec(item) for item in args.cameras]
    for spec in cameras:
        spec.jpeg_quality = args.jpeg_quality

    occlusion = OcclusionCfg(
        mode=args.mode,
        blur_ksize=args.blur_ksize,
        mosaic_block=args.mosaic_block,
        mask_expand=args.mask_expand,
        debug_overlay=args.debug_overlay,
    )

    print(f"[run_h5] building predictor (ckpt={args.checkpoint})", flush=True)
    predictor = build_predictor(
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe_path,
        max_num_objects=args.max_objects,
        default_output_prob_thresh=args.prob_thresh,
    )

    result = process_hdf5(
        predictor,
        args.src,
        args.dst,
        cameras=cameras,
        occlusion=occlusion,
        work_dir=args.work_dir,
        keep_work_dir=args.keep_work_dir,
    )
    print(f"\n[run_h5] summary for {result.src_path}", flush=True)
    print(f"  -> {result.dst_path}  ({result.elapsed_sec:.1f}s total)", flush=True)
    for s in result.cameras:
        print(
            f"  [{s.camera}] frames={s.n_frames} "
            f"hit={s.hit_ratio*100:.1f}% avg_obj={s.avg_objects:.2f} "
            f"bytes {s.original_bytes/1e6:.1f}MB -> {s.rewritten_bytes/1e6:.1f}MB "
            f"in {s.elapsed_sec:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
