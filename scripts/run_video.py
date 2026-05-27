"""CLI: run SAM 3.1 head/face occlusion on a single mp4."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from sam31_head_mask import (  # noqa: E402
    OcclusionCfg,
    build_predictor,
    process_video,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, help="Input mp4 path.")
    p.add_argument("--out", required=True, help="Output mp4 path.")
    p.add_argument("--checkpoint", required=True, help="Path to sam3.1_multiplex.pt")
    p.add_argument("--bpe_path", default=None, help="Override bpe_simple_vocab path.")
    p.add_argument("--prompt", default="human head")
    p.add_argument("--mode", default="blur", choices=["blur", "mosaic", "fill"])
    p.add_argument("--blur_ksize", type=int, default=51)
    p.add_argument("--mosaic_block", type=int, default=24)
    p.add_argument("--mask_expand", type=int, default=10)
    p.add_argument("--prob_thresh", type=float, default=0.5)
    p.add_argument("--max_objects", type=int, default=8)
    p.add_argument(
        "--restrict_region", default=None,
        help='Optional "x1,y1,x2,y2" — masks outside are ignored.',
    )
    p.add_argument("--debug_overlay", action="store_true",
                   help="Draw mask contours in green for QA.")
    p.add_argument("--fourcc", default="mp4v")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    occlusion = OcclusionCfg(
        mode=args.mode,
        blur_ksize=args.blur_ksize,
        mosaic_block=args.mosaic_block,
        mask_expand=args.mask_expand,
        debug_overlay=args.debug_overlay,
    )

    restrict_region = None
    if args.restrict_region:
        restrict_region = tuple(int(v) for v in args.restrict_region.split(","))
        if len(restrict_region) != 4:
            raise SystemExit("--restrict_region must be 'x1,y1,x2,y2'")

    print(f"[run_video] building predictor (ckpt={args.checkpoint})", flush=True)
    predictor = build_predictor(
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe_path,
        max_num_objects=args.max_objects,
        default_output_prob_thresh=args.prob_thresh,
    )

    result = process_video(
        predictor,
        args.video,
        args.out,
        prompt=args.prompt,
        occlusion=occlusion,
        restrict_region=restrict_region,
        fourcc=args.fourcc,
    )
    print(
        f"[run_video] done: frames={result.frames_written} "
        f"avg_obj={result.avg_objects:.2f} hit={result.hit_ratio*100:.1f}% "
        f"prop_fps={result.fps_eff:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
