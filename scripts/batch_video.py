"""CLI: process many mp4 files in one process (the predictor is loaded once).

Reads file paths line-by-line from a manifest (or recursively scans a directory)
and writes a redacted copy of each to ``--output_dir`` with the configured
suffix. Progress is persisted to a JSON state file so re-runs skip completed
items.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import List

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent.parent))

from sam31_head_mask import (  # noqa: E402
    OcclusionCfg,
    build_predictor,
    process_video,
)


_STOP = False


def _on_sig(signum, frame):  # noqa: D401
    global _STOP
    _STOP = True
    print(f"\n[batch_video] received signal {signum}; will exit after current item\n", flush=True)


signal.signal(signal.SIGTERM, _on_sig)
signal.signal(signal.SIGINT, _on_sig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="Text file: one input mp4 path per line.")
    src.add_argument("--input_dir", help="Recursively glob *.mp4 under this directory.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--state", required=True, help="Resumable progress JSON.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bpe_path", default=None)
    p.add_argument("--suffix", default="head_blur", help="Output filename suffix.")
    p.add_argument("--prompt", default="human head")
    p.add_argument("--mode", default="blur", choices=["blur", "mosaic", "fill"])
    p.add_argument("--blur_ksize", type=int, default=51)
    p.add_argument("--mosaic_block", type=int, default=24)
    p.add_argument("--mask_expand", type=int, default=10)
    p.add_argument("--prob_thresh", type=float, default=0.5)
    p.add_argument("--max_objects", type=int, default=8)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args(argv)


def _enumerate_inputs(args) -> List[Path]:
    if args.manifest:
        paths = [
            Path(line.strip())
            for line in Path(args.manifest).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        paths = sorted(Path(args.input_dir).rglob("*.mp4"))
    end = len(paths) if args.end < 0 else args.end
    paths = paths[args.start:end]
    if args.limit and len(paths) > args.limit:
        paths = paths[: args.limit]
    return paths


def _load_state(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"processed": {}, "failed": {}}
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        p.rename(p.with_suffix(p.suffix + ".corrupt." + str(int(time.time()))))
        return {"processed": {}, "failed": {}}


def _save_state(path: str, state: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    os.replace(tmp, p)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    paths = _enumerate_inputs(args)
    print(f"[batch_video] {len(paths)} input(s) selected", flush=True)
    if args.dry_run:
        for p in paths[:15]:
            stem = p.stem
            out = Path(args.output_dir) / f"{stem}_{args.suffix}.mp4"
            print(f"  would write {p} -> {out}")
        if len(paths) > 15:
            print(f"  ... ({len(paths)-15} more)")
        return

    state = _load_state(args.state)
    done = {k for k, v in state.get("processed", {}).items() if v.get("ok")}
    todo = [p for p in paths if str(p) not in done]
    print(f"[batch_video] {len(paths)-len(todo)} already done, {len(todo)} to do", flush=True)

    print(f"[batch_video] building predictor (ckpt={args.checkpoint})", flush=True)
    t0 = time.time()
    predictor = build_predictor(
        checkpoint_path=args.checkpoint,
        bpe_path=args.bpe_path,
        max_num_objects=args.max_objects,
        default_output_prob_thresh=args.prob_thresh,
    )
    print(f"[batch_video] predictor ready in {time.time()-t0:.1f}s", flush=True)

    occlusion = OcclusionCfg(
        mode=args.mode,
        blur_ksize=args.blur_ksize,
        mosaic_block=args.mosaic_block,
        mask_expand=args.mask_expand,
    )

    ok, fail = 0, 0
    t_batch = time.time()
    for i, src in enumerate(todo, 1):
        if _STOP:
            print(f"[batch_video] graceful stop before {src}", flush=True)
            break
        out = Path(args.output_dir) / f"{src.stem}_{args.suffix}.mp4"
        tmp = out.with_suffix(out.suffix + ".tmp.mp4")
        print(f"\n[{i}/{len(todo)}] {src}  ->  {out}", flush=True)
        last_err = ""
        for attempt in range(1, args.retries + 2):
            try:
                t_ep = time.time()
                result = process_video(
                    predictor,
                    str(src),
                    str(tmp),
                    prompt=args.prompt,
                    occlusion=occlusion,
                )
                os.replace(tmp, out)
                state["processed"][str(src)] = {
                    "ok": True,
                    "out": str(out),
                    "frames": result.frames_written,
                    "avg_objects": result.avg_objects,
                    "hit_ratio": result.hit_ratio,
                    "prop_fps": result.fps_eff,
                    "elapsed_sec": time.time() - t_ep,
                    "prompt": args.prompt,
                    "mode": args.mode,
                    "ts": int(time.time()),
                }
                state.get("failed", {}).pop(str(src), None)
                _save_state(args.state, state)
                ok += 1
                break
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                print(f"  [attempt {attempt}] ERROR: {last_err}", flush=True)
                traceback.print_exc()
                if tmp.exists():
                    try: tmp.unlink()
                    except OSError: pass
                if attempt > args.retries:
                    break
                time.sleep(2 * attempt)
        else:
            pass
        if str(src) not in state["processed"] or not state["processed"][str(src)].get("ok"):
            fail += 1
            state.setdefault("failed", {})[str(src)] = {
                "error": last_err, "ts": int(time.time()),
            }
            _save_state(args.state, state)

    el = time.time() - t_batch
    print(f"\n[batch_video] done: ok={ok} fail={fail} in {el:.1f}s ({el/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
