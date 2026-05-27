"""SAM 3.1 multiplex predictor construction.

Thin wrapper around `sam3.model_builder.build_sam3_multiplex_video_predictor`
that

  * applies the small upstream-bug monkey-patch for
    ``Sam3MultiplexTrackingWithInteractivity.init_state(offload_state_to_cpu=…)``;
  * picks a sensible default ``bpe_path`` so callers do not need to know
    where the (namespace) sam3 package lives.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import sam3 as _sam3_pkg
    _DEFAULT_BPE = os.path.join(
        next(iter(_sam3_pkg.__path__)), "assets", "bpe_simple_vocab_16e6.txt.gz"
    )
except Exception:  # noqa: BLE001
    _DEFAULT_BPE = None


DEFAULT_BPE_FILE: Optional[str] = _DEFAULT_BPE


def _patch_offload_kw() -> None:
    """Make `Sam3MultiplexTrackingWithInteractivity.init_state` tolerant of the
    `offload_state_to_cpu` kwarg that `Sam3BasePredictor.start_session` always
    passes (upstream codebase mismatch).
    """
    from sam3.model.sam3_multiplex_tracking import (
        Sam3MultiplexTrackingWithInteractivity,
    )

    if getattr(Sam3MultiplexTrackingWithInteractivity, "_offload_patched", False):
        return
    _orig = Sam3MultiplexTrackingWithInteractivity.init_state

    def _patched(self, *a, **kw):
        kw.pop("offload_state_to_cpu", None)
        return _orig(self, *a, **kw)

    Sam3MultiplexTrackingWithInteractivity.init_state = _patched
    Sam3MultiplexTrackingWithInteractivity._offload_patched = True  # type: ignore[attr-defined]


def build_predictor(
    checkpoint_path: str,
    *,
    bpe_path: Optional[str] = None,
    max_num_objects: int = 8,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    compile: bool = False,
    warm_up: bool = False,
    default_output_prob_thresh: float = 0.5,
):
    """Build a SAM 3.1 multiplex video predictor.

    The predictor exposes ``handle_request`` / ``handle_stream_request`` and is
    safe to reuse across many videos by alternating
    ``start_session``/``add_prompt``/``propagate_in_video``/``close_session``.

    Parameters
    ----------
    checkpoint_path:
        Path to ``sam3.1_multiplex.pt``. The official weights are gated on
        Hugging Face under ``facebook/sam3.1``; several community mirrors
        republish the identical file (see README).
    bpe_path:
        Path to ``bpe_simple_vocab_16e6.txt.gz``. Defaults to the file shipped
        with the installed ``sam3`` package.
    max_num_objects:
        Maximum simultaneously tracked instances per video.
    use_fa3, use_rope_real:
        Flash-Attention-3 / real-valued RoPE. Disabled by default since most
        sites do not have ``flash-attn-3`` installed.
    compile, warm_up:
        ``torch.compile``-based optimisations. Disabled by default to keep
        startup snappy; turn on for long-running services.
    default_output_prob_thresh:
        Sigmoid threshold for mask binarisation.
    """
    _patch_offload_kw()
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if bpe_path is None:
        bpe_path = DEFAULT_BPE_FILE
        if bpe_path is None or not os.path.exists(bpe_path):
            raise FileNotFoundError(
                "Could not locate the SAM 3.1 BPE vocabulary file; pass "
                "`bpe_path=` explicitly. Expected something like "
                "<sam3_package>/assets/bpe_simple_vocab_16e6.txt.gz."
            )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. See README §Checkpoints."
        )

    return build_sam3_multiplex_video_predictor(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        max_num_objects=max_num_objects,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        compile=compile,
        warm_up=warm_up,
        default_output_prob_thresh=default_output_prob_thresh,
    )
