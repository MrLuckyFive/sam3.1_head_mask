"""sam31_head_mask — privacy-preserving head/face occlusion using SAM 3.1.

Public entry points:
    build_predictor    — load the SAM 3.1 multiplex video predictor
    OcclusionCfg       — knobs for blur/mosaic/fill rendering
    process_video      — single-video pipeline (mp4 → mp4)
    process_hdf5       — single-hdf5 pipeline (.hdf5 → .hdf5, JPEG byte-stream layout)
"""

from .sam_runtime import DEFAULT_BPE_FILE, build_predictor  # noqa: F401
from .occlusion import OcclusionCfg, apply_occlusion  # noqa: F401
from .video_pipeline import ProcessResult, process_video  # noqa: F401
from .h5_pipeline import H5CameraSpec, process_hdf5  # noqa: F401

__version__ = "0.1.0"
