"""
src/ddfp/__init__.py
────────────────────────────────────────────────────────────────────────────
DD-FP (Domain-Decomposed Front Propagation) public API.

Backend selection (once at import time):
  - CuPy installed + GPU available  →  gpu_immersion   (Level-BFS CUDA kernel)
  - otherwise                       →  parallel_immersion  (NumPy/SciPy CPU)

Public functions
────────────────
  immersion_pipeline(vol_u8, verbose=False) → np.ndarray
      uint8 3D volume → u_DWC  (DWC guaranteed, shape = 2W-1, 2H-1, 2D-1)

  run_ddfp_2d(image_f32, verbose=False) → np.ndarray
      float32 2D image [H, W] → u_DWC [2H-1, 2W-1]
      Wrapper called by DDFPPreprocessor.
      Internally: [H,W] → [W,H,1] (uint8) → immersion_pipeline → [2W-1,2H-1].

Backend loading
───────────────
  1. cupy import succeeds + cp.cuda.runtime.getDeviceCount() > 0  →  GPU
  2. otherwise  →  CPU
  3. Decision made once at import time (no per-call overhead).
"""


from __future__ import annotations

import warnings
import numpy as np

# Backend selection (once at import time)

_BACKEND: str = "cpu"   # "gpu" | "cpu"

try:
    import cupy as cp
    _n_devices = cp.cuda.runtime.getDeviceCount()
    if _n_devices > 0:
        from .gpu_immersion import immersion_pipeline as _immersion_pipeline
        _BACKEND = "gpu"
    else:
        raise RuntimeError("no GPU")
except Exception:
    from .parallel_immersion import immersion_pipeline as _immersion_pipeline  # type: ignore
    _BACKEND = "cpu"


def get_backend() -> str:
    """Return the active backend name: 'gpu' or 'cpu'."""
    return _BACKEND


# Public API 1: immersion_pipeline (3D, uint8)

def immersion_pipeline(
    vol_u8: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """
    uint8 3D volume → u_DWC (DWC guaranteed).

    Parameters
    ----------
    vol_u8  : np.ndarray, shape (W, H, D), dtype uint8
    verbose : print progress if True

    Returns
    -------
    u_dwc : np.ndarray, shape (2W-1, 2H-1, 2D-1), dtype float32
            Digitally Well-Composed guaranteed (Boutry et al. 2015 Theorem 2)

    Notes
    -----
    GPU backend uses cupy arrays internally; return value is always
    a numpy array (CPU memory).
    """
    if vol_u8.dtype != np.uint8:
        vol_u8 = np.clip(vol_u8, 0, 255).astype(np.uint8)

    result = _immersion_pipeline(vol_u8, verbose=verbose)

    # Older parallel_immersion may return (u_dwc, ford) tuple; take first element.
    if isinstance(result, tuple):
        result = result[0]

    # Convert cupy array to numpy if needed
    if hasattr(result, "get"):
        result = result.get()
    if hasattr(result, "__array__"):
        result = np.asarray(result)

    return result.astype(np.float32)


# Public API 2: run_ddfp_2d (2D wrapper for DDFPPreprocessor)

def run_ddfp_2d(
    image: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """
    2D float32 image [H, W] → u_DWC [2H-1, 2W-1].

    Called by DDFPPreprocessor.__call__.

    Conversion:
      [H, W] float32 [0,1]
        → × 255 → uint8
        → [W, H, 1] uint8  (axis order for immersion_pipeline)
        → immersion_pipeline → [2W-1, 2H-1, 1] float32
        → [:, :, 0] → [2W-1, 2H-1] float32
        → ÷ 255 → [0, 1] normalised

    Note: immersion_pipeline expects (W, H, D) axis order.
          Image (H, W) → transpose → (W, H) → unsqueeze → (W, H, 1).

    Parameters
    ----------
    image   : np.ndarray, shape (H, W), float32 in [0, 1]
    verbose : print progress if True

    Returns
    -------
    np.ndarray, shape (2H-1, 2W-1), float32 in [0, 1], DWC guaranteed
    """
    if image.ndim != 2:
        raise ValueError(f"[run_ddfp_2d] only 2D input supported, got shape={image.shape}")
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    H, W = image.shape

    # float32 [0,1] → uint8 [0,255]

    vol_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    # (H, W) → (W, H, 1) to match immersion_pipeline axis order (W, H, D)
    vol_3d = vol_u8.T[:, :, np.newaxis]       # shape (W, H, 1)

    # compute u_DWC → shape (2W-1, 2H-1, 1)
    u_dwc_3d = immersion_pipeline(vol_3d, verbose=verbose)

    # (2W-1, 2H-1, 1) → (2W-1, 2H-1) → transpose → (2H-1, 2W-1)
    u_dwc_2d = u_dwc_3d[:, :, 0].T           # shape (2H-1, 2W-1)

    # [0, 255] → [0, 1] normalise
    result = u_dwc_2d / 255.0
    return result.astype(np.float32)


__all__ = ["immersion_pipeline", "run_ddfp_2d", "get_backend"]