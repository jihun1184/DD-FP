"""
src/preprocessing/preprocessor.py
-----------------------------------------------------------------------------
Three preprocessing conditions -- all produce identical (2H-1)x(2W-1) output.

Strategy C (recommended):
  Unify all conditions to DD-FP expanded grid size.
  Eliminates reviewer objection of "resolution-driven performance difference".

  Condition    Method                          DWC guarantee   Cache
  ------------------------------------------------------------------
  no_interp    nearest-neighbor zoom           No              .npy
  naive_interp bilinear zoom                   No              .npy
  ddfp         DD-FP front propagation         Yes (delta>=1)  .npy

Output size: H x W  ->  (2H-1) x (2W-1)

Label handling: binary masks are always expanded with nearest-neighbor then
re-binarised.

Cache mechanism (shared by all three conditions)
------------------------------------------------
All preprocessors accept an optional cache_dir via cfg:
  no_interp    -> cfg.preprocessing.no_interp_cache_dir    (default None)
  naive_interp -> cfg.preprocessing.naive_interp_cache_dir (default None)
  ddfp         -> cfg.preprocessing.ddfp_cache_dir          (default None)

When cache_dir is set and sample_id is provided:
  HIT  -- load {cache_dir}/{sample_id}.npy, skip recomputation
  MISS -- compute, save result, return

Cache file: {cache_dir}/{sample_id}.npy  (image only; labels are cheap).
Shape mismatch in a cached file raises RuntimeError (stale cache guard).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import zoom as scipy_zoom


# -----------------------------------------------------------------------------
# Shape helper
# -----------------------------------------------------------------------------

def expanded_shape(h: int, w: int) -> tuple[int, int]:
    """DD-FP expanded grid target size."""
    return 2 * h - 1, 2 * w - 1


def zoom_to_expanded(arr: np.ndarray, order: int,
                     is_label: bool = False) -> np.ndarray:
    """
    arr [H, W] -> [(2H-1), (2W-1)] float32

    order    : 0=nearest, 1=bilinear
    is_label : if True, re-binarise output at threshold 0.5
    """
    h, w = arr.shape
    if h < 2 or w < 2:          # 1px edge case
        return arr.astype(np.float32)
    th, tw = expanded_shape(h, w)
    out = scipy_zoom(arr, (th / h, tw / w), order=order, prefilter=False)
    out = out.astype(np.float32)
    if is_label:
        out = (out >= 0.5).astype(np.float32)
    return out


# -----------------------------------------------------------------------------
# Base  (shared cache logic lives here)
# -----------------------------------------------------------------------------

class BasePreprocessor(ABC):
    """
    Common interface for all three preprocessing conditions.

    __call__(image, label, sample_id)
      image     : [H, W] float32 in [0, 1]
      label     : [H, W] float32 binary {0, 1}, or None
      sample_id : cache key (str); cache is skipped if None
    Returns: (image_out [(2H-1)x(2W-1)], label_out same size or None)

    Cache protocol
    --------------
    Subclasses set self._cache_dir (Path or None) in __init__.
    __call__ should call self._load_cache / self._save_cache around
    self._compute(image).
    The helper method self._cached_call(image, sample_id) encapsulates
    the full hit/miss/save flow so subclasses need only implement _compute.
    """

    # subclasses set this in __init__
    _cache_dir: Path | None = None

    @abstractmethod
    def _compute(self, image: np.ndarray) -> np.ndarray:
        """Perform the actual (expensive) preprocessing. No caching here."""
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        image: np.ndarray,
        label: np.ndarray | None = None,
        sample_id: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # -- input validation -----------------------------------------------------

    def _check(self, arr: np.ndarray, tag: str) -> np.ndarray:
        if arr.ndim != 2:
            raise ValueError(f"[{tag}] Only 2D input supported. shape={arr.shape}")
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr

    def _upsample_label(self, label: np.ndarray | None) -> np.ndarray | None:
        if label is None:
            return None
        label = self._check(label, "label")
        return zoom_to_expanded(label, order=0, is_label=True)

    # -- cache helpers (shared by all three preprocessors) --------------------

    def _cache_path(self, sample_id: str | None) -> Path | None:
        """Return cache file path, or None if caching is disabled."""
        if self._cache_dir is None or sample_id is None:
            return None
        return self._cache_dir / f"{sample_id}.npy"

    def _verify_shape(self, out: np.ndarray, h: int, w: int) -> None:
        """Raise RuntimeError if cached/computed shape does not match."""
        target = expanded_shape(h, w)
        if out.shape != target:
            raise RuntimeError(
                f"[{self.name}] shape mismatch: got {out.shape}, "
                f"expected {target} (input {h}x{w})"
            )

    def _cached_call(self, image: np.ndarray,
                     sample_id: str | None) -> np.ndarray:
        """
        Full cache hit/miss/save flow.

        1. If cache file exists -> load and return (skip _compute).
        2. Otherwise -> call _compute -> save to cache -> return.
        3. If _cache_dir is None or sample_id is None -> call _compute directly.
        """
        h, w  = image.shape
        cpath = self._cache_path(sample_id)

        if cpath is not None and cpath.exists():
            # Cache HIT
            img_out = np.load(cpath).astype(np.float32)
            self._verify_shape(img_out, h, w)
            return img_out

        # Cache MISS (or caching disabled)
        img_out = self._compute(image)
        self._verify_shape(img_out, h, w)

        if cpath is not None:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.save(cpath, img_out)

        return img_out


# -----------------------------------------------------------------------------
# Condition 1: no_interp  (nearest-neighbour)
# -----------------------------------------------------------------------------

class NoInterpPreprocessor(BasePreprocessor):
    """
    Nearest-neighbour zoom -> (2H-1)x(2W-1).
    No DWC guarantee.

    Cache: cfg.preprocessing.no_interp_cache_dir (default None)
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        cache = getattr(cfg.preprocessing, "no_interp_cache_dir", None)
        self._cache_dir = Path(cache) if cache is not None else None

    def _compute(self, image: np.ndarray) -> np.ndarray:
        return zoom_to_expanded(image, order=0)

    def __call__(self, image, label=None, sample_id=None):
        image = self._check(image, "NoInterp")
        img_out = self._cached_call(image, sample_id)
        return img_out, self._upsample_label(label)


# -----------------------------------------------------------------------------
# Condition 2: naive_interp  (bilinear)
# -----------------------------------------------------------------------------

class NaiveInterpPreprocessor(BasePreprocessor):
    """
    Bilinear zoom -> (2H-1)x(2W-1).
    No DWC guarantee.

    Parameters
    ----------
    cfg.preprocessing.naive_mode           : "bilinear" | "nearest"
    cfg.preprocessing.naive_interp_cache_dir : cache root dir (default None)
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        mode = getattr(cfg.preprocessing, "naive_mode", "bilinear")
        self.order = 1 if mode == "bilinear" else 0
        cache = getattr(cfg.preprocessing, "naive_interp_cache_dir", None)
        self._cache_dir = Path(cache) if cache is not None else None

    def _compute(self, image: np.ndarray) -> np.ndarray:
        try:
            return zoom_to_expanded(image, order=self.order)
        except Exception as e:
            raise RuntimeError(f"[NaiveInterp] zoom failed: {e}") from e

    def __call__(self, image, label=None, sample_id=None):
        image = self._check(image, "NaiveInterp")
        img_out = self._cached_call(image, sample_id)
        return img_out, self._upsample_label(label)


# -----------------------------------------------------------------------------
# Condition 3: ddfp  (DWC guaranteed, cache-first)
# -----------------------------------------------------------------------------

class DDFPPreprocessor(BasePreprocessor):
    """
    DD-FP: H x W -> (2H-1) x (2W-1), DWC guaranteed (Theorem 3.1).

    Implementation: src.ddfp.run_ddfp_2d (CuPy GPU or NumPy CPU auto-selected)

    Parameters
    ----------
    cfg.preprocessing.ddfp_overlap   : delta (>= 1 required, Theorem 3.1)
    cfg.preprocessing.ddfp_cache_dir : cache root directory (default None)
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        self.overlap = int(getattr(cfg.preprocessing, "ddfp_overlap", 1))
        cache = getattr(cfg.preprocessing, "ddfp_cache_dir", None)
        self._cache_dir = Path(cache) if cache is not None else None

        if self.overlap < 1:
            raise ValueError(
                f"[DDFP] ddfp_overlap >= 1 required (Theorem 3.1). "
                f"Got delta={self.overlap}"
            )

        try:
            from src.ddfp import run_ddfp_2d, get_backend
            self._run_fn  = run_ddfp_2d
            self._backend = get_backend()
        except ImportError as e:
            raise ImportError(
                "[DDFP] src.ddfp module not found. "
                "Check that src/ddfp/ is present."
            ) from e

    def _compute(self, image: np.ndarray) -> np.ndarray:
        """Call run_ddfp_2d directly (GPU/CPU auto-selected)."""
        return self._run_fn(image, verbose=False)

    def __call__(self, image, label=None, sample_id=None):
        image = self._check(image, "DDFP")
        img_out = self._cached_call(image, sample_id)
        return img_out, self._upsample_label(label)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

_REGISTRY: dict[str, type[BasePreprocessor]] = {
    "no_interp":    NoInterpPreprocessor,
    "naive_interp": NaiveInterpPreprocessor,
    "ddfp":         DDFPPreprocessor,
}


def get_preprocessor(cfg: SimpleNamespace) -> BasePreprocessor:
    ptype = cfg.preprocessing.type
    if ptype not in _REGISTRY:
        raise ValueError(
            f"[Preprocessor] Unknown type: '{ptype}'. "
            f"Valid: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[ptype](cfg)