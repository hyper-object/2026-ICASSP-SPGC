from __future__ import annotations
import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import matplotlib.pyplot as plt


ArrayLike = Union[np.ndarray, torch.Tensor]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Torch/NumPy -> NumPy (no copy if possible)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

def _ensure_chw(x: ArrayLike) -> np.ndarray:
    """
    Accept CHW or HWC and return CHW.
    For single-channel images, supports C=1 layouts too.
    """
    arr = _to_numpy(x)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    # If HWC
    if arr.shape[0] not in (1, 3, 4):  # likely H
        arr = np.transpose(arr, (2, 0, 1))
    return arr

def _chw_to_hwc01(x: ArrayLike, eps: float = 1e-8) -> np.ndarray:
    """CHW -> HWC in [0,1] (per-image min/max normalize for display only)."""
    chw = _ensure_chw(x).astype(np.float32)
    hwc = np.transpose(chw, (1, 2, 0))
    vmin, vmax = np.nanmin(hwc), np.nanmax(hwc)
    if vmax <= vmin + eps:
        return np.zeros_like(hwc, dtype=np.float32)
    return (hwc - vmin) / (vmax - vmin + eps)

def _is_single_channel(chw: np.ndarray) -> bool:
    return chw.shape[0] == 1