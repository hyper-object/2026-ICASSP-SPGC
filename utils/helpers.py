from __future__ import annotations
import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import colour 

import torch



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

def _as_hwc(x: np.ndarray) -> np.ndarray:
    """
    Accepts (H,W,C) or (C,H,W). Returns (H,W,C).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")
    if x.shape[0] in (1, 3) and x.shape[0] < x.shape[-1]:
        # Likely CHW
        return np.transpose(x, (1, 2, 0))
    return x  # already HWC

def _check_shapes(a: np.ndarray, b: np.ndarray):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

def _apply_mask(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flattens masked pixels across H and W (keeps channels).
    Returns arrays shaped (N, C) where N is number of valid pixels.
    """
    H, W, C = a.shape
    if mask is None:
        A = a.reshape(-1, C)
        B = b.reshape(-1, C)
        # drop NaN pairs
        mvalid = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
        return A[mvalid], B[mvalid]
    m = _to_numpy(mask).astype(bool)
    if m.shape != (H, W):
        raise ValueError(f"mask must be (H,W) = {(H,W)}, got {m.shape}")
    A = a[m].reshape(-1, C)
    B = b[m].reshape(-1, C)
    mvalid = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    return A[mvalid], B[mvalid]

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.nanmean(d * d))

def _to_hwc(x: torch.Tensor) -> np.ndarray:
    """(N,C,H,W) -> (H,W,C) numpy for first item."""
    return x.detach().permute(1, 2, 0).contiguous().cpu().numpy()


def _deltaE00_mean(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    """Mean CIEDE2000 between two sRGB images in [0,1]."""
    XYZ1 = colour.sRGB_to_XYZ(rgb1)
    XYZ2 = colour.sRGB_to_XYZ(rgb2)
    Lab1 = colour.XYZ_to_Lab(XYZ1)
    Lab2 = colour.XYZ_to_Lab(XYZ2)
    dE = colour.difference.delta_E(Lab1.reshape(-1, 3), Lab2.reshape(-1, 3), method="CIE 2000")
    return float(np.mean(dE))