from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
from .helpers import _to_numpy, _as_hwc, _check_shapes, _apply_mask

ArrayLike = Union[np.ndarray]

def rmse(
    ref: ArrayLike,
    est: ArrayLike,
    per_channel: bool = False,
    mask: Optional[np.ndarray] = None,
) -> Union[float, np.ndarray]:
    """
    Root Mean Squared Error between ref and est.

    Args
    ----
    ref, est : (H,W,C) or (C,H,W) arrays
    per_channel : if True, returns RMSE per channel (C,)
    mask : optional (H,W) boolean ROI

    Returns
    -------
    float or (C,) np.ndarray
    """
    A = _as_hwc(_to_numpy(ref)).astype(np.float64)
    B = _as_hwc(_to_numpy(est)).astype(np.float64)
    _check_shapes(A, B)
    if mask is None and not np.isnan(A).any() and not np.isnan(B).any():
        d = A - B
        if per_channel:
            return np.sqrt(np.mean(d.reshape(-1, A.shape[2])**2, axis=0))
        return float(np.sqrt(np.mean(d * d)))
    # masked / NaN-safe path
    Af, Bf = _apply_mask(A, B, mask)
    if per_channel:
        return np.sqrt(np.mean((Af - Bf) ** 2, axis=0))
    return float(np.sqrt(np.mean((Af - Bf) ** 2)))


def psnr(
    ref: ArrayLike,
    est: ArrayLike,
    data_range: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).

    Args
    ----
    ref, est : (H,W,C) or (C,H,W)
    data_range : peak-to-peak value of the reference.
        If None, infer from dtype/value range:
        - float -> use max(ref) - min(ref)
        - integer -> use max of dtype (e.g., 255 for uint8)
    mask : optional (H,W)
    """
    A = _as_hwc(_to_numpy(ref)).astype(np.float64)
    B = _as_hwc(_to_numpy(est)).astype(np.float64)
    _check_shapes(A, B)

    if data_range is None:
        if np.issubdtype(A.dtype, np.floating):
            vmin = np.nanmin(A)
            vmax = np.nanmax(A)
            data_range = float(max(vmax - vmin, eps))
        else:
            info = np.iinfo(A.dtype)
            data_range = float(info.max - info.min)
    if mask is None:
        mse = np.nanmean((A - B) ** 2) if np.isnan(A).any() or np.isnan(B).any() else np.mean((A - B) ** 2)
    else:
        Af, Bf = _apply_mask(A, B, mask)
        mse = float(np.nanmean((Af - Bf) ** 2))
    mse = max(mse, eps)
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def _gaussian_kernel2d(win_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(win_size) - win_size // 2
    g1d = np.exp(-0.5 * (ax / sigma) ** 2)
    g1d /= g1d.sum()
    g2d = np.outer(g1d, g1d)
    g2d /= g2d.sum()
    return g2d.astype(np.float64)

def _pad_reflect(img: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad)), mode="reflect")

def _filter2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2D 'same' convolution with reflect padding. img: (H,W), kernel: (k,k)
    """
    H, W = img.shape
    k = kernel.shape[0]
    pad = k // 2
    src = _pad_reflect(img, pad)
    out = np.zeros_like(img, dtype=np.float64)
    # naive implementation; OK for analysis / validation
    for i in range(H):
        ii = i + pad
        for j in range(W):
            jj = j + pad
            region = src[ii - pad:ii + pad + 1, jj - pad:jj + pad + 1]
            out[i, j] = float(np.sum(region * kernel))
    return out

def ssim(
    ref: ArrayLike,
    est: ArrayLike,
    data_range: Optional[float] = None,
    win_size: int = 11,
    sigma: float = 1.5,
    K: Tuple[float, float] = (0.01, 0.03),
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Structural Similarity Index (mean over channels).

    Args
    ----
    ref, est : (H,W,C) or (C,H,W)
    data_range : L in SSIM; if None, inferred like PSNR
    win_size, sigma : Gaussian window parameters
    K : (K1, K2) constants (default sRGB spec-like)
    mask : optional (H,W) ROI for averaging

    Returns
    -------
    float : mean SSIM across channels
    """
    A = _as_hwc(_to_numpy(ref)).astype(np.float64)
    B = _as_hwc(_to_numpy(est)).astype(np.float64)
    _check_shapes(A, B)
    H, W, C = A.shape

    if data_range is None:
        if np.issubdtype(A.dtype, np.floating):
            vmin = np.nanmin(A)
            vmax = np.nanmax(A)
            data_range = float(max(vmax - vmin, 1e-12))
        else:
            info = np.iinfo(A.dtype); data_range = float(info.max - info.min)
    K1, K2 = K
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    kernel = _gaussian_kernel2d(win_size, sigma)

    ssim_maps = []
    for ch in range(C):
        x = A[..., ch]
        y = B[..., ch]

        mu_x = _filter2d(x, kernel)
        mu_y = _filter2d(y, kernel)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = _filter2d(x * x, kernel) - mu_x2
        sigma_y2 = _filter2d(y * y, kernel) - mu_y2
        sigma_xy = _filter2d(x * y, kernel) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        ssim_map = num / (den + 1e-12)
        ssim_maps.append(ssim_map)

    ssim_map = np.stack(ssim_maps, axis=-1).mean(axis=-1)  # average channels

    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        if m.shape != (H, W):
            raise ValueError(f"mask shape {m.shape} must be {(H,W)}")
        vals = ssim_map[m]
    else:
        vals = ssim_map.reshape(-1)

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def sam(
    ref: ArrayLike,
    est: ArrayLike,
    reduction: str = "mean",  # {"mean","median","none"}
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Union[float, np.ndarray]:
    """
    Spectral Angle Mapper (degrees). Works per-pixel for images/cubes.

    Args
    ----
    ref, est : (H,W,C) or (C,H,W) or (C,)
    reduction : 'mean' | 'median' | 'none' (returns map of shape (H,W))
    mask : optional (H,W) ROI

    Returns
    -------
    float or (H,W) map
    """
    A = _to_numpy(ref).astype(np.float64)
    B = _to_numpy(est).astype(np.float64)

    # Vector case
    if A.ndim == 1 and B.ndim == 1:
        num = float(np.dot(A, B))
        den = float(np.linalg.norm(A) * np.linalg.norm(B) + eps)
        return float(np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0))))

    # Image/cube case
    A = _as_hwc(A); B = _as_hwc(B); _check_shapes(A, B)
    H, W, C = A.shape
    a = A.reshape(-1, C)
    b = B.reshape(-1, C)

    num = np.sum(a * b, axis=1)
    den = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + eps
    ang = np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))
    ang_map = ang.reshape(H, W)

    if reduction == "none":
        if mask is None:
            return ang_map
        m = _to_numpy(mask).astype(bool)
        if m.shape != (H,W):
            raise ValueError(f"mask shape {m.shape} must be {(H,W)}")
        out = np.full_like(ang_map, np.nan, dtype=np.float64)
        out[m] = ang_map[m]
        return out

    vals = ang if mask is None else ang_map[_to_numpy(mask).astype(bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals) if reduction == "mean" else np.median(vals))


def sid(
    ref: ArrayLike,
    est: ArrayLike,
    reduction: str = "mean",  # {"mean","median","none"}
    mask: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Union[float, np.ndarray]:
    """
    Spectral Information Divergence (non-symmetric). We report SID(ref||est)+SID(est||ref).

    Args
    ----
    ref, est : (H,W,C) or (C,H,W) or (C,)
    reduction : 'mean' | 'median' | 'none' (returns map)
    mask : optional (H,W)

    Returns
    -------
    float or (H,W) map
    """
    def _sid_pair(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        # Normalize to probability-like spectra
        ps = p / (np.sum(p, axis=-1, keepdims=True) + eps)
        qs = q / (np.sum(q, axis=-1, keepdims=True) + eps)
        # Element-wise safe logs
        sid_pq = np.sum(ps * np.log((ps + eps) / (qs + eps)), axis=-1)
        sid_qp = np.sum(qs * np.log((qs + eps) / (ps + eps)), axis=-1)
        return sid_pq + sid_qp

    A = _to_numpy(ref).astype(np.float64)
    B = _to_numpy(est).astype(np.float64)

    if A.ndim == 1 and B.ndim == 1:
        return float(_sid_pair(A[None, :], B[None, :])[0])

    A = _as_hwc(A); B = _as_hwc(B); _check_shapes(A, B)
    H, W, C = A.shape
    a = A.reshape(-1, C)
    b = B.reshape(-1, C)
    sid_vals = _sid_pair(a, b).reshape(H, W)

    if reduction == "none":
        if mask is None:
            return sid_vals
        m = _to_numpy(mask).astype(bool)
        out = np.full_like(sid_vals, np.nan, dtype=np.float64)
        out[m] = sid_vals[m]
        return out

    vals = sid_vals if mask is None else sid_vals[_to_numpy(mask).astype(bool)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals) if reduction == "mean" else np.median(vals))


def ergas(
    ref: ArrayLike,
    est: ArrayLike,
    scale: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).
    Lower is better.

    Formula:
        ERGAS = 100 * scale * sqrt( (1/C) * sum_c ( RMSE_c / mean_ref_c )^2 )

    where:
      - RMSE_c is per-channel RMSE,
      - mean_ref_c is mean of the reference channel,
      - scale is the spatial resolution ratio (e.g., HR/LR); use 1.0 if same resolution.

    Args
    ----
    ref, est : (H,W,C) or (C,H,W)
    scale : ratio of resolutions (HR/LR). If not applicable, leave 1.0.

    Returns
    -------
    float : ERGAS value
    """
    A = _as_hwc(_to_numpy(ref)).astype(np.float64)
    B = _as_hwc(_to_numpy(est)).astype(np.float64)
    _check_shapes(A, B)
    H, W, C = A.shape

    # Per-channel RMSE
    rmse_c = np.sqrt(np.mean((A - B) ** 2, axis=(0, 1)))
    mu_c   = np.mean(A, axis=(0, 1))

    # Avoid division by zero
    ratios = rmse_c / (np.abs(mu_c) + eps)
    val = 100.0 * float(scale) * float(np.sqrt(np.mean(ratios ** 2)))
    return val

