from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np

from utils.metrics import sam as sam_metric, sid as sid_metric, ergas as ergas_metric
from utils.metrics import psnr as psnr_metric, ssim as ssim_metric
from utils.visualizations import render_srgb_preview  # returns sRGB float [0,1]

# --- scaling helpers (all return [0,1]) ---
# def _exp_score(x_mean: float, tau: float) -> float:
#     return float(np.exp(-float(x_mean) / float(tau)))

def _exp_score(x_mean: float, tau: float, min_val: float = 1e-6) -> float:
    score = np.exp(-float(x_mean) / float(tau))
    return float(np.clip(score, min_val, 1.0))  # avoid 0.0

def _normalize_psnr(psnr_db: float, lo: float = 20.0, hi: float = 50.0) -> float:
    return float(np.clip((psnr_db - lo) / (hi - lo), 0.0, 1.0))

def _deltaE00_mean(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    import colour
    XYZ1 = colour.sRGB_to_XYZ(rgb1); XYZ2 = colour.sRGB_to_XYZ(rgb2)
    Lab1 = colour.XYZ_to_Lab(XYZ1);  Lab2 = colour.XYZ_to_Lab(XYZ2)
    dE = colour.difference.delta_E(Lab1.reshape(-1,3), Lab2.reshape(-1,3), method="CIE 2000")
    return float(np.mean(dE))

def evaluate_pair_ssc(
    gt_cube: np.ndarray,       # HxWxC or CxHxW reflectance [0,1]
    pr_cube: np.ndarray,       # HxWxC or CxHxW
    wl_nm: np.ndarray,         # (C,)
    mask: Optional[np.ndarray] = None,  # HxW bool

    # weights for the 3 groups (spectral, spatial, color)
    weights: Tuple[float, float, float] = (0.5, 0.35, 0.15),

    # spectral scaling τ
    taus = dict(sam=5.0, sid=0.02, ergas=3.0, de=3.0),

    # PSNR normalization range
    psnr_range: Tuple[float, float] = (20.0, 50.0),
) -> Dict[str, float]:
    """
    Returns all subscores in [0,1] and the final SSC in [0,1].
    """
    # --- spectral metrics (means over mask) ---
    sam_mean = sam_metric(gt_cube, pr_cube, reduction="mean", mask=mask)      # deg (↓)
    sid_mean = sid_metric(gt_cube, pr_cube, reduction="mean", mask=mask)      # (↓)
    erg_val  = ergas_metric(gt_cube, pr_cube, scale=1.0)                      # (↓)

    S_SAM    = _exp_score(sam_mean, taus["sam"])
    S_SID    = _exp_score(sid_mean, taus["sid"])
    S_ERGAS  = _exp_score(erg_val,  taus["ergas"])
    S_spec   = (S_SAM * S_SID * S_ERGAS) ** (1/3)

    # --- spatial/color via sRGB render (D65) ---
    gt_rgb = render_srgb_preview(gt_cube, wl_nm, clip=True, title=None)
    pr_rgb = render_srgb_preview(pr_cube, wl_nm, clip=True, title=None)

    psnr_val = psnr_metric(gt_rgb, pr_rgb, data_range=1.0, mask=mask)
    ssim_val = ssim_metric(gt_rgb, pr_rgb, data_range=1.0, mask=mask)

    S_PSNR   = _normalize_psnr(psnr_val, *psnr_range)
    S_spat   = 0.5 * (S_PSNR + float(ssim_val))

    dE_mean  = _deltaE00_mean(gt_rgb, pr_rgb)
    S_color  = _exp_score(dE_mean, taus["de"])

    # --- final SSC (weighted geometric mean) ---
    ws, wp, wc = weights
    SSC = (S_spec**ws * S_spat**wp * S_color**wc) ** (1.0 / (ws + wp + wc))
    # SSC = (ws * S_spec + wp * S_spat + wc * S_color) / (ws + wp + wc)


    return dict(
        # raw metrics
        SAM_deg=float(sam_mean), SID=float(sid_mean), ERGAS=float(erg_val),
        PSNR_dB=float(psnr_val), SSIM=float(ssim_val), DeltaE00=float(dE_mean),

        # group subscores (0..1)
        S_SAM=float(S_SAM), S_SID=float(S_SID), S_ERGAS=float(S_ERGAS),
        S_PSNR=float(S_PSNR), S_SSIM=float(ssim_val),
        S_SPEC=float(S_spec), S_SPAT=float(S_spat), S_COLOR=float(S_color),

        # final
        SSC=float(SSC),
    )


# ################ USAGE EXAMPLE ################
# # gt_cube, pr_cube: (H,W,61) reflectance in [0,1]
# # wl_61: np.array of wavelengths
# # mask: optional HxW bool
# from utils.leaderboard_ssc import evaluate_pair_ssc

# res = evaluate_pair_ssc(gt_cube, pr_cube, wl_61, mask=None)
# print(res["SSC"], res)
# ################ USAGE EXAMPLE ################
