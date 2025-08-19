from __future__ import annotations
import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

from .helpers import _to_numpy, _ensure_chw, _chw_to_hwc01, _is_single_channel


ArrayLike = Union[np.ndarray, torch.Tensor]

def visualize_sample_overview(
    sample: dict,
    index: int = 0,
    cube_bands: Tuple[int, int, int] = (10, 20, 30),
    figsize: Tuple[int, int] = (20, 5),
    mosaic_cmap: str = "gray",
    title: Optional[str] = None,
):
    """
    Visualize a single item from a batch-like sample dict with keys:
      "rgb", "cube", "mosaic", "rgb_2", "rgb_4", "id" (optional)

    - Each value is either CHW or BCHW (Torch/NumPy). This function handles both.
    - The HSI cube is shown as a 3-band false-color composite.

    Args
    ----
    sample: dict
        Batch from your DataLoader (tensors or ndarrays).
    index: int
        Which item in the batch to show.
    cube_bands: (int,int,int)
        Indices of HSI bands to compose as false-color RGB.
    figsize: (w,h)
        Matplotlib figure size in inches.
    mosaic_cmap: str
        Colormap for mosaic (single-channel) display.
    title: str or None
        Optional suptitle override. If None, tries to use sample['id'].
    """
    # Pull the k-th item from batch for each modality
    def _pick(k, key):
        x = sample[key]
        arr = _to_numpy(x)
        if arr.ndim == 4:  # BCHW
            arr = arr[k]
        return arr

    rgb     = _pick(index, "rgb")
    cube    = _pick(index, "cube")
    rgb_2   = _pick(index, "rgb_2")
    rgb_4   = _pick(index, "rgb_4")
    mosaic  = _pick(index, "mosaic")

    # Convert to viewable HWC in [0,1]
    rgb_hwc    = _chw_to_hwc01(rgb)
    rgb2_hwc   = _chw_to_hwc01(rgb_2)
    rgb4_hwc   = _chw_to_hwc01(rgb_4)

    # Mosaic (C=1) -> HWC
    mosaic_chw = _ensure_chw(mosaic)
    if not _is_single_channel(mosaic_chw):
        raise ValueError(f"'mosaic' is expected to be single-channel; got C={mosaic_chw.shape[0]}")
    mosaic_hwc = np.transpose(mosaic_chw, (1, 2, 0))
    # normalize grayscale to [0,1] for display
    mmin, mmax = np.nanmin(mosaic_hwc), np.nanmax(mosaic_hwc)
    if mmax > mmin:
        mosaic_hwc = (mosaic_hwc - mmin) / (mmax - mmin)

    # False-color from cube
    C, H, W = cube.shape
    bsel = tuple(int(b) for b in cube_bands)
    if any(b < 0 or b >= C for b in bsel):
        raise ValueError(f"cube_bands {cube_bands} out of range for C={C}")
    cube_fc = np.stack([cube[bsel[0]], cube[bsel[1]], cube[bsel[2]]], axis=-1)
    # Normalize composite
    vmin, vmax = np.nanmin(cube_fc), np.nanmax(cube_fc)
    cube_fc = (cube_fc - vmin) / (max(vmax - vmin, 1e-8))

    # Plot overview
    fig, axs = plt.subplots(1, 5, figsize=figsize)
    axs[0].imshow(rgb_hwc);             axs[0].set_title("RGB"); axs[0].axis("off")
    axs[1].imshow(cube_fc);             axs[1].set_title(f"HSI false-color {bsel}"); axs[1].axis("off")
    axs[2].imshow(mosaic_hwc.squeeze(), cmap=mosaic_cmap); axs[2].set_title("Mosaic"); axs[2].axis("off")
    axs[3].imshow(rgb2_hwc);            axs[3].set_title("RGB_2"); axs[3].axis("off")
    axs[4].imshow(rgb4_hwc);            axs[4].set_title("RGB_4"); axs[4].axis("off")

    # Suptitle
    if title is None:
        sid = sample.get("id", None)
        # 'id' may be a list/array of strings (B,)
        if isinstance(sid, (list, tuple, np.ndarray)) and len(sid) > index:
            title = str(sid[index])
        elif isinstance(sid, str):
            title = sid
    if title:
        fig.suptitle(f"Sample: {title}", fontsize=12)

    fig.tight_layout()
    plt.show()


def visualize_hsi_grid(
    cube: ArrayLike,
    wl: Optional[Sequence[float]] = None,
    cols: int = 9,
    figsize: Tuple[int, int] = (16, 12),
    cmap: str = "gray",
    contrast: str = "percentile",   # {"percentile", "minmax", "global"}
    p_low: float = 2.0,
    p_high: float = 98.0,
    suptitle: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """
    Plot all bands of an HSI cube (C,H,W) or (B,C,H,W)[0] as a grid.

    Args
    ----
    cube : array-like
        Hyperspectral cube (C,H,W) or BCHW (the first sample will be used).
    wl : sequence of float or None
        Optional wavelengths for per-plot titles. Length must equal C if given.
    cols : int
        Number of columns in the grid.
    figsize : (w,h)
        Figure size in inches.
    cmap : str
        Matplotlib colormap for band images.
    contrast : {"percentile","minmax","global"}
        Per-band display scaling:
         - "percentile": vmin/vmax = [p_low, p_high] % per-band
         - "minmax": per-band min/max
         - "global": single vmin/vmax from global percentiles
    p_low, p_high : float
        Percentiles for "percentile" or "global".
    suptitle : str or None
        Overall title above the grid.
    show : bool
        Whether to call plt.show().
    save_path : str or None
        If provided, saves the figure here.
    dpi : int
        DPI for saving.

    Returns
    -------
    fig, axes : Matplotlib objects
    """
    arr = _to_numpy(cube).astype(np.float32)  # C,H,W
    if arr.ndim == 4:  # BCHW -> use first
        arr = arr[0]
    C, H, W = arr.shape

    # Compute global contrast if needed
    if contrast == "global":
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size == 0:
            vmin_g, vmax_g = 0.0, 1.0
        else:
            vmin_g = np.percentile(finite_vals, p_low)
            vmax_g = np.percentile(finite_vals, p_high)
            if vmax_g <= vmin_g:
                vmax_g = vmin_g + 1e-6

    rows = int(math.ceil(C / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for b in range(C):
        r, c = b // cols, b % cols
        ax = axes[r, c]
        band = arr[b]

        if contrast == "percentile":
            vals = band[np.isfinite(band)]
            if vals.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.percentile(vals, p_low))
                vmax = float(np.percentile(vals, p_high))
                if vmax <= vmin: vmax = vmin + 1e-6
        elif contrast == "minmax":
            vals = band[np.isfinite(band)]
            if vals.size == 0:
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
                if vmax <= vmin: vmax = vmin + 1e-6
        elif contrast == "global":
            vmin, vmax = vmin_g, vmax_g
        else:
            raise ValueError("contrast must be one of {'percentile','minmax','global'}.")

        ax.imshow(band, cmap=cmap, vmin=vmin, vmax=vmax)
        if wl is not None and len(wl) == C:
            ax.set_title(f"{wl[b]:.0f} nm", fontsize=8)
        else:
            ax.set_title(f"b{b:02d}", fontsize=8)
        ax.axis("off")

    # Hide any unused cells
    total = rows * cols
    for k in range(C, total):
        r, c = k // cols, k % cols
        axes[r, c].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98] if suptitle else None)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def interactive_hsi_viewer(
    cube: ArrayLike,
    wl: Optional[Sequence[float]] = None,
    init_band: int = 0,
    figsize: Tuple[int, int] = (12, 5),
    cmap: str = "gray",
    contrast: str = "percentile",  # {"percentile","minmax","global"}
    p_low: float = 2.0,
    p_high: float = 98.0,
):
    """
    Interactive HSI viewer:
      - Left: current band image (with chosen contrast scaling)
      - Right: spectrum at the clicked pixel (with band marker)
      - Bottom: slider to change band

    Args
    ----
    cube : (C,H,W) or (H,W,C) tensor/ndarray
    wl   : optional list/array of wavelengths (len==C); used for axis labels
    init_band : initial band index to display
    figsize, cmap, contrast, p_low, p_high : display options
    """
    arr = _to_numpy(cube).astype(np.float32)   # C,H,W
    C, H, W = arr.shape
    b = int(np.clip(init_band, 0, C - 1))

    if wl is None:
        wl = np.arange(C)
        wl_lab = "Band index"
    else:
        wl = np.asarray(wl)
        wl_lab = "Wavelength (nm)"

    # Precompute global vmin/vmax for "global" contrast
    if contrast == "global":
        finite_vals = arr[np.isfinite(arr)]
        if finite_vals.size == 0:
            vmin_g, vmax_g = 0.0, 1.0
        else:
            vmin_g = np.percentile(finite_vals, p_low)
            vmax_g = np.percentile(finite_vals, p_high)
            if vmax_g <= vmin_g:
                vmax_g = vmin_g + 1e-6

    def band_limits(band_img):
        vals = band_img[np.isfinite(band_img)]
        if vals.size == 0:
            return 0.0, 1.0
        if contrast == "percentile":
            vmin = float(np.percentile(vals, p_low))
            vmax = float(np.percentile(vals, p_high))
            if vmax <= vmin: vmax = vmin + 1e-6
            return vmin, vmax
        elif contrast == "minmax":
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if vmax <= vmin: vmax = vmin + 1e-6
            return vmin, vmax
        elif contrast == "global":
            return vmin_g, vmax_g
        else:
            return float(np.min(vals)), float(np.max(vals))

    # Figure layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[20, 1], width_ratios=[1, 1])
    ax_img   = fig.add_subplot(gs[0, 0])
    ax_spec  = fig.add_subplot(gs[0, 1])
    ax_sldr  = fig.add_subplot(gs[1, :])

    # Initial image
    vmin, vmax = band_limits(arr[b])
    im = ax_img.imshow(arr[b], cmap=cmap, vmin=vmin, vmax=vmax)
    ax_img.set_title(f"Band {b} ({wl[b]:.0f} nm)" if wl_lab.startswith("Wavelength") else f"Band {b}")
    ax_img.axis("off")

    # Spectrum plot init (pick center pixel)
    r0, c0 = H // 2, W // 2
    spec_line, = ax_spec.plot(wl, arr[:, r0, c0], lw=1.5)
    band_marker = ax_spec.axvline(wl[b], color="r", ls="--", lw=1.0)
    ax_spec.set_xlabel(wl_lab)
    ax_spec.set_ylabel("Reflectance / Radiance")
    ax_spec.set_title(f"Spectrum @ (r={r0}, c={c0})")
    ax_spec.grid(True, alpha=0.25)

    # A marker on the image to indicate picked pixel
    img_marker, = ax_img.plot([c0], [r0], marker="o", ms=6, mec="w", mfc="none", mew=1.5)

    # Slider
    sldr = Slider(ax=ax_sldr, label="Band", valmin=0, valmax=C - 1, valinit=b, valstep=1)

    def update_image(new_b):
        new_b = int(new_b)
        im.set_data(arr[new_b])
        vmin, vmax = band_limits(arr[new_b])
        im.set_clim(vmin=vmin, vmax=vmax)
        ax_img.set_title(f"Band {new_b} ({wl[new_b]:.0f} nm)" if wl_lab.startswith("Wavelength") else f"Band {new_b}")
        band_marker.set_xdata([wl[new_b], wl[new_b]])
        fig.canvas.draw_idle()

    def on_slide(val):
        update_image(val)

    def on_click(event):
        nonlocal r0, c0
        if event.inaxes is not ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return
        r0 = int(round(event.ydata))
        c0 = int(round(event.xdata))
        r0 = np.clip(r0, 0, H - 1)
        c0 = np.clip(c0, 0, W - 1)
        img_marker.set_data([c0], [r0])
        spec_line.set_ydata(arr[:, r0, c0])
        ax_spec.set_title(f"Spectrum @ (r={r0}, c={c0})")
        fig.canvas.draw_idle()

    sldr.on_changed(on_slide)
    cid = fig.canvas.mpl_connect("button_press_event", on_click)

    plt.tight_layout()
    plt.show()

    # Return handles in case caller wants to tweak
    return {
        "fig": fig, "ax_img": ax_img, "ax_spec": ax_spec, "slider": sldr,
        "im": im, "spec_line": spec_line, "band_marker": band_marker, "img_marker": img_marker,
        "click_cid": cid
    }


def plot_spectral_profiles(
    cube: ArrayLike,
    wl: Optional[Sequence[float]] = None,
    coords: Optional[Iterable[Tuple[int, int]]] = None,
    mask: Optional[ArrayLike] = None,
    reduce: str = "median",     # {"median","mean"}
    colors: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Show ROI/points on an RGB proxy (left) and plot spectra (right).

    Args
    ----
    cube : (C,H,W) or (H,W,C)
    wl   : wavelengths (len==C)
    coords : iterable of (row, col) pixel coordinates
    mask : boolean (H,W) mask to aggregate (median/mean) over a region
    reduce : aggregation for mask
    colors : optional colors for each series (coords + optional mask)
    """
    arr = _to_numpy(cube).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] < arr.shape[-1]:
        C, H, W = arr.shape  # assume (C,H,W)
    elif arr.ndim == 3:
        H, W, C = arr.shape
        arr = arr.transpose(2, 0, 1)  # -> (C,H,W)
    else:
        raise ValueError("cube must be (C,H,W) or (H,W,C)")

    if wl is None:
        wl = np.arange(C)
        xlab = "Band index"
    else:
        wl = np.asarray(wl)
        xlab = "Wavelength (nm)"

    series = []
    labels = []

    # --- Spectra from selected points ---
    if coords:
        for i, (r, c) in enumerate(coords):
            r = int(np.clip(r, 0, H - 1))
            c = int(np.clip(c, 0, W - 1))
            series.append(arr[:, r, c])
            labels.append(f"pt({r},{c})")

    # --- ROI mask ---
    if mask is not None:
        m = _to_numpy(mask).astype(bool)
        if m.shape != (H, W):
            raise ValueError(f"mask shape {m.shape} must be (H,W)=({H},{W})")
        pix = arr[:, m]  # (C, N)
        if pix.shape[1] == 0:
            roi_spec = np.zeros(C, dtype=np.float32)
        else:
            roi_spec = np.median(pix, axis=1) if reduce == "median" else np.mean(pix, axis=1)
        series.append(roi_spec)
        labels.append(f"ROI-{reduce} (n={pix.shape[1]})")

    if not series:
        raise ValueError("Provide at least one coord or a mask.")

    # --- Figure with 2 panels ---
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Left: RGB proxy (using 3 bands if available, else grayscale)
    if C >= 3:
        rgb_proxy = arr[[0, 1, 2], :, :].transpose(1, 2, 0)
    else:
        rgb_proxy = arr[0]
    rgb_proxy = (rgb_proxy - rgb_proxy.min()) / (rgb_proxy.max() - rgb_proxy.min() + 1e-8)
    axs[0].imshow(rgb_proxy)
    axs[0].set_title("ROI / Selected Points")

    # Overlay mask boundary
    if mask is not None:
        axs[0].contour(m, colors='red', linewidths=1)

    # Overlay points
    if coords:
        for i, (r, c) in enumerate(coords):
            axs[0].plot(c, r, 'o', markerfacecolor='none',
                        markeredgecolor='yellow', markersize=8, lw=1.5)

    axs[0].axis("off")

    # Right: spectral profiles
    if colors is None or len(colors) < len(series):
        colors = [None] * len(series)
    for y, lab, col in zip(series, labels, colors):
        axs[1].plot(wl, y, label=lab, color=col, lw=1.5)
    axs[1].set_xlabel(xlab)
    axs[1].set_ylabel("Reflectance / Radiance")
    axs[1].grid(True, alpha=0.3)
    if title:
        axs[1].set_title(title)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def sam_mask(cube_chw, ref_spec, angle_deg=8.0, eps=1e-8):
    """
    cube_chw: (C,H,W), ref_spec: (C,)
    Returns: mask (H,W) of pixels within angle threshold.
    """
    C, H, W = cube_chw.shape
    X = cube_chw.reshape(C, -1)         # C x N
    r = ref_spec.reshape(-1, 1)         # C x 1
    # cosine similarity
    dot = (X * r).sum(axis=0)
    nX = np.linalg.norm(X, axis=0) + eps
    nr = np.linalg.norm(r, axis=0) + eps
    cosang = np.clip(dot / (nX * nr), -1.0, 1.0)
    ang = np.arccos(cosang) * 180/np.pi
    mask = (ang <= angle_deg).reshape(H, W)
    return mask, ang.reshape(H, W)

def kmeans_mask(cube_chw, k=3, pick=0, whiten=True, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    C, H, W = cube_chw.shape
    X = cube_chw.reshape(C, -1).T  # N x C
    if whiten:
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Z = PCA(n_components=6, random_state=seed).fit_transform(X)  # N x 6
    km = KMeans(n_clusters=k, n_init='auto', random_state=seed).fit(Z)
    labels = km.labels_.reshape(H, W)
    mask = (labels == pick)
    return mask, labels


def render_srgb_preview(
    cube: ArrayLike,
    wl: Sequence[float],
    vis_range: Tuple[float, float] = (400.0, 700.0),
    illuminant: str = "D65",
    observer: str = "CIE 1931 2 Degree Standard Observer",
    clip: bool = True,
    figsize: Tuple[int, int] = (6, 6),
    title: Optional[str] = None,
):
    """
    Convert reflectance HSI to sRGB and display it.
    This is a thin wrapper—uses your existing hsi_reflectance_to_srgb() if available,
    otherwise falls back to an internal implementation (requires `colour-science`).

    Args
    ----
    cube : (C,H,W) or (H,W,C) reflectance in [0,1]
    wl   : (C,) wavelengths in nm
    vis_range, illuminant, observer, clip : rendering params
    """
    # Try to reuse your project function if present
    _hsi2srgb = None
    try:
        # adjust import path if your function lives elsewhere
        from utils import hsi_reflectance_to_srgb as _hsi2srgb  # type: ignore
    except Exception:
        pass

    arr = _to_numpy(cube).astype(np.float32)       # C,H,W
    arr = np.transpose(arr, (1, 2, 0))              # H,W,C for the renderer
    wl  = np.asarray(wl, dtype=float)

    if _hsi2srgb is None:
        # Minimal fallback using colour-science (matches what you used before)
        import colour

        def _load_cmf_ill_res(wl_nm, vis_range=(400.0, 700.0),
                              illuminant="D65",
                              observer="CIE 1931 2 Degree Standard Observer",
                              base_shape=(380.0, 780.0, 1.0)):
            wl_nm = np.asarray(wl_nm, dtype=float)
            vmin, vmax = vis_range
            vis_mask = (wl_nm >= vmin) & (wl_nm <= vmax)
            wl_vis = wl_nm[vis_mask]
            start, end, step = base_shape
            n_steps = int(np.floor((end - start) / step)) + 1
            wl_base = start + step * np.arange(n_steps)

            cmfs = colour.MSDS_CMFS[observer].copy().align(colour.SpectralShape(start, end, step))
            illu = colour.SDS_ILLUMINANTS[illuminant].copy().align(colour.SpectralShape(start, end, step))

            xbar_full = cmfs.values[..., 0]
            ybar_full = cmfs.values[..., 1]
            zbar_full = cmfs.values[..., 2]
            E_full = illu.values

            xbar = np.interp(wl_vis, wl_base, xbar_full)
            ybar = np.interp(wl_vis, wl_base, ybar_full)
            zbar = np.interp(wl_vis, wl_base, zbar_full)
            E = np.interp(wl_vis, wl_base, E_full)
            return wl_vis, xbar, ybar, zbar, E, vis_mask

        def _hsi2srgb(R, wl_nm, vis_range=(400.,700.), illuminant="D65", observer=observer, clip=True):
            wl, xbar, ybar, zbar, E, mask = _load_cmf_ill_res(wl_nm, vis_range, illuminant, observer)
            Rv = R[..., mask]
            w = np.gradient(wl)
            k = 1.0 / np.sum(E * ybar * w)
            Wx = k * (E * xbar * w); Wy = k * (E * ybar * w); Wz = k * (E * zbar * w)
            X = np.tensordot(Rv, Wx, axes=([2], [0]))
            Y = np.tensordot(Rv, Wy, axes=([2], [0]))
            Z = np.tensordot(Rv, Wz, axes=([2], [0]))
            M = np.array([[ 3.2406, -1.5372, -0.4986],
                          [-0.9689,  1.8758,  0.0415],
                          [ 0.0557, -0.2040,  1.0570]], dtype=np.float64)
            RGB_lin = np.stack([X, Y, Z], axis=-1) @ M.T
            RGB_lin = np.clip(RGB_lin, 0.0, None)
            a = 0.055; threshold = 0.0031308
            RGB = np.where(RGB_lin <= threshold, 12.92 * RGB_lin,
                           (1 + a) * np.power(RGB_lin, 1/2.4) - a)
            if clip: RGB = np.clip(RGB, 0.0, 1.0)
            return RGB.astype(np.float32)

    # Compute sRGB
    if _hsi2srgb is not None:
        rgb = _hsi2srgb(arr, wl, vis_range=vis_range, illuminant=illuminant, observer=observer, clip=clip)
    else:
        rgb = _hsi2srgb(arr, wl, vis_range=vis_range, illuminant=illuminant, observer=observer, clip=clip)

    # Show
    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

    return rgb





