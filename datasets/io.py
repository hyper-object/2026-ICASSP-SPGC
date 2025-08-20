# datasets/io.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image

__all__ = [
    "list_by_ext",
    "build_stem_map",
    "intersect_modalities",
    "read_h5_cube",
    "read_rgb_image",
]

def list_by_ext(root: Path, exts: Sequence[str]) -> List[Path]:
    exts = [e.lower() for e in exts]
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files

def build_stem_map(root: Path, exts: Sequence[str]) -> Dict[str, Path]:
    """
    Build {stem: path} for a folder, preferring earlier extensions in `exts` on collisions.
    """
    root = Path(root)
    exts = [e.lower() for e in exts]
    by_stem: Dict[str, Path] = {}
    for p in root.iterdir():
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in exts:
            continue
        stem = p.stem
        if stem not in by_stem:
            by_stem[stem] = p
        else:
            # prefer higher priority ext (smaller index)
            old = by_stem[stem]
            if exts.index(ext) < exts.index(old.suffix.lower()):
                by_stem[stem] = p
    return by_stem

def intersect_modalities(maps: Mapping[str, Dict[str, Path]]) -> List[str]:
    """
    Given a dict of {modality: {stem: path}}, return stems present in ALL modalities.
    """
    it = iter(maps.values())
    common = set(next(it).keys())
    for d in it:
        common &= set(d.keys())
    stems = sorted(common)
    return stems

def read_h5_cube(path: Path, dataset_name: str = "cube") -> np.ndarray:
    """
    Read HDF5 cube as HxWxC float32. Accepts CxHxW as well and transposes.
    """
    with h5py.File(path, "r") as f:
        if dataset_name not in f:
            raise KeyError(f"'{dataset_name}' not in {path}. Keys: {list(f.keys())}")
        arr = np.array(f[dataset_name], dtype=np.float32)
    # accept common layouts
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D cube, got {arr.shape} in {path}")
    # If data is C,H,W -> transpose to H,W,C
    if arr.shape[0] in (31, 61, 62, 448) and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr  # H,W,C

def read_rgb_image(path: Path) -> np.ndarray:
    """
    Read RGB as float32 HxWx3 in [0,1] without per-image min-max normalization.
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def read_mosaic(path: Path) -> np.ndarray:
    """
    Read a mosaic saved as .npy.
    Returns HxWx1 float32 array in [0,1].
    """
    arr = np.load(path).astype(np.float32)

    # If it's 2D, add channel dim -> (H,W,1)
    if arr.ndim == 2:
        arr = arr[..., None]

    # Normalize if values look like 8-bit integers
    if arr.max() > 1.0:
        arr = arr / 255.0

    return arr