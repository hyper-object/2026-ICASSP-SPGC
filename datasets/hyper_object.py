# datasets/rgb_hsi.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .base import HSIDataset
from .io import read_rgb_image, read_h5_cube, read_mosaic
from .pairing import ModalitySpec, build_index

__all__ = ["HyperObjectDataset"]


class HyperObjectDataset(HSIDataset):
    """
    Returns a dict:
      {
        "mosaic": (1,H,W) float32,
        "rgb_2":  (3,H,W) float32,
        "cube":   (C,H,W) float32,
        "id":     str
      }
    """

    def __init__(
        self,
        data_root: str,
        train: bool = True,
        transforms: Optional[Callable] = None
    ) -> None:
        super().__init__(root=data_root, transforms=transforms)

        mosaic_path=ModalitySpec(root=Path(f"{data_root}/{'train' if train else 'test'}/mosaic"), exts=(".npy",))
        rgb_2_path=ModalitySpec(root=Path(f"{data_root}/{'train' if train else 'test'}/rgb_2"),    exts=(".png", ".jpg"))
        hsi_61_path=ModalitySpec(root=Path(f"{data_root}/{'train' if train else 'test'}/hsi_61"), exts=(".h5",))

        (self.ids, self._maps) = build_index(
            {
                "mosaic": mosaic_path,
                "rgb_2": rgb_2_path,
                "hsi": hsi_61_path,
            })

    def __len__(self) -> int:
        return len(self.ids)

    def _load_(self, stem: str):
        p_mosaic = self._maps["mosaic"][stem]
        p_rgb_2 = self._maps["rgb_2"][stem]
        p_hsi = self._maps["hsi"][stem]

        mosaic = read_mosaic(p_mosaic)       # (H,W,1) float32 [0,1]
        rgb_2 = read_rgb_image(p_rgb_2)         # (H,W,3) float32 [0,1]
        cube = read_h5_cube(p_hsi, 'cube')      # (H,W,C)

        # HWC -> CHW
        mosaic_t = torch.from_numpy(np.transpose(mosaic, (2, 0, 1)))
        rgb_2_t = torch.from_numpy(np.transpose(rgb_2, (2, 0, 1)))
        cube_t = torch.from_numpy(np.transpose(cube, (2, 0, 1)))    # C,H,W
        return mosaic_t, rgb_2_t, cube_t

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        mosaic_t, rgb_2_t, cube_t = self._load_(stem)

        # Apply transforms
        if self.transforms is not None:
            # joint transform expects dict
            out = self.transforms({"mosaic": mosaic_t, "rgb_2": rgb_2_t, "cube": cube_t, "id": stem})
            mosaic_t, rgb_2_t, cube_t = out["mosaic"], out["rgb_2"], out["cube"]


        return {
            "mosaic": mosaic_t,
            "rgb_2": rgb_2_t,
            "cube": cube_t,
            "id": stem
        }
