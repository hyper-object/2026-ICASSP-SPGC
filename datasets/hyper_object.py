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
        "input": "mosaic" (1,H,W) float32 or "rgb_2"  (3,H,W) float32,
        "output":  "cube"   (C,H,W) float32,
        "id":     str
      }
    """

    def __init__(
        self,
        track: int,
        data_root: str,
        train: bool = True,
        transforms: Optional[Callable] = None,
        submisison: bool = False
    ) -> None:
        super().__init__(root=data_root, transforms=transforms)
        self.track = track 

        if submisison:
            hsi_61_path=ModalitySpec(root=Path(f"{data_root}/{'train' if train else 'test_original'}/hsi_61"), exts=(".h5",))
        else:
            hsi_61_path=ModalitySpec(root=Path(f"{data_root}/{'train' if train else 'test-public'}/hsi_61"), exts=(".h5",))

        if track == 1:
            mosaic_path=ModalitySpec(root=Path(f"{data_root}/{'train' if train else 'test-public'}/mosaic"), exts=(".npy",))
            (self.ids, self._maps) = build_index(
                {
                    "mosaic": mosaic_path,
                    "hsi": hsi_61_path,
                })
        elif track == 2:
            if submisison:
                rgb_2_path=ModalitySpec(root=Path(f"{data_root}/{'train' if train else 'test_original'}/rgb_2"),    exts=(".png", ".jpg"))
            else:
                rgb_2_path=ModalitySpec(root=Path(f"{data_root}/{'train' if train else 'test-public'}/rgb_2"),    exts=(".png", ".jpg"))
                
            (self.ids, self._maps) = build_index(
                {
                    "rgb_2": rgb_2_path,
                    "hsi": hsi_61_path,
                })

    def __len__(self) -> int:
        return len(self.ids)

    def _load_(self, stem: str):
        p_hsi = self._maps["hsi"][stem]
        cube = read_h5_cube(p_hsi, 'cube')                          # (H,W,C)
        cube_t = torch.from_numpy(np.transpose(cube, (2, 0, 1)))    # C,H,W

        if self.track == 1:
            p_mosaic = self._maps["mosaic"][stem]
            mosaic = read_mosaic(p_mosaic)                                  # (H,W,1) float32 [0,1]
            mosaic_t = torch.from_numpy(np.transpose(mosaic, (2, 0, 1)))    # 1,H,W
            return mosaic_t, cube_t
        elif self.track == 2:
            p_rgb_2 = self._maps["rgb_2"][stem]
            rgb_2 = read_rgb_image(p_rgb_2)                             # (H,W,3) float32 [0,1]
            rgb_2_t = torch.from_numpy(np.transpose(rgb_2, (2, 0, 1)))  # C,H,W
            return rgb_2_t, cube_t
            

    def __getitem__(self, idx: int):
        stem = self.ids[idx]
        input_data, output_data = self._load_(stem)

        # Apply transforms
        if self.transforms is not None:
            # joint transform expects dict
            out = self.transforms({"input_data": input_data, "output_data": output_data,  "id": stem})
            input_data, output_data = out["input_data"], out["output_data"]


        return {
            "input": input_data,              # either mosaic or rgb_2 depending on track
            "output": output_data,     # hsi (61 bands)
            "id": stem
        }
