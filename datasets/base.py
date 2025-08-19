# datasets/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

__all__ = ["HSIDataset", "JointTransform"]

class HSIDataset(Dataset):
    """
    Thin Dataset base that mirrors torchvision semantics:
    - Optionally accepts a 'transforms' callable that can operate jointly
      on input(s) and target(s).
    - Or separate 'transform' and 'target_transform'.
    """
    _repr_indent = 4

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        if transforms is not None and (transform is not None or target_transform is not None):
            raise ValueError("Pass either `transforms` or (`transform` and/or `target_transform`), not both.")
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

    # subclasses implement __getitem__, __len__

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {len(self)}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)


class JointTransform:
    """
    Helper to apply a single callable to (inputs, target) jointly.
    Your callable should take and return a dict of arrays/tensors.
    """
    def __init__(self, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self.fn = fn

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.fn(batch)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.fn})"
