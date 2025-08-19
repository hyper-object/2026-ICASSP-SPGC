# datasets/pairing.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .io import build_stem_map, intersect_modalities

@dataclass(frozen=True)
class ModalitySpec:
    root: Path
    exts: Sequence[str]

def build_index(
    specs: Dict[str, ModalitySpec],
    id_list_path: Optional[Path] = None,
) -> List[str]:
    """
    Build a list of sample IDs (stems) present in ALL modalities.
    Optionally constrain by an id list file (one stem per line).
    """
    maps = {name: build_stem_map(spec.root, spec.exts) for name, spec in specs.items()}
    stems = intersect_modalities(maps)

    if id_list_path is not None:
        keep = set([ln.strip() for ln in Path(id_list_path).read_text().splitlines() if ln.strip()])
        stems = [s for s in stems if s in keep]

    if not stems:
        raise RuntimeError("No common stems across modalities (after filtering).")
    return stems, maps
