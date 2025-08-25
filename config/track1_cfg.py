from typing import Optional, Tuple, Dict, Any, List, Literal
import numpy as np
from dataclasses import dataclass, field

@dataclass
class TrainerCfg:
    out_dir: str = "runs/track1/mosaic2hsi_baseline_v3"
    epochs: int = 1000
    amp: bool = True
    save_best: bool = True
    psnr_range: Tuple[float, float] = (20.0, 50.0)  # for reporting scale only
    log_csv_name: str = "train_log.csv"
    wl_61: np.ndarray = field(default_factory=lambda: np.arange(400, 1001, 10))  # 61 bands

    # Optimizer & scheduler settings
    lr: float = 2e-4
    weight_decay: float = 1e-4
    scheduler_type: Literal["cosine", "none"] = "cosine"
    eta_min: float = 1e-6
    lambda_sam: float = 0.1  # SAM loss weight

    # model parameters
    base_ch: int = 64
    n_blocks: int = 8
    out_bands: int = 61  # number of output spectral bands