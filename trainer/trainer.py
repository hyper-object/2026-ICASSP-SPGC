from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import csv
import time
import numpy as np
import colour

import torch
from torch.utils.data import DataLoader

# ---- bring your stuff ----
# - loss: ReconLoss (L1 + SAM) or any callable loss(pred, target, mask=None)
# - metrics: rmse/sam/sid/ergas/psnr/ssim
# - render: function to convert HSI cube -> sRGB under D65
from .losses import ReconLoss

from config.track1_cfg import TrainerCfg
from utils.helpers import _to_hwc
from utils.metrics import sam, sid, ergas
from utils.metrics import psnr, ssim
from utils.visualizations import render_srgb_preview  # returns HxWx3 float [0,1]

class Trainer:
    """
    Generic trainer for RAW mosaic -> HSI models.

    Expects each batch dict with:
      - "mosaic": (N,1,H,W) float in [0,1]
      - "cube":   (N,C,H,W) float in [0,1] (C=61 for your case)
      - Optional "mask": (N,1,H,W) bool/float (ROI), used only for metrics if present
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cfg: TrainerCfg = TrainerCfg(),
    ):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader   
        self.device = device 

        self.model = model
        self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler() if (self.cfg.amp and self.device.type == "cuda") else None
        self.loss_fn = loss_fn if loss_fn is not None else ReconLoss(lambda_sam=0.1)
        
        self.wl_nm = cfg.wl_61  # wavelength vector for rendering
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        if self.cfg.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.eta_min)
        else:
            self.scheduler = None

        # I/O
        self.out_dir = Path(self.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_best = self.out_dir / "model_best.pt"
        self.ckpt_last = self.out_dir / "model_last.pt"
        self.log_csv = self.out_dir / self.cfg.log_csv_name

        # CSV header
        if not self.log_csv.exists():
            with open(self.log_csv, "w", newline="") as f:
                w = csv.writer(f)
                header = [
                    "epoch", "lr", "train_loss", "val_loss",
                    "SAM_deg", "SID", "ERGAS",
                    "PSNR_dB", "SSIM", 
                ]
                w.writerow(header)

        self.best_val = float("inf")


    def _forward_loss(self, input_img: torch.Tensor, output_cube: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.scaler is None:
            pred = self.model(input_img)
            loss = self.loss_fn(pred, output_cube)
            return pred, loss
        else:
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                pred = self.model(input_img)
                loss = self.loss_fn(pred, output_cube)
            return pred, loss

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        running = 0.0
        n_samples = 0
        t0 = time.time()

        for batch in self.train_loader:
            input_img  :   torch.Tensor = batch["input"].to(self.device, non_blocking=True)     # (N,c(1 or 3),H,W)
            output_cube:   torch.Tensor = batch["output"].to(self.device, non_blocking=True)    # (N,C,H,W)

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is None:
                pred, loss = self._forward_loss(input_img, output_cube)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    pred, loss = self._forward_loss(input_img, output_cube)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            running += float(loss.item()) * input_img.size(0)
            n_samples += input_img.size(0)

        if self.scheduler is not None:
            self.scheduler.step()

        avg = running / max(n_samples, 1)
        dt = time.time() - t0
        return avg

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Returns dict with: val_loss, SAM_deg, SID, ERGAS, PSNR_dB, SSIM, (DeltaE00 optional).
        """
        self.model.eval()

        loss_sum = 0.0
        n_samples = 0

        sam_list: List[float] = []
        sid_list: List[float] = []
        erg_list: List[float] = []
        psnr_list: List[float] = []
        ssim_list: List[float] = []


        for batch in self.val_loader:
            input_img  :   torch.Tensor = batch["input"].to(self.device, non_blocking=True)
            output_cube:   torch.Tensor = batch["output"].to(self.device, non_blocking=True)

            # forward (no grad)
            pred_cube = self.model(input_img).clamp(0, 1)
            loss = self.loss_fn(pred_cube, output_cube)

            loss_sum += float(loss.item()) * input_img.size(0)
            n_samples += input_img.size(0)

            # per-sample metrics
            for i in range(pred_cube.size(0)):
                # --- spectral metrics (means over mask) ---
                sam_mean = sam(output_cube[i].detach().cpu().numpy(), pred_cube[i].detach().cpu().numpy(), reduction="mean", mask=mask)      # deg (↓)
                sid_mean = sid(output_cube[i].detach().cpu().numpy(), pred_cube[i].detach().cpu().numpy(), reduction="mean", mask=mask)      # (↓)
                erg_val  = ergas(output_cube[i].detach().cpu().numpy(), pred_cube[i].detach().cpu().numpy(), scale=1.0)                      # (↓)

                psnr_val = psnr(output_cube[i].detach().cpu().numpy(), pred_cube[i].detach().cpu().numpy(), data_range=1.0, mask=mask)
                ssim_val = ssim(output_cube[i].detach().cpu().numpy(), pred_cube[i].detach().cpu().numpy(), data_range=1.0, mask=mask)

                dE_mean  = _deltaE00_mean(output_cube[i].detach().cpu().numpy(), pred_cube[i].detach().cpu().numpy())


                sam_list.append(scores["SAM_deg"])
                sid_list.append(scores["SID"])
                erg_list.append(scores["ERGAS"])
                psnr_list.append(scores["PSNR_dB"])
                ssim_list.append(scores["SSIM"])


        out: Dict[str, float] = {}
        out["val_loss"] = loss_sum / max(n_samples, 1)
        out["SAM_deg"]  = float(np.mean(sam_list)) if sam_list else float("nan")
        out["SID"]      = float(np.mean(sid_list)) if sid_list else float("nan")
        out["ERGAS"]    = float(np.mean(erg_list)) if erg_list else float("nan")
        out["PSNR_dB"]  = float(np.mean(psnr_list)) if psnr_list else float("nan")
        out["SSIM"]     = float(np.mean(ssim_list)) if ssim_list else float("nan")
        return out

    def _current_lr(self) -> float:
        if self.optimizer.param_groups:
            return float(self.optimizer.param_groups[0].get("lr", 0.0))
        return 0.0

    def _log_csv(self, epoch: int, train_loss: float, val_stats: Dict[str, float]):
        row = [
            epoch,
            self._current_lr(),
            train_loss,
            val_stats.get("val_loss", float("nan")),
            val_stats.get("SAM_deg", float("nan")),
            val_stats.get("SID", float("nan")),
            val_stats.get("ERGAS", float("nan")),
            val_stats.get("PSNR_dB", float("nan")),
            val_stats.get("SSIM", float("nan")),
        ]
        with open(self.log_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _print_epoch(self, epoch: int, train_loss: float, val_stats: Dict[str, float]):
        parts = [
            # --- Epoch and Learning Info ---
            f"[{epoch:03d}]",
            f"lr: {self._current_lr():.2e}",
            f"train: {train_loss:7.4f}",
            f"val: {val_stats.get('val_loss', float('nan')):7.4f} | ",

            # --- Core Reconstruction Metrics ---
            f"SAM(deg): {val_stats.get('SAM_deg', float('nan')):6.2f}",
            f"SID: {val_stats.get('SID', float('nan')):7.4f}",
            f"ERGAS: {val_stats.get('ERGAS', float('nan')):6.3f}",
            f"PSNR(dB): {val_stats.get('PSNR_dB', float('nan')):6.2f}",
            f"SSIM: {val_stats.get('SSIM', float('nan')):5.3f} | ",
        ]
        print("  ".join(parts))

    def _save_checkpoint(self, epoch: int, is_best: bool):
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(state, self.ckpt_last)
        if is_best and self.cfg.save_best:
            torch.save(state, self.ckpt_best)
            print(f"[Saved BEST model @ epoch {epoch}] → {self.ckpt_best}")

    def fit(self):
        print(f"Start training for {self.cfg.epochs} epochs. Logs → {self.log_csv}")
        for ep in range(1, self.cfg.epochs + 1):
            train_loss = self.train_epoch(ep)
            val_stats = self.validate(ep)
            self._print_epoch(ep, train_loss, val_stats)
            self._log_csv(ep, train_loss, val_stats)

            is_best = val_stats["val_loss"] < self.best_val
            if is_best:
                self.best_val = val_stats["val_loss"]
            self._save_checkpoint(ep, is_best)
