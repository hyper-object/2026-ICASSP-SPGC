from __future__ import absolute_import, division, print_function
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms


from datasets.hyper_object import HyperObjectDataset
from datasets.base import JointTransform
from datasets.transform import random_flip



from utils.track2 import metrics
from baselines import mstpp_up

from utils.leaderboard_ssc import evaluate_pair_ssc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Paths
data_dir = '/mnt/data/2026-Hyper-Object-Data'
model_path = 'runs/track2/saved-models/exp-MST_Plus_Plus_Up-CIE/model.pt'
results_dir = 'runs/track2/results'
os.makedirs(results_dir, exist_ok=True)


BATCH_SIZE = 1
MODEL_NAME = 'MST_Plus_Plus_Up'
UPSCALE_FACTOR = 2
BAND_WAVELENGTHS = np.arange(400, 1001, 10)  # 61 bands

# Output files
results_file = os.path.join(results_dir, 'results.csv')
print(f"Results will be saved to: {results_file}")





ds_val = HyperObjectDataset(
        data_root=f'{data_dir}/track2',
        track=2,
        train=False,
)

test_loader = torch.utils.data.DataLoader(
    dataset=ds_val,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True if device == 'cuda' else False
)

print(f"Test dataset loaded with {len(ds_val)} samples.")


print(f"Loading checkpoint from: {model_path}")

model = mstpp_up.MST_Plus_Plus_LateUpsample(
    in_channels=3, out_channels=61, n_feat=61, stage=3, upscale_factor=UPSCALE_FACTOR
)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print(f"Model '{MODEL_NAME}' loaded and set to evaluation mode.")



def evaluate_return_mode(model, test_loader, return_hr=False, upscale_factor=2, device='cpu'):
    model.return_hr = return_hr
    all_results = []

    model.to(device)
    model.eval()

    for k, data in enumerate(test_loader):
        x, y = data['input'], data['output']
        x, y = x.float().to(device), y.float().to(device)

        # print(f"\n" + "*"*50)
        # print(f"--- Processing Sample {k+1}/{len(test_loader)} ---")
        # print(f"Input RGB (x) shape:          {x.shape}")
        # print(f"Ground Truth HSI (y) shape:   {y.shape}")

        with torch.no_grad():
            pred = model(x)
        
        # print(f"Model Output (pred) shape:      {pred.shape}")

        if return_hr:
            pred_metrics = pred
            y_metrics = y
        else:
            pred_metrics = pred
            y_metrics = F.interpolate(y, size=pred.shape[-2:], mode='area')

        gt_cube_np = y_metrics.squeeze(0).cpu().numpy()
        pr_cube_np = pred_metrics.squeeze(0).cpu().numpy()


        # print("--- Shapes for Metrics Calculation ---")
        # print(f"Prediction for metrics shape:   {pred_metrics.shape} (Torch Tensor)")
        # print(f"Ground Truth for metrics shape: {y_metrics.shape} (Torch Tensor)")
        # print(f"NumPy Pred Cube shape:          {pr_cube_np.shape} (NumPy Array)")
        # print(f"NumPy GT Cube shape:            {gt_cube_np.shape} (NumPy Array)")
        # print("*"*50)


        scores = evaluate_pair_ssc(
            gt_cube=gt_cube_np,
            pr_cube=pr_cube_np,
            wl_nm=BAND_WAVELENGTHS,
        )

        filename_no_ext = data['id'][0]
        scores['file'] = filename_no_ext
        all_results.append(scores)

        mode_str = "HR" if return_hr else "LR"
        print(f"[{mode_str}] Test [{k+1}/{len(test_loader)}]: {scores['file']} | "
              f"SSC: {scores['SSC']:.4f} | "
              f"SAM: {scores['SAM_deg']:.2f} | SID: {scores['SID']:.4f} | ERGAS: {scores['ERGAS']:.3f} | "
              f"PSNR: {scores['PSNR_dB']:.2f} | SSIM: {scores['SSIM']:.3f} | dE00: {scores['DeltaE00']:.2f}")

    return all_results




print("\nRunning HR evaluation (spatially upscaled output)...")
results_hr_list = evaluate_return_mode(model, test_loader, return_hr=True, upscale_factor=UPSCALE_FACTOR, device=device)

if results_hr_list:
    avg_scores = {}
    metric_keys = ['SSC', 'SAM_deg', 'SID', 'ERGAS', 'PSNR_dB', 'SSIM', 'DeltaE00']
    for key in metric_keys:
        avg_scores[key] = np.mean([res[key] for res in results_hr_list])

    print("\n" + "="*50)
    print("Average Scores on Test Set (HR Mode):")
    print(f"  - SSC:       {avg_scores['SSC']:.4f}")
    print(f"  - SAM (deg): {avg_scores['SAM_deg']:.4f}")
    print(f"  - SID:       {avg_scores['SID']:.4f}")
    print(f"  - ERGAS:     {avg_scores['ERGAS']:.4f}")
    print(f"  - PSNR (dB): {avg_scores['PSNR_dB']:.4f}")
    print(f"  - SSIM:      {avg_scores['SSIM']:.4f}")
    print(f"  - DeltaE00:  {avg_scores['DeltaE00']:.4f}")
    print("="*50)

