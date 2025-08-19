
# 2026-ICASSP-SPGC

Utilities and helper functions for the Hyper-Object Challenge:
Reconstructing Hyperspectral Cubes of Everyday Objects from Low-Cost Inputs.

- Website: https://hyper-object.github.io/

This repository provides lightweight dataset utilities, transforms, and loaders to help you train and evaluate models for the Hyper-Object Challenge. Baseline models and evaluation scripts will be added later (see placeholders below).


## Challenge at a glance

Two tracks are offered. Both aim to reconstruct high-fidelity hyperspectral cubes spanning 400–1000 nm (61 bands).

- Track 1 — Spectral Reconstruction from Mosaic Images
  - Input: single-channel spectral mosaic image (one band per pixel).
  - Output: full hyperspectral cube with C = 61 bands.

- Track 2 — Joint Spatial & Spectral Super-Resolution
  - Input: low-resolution RGB image captured with a commodity camera.
  - Output: high-resolution hyperspectral cube with C ≫ 3 and spatial upscaling.

Ranking uses a composite score that aggregates RMSE, SAM, PSNR, SSIM, and EGRAS.


## How to participate

1. Read the overview and rules on the website: https://hyper-object.github.io/
2. Join the Kaggle competitions:
   - Track 1: https://www.kaggle.com/competitions/2026-icassp-hyper-object-challenge-track-1
   - Track 2: https://www.kaggle.com/competitions/2026-icassp-hyper-object-challenge-track-2
3. Download the data from Kaggle.
4. Use this repository to load and preprocess the data for your experiments.
5. Train your model and submit predictions to Kaggle to appear on the leaderboard.


## Quick Start
1. Clone this repo `git clone https://github.com/hyper-object/2026-ICASSP-SPGC`
2. Download the dataset from `https://www.kaggle.com/datasets/tusharkute/maharashtra-districts`

The folder structure should look like this:
```
  |-- data
      |-- test
          |-- hsi_61
              |-- Count: 19 *.h5
          |-- hsi_448
              |-- Count: 19 *.h5
          |-- mosaic
              |-- Count: 19 *.png
          |-- rgb
              |-- Count: 19 *.png
          |-- rgb_2
              |-- Count: 19 *.png
          |-- rgb_4
              |-- Count: 19 *.png
      |-- train
          |-- hsi_61
              |-- Count: 167 *.h5
          |-- hsi_448
              |-- Count: 167 *.h5
          |-- mosaic
              |-- Count: 167 *.png
          |-- rgb
              |-- Count: 167 *.png
          |-- rgb_2
              |-- Count: 167 *.png
          |-- rgb_4
              |-- Count: 167 *.png
```


## Loading Data

The examples below demonstrate how to load the Hyper-Object dataset and apply paired transforms.

    # Imports:
    import torch
    from torch.utils.data import DataLoader

    from datasets.hyper_object import HyperObjectDataset
    from datasets.pairing import ModalitySpec
    from datasets.base import JointTransform
    from datasets.transform import random_flip

  
    # Create the dataset and dataloader:
    ds = HyperObjectDataset(
        data_root="data",
        train=True,
        transforms=JointTransform(random_flip),
    )

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


## Baselines

(TBD)


## Evaluation

(TBD)



## Contact

- Questions about the challenge: hyper.skin.uoft@gmail.com
- Website: https://hyper-object.github.io/
- Kaggle:
  - Track 1: https://www.kaggle.com/competitions/2026-icassp-hyper-object-challenge-track-1
  - Track 2: https://www.kaggle.com/competitions/2026-icassp-hyper-object-challenge-track-2
