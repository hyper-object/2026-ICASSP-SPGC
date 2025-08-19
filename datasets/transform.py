import random
import torch

def random_flip(batch):
    """
    Random horizontal and vertical flips applied consistently across all modalities.
    Applies to: rgb, cube, rgb_2, rgb_4, mosaic
    """
    keys = ["rgb", "cube", "rgb_2", "rgb_4", "mosaic"]

    # horizontal flip
    if random.random() < 0.5:
        for k in keys:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = torch.flip(batch[k], dims=[2])  # flip width

    # vertical flip
    if random.random() < 0.5:
        for k in keys:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = torch.flip(batch[k], dims=[1])  # flip height

    return batch


def random_crop(batch, ps=256):
    """
    Random crop applied consistently across all modalities.
    Applies to: rgb, cube, rgb_2, rgb_4, mosaic
    """
    keys = ["rgb", "cube", "rgb_2", "rgb_4", "mosaic"]

    # assume all tensors have shape (C,H,W)
    _, H, W = batch["rgb"].shape
    if H >= ps and W >= ps:
        r = random.randint(0, H - ps)
        c = random.randint(0, W - ps)
        for k in keys:
            if k in batch and isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k][:, r:r+ps, c:c+ps]

    return batch
