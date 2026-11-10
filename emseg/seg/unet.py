from __future__ import annotations

import numpy as np
import torch
from torch_em.model import UNet2d

from emseg.seg.targets import METHOD_CHANNELS


def build_model(method: str, initial_features: int = 32, depth: int = 4, n_aux: int = 0) -> torch.nn.Module:
    """n_aux: extra auxiliary output channels (e.g. 6 LSD channels) appended after the method channels.
    Post-processing only ever reads the first METHOD_CHANNELS[method] channels."""
    return UNet2d(in_channels=1, out_channels=METHOD_CHANNELS[method] + n_aux, depth=depth,
                  initial_features=initial_features, final_activation=None)


def lsd_mse(logits_aux: torch.Tensor, target_aux: torch.Tensor, fg: torch.Tensor, eps: float = 1e-6):
    """Foreground-masked MSE between sigmoid(aux logits) and LSD targets in [0,1]. fg (B,1,H,W) float."""
    d = (torch.sigmoid(logits_aux) - target_aux) ** 2 * fg
    return d.sum() / (fg.sum() * logits_aux.shape[1] + eps)


def masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """Channel-wise BCE + soft Dice, both restricted to mask==1 pixels."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * mask).sum() / (mask.sum() + eps)
    p = torch.sigmoid(logits) * mask; t = target * mask
    inter = (p * t).sum(dim=(0, 2, 3)); denom = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
    dice = 1 - (2 * inter + eps) / (denom + eps)
    return bce + dice.mean()


def upsample_image(image: np.ndarray, scale: int) -> np.ndarray:
    """Bilinear x`scale` up-sampling (align_corners=False) of a (H, W) float image; identity for scale 1."""
    if scale == 1:
        return image
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False)[0, 0].numpy()


def upsample_label(label: np.ndarray, scale: int) -> np.ndarray:
    """Nearest-neighbour x`scale` up-sampling: out[i, j] = label[i // scale, j // scale]."""
    if scale == 1:
        return label
    return np.repeat(np.repeat(label, scale, axis=0), scale, axis=1)


@torch.no_grad()
def predict_slice(model: torch.nn.Module, image: np.ndarray, device, pad_to: int = 16, scale: int = 1) -> np.ndarray:
    """Full-slice inference (image normalized outside). Returns (C, H, W) sigmoid probabilities on the ORIGINAL grid.
    scale > 1: the image is up-sampled x`scale` before the network and the output is sub-sampled [::scale, ::scale]
    (offsets were multiplied by `scale` in training, so the sub-sampled affinity channels carry the standard offsets)."""
    H0, W0 = image.shape
    image = upsample_image(image, scale)
    H, W = image.shape
    ph, pw = (-H) % pad_to, (-W) % pad_to
    x = np.pad(image, ((0, ph), (0, pw)), mode="reflect")
    x = torch.from_numpy(x[None, None].astype(np.float32)).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        y = model(x)
    y = torch.sigmoid(y.float())[0, :, :H, :W]
    if scale > 1:
        y = y[:, ::scale, ::scale]
    y = y.cpu().numpy()
    assert y.shape[1:] == (H0, W0), (y.shape, H0, W0)
    return y

