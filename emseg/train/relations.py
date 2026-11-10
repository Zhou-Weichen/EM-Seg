"""Instance-relation losses (P15) from pos/neg id maps (see data3).

pair_targets: disaffinity targets for given offsets, torch_em AffinityTransform convention (channel o compares p with
p + o; 1 = different objects). Pairs that are neither "same" (equal pos id >= 0) nor "different" (different neg ids,
both >= 0) are masked out.
supcon: supervised contrastive loss on N sampled points per image; positives = equal pos id, negatives = different neg
ids, others ignored; points half uniform over supervised pixels, half in proportion to the membrane prior (hard pairs).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F



def read(feat: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Bilinear read-out of feat (B,C,h,w) at normalised points pts (B,N,2) -> (B,N,C)."""
    return F.grid_sample(feat.float(), pts[:, None], mode="bilinear", padding_mode="border", align_corners=False)[:, :, 0].transpose(1, 2)

OFFSETS = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, 0], [0, -9], [-27, 0], [0, -27]]   # first 6 = emseg.seg.targets.OFFSETS_2D


def _shift(a: torch.Tensor, dy: int, dx: int, fill):
    """b[..., y, x] = a[..., y + dy, x + dx] (fill outside)."""
    H, W = a.shape[-2:]
    out = torch.full_like(a, fill)
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[..., ys, xs] = a[..., yd, xd]
    return out


def pair_targets(pos, neg, valid, offsets=OFFSETS):
    """pos/neg (B,H,W) int64, valid (B,H,W) bool -> target, mask (B,K,H,W) float."""
    ts, ms = [], []
    for dy, dx in offsets:
        p2, n2, v2 = _shift(pos, dy, dx, -1), _shift(neg, dy, dx, -1), _shift(valid, dy, dx, False)
        same = (pos >= 0) & (pos == p2)
        diff = (neg >= 0) & (n2 >= 0) & (neg != n2)
        ts.append(diff.float()); ms.append((valid & v2 & (same | diff)).float())
    return torch.stack(ts, 1), torch.stack(ms, 1)


class ProjHead(nn.Module):
    def __init__(self, cin: int, dim: int = 64, hidden: int = 256):
        super().__init__(); self.mlp = nn.Sequential(nn.Linear(cin, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)


def sample_sup_points(valid, prior, n: int, eps: float = 0.02):
    """-> integer pixel coords (B,n) flat indices and normalised (x,y) points (B,n,2) at pixel centres."""
    B, H, W = valid.shape
    v = valid.reshape(B, -1).float()
    wu = v + 1e-12; wp = v * (prior.reshape(B, -1) + eps) + 1e-12
    idx = torch.cat([torch.multinomial(wu, n // 2, replacement=True), torch.multinomial(wp, n - n // 2, replacement=True)], 1)
    yy, xx = (idx // W).float() + 0.5, (idx % W).float() + 0.5
    return idx, torch.stack([2 * xx / W - 1, 2 * yy / H - 1], -1)


def supcon(emb, pid, nid, tau: float = 0.1):
    """emb (B,N,D) normalised; pid/nid (B,N) int64. Returns (loss, n_anchors)."""
    s = emb @ emb.transpose(1, 2) / tau
    eye = torch.eye(emb.shape[1], dtype=torch.bool, device=emb.device)[None]
    pos = (pid[:, :, None] == pid[:, None, :]) & (pid[:, :, None] >= 0) & ~eye
    neg = (nid[:, :, None] != nid[:, None, :]) & (nid[:, :, None] >= 0) & (nid[:, None, :] >= 0)
    keep = pos | neg
    s = s.masked_fill(~keep, -1e4)
    log_den = torch.logsumexp(s, -1)
    log_num = torch.logsumexp(s.masked_fill(~pos, -1e4), -1)
    ok = pos.any(-1) & neg.any(-1)
    l = (log_den - log_num)[ok]
    return (l.mean() if l.numel() else s.new_zeros(())), int(ok.sum())
