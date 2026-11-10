"""Frozen feature extractors for probing / evaluation. Each has `.feat` (channels) and `.features(img (B,1,H,W)) -> (B,C,H,W)`.

  hand          : 5x5 intensity neighbourhood (25 ch) + ridge prior σ 1,2 (2 ch) — handcrafted reference, no learning
  dec:<ckpt|none>: PretrainNet decoder output (32 ch, stride 1)  — what emseg.probe used originally
  enc:<ckpt|none>: PretrainNet encoder stages 0+1 (stride 4/8, bilinear to stride 1) + decoder output
  ft:<state_dict>: decoder output of an emseg.finetune model (supervised reference)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from emseg.models.pretrainnet import PretrainNet
from emseg.seg.prior import ridge_prior


def load_net(ckpt: str) -> tuple[PretrainNet, int]:
    if ckpt and ckpt != "none":
        ck = torch.load(ckpt, map_location="cpu"); args = ck.get("args", {})
        net = PretrainNet(args.get("arch", "convnextv2_base"), inject=args.get("inject", False), feat=args.get("feat", 32))
        net.load_state_dict({k: v for k, v in ck["model"].items() if not k.startswith("extra.")}); return net, ck.get("iter", -1)
    return PretrainNet(), 0


class Hand(nn.Module):
    feat = 27

    @torch.no_grad()
    def features(self, img):
        p = F.unfold(F.pad(img, (2, 2, 2, 2), mode="reflect"), 5).view(img.shape[0], 25, *img.shape[-2:])
        pr = torch.stack([torch.from_numpy(ridge_prior(i[0].float().cpu().numpy(), sigmas=(s,))) for i in img for s in (1.0, 2.0)]).view(img.shape[0], 2, *img.shape[-2:])
        return torch.cat([p, pr.to(img.device, img.dtype)], 1)


class Dec(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net; self.feat = net.feat

    @torch.no_grad()
    def features(self, img):
        return self.net.forward(torch.cat([img, torch.zeros_like(img)], 1))["feat"]


class Enc(nn.Module):
    def __init__(self, net):
        super().__init__(); self.net = net; self.feat = net.dims[0] + net.dims[1] + net.feat

    @torch.no_grad()
    def features(self, img):
        x = torch.cat([img, torch.zeros_like(img)], 1)
        _, feats = self.net.encode(x)
        up = [F.interpolate(f, size=img.shape[-2:], mode="bilinear", align_corners=False) for f in feats[:2]]
        return torch.cat(up + [self.net.forward(x)["feat"]], 1)


def build_features(spec: str, dev) -> tuple[nn.Module, int]:
    if spec == "hand":
        return Hand().to(dev).eval(), 0
    kind, ck = spec.split(":", 1)
    if kind in ("dec", "enc"):
        net, it = load_net(ck)
        return (Dec if kind == "dec" else Enc)(net).to(dev).eval(), it
    if kind == "ft":
        net = PretrainNet(extra_heads={"aff": 7}); net.load_state_dict(torch.load(ck, map_location="cpu"))
        return Dec(net).to(dev).eval(), -1
    raise ValueError(spec)
