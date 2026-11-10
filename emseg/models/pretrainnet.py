"""PretrainNet: ConvNeXt-V2 encoder (timm) + full-resolution side path + FPN decoder to stride 1 + heads.

Input channels: [image, mask flag] (mask flag = 0 at downstream / inference). Full-resolution path (stride 1, 32 ch)
and a stride-2 path feed the two finest decoder levels; the encoder stem (stride 4) alone would drop 1–2 px membranes.
`inject=True` additionally adds a learnable strided projection of the full-res features to the input of every encoder
stage (ablation arm).
Outputs: dict(feat (B,32,H,W), p4 (B,dec_ch,H/4,W/4) [FPN output, stride 4], rec (B,1,H,W), zpred (B,2,H,W) [channel 0 = z-1,
1 = z+1]) plus optional heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    return nn.GroupNorm(min(8, c), c)


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False), gn(cout), nn.GELU())


class PretrainNet(nn.Module):
    def __init__(self, arch: str = "convnextv2_base", in_ch: int = 2, feat: int = 32, dec_ch: int = 128, inject: bool = False,
                 pretrained: bool = False, drop_path: float = 0.1, extra_heads: dict | None = None):
        super().__init__()
        import timm
        self.enc = timm.create_model(arch, pretrained=pretrained, in_chans=in_ch, num_classes=0, drop_path_rate=drop_path)
        dims = [fi["num_chs"] for fi in self.enc.feature_info]            # [128,256,512,1024] for base
        self.dims, self.inject, self.feat = dims, inject, feat
        self.fr = nn.Sequential(ConvBlock(in_ch, feat), ConvBlock(feat, feat))                 # stride 1
        self.hr2 = nn.Sequential(nn.Conv2d(feat, 2 * feat, 3, stride=2, padding=1, bias=False), gn(2 * feat), nn.GELU())  # stride 2
        if inject:
            # stage i input: stride 4 * 2^(i-1) for i>0 (downsample inside stage), stride 4 with dims[0] channels for i=0
            self.inj = nn.ModuleList()
            for i in range(len(dims)):
                stride = 4 if i == 0 else 4 * 2 ** (i - 1); cin = dims[0] if i == 0 else dims[i - 1]
                self.inj.append(nn.Conv2d(feat, cin, kernel_size=stride, stride=stride))
        self.lat = nn.ModuleList([nn.Conv2d(d, dec_ch, 1) for d in dims])
        self.smooth = nn.ModuleList([ConvBlock(dec_ch, dec_ch) for _ in dims[:-1]])
        self.up2 = ConvBlock(dec_ch + 2 * feat, 2 * feat)                 # stride 2 level
        self.up1 = nn.Sequential(ConvBlock(2 * feat + feat, feat), ConvBlock(feat, feat))   # stride 1 level
        self.rec = nn.Conv2d(feat, 1, 1)
        self.zpred = nn.Conv2d(feat, 2, 1)
        self.extra = nn.ModuleDict({k: nn.Conv2d(feat, v, 1) for k, v in (extra_heads or {}).items()})

    def encode(self, x):
        fr = self.fr(x)
        y = self.enc.stem(x); feats = []
        for i, stage in enumerate(self.enc.stages):
            if self.inject:
                y = y + self.inj[i](fr)
            y = stage(y); feats.append(y)
        return fr, feats

    def forward(self, x):
        fr, feats = self.encode(x)
        p = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            p = self.lat[i](feats[i]) + F.interpolate(p, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[i](p)
        p4 = p
        h2 = self.hr2(fr)
        p = self.up2(torch.cat([F.interpolate(p, size=h2.shape[-2:], mode="bilinear", align_corners=False), h2], 1))
        f = self.up1(torch.cat([F.interpolate(p, size=fr.shape[-2:], mode="bilinear", align_corners=False), fr], 1))
        out = {"feat": f, "p4": p4, "rec": self.rec(f), "zpred": self.zpred(f)}
        for k, head in self.extra.items():
            out[k] = head(f)
        return out

    @torch.no_grad()
    def features(self, img: torch.Tensor) -> torch.Tensor:
        """Downstream feature extractor: img (B,1,H,W) normalised, mask flag = 0."""
        x = torch.cat([img, torch.zeros_like(img)], 1)
        return self.forward(x)["feat"]
