"""Frozen-feature probe: how much membrane / affinity information the pretrained features carry.

Backbone + decoder frozen (mask flag = 0); a 1x1 head (fg + 6 standard disaffinities) is trained for --steps on the
training slices of --domain; evaluated on its first --val-slices validation slices with default MWS (VOI sum) and the
1/3/9 px pair diagnostics (miss = true boundary judged 'same', cut = within-object pair judged 'different').
--ckpt none  -> randomly initialised backbone (reference).
--feat <spec> overrides --ckpt with any extractor of emseg.models.features (hand | dec:<ckpt> | enc:<ckpt> | ft:<sd>).
Also reports a training-free emergence measure on the same val features: pair_auc{d} = ROC AUC of the cosine distance of
(per-channel standardised) features separating different-object from same-object pixel pairs at distance d (8 directions).

python -m emseg.probe --ckpt <run>/ckpt/last.pt --domain snemi --out <run>/probe.json
"""
from __future__ import annotations

import argparse, json, time

import numpy as np
import torch
import torch.nn.functional as F

from emseg.eval.domains import get_domain
from emseg.seg.crops import SliceCropDataset, normalize
from emseg.seg.unet import masked_bce_dice
from emseg.seg.mws import gt_of, voi_sum, mws
from emseg.seg.targets import OFFSETS_2D, DEFAULTS
from emseg.models.features import build_features


def pair_auc(f, gt, ds=(3, 9, 27), n=200_000, seed=0):
    """f (C,H,W) torch on device; gt (H,W) instance ids (0 = ignore)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); C, H, W = f.shape
    f = f - f.mean((1, 2), keepdim=True); f = f / (f.std((1, 2), keepdim=True) + 1e-6); f = F.normalize(f, dim=0)
    g = torch.from_numpy(gt.astype(np.int64)).to(f.device); out = {}
    for d in ds:
        dirs = np.array([[0, d], [d, 0], [d, d], [d, -d], [0, -d], [-d, 0], [-d, -d], [-d, d]])
        k = dirs[rng.integers(8, size=n)]; y = rng.integers(d, H - d, size=n); x = rng.integers(d, W - d, size=n)
        y, x, y2, x2 = [torch.from_numpy(v).to(f.device) for v in (y, x, y + k[:, 0], x + k[:, 1])]
        a, b = g[y, x], g[y2, x2]; ok = (a > 0) & (b > 0)
        dist = 1 - (f[:, y, x] * f[:, y2, x2]).sum(0)
        lab = (a != b)[ok].cpu().numpy(); s = dist[ok].float().cpu().numpy()
        out[f"pair_auc{d}"] = float(roc_auc_score(lab, s)) if 0 < lab.sum() < len(lab) else float("nan")
    return out


@torch.no_grad()
def feats_full(model, img, dev, tile=1024, pad_to=32):
    """Full-slice features by tiling (overlap 64) to bound memory. img (H,W) normalised -> (C,H,W) on device."""
    H, W = img.shape; ov = 64
    C = model.feat; out = torch.zeros(C, H, W, device=dev); cnt = torch.zeros(1, H, W, device=dev)
    ys = list(range(0, max(H - tile, 0) + 1, tile - 2 * ov)) or [0]; xs = list(range(0, max(W - tile, 0) + 1, tile - 2 * ov)) or [0]
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    for y in ys:
        for x in xs:
            y1, x1 = min(y + tile, H), min(x + tile, W)
            crop = img[y:y1, x:x1]; ph, pw = (-crop.shape[0]) % pad_to, (-crop.shape[1]) % pad_to
            t = torch.from_numpy(np.pad(crop, ((0, ph), (0, pw)), mode="reflect")[None, None].astype(np.float32)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = model.features(t)[0, :, :crop.shape[0], :crop.shape[1]].float()
            out[:, y:y1, x:x1] += f; cnt[:, y:y1, x:x1] += 1
    return out / cnt


def pair_diag(prob, gt, ks=(1, 3, 9)):
    rows = {}
    for k in ks:
        pr = prob[1 + OFFSETS_2D.index([0, -k])][:, k:]; g0, g1 = gt[:, k:], gt[:, :-k]
        valid = (g0 > 0) & (g1 > 0); diff = g0 != g1
        rows[f"miss{k}"] = float(((pr < 0.5) & diff & valid).sum() / max((diff & valid).sum(), 1))
        rows[f"cut{k}"] = float(((pr > 0.5) & ~diff & valid).sum() / max((~diff & valid).sum(), 1))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=""); p.add_argument("--feat", default=""); p.add_argument("--out", required=True); p.add_argument("--sample", default="B", help="cremi sample")
    p.add_argument("--domain", default="snemi"); p.add_argument("--steps", type=int, default=2000); p.add_argument("--batch", type=int, default=4)
    p.add_argument("--patch", type=int, default=512); p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--val-slices", type=int, default=5)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--seed", type=int, default=0); p.add_argument("--scale", type=float, default=1.0, help="input rescale factor (domains of unknown pixel size)")
    a = p.parse_args(argv)
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda")
    spec = a.feat or f"dec:{a.ckpt}"; model, it = build_features(spec, dev)
    for q in model.parameters():
        q.requires_grad_(False)
    kw = {"mip": 1} if a.domain == "mouse30um" else {"sample": a.sample} if a.domain == "cremi" else {}
    dom = get_domain(a.domain, **kw); splits = dom.splits("interim" if a.domain != "snemi" else "protocol"); mask = a.domain == "mouse30um"
    ds = SliceCropDataset(dom, splits["train"], "affinity_mws", patch=a.patch, n_samples=a.steps * a.batch, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, drop_last=True)
    head = torch.nn.Conv2d(model.feat, 1 + len(OFFSETS_2D), 1).to(dev); opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.05)
    t0 = time.time(); losses = []; n = 0
    for img, tgt, msk, _ in dl:
        img, tgt, msk = img.to(dev), tgt.to(dev), msk.to(dev)
        if a.scale != 1.0:
            img = F.interpolate(img, scale_factor=a.scale, mode="bilinear", align_corners=False)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.features(img).float()
        if a.scale != 1.0:
            f = F.interpolate(f, size=tgt.shape[-2:], mode="bilinear", align_corners=False)
        loss = masked_bce_dice(head(f), tgt, msk)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); losses.append(float(loss)); n += 1
        if n >= a.steps:
            break
    # evaluation
    res = []
    for z in splits["val"][: a.val_slices]:
        img = normalize(dom.load_image(z)); gt = gt_of(dom, z, mask)
        if a.scale != 1.0:
            t = torch.from_numpy(img)[None, None]; img_s = F.interpolate(t, scale_factor=a.scale, mode="bilinear", align_corners=False)[0, 0].numpy()
        else:
            img_s = img
        f = feats_full(model, img_s, dev)
        if a.scale != 1.0:
            f = F.interpolate(f[None], size=img.shape, mode="bilinear", align_corners=False)[0]
        with torch.no_grad():
            prob = torch.sigmoid(head(f[None]))[0].cpu().numpy()
        seg = mws(prob[1:].copy(), [list(o) for o in OFFSETS_2D], DEFAULTS["affinity_mws"], prob[0])
        res.append({"z": int(z), "voi_sum": voi_sum(seg, gt), **pair_diag(prob, gt), **pair_auc(f, gt)})
    summ = {k: float(np.mean([r[k] for r in res])) for k in res[0] if k != "z"}
    out = {"ckpt": a.ckpt, "feat": spec, "ckpt_iter": it, "domain": a.domain, "scale": a.scale, "probe_steps": a.steps, "final_train_loss": float(np.mean(losses[-100:])),
           "val": summ, "per_slice": res, "elapsed": time.time() - t0}
    json.dump(out, open(a.out, "w"), indent=2)
    print("PROBE", json.dumps({k: round(v, 4) for k, v in summ.items()}), f"ckpt_iter={it}"); print("exit=0", flush=True)


if __name__ == "__main__":
    main()
