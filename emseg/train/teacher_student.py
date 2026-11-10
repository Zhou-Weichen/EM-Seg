"""Teacher / student trainer. One driver, three label sources:

  --exp gold         ground-truth cell instances of the labelled datasets (the teacher)
  --exp pseudo       ground truth where available + teacher pseudo-labels (--pseudo-root) elsewhere (the student)
  --exp superpixels  ridge-prior superpixels on all EM training images, no labels (weak baseline)
Loss = masked BCE+Dice on fg + 8 disaffinities (1/3/9/27 px) at stride 1 + supervised contrastive loss on sampled
points at two levels (p4 stride 4, feat stride 1). SNEMI3D held out by default; mouse30um is never in the corpus.
Self-training options (Depth Anything V2/3 recipe): --ignore-top q drops the q fraction of largest-loss pixels of every
pseudo-labelled image (its noisy/implausible regions); --conf adds a per-pixel confidence head (extra.conf, 1 ch; loss
exp(-s)*L + lambda*s, Kendall & Gal) applied to pseudo-labelled images; --gt-sched "f0:t,f1" keeps the ground-truth
sampling fraction at f0 until fraction t of the iterations, then f1 (GT first, teacher labels later).
ckpt "model" = PretrainNet incl. extra.aff (first 7 channels = fg + emseg.seg.targets.OFFSETS_2D, so it can be used zero-shot).

torchrun --nproc_per_node N -m emseg.train.teacher_student --run <name> --exp gold --arch convnextv2_large --pretrained
"""
from __future__ import annotations

import argparse, json, os, sys, time, random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from emseg.paths import emdata_root, runs_root
import torch.nn as nn

from emseg import __version__
from emseg.train.data import InstIndex, InstCropDataset, parse_sched
from emseg.models.pretrainnet import PretrainNet
from emseg.train.relations import OFFSETS, pair_targets, ProjHead, sample_sup_points, supcon, read

EXPS = {"gold": dict(sources={"gt": "cell", "other": ""}, gt_frac=None),
        "pseudo": dict(sources={"gt": "cell", "other": "pl"}, gt_frac=0.5),
        "superpixels": dict(sources={"gt": "", "other": "sp"}, gt_frac=None)}
PL = 2   # src code of pseudo-labelled images in emseg.train.data


def _dice(logits, tgt, msk, sel=None, eps=1e-6):
    p = torch.sigmoid(logits) * msk; t = tgt * msk
    if sel is not None:
        p, t = p[sel], t[sel]
    return (1 - (2 * (p * t).sum((0, 2, 3)) + eps) / ((p * p).sum((0, 2, 3)) + (t * t).sum((0, 2, 3)) + eps)).mean()


def masked_bce_dice(logits, tgt, msk, pl=None, ignore_top=0.0, conf=None, conf_lambda=1.0, conf_norm=False, src_norm=False, pl_weight=1.0, eps=1e-6):
    """logits/tgt/msk (B,K,H,W); pl (B,) bool marks pseudo-labelled images; conf (B,1,H,W) log-variance-like score.
    conf_norm=False: Kendall-Gal form exp(-s)*L + lambda*s (degenerates to a global ~40x up-weighting of pl pixels when L << 1). conf_norm=True: per-image weights w = exp(-s) / mean_valid(exp(-s)) (mean 1, so pl and GT pixels
    keep equal average weight) and an entropy regulariser lambda * mean(w log w); optimum w ∝ exp(-L/lambda).
    Returns (loss, mask actually used, mean conf weight on pl images (conf_norm: std of w))."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, tgt, reduction="none") * msk
    if pl is not None and pl.any() and ignore_top > 0:                      # Depth Anything V2: drop the top-q largest-loss pixels per pseudo-labelled image
        with torch.no_grad():
            pix = bce.sum(1) / msk.sum(1).clamp(min=1); anyv = msk.sum(1) > 0            # (B,H,W)
            keep = torch.ones_like(anyv)
            for b in torch.nonzero(pl).flatten().tolist():
                v = pix[b][anyv[b]]
                if v.numel() > 100:
                    thr = torch.quantile(v.float()[torch.randperm(v.numel(), device=v.device)[:200_000]], 1 - ignore_top)
                    keep[b] = ~(anyv[b] & (pix[b] > thr))
        msk = msk * keep[:, None].float(); bce = bce * keep[:, None].float()
    w_mean = bce.new_zeros(())
    if conf is not None and pl is not None and pl.any():                     # confidence weighting on pseudo-labelled images only
        plw = pl.float()[:, None, None, None]
        if conf_norm:
            s = conf.clamp(-6, 6); v = msk[:, :1]                              # v: valid pixels (B,1,H,W)
            e = torch.exp(-s) * v; w = e / (e.sum((2, 3), keepdim=True) / v.sum((2, 3), keepdim=True).clamp(min=1)).clamp(min=eps)
            reg = (w * torch.log(w.clamp(min=eps)) * v).sum((2, 3), keepdim=True) / v.sum((2, 3), keepdim=True).clamp(min=1)   # mean w log w per image
            bce = bce * (1 - plw) + (bce * w + conf_lambda * reg * msk) * plw               # reg enters l_bce as the mask-weighted mean over pl images
            n = (v * plw).sum().clamp(min=1); mw = (w * v * plw).sum() / n
            w_mean = torch.sqrt(((w - mw) ** 2 * v * plw).sum() / n)         # std of w on pl pixels (mean is 1 by construction)
        else:
            s = conf.clamp(-4, 4); w = torch.exp(-s)
            bce = bce * (1 - plw) + (bce * w + conf_lambda * s * msk) * plw
            w_mean = (w * msk[:, :1] * plw).sum() / (msk[:, :1] * plw).sum().clamp(min=1)
    if src_norm and pl is not None and pl.any() and (~pl).any():             # per-source normalisation: L = L_gt + pl_weight * L_pl (each a mean over its own pixels)
        gtw = (~pl).float()[:, None, None, None]; plw = pl.float()[:, None, None, None]
        l_bce = (bce * gtw).sum() / (msk * gtw).sum().clamp(min=1) + pl_weight * (bce * plw).sum() / (msk * plw).sum().clamp(min=1)
        return l_bce + _dice(logits, tgt, msk, ~pl) + pl_weight * _dice(logits, tgt, msk, pl), msk, w_mean
    scale = pl_weight if (src_norm and pl is not None and pl.all()) else 1.0  # all-pl batch under src_norm: weight it as a whole
    l_bce = bce.sum() / msk.sum().clamp(min=1)
    return scale * (l_bce + _dice(logits, tgt, msk)), msk, w_mean


class Net(nn.Module):
    def __init__(self, net: PretrainNet, dim: int = 64):
        super().__init__()
        self.net = net
        self.proj = nn.ModuleDict({"p4": ProjHead(net.lat[0].out_channels, dim), "feat": ProjHead(net.feat, dim)})

    def forward(self, x, pts):
        o = self.net(x)
        return o["aff"], o.get("conf"), {l: self.proj[l](read(o[l], pts)) for l in self.proj}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--exp", required=True, choices=list(EXPS))
    p.add_argument("--manifest", default=str(emdata_root() / "manifests/em_only.csv"), help="comma-separated csv list"); p.add_argument("--data-root", default=str(emdata_root()))
    p.add_argument("--pseudo-root", default=""); p.add_argument("--exclude", default="snemi3d")
    p.add_argument("--arch", default="convnextv2_base"); p.add_argument("--pretrained", action="store_true", help="timm ImageNet weights for the encoder (when no --init)")
    p.add_argument("--init", default="", help="PretrainNet checkpoint to start from (encoder + decoder; heads only with --init-heads)")
    p.add_argument("--iters", type=int, default=20_000); p.add_argument("--batch", type=int, default=16, help="per GPU"); p.add_argument("--patch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4, help="for total batch 32"); p.add_argument("--wd", type=float, default=0.05); p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--workers", type=int, default=12); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--points", type=int, default=1024); p.add_argument("--tau", type=float, default=0.1); p.add_argument("--con-lambda", type=float, default=1.0)
    p.add_argument("--sp-merge", type=float, default=0.6)
    p.add_argument("--ignore-top", type=float, default=0.0, help="fraction of largest-loss pixels ignored per pseudo-labelled image (DA-V2: 0.1)")
    p.add_argument("--conf", action="store_true", help="confidence head on pseudo-labelled images"); p.add_argument("--conf-lambda", type=float, default=1.0)
    p.add_argument("--conf-norm", action="store_true", help="normalised confidence weights (mean 1 per image) + entropy regulariser instead of the Kendall-Gal form")
    p.add_argument("--init-heads", action="store_true", help="also load matching extra.* heads from --init (default: encoder/FPN only)")
    p.add_argument("--src-norm", action="store_true", help="normalise the affinity loss per source: L = L_gt + pl_weight * L_pl (pixel counts of one source cannot swamp the other)")
    p.add_argument("--pl-weight", type=float, default=1.0, help="weight of the pseudo-label loss term under --src-norm")
    p.add_argument("--pl-ramp", type=int, default=0, help="linearly ramp pl-weight from 0 over this many iterations")
    p.add_argument("--gt-power", type=float, default=None, help="GT datasets sampled with mass ∝ n_rows**power (default: manifest weights)")
    p.add_argument("--gt-sched", default="", help='"f0:t,f1": GT sampling fraction f0 until fraction t of the run, then f1 (overrides the exp default)')
    p.add_argument("--pl-power", type=float, default=0.5, help="pseudo-label datasets sampled with mass ∝ n_rows^power")
    p.add_argument("--pl-min-inst", type=int, default=8); p.add_argument("--pl-max-frac", type=float, default=0.6, help="drop pseudo-labelled rows with fewer instances / one dominating instance (needs <pseudo_root>/stats.csv)")
    p.add_argument("--ckpt-every", type=int, default=1000); p.add_argument("--snap-every", type=int, default=5000); p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--prefetch", type=int, default=2, help="batches queued per worker (was 4; each queued batch is shared memory)"); p.add_argument("--resume", default=""); p.add_argument("--smoke", action="store_true")
    a = p.parse_args(argv)
    cfg = EXPS[a.exp]
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda"); main_proc = rank == 0
    random.seed(a.seed + rank); np.random.seed(a.seed + rank); torch.manual_seed(a.seed)
    out = runs_root() / a.run
    if main_proc:
        (out / "ckpt").mkdir(parents=True, exist_ok=True)
    sched = parse_sched(a.gt_sched) if a.gt_sched else None
    index = InstIndex(a.manifest, cfg["sources"], exclude=[d for d in a.exclude.split(",") if d], gt_frac=cfg["gt_frac"], pseudo_root=a.pseudo_root, pl_power=a.pl_power, pl_min_inst=a.pl_min_inst, pl_max_frac=a.pl_max_frac, gt_power=a.gt_power)
    ds = InstCropDataset(index, a.data_root, a.pseudo_root, patch=a.patch, n_samples=a.iters * a.batch + 1000, seed=a.seed * 1000 + rank, sp_merge=a.sp_merge, gt_sched=sched)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=a.workers > 0, prefetch_factor=a.prefetch if a.workers > 0 else None)
    extra_heads = {"aff": 1 + len(OFFSETS)}
    if a.conf:
        extra_heads["conf"] = 1
    net = PretrainNet(a.arch, feat=32, pretrained=a.pretrained and not a.init, extra_heads=extra_heads)
    init_iter = None
    if a.init:
        ck = torch.load(a.init, map_location="cpu"); init_iter = ck.get("iter")
        own = net.state_dict()
        sd = {k: v for k, v in ck["model"].items() if (a.init_heads and k in own and own[k].shape == v.shape) or not k.startswith("extra.")}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        assert all(k.startswith("extra.") for k in missing) and not unexpected, (missing, unexpected)
        if main_proc: print(f"init from {a.init}: {len(sd)} tensors loaded, heads not loaded: {sorted({k.split('.')[1] for k in missing})}", flush=True)
    model = Net(net).to(dev)
    dmodel = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index], find_unused_parameters=True) if ddp else model
    lr = a.lr * (a.batch * world) / 32
    decay = [q for n, q in model.named_parameters() if q.ndim > 1 and not n.endswith(".bias")]
    no_decay = [q for n, q in model.named_parameters() if not (q.ndim > 1 and not n.endswith(".bias"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    lr_at = lambda it: lr * it / a.warmup if it < a.warmup else lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * (it - a.warmup) / max(1, a.iters - a.warmup))))
    it = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False); model.load_state_dict(ck["full"]); opt.load_state_dict(ck["opt"]); it = ck["iter"]; ds.offset = it * a.batch   # keep the GT schedule on the global clock
    if main_proc:
        n_src = {s: sum(1 for r in index.rows if r[8] == s) for s in ("gt", "sp", "pl")}
        n_params = sum(q.numel() for q in model.net.parameters())
        json.dump({**vars(a), **{k: v for k, v in cfg.items()}, "offsets": OFFSETS, "world": world, "lr_eff": lr, "init_iter": init_iter, "n_rows": len(index.rows),
                   "rows_per_source": n_src, "rows_per_dataset": index.n_per_dataset, "datasets": index.datasets, "n_params_M": n_params / 1e6, "extra_heads": extra_heads,
                   "emseg": __version__, "torch": torch.__version__, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(out / "config.json", "w"), indent=2)
        logf = open(out / "log.jsonl", "a")
    def log(**k):
        if main_proc:
            k["time"] = time.time(); logf.write(json.dumps(k) + "\n"); logf.flush()
    model.train(); t0 = t_last = time.time(); acc = {}; n_skip = 0
    for b in dl:
        if it >= a.iters:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        tgt, msk = pair_targets(b["pos"], b["neg"], b["valid"])
        tgt = torch.cat([b["fg"][:, None], tgt], 1); msk = torch.cat([b["fg_valid"][:, None].float(), msk], 1)
        idx, pts = sample_sup_points(b["valid"], b["prior"], a.points)
        pid = b["pos"].reshape(b["pos"].shape[0], -1).gather(1, idx); nid = b["neg"].reshape(b["neg"].shape[0], -1).gather(1, idx)
        pl = b["src"] == PL
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, conf, emb = dmodel(torch.cat([b["img"], torch.zeros_like(b["img"])], 1), pts)
        plw_it = a.pl_weight * (min(1.0, it / a.pl_ramp) if a.pl_ramp > 0 else 1.0)
        l_aff, msk_used, w_conf = masked_bce_dice(logits.float(), tgt, msk, pl, a.ignore_top, conf.float() if conf is not None else None, a.conf_lambda, a.conf_norm, a.src_norm, plw_it)
        l_con = {}; loss = l_aff
        for l, e in emb.items():
            l_con[l], n_anc = supcon(e.float(), pid, nid, a.tau); loss = loss + a.con_lambda * l_con[l]
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        else:
            n_skip += 1; log(event="skip_step", iter=it)
        it += 1
        for k, v in {"loss": float(loss), "aff": float(l_aff), **{f"con_{l}": float(v) for l, v in l_con.items()}, "anchors": n_anc / b["img"].shape[0],
                     "mask_frac": float(msk_used[:, 1:].mean()), "pos_frac": float((tgt[:, 1:] * msk[:, 1:]).sum() / msk[:, 1:].sum().clamp(min=1)),
                     "pl_frac": float(pl.float().mean()), "conf_w": float(w_conf), "pl_w": plw_it}.items():
            acc.setdefault(k, []).append(v)
        if it % a.log_every == 0:
            dt = time.time() - t_last; t_last = time.time(); r = {k: float(np.mean(v)) for k, v in acc.items()}; acc = {}
            log(event="train", iter=it, **r, lr=lr_at(it), img_per_s=a.log_every * a.batch * world / dt, mem_gb=torch.cuda.max_memory_allocated() / 2**30, elapsed=time.time() - t0, n_skipped=n_skip)
            if main_proc:
                print(f"it {it} " + " ".join(f"{k} {v:.4f}" for k, v in r.items()) + f" {a.log_every * a.batch * world / dt:.1f} img/s", flush=True)
        if main_proc and (it % a.ckpt_every == 0 or it == a.iters):
            meta = {"iter": it, "args": {**vars(a), "feat": 32, "extra_heads": extra_heads}, "offsets": OFFSETS}
            torch.save({"model": model.net.state_dict(), "full": model.state_dict(), "opt": opt.state_dict(), **meta}, out / "ckpt/last.pt")
            if it % a.snap_every == 0 or it == a.iters:
                torch.save({"model": model.net.state_dict(), **meta}, out / f"ckpt/step_{it:06d}.pt")
        if a.smoke and it >= 30:
            break
    log(event="train_done", iters=it, elapsed=time.time() - t0, n_skipped=n_skip)
    if main_proc:
        print("exit=0", flush=True)
    if ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback; traceback.print_exc(); print("exit=1", flush=True); sys.exit(1)
