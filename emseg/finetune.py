"""Full fine-tune of a pretrained PretrainNet on one 2D domain with the affinity_mws target (fg + 6 disaffinities).

Protocol: 20k iterations, batch 4, 512² crops, MWS grid selected on all validation slices by mean VOI sum, then test;
arms differ only in the initialisation (--ckpt none = random init). --zero-shot skips training and evaluates the
checkpoint's own affinity head.

python -m emseg.finetune --run <name> --domain snemi --ckpt <run>/ckpt/step_040000.pt [--zero-shot]
"""
from __future__ import annotations

import argparse, json, sys, time, hashlib, random
import multiprocessing as mp

import numpy as np, pandas as pd, tifffile, torch

from emseg.eval.domains import get_domain
from emseg.eval.errclass import error_classes
from emseg.eval.gt import relabel_2d
from emseg.eval.metrics import slice_metrics
from emseg.paths import runs_root
from emseg.seg.crops import SliceCropDataset, normalize
from emseg.seg.unet import masked_bce_dice
from emseg.seg.mws import gt_of, voi_sum, SCHEME, mws
from emseg.seg.targets import GRIDS, DEFAULTS, OFFSETS_2D
from emseg.models.pretrainnet import PretrainNet

N_CH = 1 + len(OFFSETS_2D)


def build(ckpt: str, zero_shot: bool = False):
    if zero_shot:                                   # keep the pretraining affinity head (first N_CH channels = fg + OFFSETS_2D)
        ck = torch.load(ckpt, map_location="cpu"); args = ck.get("args", {})
        model = PretrainNet(args.get("arch", "convnextv2_base"), feat=args.get("feat", 32), extra_heads=args["extra_heads"])
        model.load_state_dict(ck["model"]); return model, ck.get("iter", -1)
    if ckpt and ckpt != "none":
        ck = torch.load(ckpt, map_location="cpu"); args = ck.get("args", {})
        model = PretrainNet(args.get("arch", "convnextv2_base"), inject=args.get("inject", False), feat=args.get("feat", 32), extra_heads={"aff": N_CH})
        missing, unexpected = model.load_state_dict({k: v for k, v in ck["model"].items() if not k.startswith("extra.")}, strict=False)
        assert all(k.startswith("extra.") for k in missing) and not unexpected, (missing, unexpected)
        return model, ck.get("iter", -1)
    return PretrainNet(extra_heads={"aff": N_CH}), 0


def logits(model, img):
    return model(torch.cat([img, torch.zeros_like(img)], 1))["aff"][:, :N_CH]


@torch.no_grad()
def infer(model, img, dev, tile=1024, ov=64, pad_to=32, scale=1.0):
    """Tiled full-slice prediction: img (H,W) normalised -> probabilities (N_CH,H,W) numpy. scale != 1: predict on the
    rescaled image and resize the probabilities back (affinity offsets are then in rescaled pixels)."""
    if scale != 1.0:
        import torch.nn.functional as F
        H0, W0 = img.shape
        t = F.interpolate(torch.from_numpy(img.astype(np.float32))[None, None], scale_factor=scale, mode="bilinear", align_corners=False, antialias=scale < 1)
        pr = infer(model, t[0, 0].numpy(), dev, tile, ov, pad_to)
        return F.interpolate(torch.from_numpy(pr)[None], size=(H0, W0), mode="bilinear", align_corners=False)[0].numpy()
    H, W = img.shape; out = torch.zeros(N_CH, H, W, device=dev); cnt = torch.zeros(1, H, W, device=dev)
    ys = list(range(0, max(H - tile, 0) + 1, tile - 2 * ov)) or [0]; xs = list(range(0, max(W - tile, 0) + 1, tile - 2 * ov)) or [0]
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    for y in ys:
        for x in xs:
            y1, x1 = min(y + tile, H), min(x + tile, W)
            crop = img[y:y1, x:x1]; ph, pw = (-crop.shape[0]) % pad_to, (-crop.shape[1]) % pad_to
            t = torch.from_numpy(np.pad(crop, ((0, ph), (0, pw)), mode="reflect")[None, None].astype(np.float32)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = torch.sigmoid(logits(model, t).float())[0, :, :crop.shape[0], :crop.shape[1]]
            out[:, y:y1, x:x1] += p; cnt[:, y:y1, x:x1] += 1
    return (out / cnt).cpu().numpy()


_V = None


def _grid_job(prm):
    return prm, float(np.mean([voi_sum(mws(d["prob"][1:].copy(), [list(o) for o in OFFSETS_2D], prm, d["prob"][0]), d["gt"]) for d in _V]))


def main(argv=None):
    global _V
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--domain", required=True, choices=sorted(SCHEME))
    p.add_argument("--seed", type=int, default=0); p.add_argument("--iters", type=int, default=20000); p.add_argument("--batch", type=int, default=4)
    p.add_argument("--patch", type=int, default=512); p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=0.05); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--val-every", type=int, default=2000); p.add_argument("--val-slices", type=int, default=5); p.add_argument("--grid-workers", type=int, default=8)
    p.add_argument("--zero-shot", action="store_true", help="no training: evaluate the checkpoint's own affinity head"); p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--sample", default="B", help="cremi sample")
    a = p.parse_args(argv)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda"); out = runs_root() / a.run; (out / "ckpt").mkdir(parents=True, exist_ok=True)
    kw = {"mip": 1} if a.domain == "mouse30um" else {"sample": a.sample} if a.domain == "cremi" else {}
    dom = get_domain(a.domain, **kw); splits = dom.splits(SCHEME[a.domain]); mask = a.domain == "mouse30um"
    model, ck_it = build(a.ckpt, a.zero_shot); model = model.to(dev)
    predict = lambda m, im, d: infer(m, im, d, scale=a.scale)
    json.dump({**vars(a), "ckpt_iter": ck_it, "splits": splits, "torch": torch.__version__, "start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(out / "config.json", "w"), indent=2)
    logf = open(out / "log.jsonl", "a")
    def log(**k):
        k["time"] = time.time(); logf.write(json.dumps(k) + "\n"); logf.flush()
    if a.zero_shot:
        a.iters = 0; torch.save(model.state_dict(), out / "ckpt/best.pt"); torch.save(model.state_dict(), out / "ckpt/last.pt")
    ds = SliceCropDataset(dom, splits["train"], "affinity_mws", patch=a.patch, n_samples=max(a.iters, 1) * a.batch, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    decay = [q for n, q in model.named_parameters() if q.ndim > 1 and not n.endswith(".bias")]
    no_decay = [q for n, q in model.named_parameters() if not (q.ndim > 1 and not n.endswith(".bias"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr)
    lr_at = lambda it: a.lr * it / a.warmup if it < a.warmup else a.lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * (it - a.warmup) / max(1, a.iters - a.warmup))))
    it, best, t0, run_l, n_skip = 0, float("inf"), time.time(), [], 0
    val_sub = splits["val"][: a.val_slices]; model.train()
    for img, tgt, msk, _ in (dl if a.iters > 0 else []):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        img, tgt, msk = img.to(dev, non_blocking=True), tgt.to(dev), msk.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = logits(model, img)
        loss = masked_bce_dice(lg.float(), tgt, msk)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step(); run_l.append(float(loss.detach()))
        else:
            n_skip += 1; log(event="skip_step", iter=it)
        it += 1
        if it % 100 == 0:
            log(event="train", iter=it, loss=float(np.mean(run_l)) if run_l else None, lr=lr_at(it), elapsed=time.time() - t0, n_skipped=n_skip); run_l = []
        if it % a.val_every == 0 or it == a.iters:
            model.eval()
            vs = []
            for z in val_sub:
                pr = predict(model, normalize(dom.load_image(z)), dev)
                vs.append(voi_sum(mws(pr[1:].copy(), [list(o) for o in OFFSETS_2D], DEFAULTS["affinity_mws"], pr[0]), gt_of(dom, z, mask)))
            v = float(np.mean(vs))
            log(event="val", iter=it, voi_sum=v); torch.save(model.state_dict(), out / "ckpt/last.pt")
            if v < best:
                best = v; torch.save(model.state_dict(), out / "ckpt/best.pt"); log(event="best", iter=it, voi_sum=v)
            print(f"it {it} val {v:.4f} best {best:.4f} {time.time() - t0:.0f}s", flush=True); model.train()
        if it >= a.iters:
            break
    log(event="train_done", iters=it, best_val=best, elapsed=time.time() - t0, n_skipped=n_skip); del dl, ds
    # ---------------- selection on all val slices: MWS grid
    model.load_state_dict(torch.load(out / "ckpt/best.pt", map_location=dev)); model.eval()
    _V = [{"prob": predict(model, normalize(dom.load_image(z)), dev), "gt": gt_of(dom, z, mask)} for z in splits["val"]]
    grid = []
    with mp.get_context("fork").Pool(a.grid_workers) as pool:
        for prm, v in pool.imap_unordered(_grid_job, GRIDS["affinity_mws"]):
            grid.append({"params": prm, "val_voi_sum": v}); log(event="grid", **grid[-1])
    sel = min(grid, key=lambda g: g["val_voi_sum"])
    json.dump({"grid": grid, "selected": sel, "criterion": "min mean per-slice val VOI sum"}, open(out / "val_grid.json", "w"), indent=2)
    _V = None
    # ---------------- test
    rows = []; pdir = out / "pred" / "test"; pdir.mkdir(parents=True, exist_ok=True)
    for z in splits["test"]:
        prob = predict(model, normalize(dom.load_image(z)), dev); gt = gt_of(dom, z, mask)
        seg = relabel_2d(mws(prob[1:].copy(), [list(o) for o in OFFSETS_2D], sel["params"], prob[0]))
        tifffile.imwrite(pdir / f"pred_{z:04d}.tif", seg.astype(np.uint32), compression="zlib")
        rows.append({"slice": z, **slice_metrics(seg, gt), **error_classes(seg, gt, prob[1:3])}); log(event="test_slice", z=z)
    df = pd.DataFrame(rows); df.to_csv(out / "test_per_slice.csv", index=False)
    summ = df.mean(numeric_only=True).drop("slice").to_dict()
    sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
    json.dump({"run": a.run, "status": "Completed", "end": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ckpt": a.ckpt, "best_val_5": best, "selected": sel,
               "test_summary": summ, "ckpt_sha256": {k: sha(out / f"ckpt/{k}.pt") for k in ("best", "last")}}, open(out / "done.json", "w"), indent=2)
    print("DONE test voi_sum", round(summ["voi_sum"], 4), "val(all)", round(sel["val_voi_sum"], 4)); print("exit=0", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback; traceback.print_exc(); print("exit=1", flush=True); sys.exit(1)
