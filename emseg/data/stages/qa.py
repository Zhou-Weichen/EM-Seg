"""Stage 4 - per-image quality metrics (every 2D image of em/, em2d/, natural/) -> qa/images.csv
Metrics (computed on the stored uint8 raw and the v2 labels/masks):
  valid_frac        fraction of pixels where any class is annotated (mask)
  raw_mean/raw_std  intensity statistics over the image
  sat_frac          fraction of the non-padding pixels at 0 or 255 (clipping / over-exposure after normalisation)
  pad_frac          fraction of border padding (constant regions connected to the image border)
  lap_var           variance of the Laplacian of the 2x-downsampled image (sharpness; blur detection)
  for each class c: n_<c> (instances), cov_<c> (labelled fraction of the annotated area), maxinst_<c> (largest
                    instance / annotated area), medarea_<c> (median instance area, px)
  bnd_contrast      dense 'cell' class only: median raw on instance boundaries minus median raw in instance
                    interiors (EM membranes are dark -> negative; strongly positive = misaligned / inverted labels)
  md5               of the raw bytes (exact duplicates)
  phash             64-bit DCT perceptual hash of the whole image (near duplicates)
usage: python -m emseg.data.stages.qa [--workers 48]"""
import argparse, csv, glob, hashlib, json
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from scipy import ndimage as ndi
from scipy.fft import dctn

from emseg.data.standard import DATA_ROOT, padding_mask


def phash(img):
    h, w = img.shape
    small = img[: h - h % 32 or h, : w - w % 32 or w].astype(np.float32)
    small = small.reshape(32, small.shape[0] // 32, 32, small.shape[1] // 32).mean((1, 3)) if min(h, w) >= 32 else np.resize(small, (32, 32))
    d = dctn(small, norm="ortho")[:8, :8].ravel()[1:]
    bits = d > np.median(d)
    return "%016x" % int("".join("1" if b else "0" for b in np.r_[bits, False]), 2)


def metrics(raw, labels, masks):
    r = {}
    anymask = np.zeros(raw.shape, bool)
    for m in masks.values():
        anymask |= m > 0
    r["valid_frac"] = round(float(anymask.mean()), 4)
    r["raw_mean"] = round(float(raw.mean()), 2); r["raw_std"] = round(float(raw.std()), 2)
    tissue = padding_mask(raw) > 0                       # saturation is measured on the imaged area only: border padding
    r["pad_frac"] = round(float(1 - tissue.mean()), 4)   # (constant 0 after normalisation) is not over-exposure
    r["sat_frac"] = round(float(((raw == 0) | (raw == 255))[tissue].mean()), 4) if tissue.any() else 1.0
    sm = raw[::2, ::2].astype(np.float32)
    r["lap_var"] = round(float(ndi.laplace(sm).var()), 2)
    for c, lab in labels.items():
        m = masks[c] > 0; area = max(int(m.sum()), 1)
        ids, cnt = np.unique(lab[m], return_counts=True); cnt = cnt[ids > 0]
        r[f"n_{c}"] = int(cnt.size); r[f"cov_{c}"] = round(float(cnt.sum() / area), 4)
        r[f"maxinst_{c}"] = round(float(cnt.max() / area), 4) if cnt.size else 0.0
        r[f"medarea_{c}"] = float(np.median(cnt)) if cnt.size else 0.0
        if c == "cell" and cnt.size >= 2:
            l = lab
            bd = np.zeros(l.shape, bool)
            bd[:-1] |= l[:-1] != l[1:]; bd[1:] |= l[:-1] != l[1:]; bd[:, :-1] |= l[:, :-1] != l[:, 1:]; bd[:, 1:] |= l[:, :-1] != l[:, 1:]
            bd &= m
            inner = (l > 0) & m & ~ndi.binary_dilation(bd, iterations=3)
            r["bnd_contrast"] = round(float(np.median(raw[bd]) - np.median(raw[inner])), 1) if bd.any() and inner.any() else ""
    r["md5"] = hashlib.md5(np.ascontiguousarray(raw).tobytes()).hexdigest()
    r["phash"] = phash(raw)
    return r


def do_volume(p):
    rows = []
    with h5py.File(p, "r") as f:
        a = f.attrs; classes = json.loads(a["annotated_classes"]); res = list(a["resolution_nm"])
        Z, H, W = f["raw"].shape; kept = list(a["kept_z"]) if "kept_z" in a else list(range(Z))
        for z in range(Z):
            labels = {c: f[f"labels/{c}"][z] for c in classes}; masks = {c: f[f"masks/{c}"][z] for c in classes}
            rows.append({"uid": f"{a['dataset']}/{a['volume']}/{z}", "dataset": a["dataset"], "volume": a["volume"],
                         "file": str(p.relative_to(DATA_ROOT) if hasattr(p, "relative_to") else p).replace(str(DATA_ROOT) + "/", ""),
                         "key": "", "z": z, "z_src": int(kept[z]), "H": H, "W": W, "res_xy_nm": res[-1], "res_z_nm": res[0],
                         "classes": "|".join(classes), "quality": str(a.get("quality", "gold")), "modality": a["modality"],
                         "organism": a["organism"], **metrics(f["raw"][z], labels, masks)})
    return rows


def do_shard(args):
    p, ds = args; rows = []
    with h5py.File(p, "r") as f:
        fa = f.attrs
        for k in f.keys():
            g = f[k]; classes = json.loads(g.attrs["annotated_classes"])
            raw = g["raw"][:]; res = np.atleast_1d(g.attrs.get("resolution_nm", -1.0))
            rows.append({"uid": f"{ds}/{k}", "dataset": ds, "volume": k.split("__")[0] if "__" in k else "", "file": p.replace(str(DATA_ROOT) + "/", ""),
                         "key": k, "z": "", "z_src": "", "H": raw.shape[0], "W": raw.shape[1],
                         "res_xy_nm": "" if res[-1] < 0 else float(res[-1]), "res_z_nm": "", "classes": "|".join(classes),
                         "quality": str(fa.get("quality", "gold")), "modality": fa.get("modality", "natural photo (grayscale)"),
                         "organism": fa.get("organism", ""),
                         **metrics(raw, {c: g[f"labels/{c}"][:] for c in classes}, {c: g[f"masks/{c}"][:] for c in classes})})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=48); a = ap.parse_args()
    vols = sorted((DATA_ROOT / "em").glob("*/*.h5"))
    shards = [(p, p.split("/")[-2]) for p in sorted(glob.glob(str(DATA_ROOT / "em2d/*/shard_*.h5")))]
    shards += [(p, "coco_panoptic") for p in sorted(glob.glob(str(DATA_ROOT / "natural/coco_panoptic/*/shard_*.h5")))]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for rs in ex.map(do_volume, vols):
            rows += rs
        for rs in ex.map(do_shard, shards):
            rows += rs
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    out = DATA_ROOT / "qa/images.csv"; out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval=""); w.writeheader(); w.writerows(rows)
    print(len(rows), "images ->", out); print("exit=0")


if __name__ == "__main__":
    main()
