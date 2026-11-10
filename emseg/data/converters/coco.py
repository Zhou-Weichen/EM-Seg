"""COCO panoptic 2017 -> natural-image part of the corpus (grayscale, dense class-agnostic segments).

Every panoptic segment (things AND stuff) is an instance, as in dense EM segmentation every pixel belongs to some
object. iscrowd segments and VOID pixels -> label 0, mask 0. Segments are split into 4-connected components (same rule
as EM: 2D CCs), components < MIN_CC px dropped (mask 0).
Quality filters (image dropped if any fails): short side >= 320 px, VOID+crowd fraction <= 0.15, >= 5 instances after
cleaning, std of the gray image >= 10.
Output: natural/coco_panoptic/<split>/shard_NNNN.h5, groups '<image_id>' with raw (H,W) uint8, label (H,W) uint32,
masks/natural_segment (H,W) uint8 (EMDATA-v2 layout: raw, labels/natural_segment); attrs on the group: file_name, H, W, n_inst, void_frac, categories (json of segment -> category).
usage: python -m emseg.data.converters.coco <split> [--workers 32] [--limit N]
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, os, zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from emseg.data.standard import DATA_ROOT, relabel_slice, STANDARD

SRC = DATA_ROOT / "raw_downloads/natural/coco"
SHARD = 1000


def rgb2id(c):
    c = c.astype(np.uint32); return c[..., 0] + 256 * c[..., 1] + 256 * 256 * c[..., 2]


def gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"))          # ITU-R 601-2 luma


_ZIP = {}


def _open_image(split, file_name):
    d = SRC / "extracted" / f"{split}2017" / file_name
    if d.exists():
        return Image.open(d)
    z = _ZIP.get(split)
    if z is None:
        z = _ZIP[split] = zipfile.ZipFile(SRC / f"{split}2017.zip")
    return Image.open(z.open(f"{split}2017/{file_name}"))


def convert_one(args):
    split, ann, cats = args
    pan = rgb2id(np.asarray(Image.open(SRC / "extracted/annotations" / f"panoptic_{split}2017" / ann["file_name"]).convert("RGB")))
    img = gray(_open_image(split, ann["file_name"].replace(".png", ".jpg")))
    H, W = img.shape
    if min(H, W) < 320 or img.std() < 10:
        return None, "small_or_flat"
    src = np.zeros((H, W), np.int64); bad = pan == 0
    seg_cat = {}
    for k, s in enumerate(ann["segments_info"], 1):
        m = pan == s["id"]
        if s.get("iscrowd", 0):
            bad |= m; continue
        src[m] = k; seg_cat[k] = cats[s["category_id"]]
    void = float(bad.mean())
    if void > 0.15:
        return None, "void"
    lab, tiny = relabel_slice(src)
    if lab.max() < 5:
        return None, "few_instances"
    mask = (~bad & ~tiny).astype(np.uint8)
    return {"image_id": ann["image_id"], "file_name": ann["file_name"].replace(".png", ".jpg"), "raw": img, "label": lab,
            "mask": mask, "void_frac": void, "n_inst": int(lab.max()),
            "categories": json.dumps(sorted(set(seg_cat.values())))}, "ok"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("split", choices=["train", "val"]); ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    js = json.load(open(SRC / "extracted/annotations" / f"panoptic_{a.split}2017.json"))
    cats = {c["id"]: c["name"] for c in js["categories"]}
    anns = sorted(js["annotations"], key=lambda x: x["image_id"])
    if a.limit:
        anns = anns[: a.limit]
    out = DATA_ROOT / "natural/coco_panoptic" / a.split; out.mkdir(parents=True, exist_ok=True)
    rows, stats = [], {}
    shard, f, n_in = 0, None, 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for rec, why in pool.imap(convert_one, [(a.split, x, cats) for x in anns], chunksize=16):
            stats[why] = stats.get(why, 0) + 1
            if rec is None:
                continue
            if f is None or n_in >= SHARD:
                if f is not None:
                    f.close(); os.replace(tmp, path); shard += 1
                path = out / f"shard_{shard:04d}.h5"; tmp = path.with_suffix(".h5.tmp"); f = h5py.File(tmp, "w"); n_in = 0
                f.attrs.update({"standard": STANDARD, "source": "COCO panoptic 2017", "split": a.split})
            g = f.create_group(str(rec["image_id"]))
            for k, dst in (("raw", "raw"), ("label", "labels/natural_segment"), ("mask", "masks/natural_segment")):
                g.create_dataset(dst, data=rec[k], compression="gzip", compression_opts=4)
            g.attrs["annotated_classes"] = '["natural_segment"]'
            g.attrs.update({"file_name": rec["file_name"], "void_frac": rec["void_frac"], "n_inst": rec["n_inst"], "categories": rec["categories"]})
            rows.append({"file": str(path.relative_to(DATA_ROOT)), "key": str(rec["image_id"]), "H": rec["raw"].shape[0], "W": rec["raw"].shape[1],
                         "classes": "natural_segment", "n_natural_segment": rec["n_inst"], "label_frac": round(float((rec["label"] > 0).mean()), 4), "void_frac": round(rec["void_frac"], 4)})
            n_in += 1
    if f is not None:
        f.close(); os.replace(tmp, path)
    import csv
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump({"split": a.split, "n_annotations": len(anns), "n_kept": len(rows), "filter_counts": stats}, open(out / "build_summary.json", "w"), indent=2)
    print(json.dumps(stats)); print("exit=0")


if __name__ == "__main__":
    main()
