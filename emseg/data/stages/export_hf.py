"""Stage 8 - export to the Hugging Face datasets format (Parquet shards + typed features + data card), under
$EMDATA_ROOT/hf/emdata-v2/.

Why Parquet + `datasets.Features` (and not the HDF5 volumes themselves): Parquet is the Hub-native format - no loading
script, `load_dataset("parquet", ...)` / `load_dataset("<repo>")` memory-maps or streams it, shards shuffle independently,
and the schema (incl. image / label semantics) travels inside the file as `huggingface` metadata, so the label
configuration cannot drift from the data. The HDF5 volumes stay the master copy (z-continuity, provenance attrs).

Configs (= which rows of the stage-6 manifests go in; one directory each under data/):
  em_open           EM rows whose SOURCE is redistributable (registry REDISTRIBUTION == "open": CC0 / CC-BY / MIT)
                    -> the config that may be uploaded to the Hub
  em_nc             EM rows of non-commercial sources (CC BY-NC[-SA]) -> separate upload under NC terms, if ever
  em_full           every EM row of manifests/em_only.csv (local use; contains challenge / research-use data)
  em_plus_natural   manifests/em_plus_natural.csv (local use; COCO images cannot be redistributed)
Rows of split "gap" (isolation sections before val) are not exported. Sampling weights are renormalised per config
over its train rows (sum = 1). Rows are ordered by a stable hash of the uid so that every shard mixes all datasets
(streaming readers see a shuffled stream shard by shard).

Encoding: image = PNG uint8; per class: instances = PNG uint16 (per-slice consecutive ids, max observed 4129 << 65535),
valid_mask = PNG uint8 (1 = pixel annotated for this class). Partial-label convention: classes absent from
`annotated_classes` are UNKNOWN on that image (not background). Lossless, bit-exact with the HDF5 (checked by --verify).

usage: python -m emseg.data.stages.export_hf [--configs em_open em_nc em_full em_plus_natural] [--rows-per-shard 800] [--workers 32]
       python -m emseg.data.stages.export_hf --verify [--config em_open]      # roundtrip vs HDF5 + load_dataset + throughput"""
import argparse, csv, hashlib, io, json, os, time, zlib
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image as PILImage

from emseg.data.standard import DATA_ROOT, CLASSES, STANDARD
from emseg.data.registry import SOURCES, owner_of, REDISTRIBUTION
from emseg.data.log import event

OUT = DATA_ROOT / "hf/emdata-v2"
CLASS_NAMES = list(CLASSES)                         # index in ClassLabel == position here (ids in classes.json)
TIERS = ["open", "nc", "check", "restricted"]
CONFIGS = {                                         # manifest, row filter
    "em_open": ("em_only.csv", lambda r, s: s is not None and s.redistribution == "open"),
    "em_nc": ("em_only.csv", lambda r, s: s is not None and s.redistribution == "nc"),
    "em_full": ("em_only.csv", lambda r, s: True),
    "em_plus_natural": ("em_plus_natural.csv", lambda r, s: True),
}
SPLIT_NAME = {"train": "train", "val": "validation"}


def features():
    from datasets import Features, Value, Image, ClassLabel, List
    return Features({
        "uid": Value("string"), "dataset": Value("string"), "source_id": Value("string"), "volume": Value("string"),
        "z": Value("int32"), "z_src": Value("int32"), "height": Value("int32"), "width": Value("int32"),
        "res_xy_nm": Value("float32"), "res_z_nm": Value("float32"),
        "modality": Value("string"), "organism": Value("string"),
        "quality": ClassLabel(names=["gold", "silver"]), "license": Value("string"), "redistribution": ClassLabel(names=TIERS),
        "image": Image(),
        "annotated_classes": List(ClassLabel(names=CLASS_NAMES)),
        "annotations": List({"class": ClassLabel(names=CLASS_NAMES), "instances": Image(), "valid_mask": Image(), "n_instances": Value("int32")}),
        "n_instances_total": Value("int32"),
        "weight": Value("float64"),
    })


def png(arr):
    arr = np.ascontiguousarray(arr)
    if arr.dtype == np.uint8:
        im = PILImage.fromarray(arr, mode="L")
    else:
        assert arr.max(initial=0) < 65536, arr.max()
        im = PILImage.fromarray(arr.astype(np.uint16), mode="I;16")
    b = io.BytesIO(); im.save(b, format="PNG", compress_level=6)
    return {"bytes": b.getvalue(), "path": None}


def load(r):
    """(raw, {cls: (labels, mask)}) of one manifest row from the master HDF5"""
    with h5py.File(DATA_ROOT / r["file"], "r") as f:
        g = f if r["z"] != "" else f[r["key"]]
        z = int(r["z"]) if r["z"] != "" else None
        raw = g["raw"][z] if z is not None else g["raw"][:]
        out = {}
        for c in r["classes"].split("|"):
            out[c] = (g[f"labels/{c}"][z] if z is not None else g[f"labels/{c}"][:], g[f"masks/{c}"][z] if z is not None else g[f"masks/{c}"][:])
    return raw, out


def _fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v <= 0 else v


def row_record(r, src, weight):
    raw, labs = load(r)
    n = json.loads(r["n_inst"]) if r["n_inst"] else {}
    anns = [{"class": CLASS_NAMES.index(c), "instances": png(l), "valid_mask": png(m), "n_instances": int(n.get(c, int(l.max())))}
            for c, (l, m) in labs.items()]
    return {"uid": r["uid"], "dataset": r["dataset"], "source_id": src.id if src else "", "volume": Path(r["file"]).stem if r["z"] != "" else r["key"],
            "z": int(r["z"]) if r["z"] != "" else -1, "z_src": int(float(r["z_src"])) if r["z_src"] not in ("", None) else -1,
            "height": int(r["H"]), "width": int(r["W"]), "res_xy_nm": _fnum(r["res_xy_nm"]), "res_z_nm": _fnum(r["res_z_nm"]),
            "modality": r["modality"], "organism": r["organism"], "quality": ["gold", "silver"].index(r["quality"]),
            "license": src.license if src else "", "redistribution": TIERS.index(src.redistribution if src else "restricted"),
            "image": png(raw), "annotated_classes": [CLASS_NAMES.index(c) for c in labs], "annotations": anns,
            "n_instances_total": int(sum(a["n_instances"] for a in anns)), "weight": weight}


def write_shard(args):
    cfg, split, idx, nshards, rows, weights = args
    out = OUT / "data" / cfg / f"{split}-{idx:05d}-of-{nshards:05d}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats = features(); schema = feats.arrow_schema.with_metadata({"huggingface": json.dumps({"info": {"features": feats.to_dict()}})})
    recs = [row_record(r, owner_of(r["dataset"]), w) for r, w in zip(rows, weights)]
    table = pa.Table.from_pylist(recs, schema=schema)
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=200)       # PNG bytes are already compressed: zstd is cheap
    os.replace(tmp, out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    return {"config": cfg, "split": split, "path": str(out.relative_to(OUT)), "num_examples": len(recs), "num_bytes": out.stat().st_size, "sha256": h,
            "datasets": sorted({r["dataset"] for r in rows})}


def plan(cfg, rows_per_shard):
    man, keep = CONFIGS[cfg]
    rows = [r for r in csv.DictReader(open(DATA_ROOT / "manifests" / man)) if r["split"] in SPLIT_NAME and keep(r, owner_of(r["dataset"]))]
    jobs = []
    for sp, name in SPLIT_NAME.items():
        rs = sorted((r for r in rows if r["split"] == sp), key=lambda r: zlib.crc32(r["uid"].encode()))      # stable hash order
        w = np.array([float(r["weight"] or 0) for r in rs]); w = w / w.sum() if sp == "train" and w.sum() > 0 else w * 0
        n = max(1, -(-len(rs) // rows_per_shard))
        for i in range(n):
            sl = slice(i * rows_per_shard, (i + 1) * rows_per_shard)
            jobs.append((cfg, name, i, n, rs[sl], w[sl].tolist()))
    return rows, jobs


def write_card(infos):
    """README.md (HF dataset card, YAML header + body), classes.json, shard_checksums.json, sampler.py"""
    feats = features()
    from emseg.data.standard import MIN_CC, BLANK_STD, MIN_VALID_FRAC, MIN_INST_DENSE
    classes = {"standard": STANDARD, "classes": [
        {"id": CLASSES[c], "name": c, "index": i, "exhaustive": c in ("cell", "natural_segment"),
         "description": {"cell": "dense 2D cell / neurite cross-sections (every pixel of the imaged tissue belongs to one instance)",
                         "mitochondrion": "mitochondria instances", "nucleus": "nucleus instances", "fusiform_vesicle": "fusiform vesicles (UroCell)",
                         "endolysosome": "endolysosomes (UroCell, components of a binary mask)", "cilium": "cilia (Platynereis, SpongeEM)",
                         "nanoparticle": "nanoparticles in materials-science EM (EMPS)", "myelin": "myelin sheaths (AimSeg)",
                         "microvillus": "microvilli (SpongeEM)", "axon": "myelinated axons only, not an exhaustive cell tiling (AimSeg)",
                         "natural_segment": "COCO panoptic segments (things + stuff) of natural photos"}[c]}
        for i, c in enumerate(CLASS_NAMES)],
        "partial_label_convention": "a class not listed in annotated_classes is UNKNOWN on that image; its pixels must not be used as negatives",
        "instances_encoding": "PNG uint16, 0 = background / not this class, 1..n = instance ids (consecutive per image, 4-connected 2D components)",
        "valid_mask_encoding": "PNG uint8, 1 = pixel is annotated for this class (0 = padding, removed tiny fragments, or outside the annotated box)",
        "image_encoding": "PNG uint8, per source volume linear stretch of the P0.5-P99.5 range (attrs norm_p_lo / norm_p_hi in the HDF5 master)"}
    json.dump(classes, open(OUT / "classes.json", "w"), indent=1)
    by_cfg = defaultdict(lambda: defaultdict(lambda: {"num_examples": 0, "num_bytes": 0, "shards": []}))
    for i in infos:
        d = by_cfg[i["config"]][i["split"]]; d["num_examples"] += i["num_examples"]; d["num_bytes"] += i["num_bytes"]
        d["shards"].append({k: i[k] for k in ("path", "num_examples", "num_bytes", "sha256")})
    dsinfo = {cfg: {"version": "2.0.0", "standard": STANDARD, "features": feats.to_dict(), "splits": {sp: {"num_examples": d["num_examples"], "num_bytes": d["num_bytes"]} for sp, d in sps.items()},
                    "shards": [s for d in sps.values() for s in d["shards"]], "download_size": sum(d["num_bytes"] for d in sps.values())}
              for cfg, sps in by_cfg.items()}
    json.dump(dsinfo, open(OUT / "shard_checksums.json", "w"), indent=1)
    yaml_cfg = "\n".join(f"- config_name: {cfg}\n  data_files:\n" + "".join(f"  - split: {sp}\n    path: data/{cfg}/{sp}-*\n" for sp in sps) for cfg, sps in by_cfg.items())
    yaml_info = "\n".join(f"- config_name: {cfg}\n  splits:\n" + "".join(f"  - name: {sp}\n    num_examples: {d['num_examples']}\n    num_bytes: {d['num_bytes']}\n" for sp, d in sps.items())
                          + f"  download_size: {sum(d['num_bytes'] for d in sps.values())}\n" for cfg, sps in by_cfg.items())
    src_rows = "\n".join(f"| `{s.id}` | {s.name} | {s.redistribution} | {s.license} | {s.citation} | {s.homepage.split(' ')[0]} |"
                         for s in SOURCES if s.default)
    cfg_rows = "\n".join(f"| `{cfg}` | " + " / ".join(f"{sp} {d['num_examples']:,}" for sp, d in sps.items()) + f" | {sum(d['num_bytes'] for d in sps.values()) / 1e9:.1f} GB | "
                         + {"em_open": "redistributable EM sources only (CC0 / CC-BY / MIT) - the config meant for the Hub",
                            "em_nc": "non-commercial EM sources (CC BY-NC[-SA]) - separate upload under NC terms only",
                            "em_full": "all EM rows (adds challenge / research-use-only sources) - local use",
                            "em_plus_natural": "em_full + 5 % sampling mass of COCO panoptic (grayscale) - local use (COCO images are not redistributable)"}[cfg] + " |"
                         for cfg, sps in by_cfg.items())
    card = f"""---
pretty_name: EMDATA-v2 - 2D instance segmentation corpus for electron microscopy
license: other
license_name: mixed-per-source
task_categories:
- image-segmentation
tags:
- electron-microscopy
- connectomics
- instance-segmentation
- biology
size_categories:
- 10K<n<100K
configs:
{yaml_cfg}
dataset_info:
{yaml_info}
---

# EMDATA-v2: a multi-source 2D instance-segmentation corpus for electron microscopy

Every row is one 2D EM section (or one 2D image) with **per-class instance labels** in a unified vocabulary, plus a
validity mask per class, provenance, pixel size and a sampling weight. Built from {sum(1 for s in SOURCES if s.default)} public sources,
each freshly downloaded from its public release (sha256 recorded), converted to one standard ({STANDARD}),
quality-checked, de-duplicated across sources and scanned for benchmark leakage. The full build pipeline, with the
rationale of every step, is `emseg/data/README.md` of the code repository; `pipeline_log.jsonl` records every download and
conversion event.

## Configs

| config | rows (split) | size | contents |
|---|---|---|---|
{cfg_rows}

`em_open` is the only config whose sources all permit redistribution; `em_full` / `em_plus_natural` exist for local
training and are **not** uploaded. Rows of the `gap` split (isolation sections between train and validation inside a
volume) are not exported.

## Features

```python
{feats}
```

- `image`: PNG uint8, per-volume linear stretch of the P0.5-P99.5 grey range.
- `annotations[i].instances`: PNG **uint16**; 0 = not this class, 1..n = instance ids (per-image consecutive 4-connected 2D components).
  Decode with `np.array(pil_img)` -> uint16.
- `annotations[i].valid_mask`: PNG uint8, 1 = this pixel is annotated for the class (0 = padding / removed fragments / outside the annotated box).
- `annotated_classes`: **partial-label convention** - a class that is not listed is *unknown* on this image, not background
  (e.g. the background of a mitochondria dataset contains unlabelled cells). Exhaustive classes (`cell`, `natural_segment`) tile
  the whole valid area; organelle classes are sparse.
- `weight`: sampling probability of the row inside the train split of its config (sums to 1); it is sqrt(labelled pixels) per
  dataset, x0.5 for automatic (silver) labels, spread uniformly over the dataset's images. `datasets` does not sample by weight -
  see `sampler.py` for a `WeightedRandomSampler` example. Validation rows have weight 0.
- `res_xy_nm` / `res_z_nm`: null when the release does not state the pixel size (2D collections, downsampled crops).
- `quality`: gold = expert / proofread labels; silver = automatic segmentation of a connectome (randomly cut 1024x1024x8 slabs).

Class ids / descriptions: `classes.json`. Shard list with sha256: `shard_checksums.json`.

## Sources and licences

| source | name | redistribution tier | licence | citation | homepage |
|---|---|---|---|---|---|
{src_rows}

Tiers: **open** = CC0 / CC-BY / MIT (in `em_open`); **nc** = non-commercial (in `em_nc` only); **check** = licence of the
underlying data not verified per item (collections inheriting original licences) - kept out of the uploadable configs until
verified; **restricted** = challenge / research-use terms or third-party image rights - local use only.
Please cite the original publications of every source you use.

## Cleaning and splits (summary)

Converted per source volume: percentile normalisation to uint8, per-slice 4-connected relabelling, removal of padding
(< {MIN_VALID_FRAC:.0%} valid), blank (std < {BLANK_STD}) and degenerate slices, fragments < {MIN_CC} px masked out; dense classes need
>= 30 % labelled area and >= {MIN_INST_DENSE} instances per slice. Per-image QA filters (size, saturation, blur, degenerate / misaligned
labels), exact + perceptual-hash duplicates, cross-source region overlap (band-passed NCC at 32 nm), benchmark-leakage scan.
Validation = whole volumes (>= 10 volumes per dataset) or the last 5 % of a volume behind a 2-section gap; 2D collections by
source group. Our own benchmark sections (SNEMI AC4 z80-99 + AC3, CREMI z75-124) are excluded from all configs.

"""
    (OUT / "README.md").write_text(card)
    (OUT / "sampler.py").write_text('''"""Weighted sampling + decoding helpers for EMDATA-v2 (HF datasets format).
usage: ds = load_dataset("parquet", data_files={"train": "data/em_open/train-*"})["train"]
       loader = DataLoader(ds.with_transform(decode), batch_size=8, sampler=weighted_sampler(ds), collate_fn=list)"""
import numpy as np, torch
from torch.utils.data import WeightedRandomSampler


def weighted_sampler(ds, num_samples=None):
    w = torch.tensor(ds["weight"], dtype=torch.double)          # one scan of a scalar column
    return WeightedRandomSampler(w, num_samples or len(ds), replacement=True)


def decode(batch):
    """PIL -> numpy; instances uint16, valid_mask uint8; unknown classes are NOT in the dict (partial labels)"""
    out = {"image": [np.array(im, dtype=np.uint8) for im in batch["image"]], "labels": []}
    for anns in batch["annotations"]:
        out["labels"].append({a["class"]: (np.array(a["instances"], dtype=np.uint16), np.array(a["valid_mask"], dtype=np.uint8)) for a in anns})
    return out
''')


def export(a):
    OUT.mkdir(parents=True, exist_ok=True)
    infos, t0 = [], time.time()
    for cfg in a.configs:
        rows, jobs = plan(cfg, a.rows_per_shard)
        event("export_hf", "start", cfg, n_rows=len(rows), n_shards=len(jobs))
        with ProcessPoolExecutor(a.workers) as ex:
            for info in ex.map(write_shard, jobs):
                infos.append(info); print(info["path"], info["num_examples"], f"{info['num_bytes'] / 1e6:.0f} MB", flush=True)
        n = sum(i["num_examples"] for i in infos if i["config"] == cfg); b = sum(i["num_bytes"] for i in infos if i["config"] == cfg)
        event("export_hf", "done", cfg, n_rows=n, bytes=b, seconds=round(time.time() - t0))
    prev = json.load(open(OUT / "shard_checksums.json")) if (OUT / "shard_checksums.json").exists() else {}
    for cfg, d in prev.items():                                   # keep configs exported in an earlier run
        if cfg not in a.configs:
            infos += [{"config": cfg, "split": s["path"].split("/")[-1].split("-")[0], **s, "datasets": []} for s in d["shards"]]
    write_card(infos)
    print("card ->", OUT / "README.md"); print("exit=0")


def _decode_range(args):
    from datasets import load_dataset
    cfg, lo, hi = args
    ds = load_dataset("parquet", data_files={"train": str(OUT / "data" / cfg / "train-*")}, split="train")
    n = px = 0
    for i in range(lo, min(hi, len(ds))):
        ex = ds[i]; px += np.array(ex["image"]).size; [np.array(an["instances"]) for an in ex["annotations"]]; n += 1
    return n, px


def verify(a):
    """bit-exact roundtrip of 200 random rows vs the HDF5 master (via the manifest), load_dataset, DataLoader throughput"""
    import random
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"train": str(OUT / "data" / a.config / "train-*")}, split="train")
    print(a.config, "rows", len(ds), "instances feature:", ds.features["annotations"].feature["instances"])
    man = {r["uid"]: r for r in csv.DictReader(open(DATA_ROOT / "manifests" / CONFIGS[a.config][0]))}
    rng = random.Random(0); bad = []
    for i in rng.sample(range(len(ds)), 200):
        ex = ds[i]; raw, labs = load(man[ex["uid"]])
        ok = np.array_equal(np.array(ex["image"]), raw) and {CLASS_NAMES[c] for c in ex["annotated_classes"]} == set(labs)
        for an in ex["annotations"]:
            l, m = labs[CLASS_NAMES[an["class"]]]
            li = np.array(an["instances"]); ok &= li.dtype == np.uint16 and np.array_equal(li, l) and np.array_equal(np.array(an["valid_mask"]), m)
        if not ok:
            bad.append(ex["uid"])
    print("checked 200 random rows against the HDF5 master; mismatches:", len(bad), bad[:5])
    t = time.time()
    with ProcessPoolExecutor(16) as ex:                         # 16 reader processes, like DataLoader(num_workers=16)
        res = list(ex.map(_decode_range, [(a.config, i * 200, (i + 1) * 200) for i in range(16)]))
    dt = time.time() - t; n = sum(r[0] for r in res); px = sum(r[1] for r in res)
    print(f"throughput: {n / dt:.0f} img/s, {px / dt / 1e6:.0f} Mpx/s (16 processes, load_dataset + decode image + instances, incl. start-up)")
    event("export_hf", "verify", a.config, rows=len(ds), mismatches=len(bad), img_per_s=round(n / dt), mpx_per_s=round(px / dt / 1e6))
    print("exit=0" if not bad else "exit=1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS)); ap.add_argument("--rows-per-shard", type=int, default=800)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--verify", action="store_true"); ap.add_argument("--config", default="em_open")
    a = ap.parse_args()
    verify(a) if a.verify else export(a)


if __name__ == "__main__":
    main()
