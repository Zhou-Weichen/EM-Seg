"""Unified storage standard and cleaning rules (EMDATA-v2). Every converter goes through `write_volume_mc` /
`write_images_mc` (the single-class `write_volume` / `write_images` are thin wrappers).

One HDF5 file per source volume: $EMDATA_ROOT/em/<dataset>/<volume>.h5
  raw              (Z,Y,X) uint8   per-volume percentile normalisation (P0.5 -> 0, P99.5 -> 255 over valid pixels),
                                   membranes dark (polarity checked visually, attr `inverted`)
  labels/<class>   (Z,Y,X) uint32  2D instances of that class: per-slice 4-connected components of the source ids,
                                   consecutive 1..K per slice (NOT consistent across z); 0 = no instance of this class
  masks/<class>    (Z,Y,X) uint8   1 = this class is annotated at this pixel (0: outside annotated region, padding,
                                   crowd/void, removed tiny CCs, or the class is unusable on this slice)
  label_src/<class> (Z,Y,X) uint32/uint64 original source ids (3D-consistent when the source is), uncleaned
attrs: dataset, volume, annotated_classes (json list), source_url, license, modality, organism, tissue, label_scope
       (compat), resolution_nm [z,y,x], norm_p_lo/p_hi, inverted, kept_z, dropped (reason -> z list), standard, built
Unified class vocabulary: CLASSES below. Classes not listed in annotated_classes are UNKNOWN everywhere in that image
(partial-label convention): a mito-only image says nothing about cells.

label_scope:
  dense_cells       every cell / neurite profile in the annotated region is an instance (background = extracellular /
                    membrane / unannotated)  -> usable for full instance segmentation
  organelle:<name>  only that organelle class is annotated (mito, nucleus, ...); background contains un-annotated cells
  multi:<a,b,..>    several classes annotated as instances
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

STANDARD = "EMDATA-v2"
CLASSES = {"cell": 1, "mitochondrion": 2, "nucleus": 3, "fusiform_vesicle": 4, "endolysosome": 5, "cilium": 6,
           "nanoparticle": 7, "myelin": 8, "microvillus": 9, "axon": 10, "natural_segment": 100}
SCOPE2CLASS = {"dense_cells": "cell", "organelle:mito": "mitochondrion", "organelle:nucleus": "nucleus",
               "organelle:fusiform_vesicle": "fusiform_vesicle", "organelle:endolysosome": "endolysosome",
               "organelle:cilium": "cilium", "organelle:nanoparticle": "nanoparticle", "organelle:myelin": "myelin",
               "organelle:microvillus": "microvillus", "organelle:axon": "axon",
               "dense_panoptic": "natural_segment"}
DENSE_CLASSES = {"cell", "natural_segment"}   # exhaustive-partition classes: density rules apply
DATA_ROOT = Path(os.environ.get("EMDATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
MIN_CC = 10                 # 2D components smaller than this (px) -> label 0, mask 0
BLANK_STD = 2.0             # raw slice std (uint8 after normalisation) below this -> blank, dropped
MIN_VALID_FRAC = 0.25       # slices with less valid (non-padding) area are dropped
MIN_LABELED_FRAC = {"dense_cells": 0.30, "cell": 0.30}   # dense classes: below this labelled area the class is unusable on the slice
MIN_INST_DENSE = 3          # dense datasets: slices with fewer instances are dropped
PAD_RUN = 64                # constant-value runs touching the border longer than this are treated as padding


def percentiles(raw: np.ndarray, valid: np.ndarray | None = None, lo=0.5, hi=99.5, n=4_000_000, seed=0):
    rng = np.random.default_rng(seed)
    flat = raw.reshape(-1); vm = None if valid is None else valid.reshape(-1)
    idx = rng.integers(0, flat.size, size=min(n, flat.size))
    v = flat[idx] if vm is None else flat[idx][vm[idx] > 0]
    if v.size == 0:
        v = flat[idx]
    return np.percentile(v.astype(np.float64), [lo, hi])


def to_uint8(raw: np.ndarray, p_lo: float, p_hi: float, invert: bool = False) -> np.ndarray:
    x = (raw.astype(np.float32) - p_lo) / max(p_hi - p_lo, 1e-6)
    x = np.clip(x, 0, 1)
    if invert:
        x = 1 - x
    return (x * 255 + 0.5).astype(np.uint8)


def padding_mask(sl: np.ndarray) -> np.ndarray:
    """1 = real image. Padding = connected constant-intensity regions touching the border (e.g. CREMI padded, rotated
    stitching black corners)."""
    valid = np.ones(sl.shape, np.uint8)
    for v in np.unique(np.concatenate([sl[0], sl[-1], sl[:, 0], sl[:, -1]])):
        eq = sl == v
        if eq.sum() < PAD_RUN * 4:
            continue
        lab, n = ndi.label(eq)
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        for b in border[border > 0]:
            comp = lab == b
            if comp.sum() >= PAD_RUN * PAD_RUN // 4:
                valid[comp] = 0
    return valid


def relabel_slice(src: np.ndarray, min_cc: int = MIN_CC):
    """src ids (Y,X) -> (label uint32 consecutive 4-connected components of equal id, removed-tiny mask)."""
    from skimage.measure import label as cc_label
    cc = cc_label(src.astype(np.int64, copy=False), background=0, connectivity=1)
    if cc.max() == 0:
        return cc.astype(np.uint32), np.zeros(src.shape, bool)
    sizes = np.bincount(cc.ravel()); small = sizes < min_cc; small[0] = False
    tiny = small[cc]
    keep = ~small; keep[0] = False
    lut = np.zeros(sizes.size, np.uint32); lut[keep] = np.arange(1, keep.sum() + 1, dtype=np.uint32)
    return lut[cc], tiny


def _clean_slice(raw8_z, valid_z, srcs_z: dict, anns_z: dict):
    """-> (reason or None, {cls: (lab, mask, n_inst, lab_frac, status)})"""
    if valid_z.mean() < MIN_VALID_FRAC:
        return "padding", {}
    if raw8_z[valid_z > 0].std() < BLANK_STD:
        return "blank", {}
    out = {}
    for c, src in srcs_z.items():
        v = valid_z > 0
        if anns_z.get(c) is not None:
            v = v & (anns_z[c] > 0)
        lab, tiny = relabel_slice(np.where(v, src, 0))
        lab_frac = float((lab > 0)[v].mean()) if v.any() else 0.0
        status = "ok"
        if not v.any():
            status = "not_annotated"
        elif c in DENSE_CLASSES and lab_frac < MIN_LABELED_FRAC["cell"]:
            status = "sparse_label"
        elif c in DENSE_CLASSES and lab.max() < MIN_INST_DENSE:
            status = "degenerate_label"      # e.g. EMNeuron AxonEM[M] vol0: one id over the whole volume
        m = (v & ~tiny) if status == "ok" else np.zeros_like(v)
        out[c] = (lab if status == "ok" else np.zeros_like(lab), m.astype(np.uint8), int(lab.max()) if status == "ok" else 0, lab_frac, status)
    if not any(o[4] == "ok" and o[2] > 0 for o in out.values()):
        st = sorted({o[4] for o in out.values()})
        return ("no_label" if st == ["ok"] else "+".join(st)), out
    return None, out


def write_volume_mc(dataset: str, volume: str, raw: np.ndarray, labels: dict, *, resolution_nm, modality: str,
                    organism: str, tissue: str, source_url: str, license: str, ann_masks: dict | None = None,
                    ann_mask: np.ndarray | None = None, z_index=None, invert: bool | None = None, keep_src: bool = True,
                    extra_attrs: dict | None = None, out_root: Path | None = None) -> dict:
    """Clean + normalise + write one multi-class volume. labels: {class: (Z,Y,X) source ids}; ann_masks: optional
    {class: (Z,Y,X) annotated region}; ann_mask: optional region valid for all classes."""
    for c, l in labels.items():
        assert c in CLASSES, c
        assert l.shape == raw.shape, (c, l.shape, raw.shape)
    ann_masks = ann_masks or {}
    Z = raw.shape[0]; z_index = list(range(Z)) if z_index is None else list(z_index)
    valid = np.stack([padding_mask(raw[z]) for z in range(Z)])
    if ann_mask is not None:
        valid = valid & (ann_mask > 0)
    p_lo, p_hi = percentiles(raw, valid)
    r8 = to_uint8(raw, p_lo, p_hi)
    pol_flag = polarity_inverted(r8, labels["cell"]) if "cell" in labels else False   # recorded only, never applied
    invert = bool(invert)
    if invert:
        r8 = 255 - r8
    keep, dropped, rows, per = [], {}, [], {c: ([], []) for c in labels}
    class_status = {c: {} for c in labels}
    for z in range(Z):
        reason, res = _clean_slice(r8[z], valid[z], {c: l[z] for c, l in labels.items()},
                                   {c: (m[z] if m is not None else None) for c, m in ann_masks.items()})
        for c, o in res.items():
            if o[4] != "ok":
                class_status[c].setdefault(o[4], []).append(int(z_index[z]))
        if reason:
            dropped.setdefault(reason, []).append(int(z_index[z])); continue
        keep.append(z)
        for c, o in res.items():
            per[c][0].append(o[0]); per[c][1].append(o[1])
        rows.append({"z": len(keep) - 1, "z_src": int(z_index[z]), **{f"n_{c}": res[c][2] for c in res}})
    root = out_root or (DATA_ROOT / "em")
    path = root / dataset / f"{volume}.h5"; path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"dataset": dataset, "volume": volume, "file": str(path.relative_to(DATA_ROOT)), "shape_src": list(raw.shape),
               "classes": list(labels), "n_kept": len(keep), "dropped": dropped, "class_unusable": class_status, "invert": invert,
               "polarity_flag": bool(pol_flag), "p_lo": float(p_lo), "p_hi": float(p_hi), "slices": rows}
    if not keep:
        return summary
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        Y, X = raw.shape[1:]; ch = (1, min(Y, 256), min(X, 256)); kw = dict(chunks=ch, compression="gzip", compression_opts=4)
        f.create_dataset("raw", data=r8[keep], **kw)
        for c in labels:
            f.create_dataset(f"labels/{c}", data=np.stack(per[c][0]), **kw)
            f.create_dataset(f"masks/{c}", data=np.stack(per[c][1]), **kw)
            if keep_src:
                src = labels[c][keep]; src = src.astype(np.uint32 if src.max() < 2 ** 32 else np.uint64)
                f.create_dataset(f"label_src/{c}", data=src, **kw)
        import json as _json
        a = f.attrs
        a.update({"dataset": dataset, "volume": volume, "annotated_classes": _json.dumps(list(labels)),
                  "label_scope": "+".join(labels), "source_url": source_url, "license": license, "modality": modality,
                  "organism": organism, "tissue": tissue, "resolution_nm": np.asarray(resolution_nm, float),
                  "norm_p_lo": float(p_lo), "norm_p_hi": float(p_hi), "inverted": invert, "polarity_heuristic_inverted": bool(pol_flag),
                  "kept_z": np.asarray([z_index[z] for z in keep]), "dropped": repr(dropped), "class_unusable": repr(class_status),
                  "standard": STANDARD, "built": _dt.datetime.now().isoformat(timespec="seconds")})
        for k, v in (extra_attrs or {}).items():
            a[k] = v
    os.replace(tmp, path)
    return summary


def write_volume(dataset, volume, raw, label_src, *, label_scope, ann_mask=None, **kw):
    """single-class wrapper (label_scope -> unified class)"""
    kw.pop("invert", None) if kw.get("invert") is None else None
    return write_volume_mc(dataset, volume, raw, {SCOPE2CLASS[label_scope]: label_src}, ann_mask=ann_mask, **kw)


def polarity_inverted(r8: np.ndarray, lab: np.ndarray, n_slices: int = 8) -> bool:
    """True if instance boundaries are brighter than instance interiors (inverted EM contrast)."""
    zs = np.linspace(0, r8.shape[0] - 1, min(n_slices, r8.shape[0])).astype(int)
    b_vals, i_vals = [], []
    for z in zs:
        l = lab[z]
        bd = np.zeros(l.shape, bool)
        bd[:-1] |= l[:-1] != l[1:]; bd[1:] |= l[:-1] != l[1:]; bd[:, :-1] |= l[:, :-1] != l[:, 1:]; bd[:, 1:] |= l[:, :-1] != l[:, 1:]
        bd = ndi.binary_dilation(bd, iterations=1)
        inner = (l > 0) & ~ndi.binary_dilation(bd, iterations=3)
        if bd.any() and inner.any():
            b_vals.append(np.median(r8[z][bd])); i_vals.append(np.median(r8[z][inner]))
    return bool(b_vals) and float(np.median(b_vals)) > float(np.median(i_vals)) + 5


def write_images(dataset: str, items, *, label_scope: str | None = None, resolution_nm=None, modality: str, organism: str,
                 tissue: str, source_url: str, license: str, shard_size: int = 1000, extra_attrs: dict | None = None,
                 classes: list | None = None, out_sub: str = "em2d") -> dict:
    """2D image collections: items yield (key, raw2d, labels[, ann_mask2d][, attrs]) where labels is a {class: ids}
    dict or a single array (class from label_scope). Same cleaning as volumes, per-image normalisation.
    Output <out_sub>/<dataset>/shard_NNNN.h5: one group per image with raw, labels/<c>, masks/<c>; index.csv;
    build_summary.json."""
    import csv, json
    out = DATA_ROOT / out_sub / dataset; out.mkdir(parents=True, exist_ok=True)
    rows, dropped, st = [], {}, {"shard": 0, "f": None, "n": 0, "tmp": None, "path": None}
    seen_classes = set(classes or ([SCOPE2CLASS[label_scope]] if label_scope else []))

    def close():
        if st["f"] is not None:
            st["f"].close(); os.replace(st["tmp"], st["path"])
    for it in items:
        key, raw = it[0], np.asarray(it[1]); labs = it[2]; ann = it[3] if len(it) > 3 else None; at = it[4] if len(it) > 4 else {}
        if not isinstance(labs, dict):
            labs = {SCOPE2CLASS[label_scope]: labs}
        seen_classes |= set(labs)
        if raw.ndim == 3:                                   # RGB(A) -> gray
            raw = raw[..., :3].astype(np.float32).mean(-1)
        v = padding_mask(raw)
        if ann is not None:
            v = v & (np.asarray(ann) > 0)
        p_lo, p_hi = percentiles(raw, v)
        r8 = to_uint8(raw, p_lo, p_hi)
        reason, res = _clean_slice(r8, v, {c: np.asarray(l) for c, l in labs.items()}, {})
        if reason:
            dropped.setdefault(reason, []).append(str(key)); continue
        if st["f"] is None or st["n"] >= shard_size:
            close()
            st["path"] = out / f"shard_{st['shard']:04d}.h5"; st["tmp"] = st["path"].with_suffix(".h5.tmp")
            st["f"] = h5py.File(st["tmp"], "w"); st["n"] = 0; st["shard"] += 1
            st["f"].attrs.update({"dataset": dataset, "source_url": source_url, "license": license, "modality": modality, "organism": organism,
                                  "tissue": tissue, "standard": STANDARD, **(extra_attrs or {})})
        key = str(key).replace("/", "__")                 # '/' would create nested HDF5 groups: one image = one top-level group
        g = st["f"].create_group(key)
        g.create_dataset("raw", data=r8, compression="gzip", compression_opts=4)
        for c, o in res.items():
            g.create_dataset(f"labels/{c}", data=o[0], compression="gzip", compression_opts=4)
            g.create_dataset(f"masks/{c}", data=o[1], compression="gzip", compression_opts=4)
        r = at.get("resolution_nm", resolution_nm)
        g.attrs.update({"norm_p_lo": float(p_lo), "norm_p_hi": float(p_hi), "resolution_nm": -1.0 if r is None else np.asarray(r, float),
                        "annotated_classes": json.dumps(list(res)), **{k: val for k, val in at.items() if k != "resolution_nm"}})
        rows.append({"file": str(st["path"].relative_to(DATA_ROOT)), "key": str(key), "H": r8.shape[0], "W": r8.shape[1],
                     "classes": "|".join(res), **{f"n_{c}": o[2] for c, o in res.items()},
                     "res_xy_nm": "" if r is None else float(np.atleast_1d(r)[-1]), "group": at.get("group", at.get("source_subdir", ""))})
        st["n"] += 1
    close()
    if rows:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k.startswith("n_"), k))
        with open(out / "index.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summ = {"dataset": dataset, "n_items_kept": len(rows), "dropped": {k: len(x) for k, x in dropped.items()},
            "annotated_classes": sorted(seen_classes), "modality": modality, "organism": organism, "tissue": tissue,
            "source_url": source_url, "license": license, "resolution_nm": resolution_nm, "standard": STANDARD}
    json.dump({**summ, "dropped_keys": dropped}, open(out / "build_summary.json", "w"), indent=1, default=str)
    return summ
