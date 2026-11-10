"""Raw-only cut-outs from public connectome volumes (cloud-volume, anonymous HTTPS) -> unlabelled pool for self-training (P15 I3).

Like emseg.data.converters.stream but without labels: S random slabs per source, `xy` x `xy` px and `nz` consecutive sections at the mip
given in SOURCES (~8 nm/px). A slab is kept if the EM is not blank (std >= 5, <= 30 % saturated pixels) and, when the
source has a segmentation, if that segmentation covers >= 50 % of the middle section (= inside tissue). Sampling region =
the volume bounds shrunk by 10 % per side (or an explicit box). Seed fixed per source, resumable (existing files skipped).
Writes <data-root>/unlab/<source>/slab_x<x0>_y<y0>_z<z0>.h5 (`raw` (nz,H,W) uint8 + attrs), appends rows to
<data-root>/manifests/unlab.csv (em_only.csv columns, classes empty, dataset unlab_<source>, split train) and logs every
download (url, mip, bbox, sha256 of raw) to <data-root>/unlab/download_log.jsonl.

usage: python -m emseg.data.unlab <source>[,<source>...] --slabs 2000 [--xy 1024 --nz 4 --threads 8]\n(BossDB raw layers have 512x512x16 chunks: use --nz 16 and z0 aligned to 16 for ~4x more sections per byte)
"""
from __future__ import annotations

import argparse, os, sys, hashlib, json, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# name: (em url, em mip, seg url or None, seg mip, modality, organism, tissue, licence, box (x0,x1,y0,y1,z0,z1 at em mip) or None)
SOURCES = {
    "hemibrain": ("gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg", 0, "gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation", 0,
                  "FIB-SEM", "drosophila", "central brain (hemibrain v1.2)", "CC-BY 4.0 (Janelia FlyEM)", None),
    "manc": ("gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg", 0, "gs://manc-seg-v1p2/manc-seg-v1.2", 0,
             "FIB-SEM", "drosophila", "male ventral nerve cord (MANC v1.2)", "CC-BY 4.0 (Janelia FlyEM)", None),
    "malecns": ("gs://flyem-male-cns/em/em-clahe-jpeg", 0, "gs://flyem-male-cns/v0.9/segmentation", 0,
                "FIB-SEM", "drosophila", "male CNS (v0.9)", "CC-BY 4.0 (Janelia FlyEM)", None),
    "h01": ("gs://h01-release/data/20210601/4nm_raw", 1, "gs://h01-release/data/20210601/c3", 0,
            "multi-beam SEM (ATUM)", "human", "temporal cortex (H01)", "CC-BY 4.0", None),
    "minnie65": ("precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em", 0,
                 "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300/", 0,
                 "ssTEM", "mouse", "visual cortex (MICrONS minnie65)", "CC-BY 4.0", None),
    "flywire": ("gs://microns-seunglab/drosophila_v0/alignment/image_rechunked", 2, "gs://flywire_v141_m783", 0,
                "ssTEM", "drosophila", "whole brain (FAFB v14)", "CC-BY 4.0 (FlyWire)", (5100, 59200, 1440, 29600, 16, 7062)),
}
COLS = ["uid", "group", "dataset", "file", "key", "z", "z_src", "H", "W", "res_xy_nm", "res_z_nm", "classes", "n_inst", "quality", "modality", "organism", "split", "weight"]
_local = threading.local()


def _cv(url, mip):
    from cloudvolume import CloudVolume
    return CloudVolume(url, mip=mip, use_https=True, fill_missing=True, progress=False, parallel=1, bounded=False)


def _vols(name):
    """(em volume, seg volume at a coarse mip for the coverage check, seg downsampling factor (x, y, z) relative to em)."""
    d = getattr(_local, "vols", None) or {}
    if name not in d:
        em_url, em_mip, seg_url, seg_mip = SOURCES[name][:4]
        em = _cv(em_url, em_mip); seg = fac = None
        if seg_url:
            s0 = _cv(seg_url, seg_mip); mips = [m for m in s0.available_mips if m >= seg_mip]
            m = max(m for m in mips if m <= seg_mip + 3)              # coverage check at <= 8x coarser resolution (cheap)
            seg = _cv(seg_url, m); fac = np.maximum(1, np.round(np.array(seg.resolution) / np.array(em.resolution))).astype(int)   # (x, y, z)
        d[name] = (em, seg, fac); _local.vols = d
    return d[name]


def _ok_em(e):
    return e.std() >= 5 and ((e <= 2) | (e >= 253)).mean() <= 0.3


def fetch(name, x0, y0, z0, xy, nz):
    em, seg, fac = _vols(name)
    if seg is not None:
        sx, sy, sz = x0 // fac[0], y0 // fac[1], (z0 + nz // 2) // fac[2]; wx, wy = max(1, xy // fac[0]), max(1, xy // fac[1])
        s_mid = np.asarray(seg[sx:sx + wx, sy:sy + wy, sz:sz + 1])[..., 0, 0]
        if (s_mid > 0).mean() < 0.5:
            return None
    e = np.asarray(em[x0:x0 + xy, y0:y0 + xy, z0:z0 + nz])[..., 0]      # one read (chunk-aligned nz, e.g. 16, costs no more than 1 section)
    raw = np.ascontiguousarray(e.transpose(2, 1, 0))                      # (z, y, x)
    if not all(_ok_em(sl) for sl in raw):
        return None
    return raw


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("sources"); p.add_argument("--slabs", type=int, default=500); p.add_argument("--xy", type=int, default=1024); p.add_argument("--nz", type=int, default=4)
    p.add_argument("--threads", type=int, default=8); p.add_argument("--seed", type=int, default=1); p.add_argument("--data-root", default=os.environ.get("EMDATA_ROOT", str(Path(__file__).resolve().parents[2] / "data")))
    p.add_argument("--max-stall", type=int, default=1500, help="seconds without any fetch completing before the source is abandoned with what is on disk")
    p.add_argument("--sources-json", default="", help="json {name: [em_url, em_mip, seg_url|null, seg_mip, modality, organism, tissue, licence, box|null]} added to SOURCES")
    a = p.parse_args(argv)
    if a.sources_json:
        SOURCES.update({k: tuple(v) for k, v in json.load(open(a.sources_json)).items()})
    root = Path(a.data_root); man = root / "manifests" / "unlab.csv"; logp = root / "unlab" / "download_log.jsonl"; logp.parent.mkdir(parents=True, exist_ok=True); man.parent.mkdir(parents=True, exist_ok=True)
    for name in a.sources.split(","):
        em_url, em_mip, seg_url, seg_mip, mod, org, tis, lic, box = SOURCES[name]
        em, _, _ = _vols(name); res = [float(r) for r in em.resolution]        # x, y, z nm at em mip
        if box is None:
            b = em.bounds; lo, hi = np.array(b.minpt), np.array(b.maxpt); m = (hi - lo) // 10; lo, hi = lo + m, hi - m
        else:
            lo, hi = np.array(box[0::2]), np.array(box[1::2])
        out = root / "unlab" / f"unlab_{name}"; out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(a.seed + abs(hash(name)) % 1000); kept = len(list(out.glob("slab_*.h5"))); tries = 0; t0 = time.time()
        print(json.dumps({"source": name, "existing": kept, "res_nm": res, "lo": lo.tolist(), "hi": hi.tolist()}), flush=True)
        lock = threading.Lock()

        def job(c):
            x0, y0, z0 = c
            f = out / f"slab_x{x0}_y{y0}_z{z0}.h5"
            if f.exists():
                return "exists"
            try:
                raw = fetch(name, x0, y0, z0, a.xy, a.nz)
            except Exception as e:                       # transient fetch error: count as a failed try
                return f"err {type(e).__name__}: {str(e)[:80]}"
            if raw is None:
                return "rejected"
            sha = hashlib.sha256(raw.tobytes()).hexdigest()
            with h5py.File(f, "w") as h:
                d = h.create_dataset("raw", data=raw, chunks=(1, min(256, a.xy), min(256, a.xy)), compression="gzip", compression_opts=1)
                h.attrs.update({"source_url": em_url, "mip": em_mip, "bbox_xyz_vox": [int(x0), int(x0 + a.xy), int(y0), int(y0 + a.xy), int(z0), int(z0 + a.nz)],
                                "resolution_xyz_nm": res, "modality": mod, "organism": org, "tissue": tis, "license": lic, "sha256_raw": sha})
            rel = str(f.relative_to(root))
            with lock:
                with open(logp, "a") as lf:
                    lf.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source": name, "url": em_url, "mip": em_mip, "bbox_xyz_vox": [int(x0), int(x0 + a.xy), int(y0), int(y0 + a.xy), int(z0), int(z0 + a.nz)],
                                         "file": rel, "sha256_raw": sha}) + "\n")
            return "kept"

        from concurrent.futures import wait, FIRST_COMPLETED
        zal = a.nz if a.nz >= 8 else 1
        cand = lambda: (int(rng.integers(lo[0], hi[0] - a.xy)), int(rng.integers(lo[1], hi[1] - a.xy)), int(rng.integers(lo[2], hi[2] - a.nz) // zal * zal))
        stats = {"rejected": 0, "errors": 0}; last = time.time(); err_ex = []
        ex = ThreadPoolExecutor(a.threads); stall_t = time.time()
        if True:
            pending = {ex.submit(job, cand()) for _ in range(a.threads)}; tries = len(pending)
            while pending:
                done, pending = wait(pending, timeout=300, return_when=FIRST_COMPLETED)
                if not done:                                   # nothing completed for 5 min: report; abandon the source after --max-stall
                    print(json.dumps({"source": name, "stalled": len(pending), "elapsed": round(time.time() - t0)}), flush=True)
                    if time.time() - stall_t > a.max_stall:
                        print(json.dumps({"source": name, "abandoned": len(pending), "kept": kept}), flush=True); ABANDONED.append(name); break
                    continue
                stall_t = time.time()
                for fu in done:
                    r = fu.result()
                    if r == "kept": kept += 1
                    elif r == "rejected": stats["rejected"] += 1
                    elif r.startswith("err"): stats["errors"] += 1; err_ex = [r]
                    if kept + len(pending) < a.slabs and tries < 8 * a.slabs:
                        pending.add(ex.submit(job, cand())); tries += 1
                if time.time() - last > 60:
                    last = time.time()
                    print(json.dumps({"source": name, "kept": kept, "tries": tries, **stats, "err_example": err_ex, "elapsed": round(time.time() - t0)}), flush=True)
        ex.shutdown(wait=False, cancel_futures=True)
        rows = write_manifest(name, root, man)
        print(json.dumps({"source": name, "done": True, "kept": kept, "tries": tries, "rows": len(rows)}), flush=True)
    print("exit=0", flush=True); sys.stdout.flush(); os._exit(0)   # hung fetch threads must not block interpreter exit


ABANDONED = []


def manifest_rows(name, root):
    """Rows for every slab on disk for a source, from the h5 attrs (no network)."""
    rows = []
    for f in sorted((root / "unlab" / f"unlab_{name}").glob("slab_*.h5")):
        try:
            with h5py.File(f, "r") as h:
                nz, H, W = h["raw"].shape; z0 = int(h.attrs["bbox_xyz_vox"][4]); res = list(h.attrs["resolution_xyz_nm"]); mod = str(h.attrs["modality"]); org = str(h.attrs["organism"])
        except OSError:                                   # truncated file from a killed writer
            f.unlink(); continue
        rel = str(f.relative_to(root))
        rows += [dict(uid=f"unlab_{name}/{f.stem}/{z}", group="em_unlab", dataset=f"unlab_{name}", file=rel, key="", z=z, z_src=z0 + z, H=H, W=W,
                      res_xy_nm=float(res[0]), res_z_nm=float(res[2]), classes="", n_inst=0, quality="none", modality=mod, organism=org, split="train", weight=1.0) for z in range(nz)]
    return rows


_MAN_LOCK = threading.Lock()


def write_manifest(name, root, man):
    rows = manifest_rows(name, root)
    with _MAN_LOCK:
        old = pd.read_csv(man, low_memory=False) if man.exists() else pd.DataFrame(columns=COLS)
        pd.concat([old[old.dataset != f"unlab_{name}"], pd.DataFrame(rows, columns=COLS)], ignore_index=True).to_csv(man, index=False)
    return rows


if __name__ == "__main__":
    main()
