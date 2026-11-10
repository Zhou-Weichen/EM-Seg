"""Stream cut-outs from public connectome volumes (cloud-volume, anonymous HTTPS) -> EMDATA-v1 files.

Sampling: S random slabs per source, each `xy` x `xy` px and `nz` consecutive sections at the native EM/seg resolution
(many thin slabs = more appearance diversity per GB than a few big cubes, which is what 2D training needs). A slab is
kept only if the segmentation covers >= 50 % of it and the EM is not blank; otherwise it is resampled (max 6 x S tries).
Sampling region = the segmentation bounds shrunk by 10 % per side (or an explicit box). Seed fixed per source.
Labels of these sources are automatic (agglomeration) segmentations with partial / targeted proofreading
-> quality 'silver', except kharris15 (expert dense annotation, 'gold').
usage: python -m emseg.data.converters.stream <source> [--slabs 48] [--xy 1024] [--nz 8] [--threads 16]
"""
from __future__ import annotations

import argparse, json, time
import numpy as np

from emseg.data.standard import write_volume, DATA_ROOT

SOURCES = {
    # name: em url, em mip, seg url, seg mip, resolution [z,y,x], modality, organism, tissue, licence, quality, box (xyz voxels at seg mip or None)
    "hemibrain": ("gs://neuroglancer-janelia-flyem-hemibrain/emdata/clahe_yz/jpeg", 0, "gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation", 0,
                  [8, 8, 8], "FIB-SEM", "drosophila", "central brain (hemibrain v1.2)", "CC-BY 4.0 (Janelia FlyEM)", "silver", None),
    "manc": ("gs://flyem-vnc-2-26-213dba213ef26e094c16c860ae7f4be0/v3_emdata_clahe_xy/jpeg", 0, "gs://manc-seg-v1p2/manc-seg-v1.2", 0,
             [8, 8, 8], "FIB-SEM", "drosophila", "male ventral nerve cord (MANC v1.2)", "CC-BY 4.0 (Janelia FlyEM)", "silver", None),
    "malecns": ("gs://flyem-male-cns/em/em-clahe-jpeg", 0, "gs://flyem-male-cns/v0.9/segmentation", 0,
                [8, 8, 8], "FIB-SEM", "drosophila", "male CNS (v0.9)", "CC-BY 4.0 (Janelia FlyEM)", "silver", None),
    "h01": ("gs://h01-release/data/20210601/4nm_raw", 1, "gs://h01-release/data/20210601/c3", 0,
            [33, 8, 8], "multi-beam SEM (ATUM)", "human", "temporal cortex (H01, c3 segmentation)", "CC-BY 4.0", "silver", None),
    "minnie65": ("precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em", 0,
                 "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m1300/", 0,
                 [40, 8, 8], "ssTEM", "mouse", "visual cortex (MICrONS minnie65, seg m1300)", "CC-BY 4.0", "silver", None),
    "flywire": ("gs://microns-seunglab/drosophila_v0/alignment/image_rechunked", 2, "gs://flywire_v141_m783", 0,
                [40, 16, 16], "ssTEM", "drosophila", "whole brain (FAFB v14 + FlyWire v783)", "CC-BY 4.0 (FlyWire)", "silver",
                (5100, 59200, 1440, 29600, 16, 7062)),
}


def _cv(url, mip):
    from cloudvolume import CloudVolume
    return CloudVolume(url, mip=mip, use_https=True, fill_missing=True, progress=False, parallel=1, bounded=False)


def stream(name, slabs, xy, nz, threads, seed=0):
    em_url, em_mip, seg_url, seg_mip, res, mod, org, tis, lic, qual, box = SOURCES[name]
    em, seg = _cv(em_url, em_mip), _cv(seg_url, seg_mip)
    r_em, r_seg = np.array(em.resolution), np.array(seg.resolution)
    assert np.allclose(r_em, r_seg), (name, r_em, r_seg)
    if box is None:
        b = seg.bounds; lo, hi = np.array(b.minpt), np.array(b.maxpt); m = (hi - lo) // 10; lo, hi = lo + m, hi - m
    else:
        lo, hi = np.array(box[0::2]), np.array(box[1::2])
    rng = np.random.default_rng(seed); kept, tries, out = 0, 0, []
    while kept < slabs and tries < 6 * slabs:
        tries += 1
        x0 = rng.integers(lo[0], hi[0] - xy); y0 = rng.integers(lo[1], hi[1] - xy); z0 = rng.integers(lo[2], hi[2] - nz)
        done_f = DATA_ROOT / "em" / f"stream_{name}" / f"slab{kept:03d}_x{x0}_y{y0}_z{z0}.h5"
        if done_f.exists():          # resume: same seed -> same candidate sequence; finished slabs are not re-fetched
            kept += 1; continue
        if any((DATA_ROOT / "em" / f"stream_{name}").glob(f"slab*_x{x0}_y{y0}_z{z0}.h5")):
            continue                 # candidate already accepted under another index (should not happen)
        # cheap pre-check on the middle section of the segmentation
        s_mid = np.asarray(seg[x0:x0 + xy, y0:y0 + xy, z0 + nz // 2:z0 + nz // 2 + 1])[..., 0, 0]
        if (s_mid > 0).mean() < 0.5:
            continue
        s = np.asarray(seg[x0:x0 + xy, y0:y0 + xy, z0:z0 + nz])[..., 0]; e = np.asarray(em[x0:x0 + xy, y0:y0 + xy, z0:z0 + nz])[..., 0]
        if (s > 0).mean() < 0.5 or e.std() < 5:
            continue
        raw = np.ascontiguousarray(e.transpose(2, 1, 0)); lab = np.ascontiguousarray(s.transpose(2, 1, 0)).astype(np.int64)
        r = write_volume(f"stream_{name}", f"slab{kept:03d}_x{x0}_y{y0}_z{z0}", raw, lab, label_scope="dense_cells",
                         resolution_nm=[res[0], res[1], res[2]], modality=mod, organism=org, tissue=tis, source_url=f"{em_url} | {seg_url}",
                         license=lic, z_index=range(z0, z0 + nz),
                         extra_attrs={"quality": qual, "bbox_xyz_vox": [int(x0), int(x0 + xy), int(y0), int(y0 + xy), int(z0), int(z0 + nz)],
                                      "em_mip": em_mip, "seg_mip": seg_mip})
        r.pop("slices", None); out.append(r); kept += 1
        print(json.dumps({"source": name, "kept": kept, "tries": tries, "n_kept": r["n_kept"], "dropped": {k: len(v) for k, v in r["dropped"].items()}}), flush=True)
    return {"source": name, "kept": kept, "tries": tries}


KHARRIS = {  # torch_em synapseweb_hippocampus: annotated bboxes (x0,x1,y0,y1,z0,z1) and dense ROIs inside them (z,y,x)
    "spine": ((3072, 6144, 1536, 3840, 30, 80), np.s_[0:42, 784:1665, 1007:1944]),
    "oblique": ((512, 4608, 768, 4608, 6, 91), np.s_[5:75, 1243:3505, 1385:3215]),
    "apical": ((2048, 6144, 2048, 6400, 55, 167), np.s_[5:106, 217:3681, 477:3936]),
}


def kharris15():
    out = []
    for region, (bb, roi) in KHARRIS.items():
        em, an = _cv(f"s3://open-neurodata/kharris15/{region}/em", 0), _cv(f"s3://open-neurodata/kharris15/{region}/anno", 0)
        x0, x1, y0, y1, z0, z1 = bb; res = list(np.array(em.resolution)[::-1])
        e = np.asarray(em[x0:x1, y0:y1, z0:z1])[..., 0].transpose(2, 1, 0)[roi]
        s = np.asarray(an[x0:x1, y0:y1, z0:z1])[..., 0].transpose(2, 1, 0)[roi].astype(np.int64)
        r = write_volume("kharris15", region, np.ascontiguousarray(e), np.ascontiguousarray(s), label_scope="dense_cells", resolution_nm=res,
                         modality="ssTEM", organism="rat", tissue=f"hippocampus CA1 neuropil ({region} region, Harris et al. 2015)",
                         source_url=f"s3://open-neurodata/kharris15/{region}", license="CC-BY 4.0 (Open NeuroData)",
                         z_index=range(z0 + roi[0].start, z0 + roi[0].stop), extra_attrs={"quality": "gold", "dense_roi": str(roi)})
        r.pop("slices", None); out.append(r); print(json.dumps({k: v for k, v in r.items() if k != "dropped"}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("source", choices=list(SOURCES) + ["kharris15"]); ap.add_argument("--slabs", type=int, default=48)
    ap.add_argument("--xy", type=int, default=1024); ap.add_argument("--nz", type=int, default=8); ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(); t0 = time.time()
    if a.source == "kharris15":
        kharris15(); print("exit=0"); return
    print(json.dumps(stream(a.source, a.slabs, a.xy, a.nz, a.threads)), f"{time.time() - t0:.0f}s"); print("exit=0")


if __name__ == "__main__":
    main()
