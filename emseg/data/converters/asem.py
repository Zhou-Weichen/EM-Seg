"""ASEM (Gallusser et al. 2023, JCB) mitochondria: public S3 bucket asem-project (zarr v2, anonymous HTTPS).
Annotated label chunks are found by listing the label array's chunk keys; for every z-chunk row (128 sections) the yx
bounding box of the annotated chunks is cut out, and every 4th section is kept (4-5 nm isotropic -> 16-20 nm between kept
sections, less redundancy). Mito masks (binary or ids) -> 2D 4-connected instances by the standard cleaning.
ER / golgi / ccp / np labels are semantic networks and are not converted.
usage: python -m emseg.data.converters.asem [--step 4]"""
from __future__ import annotations

import argparse, json, re, time, urllib.parse, urllib.request
import numpy as np
import zarr

from emseg.data.standard import write_volume, DATA_ROOT

BUCKET = "https://asem-project.s3.amazonaws.com"
DATASETS = ["2E", "3A", "46", "58", "61", "64"]


def _get(url, tries=8):
    for t in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=120).read()
        except Exception as e:  # noqa
            print("retry", url[-80:], repr(e)[:100], flush=True); time.sleep(20 * (t + 1))
    raise RuntimeError(f"failed {url}")


def list_keys(prefix):
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        x = _get(url).decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", x)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            return keys
        token = m.group(1)


def jget(url):
    return json.loads(_get(url))


def main():
    import urllib.parse  # noqa
    ap = argparse.ArgumentParser(); ap.add_argument("--step", type=int, default=4); a = ap.parse_args()
    for d in DATASETS:
        base = f"{BUCKET}/datasets/{d}/{d}.zarr/volumes"
        lab_meta, lab_attr, raw_attr = jget(f"{base}/labels/mito/.zarray"), jget(f"{base}/labels/mito/.zattrs"), jget(f"{base}/raw/.zattrs")
        res = np.array(raw_attr["resolution"], float); off = (np.array(lab_attr["offset"]) / np.array(lab_attr["resolution"])).astype(int)
        cs = lab_meta["chunks"]
        keys = [k.rsplit("/", 1)[1] for k in list_keys(f"datasets/{d}/{d}.zarr/volumes/labels/mito/")]
        idx = np.array([[int(v) for v in k.split(".")] for k in keys if re.fullmatch(r"\d+\.\d+\.\d+", k)])
        lab_arr = zarr.open_array(f"{base}/labels/mito", mode="r", zarr_format=2)
        raw_arr = zarr.open_array(f"{base}/raw", mode="r", zarr_format=2)
        print(d, "raw", raw_arr.shape, "label", lab_arr.shape, "offset_vox", off.tolist(), flush=True)
        for iz in sorted(set(idx[:, 0])):
            sel = idx[idx[:, 0] == iz]
            y0, y1 = sel[:, 1].min() * cs[1], min((sel[:, 1].max() + 1) * cs[1], lab_meta["shape"][1])
            x0, x1 = sel[:, 2].min() * cs[2], min((sel[:, 2].max() + 1) * cs[2], lab_meta["shape"][2])
            z0, z1 = iz * cs[0], min((iz + 1) * cs[0], lab_meta["shape"][0])
            rs = raw_arr.shape                                  # label box may extend beyond the raw volume -> clip
            z1 = min(z1, rs[0] - off[0]); y1 = min(y1, rs[1] - off[1]); x1 = min(x1, rs[2] - off[2])
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue
            zs = list(range(z0, z1, a.step))
            vol = f"{d}_z{z0}-{z1}_y{y0}_x{x0}"
            if (DATA_ROOT / "em/asem_mito" / f"{vol}.h5").exists():
                continue                                       # resume
            for attempt in range(6):                           # transient S3 / aiohttp resets
                try:
                    lab = np.stack([lab_arr[z, y0:y1, x0:x1] for z in zs]).astype(np.int64)
                    raw = np.stack([raw_arr[z + off[0], y0 + off[1]:y1 + off[1], x0 + off[2]:x1 + off[2]] for z in zs]) if (lab > 0).mean() >= 0.002 else None
                    break
                except Exception as e:  # noqa
                    print("retry", vol, repr(e)[:120], flush=True); time.sleep(30 * (attempt + 1))
            else:
                print("FAILED", vol, flush=True); continue
            if raw is None:
                continue
            r = write_volume("asem_mito", vol, raw, lab, label_scope="organelle:mito", resolution_nm=list(res * [a.step, 1, 1]),
                             modality="FIB-SEM", organism="cultured cells (SUM159 / iPSC / beta cells)", tissue=f"ASEM cell {d}",
                             source_url=f"{BUCKET}/datasets/{d}/{d}.zarr", license="CC-BY 4.0 (ASEM project)",
                             z_index=[z + off[0] for z in zs], extra_attrs={"note": "every 4th section of a 4-5 nm isotropic FIB-SEM volume; raw = 'raw' (not CLAHE)"})
            r.pop("slices", None); print(json.dumps({k: v for k, v in r.items() if k in ("volume", "n_kept", "shape_src", "dropped")}), flush=True)
    print("exit=0")


if __name__ == "__main__":
    main()
