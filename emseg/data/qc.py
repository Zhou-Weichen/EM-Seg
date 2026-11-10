"""QC montages: for each built volume, 3 slices (raw | label overlay | mask) -> $EMDATA_ROOT/_qc/<dataset>__<volume>.png
usage: python -m emseg.data.qc [glob relative to DATA_ROOT, default em/*/*.h5]"""
import sys
import h5py, numpy as np
from PIL import Image
from emseg.data.standard import DATA_ROOT


def colorize(lab):
    rng = np.random.default_rng(1); lut = rng.integers(40, 255, (int(lab.max()) + 1, 3), dtype=np.uint8); lut[0] = 0
    return lut[lab]


def tile(raw, lab, mask, size=384):
    H, W = raw.shape; s = size / max(H, W); h, w = max(1, int(H * s)), max(1, int(W * s))
    r = np.stack([raw] * 3, -1); ov = (0.55 * r + 0.45 * colorize(lab)).astype(np.uint8); ov[lab == 0] = r[lab == 0]
    m = np.stack([mask * 255] * 3, -1).astype(np.uint8)
    ims = [Image.fromarray(x).resize((w, h), Image.NEAREST) for x in (r, ov, m)]
    out = Image.new("RGB", (3 * w + 8, h), (255, 255, 255))
    for i, im in enumerate(ims):
        out.paste(im, (i * (w + 4), 0))
    return out


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "em/*/*.h5"
    out = DATA_ROOT / "_qc"; out.mkdir(exist_ok=True)
    for p in sorted(DATA_ROOT.glob(pat)):
        with h5py.File(p, "r") as f:
            Z = f["raw"].shape[0]; zs = sorted(set([0, Z // 2, Z - 1]))
            rows = [tile(f["raw"][z], f["label"][z], f["mask"][z]) for z in zs]
        W = max(r.width for r in rows); im = Image.new("RGB", (W, sum(r.height + 4 for r in rows)), (255, 255, 255)); y = 0
        for r in rows:
            im.paste(r, (0, y)); y += r.height + 4
        im.save(out / f"{p.parent.name}__{p.stem}.png"); print("qc", p.parent.name, p.stem, flush=True)


if __name__ == "__main__":
    main()
