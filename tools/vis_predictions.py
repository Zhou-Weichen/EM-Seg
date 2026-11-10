"""Render test predictions of a run next to the EM image and the ground truth.

For --n slices of the run's test split: <out>/<slice>.png = [EM | EM + GT boundaries (green) | GT instances | predicted instances].
GT objects get a deterministic colour (id hash); a predicted segment takes the colour of the GT object covering most of its
area when that is >= 50 % (so matching objects share a colour), otherwise a pale grey (merged / unmatched segments).
Reads <run>/config.json (domain, sample) and <run>/pred/test/pred_XXXX.tif written by emseg.finetune.

usage: python tools/vis_predictions.py --run <name> --out <dir> [--n 4 --crop 1024 --seed 0]
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emseg.eval.domains import get_domain              # noqa: E402
from emseg.eval.evaluate import load_prediction     # noqa: E402
from emseg.eval.gt import relabel_2d                # noqa: E402


def lut_color(lab: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-colour: id -> RGB from a multiplicative hash; background 0 -> black."""
    ids = lab.astype(np.uint64)
    r = (ids * np.uint64(2654435761)) % np.uint64(256); g = (ids * np.uint64(40503)) % np.uint64(256); b = (ids * np.uint64(97)) % np.uint64(256)
    rgb = np.stack([r, g, b], -1).astype(np.uint8)
    rgb = (40 + (rgb.astype(np.uint16) * 215) // 255).astype(np.uint8)
    rgb[lab == 0] = 0
    return rgb


def match_colors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Colour each predicted segment like the GT object it overlaps most (if >= 50 % of its area), else pale grey."""
    pair, cnt = np.unique(np.stack([pred.ravel(), gt.ravel()]), axis=1, return_counts=True)
    area = np.bincount(pred.ravel()); best = np.zeros(area.size, np.int64); best_n = np.zeros(area.size)
    for (pi, gi), n in zip(pair.T, cnt):
        if n > best_n[pi]:
            best_n[pi], best[pi] = n, gi
    lut = lut_color(best)
    lut[(best_n < 0.5 * np.maximum(area, 1)) | (best == 0)] = (200, 200, 200); lut[0] = 0
    return lut[pred]


def boundaries(lab: np.ndarray) -> np.ndarray:
    b = np.zeros(lab.shape, bool)
    b[:, 1:] |= lab[:, 1:] != lab[:, :-1]; b[1:, :] |= lab[1:, :] != lab[:-1, :]
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=4); p.add_argument("--crop", type=int, default=1024); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run = Path(os.environ.get("RUNS_ROOT", REPO / "runs")) / a.run
    cfg = json.load(open(run / "config.json"))
    kw = {"mip": 1} if cfg["domain"] == "mouse30um" else {"sample": cfg.get("sample", "B")} if cfg["domain"] == "cremi" else {}
    dom = get_domain(cfg["domain"], **kw)
    rng = np.random.default_rng(a.seed); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    test = cfg["splits"]["test"]
    for z in sorted(rng.choice(test, min(a.n, len(test)), replace=False).tolist()):
        img = dom.load_image(z); gt = relabel_2d(dom.load_label(z)); pred = relabel_2d(load_prediction(run / "pred/test", z))
        H, W = img.shape; c = min(a.crop, H, W); y0 = int(rng.integers(0, H - c + 1)); x0 = int(rng.integers(0, W - c + 1))
        sl = (slice(y0, y0 + c), slice(x0, x0 + c)); img, gt, pred = img[sl], gt[sl], pred[sl]
        rgb = np.stack([img] * 3, -1).astype(np.uint8); ov = rgb.copy(); ov[boundaries(gt)] = (40, 220, 90)
        panel = np.concatenate([rgb, ov, lut_color(gt), match_colors(pred, gt)], 1)
        if c > 768:
            panel = np.asarray(Image.fromarray(panel).resize((panel.shape[1] * 768 // c, 768), Image.NEAREST))
        Image.fromarray(panel).save(out / f"{cfg['domain']}_z{z:04d}.png")
    print("written", out)


if __name__ == "__main__":
    main()
