"""Stage 5 - de-duplication and benchmark-leakage scan (reads qa/images.csv, writes dedup/*.csv, dedup/summary.json).

(a) exact duplicates      identical raw bytes (md5) in different images
(b) near duplicates       64-bit pHash Hamming distance <= HAMMING across different volumes / datasets, confirmed by
                          the correlation of 64x64 thumbnails >= NCC_THUMB (consecutive sections of one volume are
                          distinct physical sections with their own labels and are NOT treated as duplicates)
(c) region overlap        the same tissue re-released at another crop / scale: every image is resampled to 32 nm/px
                          (scales {0.5,1,2} tried when the pixel size is unknown; targets centre-cropped to <= 768 px =
                          24.6 um); textured 48x48 query patches (1.5 um) from a
                          sample of images of dataset A are template-matched (normalised cross-correlation) against a
                          sample of images of dataset B, for all ordered pairs of EM datasets; a pair is flagged when
                          >= MIN_HITS query patches from >= 2 different images match with NCC >= NCC_REGION.
                          Both sides are band-passed first (minus a Gaussian blur, sigma 3 px = 96 nm): without it,
                          smooth low-texture patches (e.g. 5 nm FIB-SEM downsampled 6x) reach NCC 0.85-0.93 against
                          unrelated data. Calibration (8 query x 12 target images): known re-releases
                          (EMNeuron FIB-25 vs FIB-25, Lucchi vs Lucchi++) top NCC 0.84-1.0; unrelated pairs <= 0.45.
                          NCC of all queries against one target is computed at once in the Fourier domain (one FFT of
                          the target, batched query FFTs, local window norms from integral images) - the first version
                          (16 nm, skimage match_template per pair) needed > 9 h.
(d) benchmark leakage     the same scan with the validation / test slices of this project's benchmarks as queries
                          (SNEMI AC4 z80-99 + AC3, CREMI A/B/C z75-124, mouse_30um mip1 z70-99) against every dataset
Decisions: (a)/(b) keep the image of the higher-priority dataset (PRIORITY order: expert dense > expert organelle >
silver), drop the other; (c) is reported per dataset pair and resolved by rule in manifest.py; (d) >= BENCH_MIN_HITS
benchmark patches matched with NCC >= NCC_BENCH (0.70, same calibration) -> the dataset is excluded from the manifests until checked (a real
leak = identical sections -> NCC near 1 on many patches; a single 0.7 hit can be chance texture similarity).
usage: python -m emseg.data.stages.dedup [--workers 48] [--per-ds 40]"""
import argparse, csv, json, random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import h5py
from pathlib import Path
import numpy as np
from skimage.feature import match_template
from skimage.transform import rescale

from emseg.data.standard import DATA_ROOT

HAMMING, NCC_THUMB = 6, 0.95
TARGET_NM, PATCH, NCC_REGION, MIN_HITS = 32.0, 48, 0.70, 3
MAX_TARGET = 768                       # target versions are centre-cropped to this size (px at TARGET_NM)
NCC_BENCH, BENCH_MIN_HITS = 0.70, 2
HP_SIGMA = 3.0                         # band-pass: image - Gaussian(3 px = 96 nm) before matching (see docstring)
MIN_WIN_STD = 1.0                      # band-passed target windows flatter than this (grey levels) are not matched
UNKNOWN_SCALES = (0.5, 1.0, 2.0)
BENCH_ROOT = DATA_ROOT / "raw_downloads/em"


def load_raw(r):
    with h5py.File(DATA_ROOT / r["file"], "r") as f:
        return f["raw"][int(r["z"])] if r["z"] != "" else f[r["key"]]["raw"][:]


def thumb(img, n=64):
    t = rescale(img.astype(np.float32), (n / img.shape[0], n / img.shape[1]), anti_aliasing=True, preserve_range=True)[:n, :n]
    t = t - t.mean(); return t / (np.linalg.norm(t) + 1e-6)


def _hp(v):
    from scipy.ndimage import gaussian_filter
    return v - gaussian_filter(v, HP_SIGMA)


def at16(img, res):
    """list of band-passed versions of img resampled to TARGET_NM (several if the pixel size is unknown)"""
    if res in ("", None) or float(res) <= 0:
        return [_hp(rescale(img.astype(np.float32), s, anti_aliasing=s < 1, preserve_range=True)) for s in UNKNOWN_SCALES]
    s = float(res) / TARGET_NM
    return [_hp(rescale(img.astype(np.float32), s, anti_aliasing=s < 1, preserve_range=True) if abs(s - 1) > 1e-3 else img.astype(np.float32))]


def query_patches(img, k=4, rng=None):
    rng = rng or random.Random(0)
    H, W = img.shape
    if H < PATCH + 8 or W < PATCH + 8:
        return []
    cands = []
    for _ in range(40):
        y, x = rng.randrange(0, H - PATCH), rng.randrange(0, W - PATCH)
        p = img[y:y + PATCH, x:x + PATCH]; cands.append((float(p.std()), p))
    cands.sort(key=lambda t: -t[0])
    return [p for s, p in cands[:k] if s > 2]           # band-passed grey levels


# ---------------- (a)+(b) ----------------
def near_dups(rows):
    by_md5 = defaultdict(list)
    for i, r in enumerate(rows):
        by_md5[r["md5"]].append(i)
    exact = [(g[0], j) for g in by_md5.values() if len(g) > 1 for j in g[1:]]
    # multi-index hashing: Hamming <= 7 => at least one of 8 bytes identical
    hs = [int(r["phash"], 16) for r in rows]; buckets = defaultdict(list)
    for i, h in enumerate(hs):
        for b in range(8):
            buckets[(b, (h >> (8 * b)) & 0xFF)].append(i)
    cand = set()
    for lst in buckets.values():
        if len(lst) > 2000:          # degenerate bucket (flat images); covered by other bytes
            continue
        for a_ in range(len(lst)):
            for b_ in range(a_ + 1, len(lst)):
                i, j = lst[a_], lst[b_]
                if bin(hs[i] ^ hs[j]).count("1") <= HAMMING:
                    ri, rj = rows[i], rows[j]
                    if (ri["dataset"], ri["volume"]) != (rj["dataset"], rj["volume"]) or ri["z"] == "":
                        cand.add((min(i, j), max(i, j)))
    return exact, sorted(cand)


def confirm(args):
    i, j, ri, rj = args
    try:
        return i, j, float((thumb(load_raw(ri)) * thumb(load_raw(rj))).sum())
    except Exception:
        return i, j, 0.0


# ---------------- (c)+(d) ----------------
def sample_rows(rows, n, seed):
    rng = random.Random(seed); rows = list(rows)
    return rows if len(rows) <= n else rng.sample(rows, n)


def build_queries(args):
    name, rs = args
    rng = random.Random(1); qs = []
    for r in rs:
        img = r["_img"] if "_img" in r else load_raw(r)
        for v in at16(img, r["res_xy_nm"])[:1] if r["res_xy_nm"] not in ("", None) and float(r["res_xy_nm"]) > 0 else at16(img, 16.0):
            for p in query_patches(v, 2, rng):
                qs.append((r["uid"], p))
    return name, qs


def target_job(args):
    """one job per target dataset: resample its sample once, match every other dataset's queries against it"""
    tname, trows, Qall = args
    targets = [(r["uid"], at16(load_raw(r), r["res_xy_nm"])) for r in trows]
    return [(q, tname, match_one(qs, targets, NCC_BENCH if q == "BENCH" else NCC_REGION)) for q, qs in Qall.items() if q != tname]


def _crop(v):
    H, W = v.shape; y, x = max(0, (H - MAX_TARGET) // 2), max(0, (W - MAX_TARGET) // 2)
    return v[y:y + MAX_TARGET, x:x + MAX_TARGET]


def ncc_max(T, P, chunk=64):
    """max normalised cross-correlation of each query patch P[k] (n,PATCH,PATCH) over all 'valid' positions of T"""
    from scipy import fft
    T = T.astype(np.float64); H, W = T.shape; p = PATCH
    if H < p or W < p:
        return np.zeros(len(P))
    n = p * p
    S1 = np.pad(T, ((1, 0), (1, 0))).cumsum(0).cumsum(1); S2 = np.pad(T * T, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    box = lambda S: S[p:, p:] - S[:-p, p:] - S[p:, :-p] + S[:-p, :-p]
    wsum, wsq = box(S1), box(S2)
    den_t = np.sqrt(np.maximum(wsq - wsum * wsum / n, 0))                 # (H-p+1, W-p+1) = window std * PATCH
    flat = den_t < MIN_WIN_STD * p                                         # padding / saturated windows: NCC undefined ->
    den_t[flat] = np.inf                                                   # score 0 (else FFT round-off / ~0 gives NCC >> 1)
    Pc = P.astype(np.float64) - P.mean(axis=(1, 2), keepdims=True)
    pn = np.sqrt((Pc ** 2).sum(axis=(1, 2))) + 1e-6
    shape = (fft.next_fast_len(H + p - 1, real=True), fft.next_fast_len(W + p - 1, real=True))
    FT = fft.rfft2(T.astype(np.float32), shape)
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        FQ = fft.rfft2(Pc[i:i + chunk, ::-1, ::-1].astype(np.float32), shape)   # float32 FFTs: 2x faster, NCC error < 1e-3                 # correlation = convolution with flipped patch
        C = fft.irfft2(FT[None] * FQ, shape)[:, p - 1:H, p - 1:W]          # 'valid' part
        out[i:i + chunk] = (C / den_t[None]).reshape(len(C), -1).max(1) / pn[i:i + chunk]
    assert out.max(initial=0) <= 1.01, out.max()                         # NCC is bounded by 1: guards the implementation
    return out


def match_one(queries, targets, thr=NCC_REGION):
    if not queries:
        return []
    uids = [q for q, _ in queries]; P = np.stack([p for _, p in queries])
    best = np.zeros(len(P)); best_t = [""] * len(P)
    for tuid, versions in targets:
        for v in versions:
            s = ncc_max(_crop(v), P)
            upd = s > best; best[upd] = s[upd]
            for k in np.nonzero(upd)[0]:
                best_t[k] = tuid
    return [(uids[k], best_t[k], round(float(best[k]), 3)) for k in range(len(P)) if best[k] >= thr]


def bench_rows():
    """benchmark val/test slices as pseudo-rows with an in-memory image"""
    out = []
    with h5py.File(BENCH_ROOT / "snemi/snemi_train.h5", "r") as f:
        for z in range(80, 100, 4):
            out.append({"uid": f"BENCH_snemi_ac4_val/{z}", "res_xy_nm": 6.0, "_img": f["volumes/raw"][z]})
    with h5py.File(BENCH_ROOT / "snemi/snemi_test.h5", "r") as f:
        for z in range(0, 100, 10):
            out.append({"uid": f"BENCH_snemi_ac3_test/{z}", "res_xy_nm": 6.0, "_img": f["volumes/raw"][z]})
    for s in "ABC":
        with h5py.File(BENCH_ROOT / f"cremi/sample_{s}_20160501.hdf", "r") as f:
            for z in range(75, 125, 10):
                out.append({"uid": f"BENCH_cremi_{s}_valtest/{z}", "res_xy_nm": 4.0, "_img": f["volumes/raw"][z]})
    try:
        from emseg.eval.domains import get_domain
        dom = get_domain("mouse30um", mip=1)
        for z in range(70, 100, 6):
            out.append({"uid": f"BENCH_mouse30um_valtest/{z}", "res_xy_nm": 16.0, "_img": dom.load_image(z)})
    except Exception as e:  # noqa
        print("mouse benchmark not loaded:", e)
    return out


BENCH_SAME_VOLUME = {"snemi_ac4": "snemi3d/ac4", "cremi_A": "cremi/A", "cremi_B": "cremi/B", "cremi_C": "cremi/C"}


def _split_neighbour(bench_uid, target_uid, max_dz=2):
    """a benchmark slice matching a training slice of the SAME benchmark volume within max_dz sections is the physical
    neighbour across the train/val boundary (SNEMI AC4 z79|z80, CREMI z74|z75; ~30-40 nm apart, NCC ~0.75-0.8), not
    re-used data; the project's own experiments use the same boundary. Identical sections would be a leak and are not
    exempted (|dz| > max_dz or another volume)."""
    b, zb = bench_uid.replace("BENCH_", "").rsplit("/", 1)
    for k, vol in BENCH_SAME_VOLUME.items():
        if b.startswith(k) and target_uid.startswith(vol):
            try:
                return abs(int(target_uid.rsplit("/", 1)[1]) - int(zb)) <= max_dz
            except ValueError:
                return False
    return False


HEAVY_OVERLAP = 0.30          # fraction of a dataset's query patches matched in another dataset -> same data re-released


def rank(ds, rows_by_ds):
    """keep-priority of a dataset: expert dense > expert organelle > silver; original release > re-release collection"""
    r = rows_by_ds[ds][0]
    tier = 1 if r["quality"] == "silver" else (3 if "cell" in r["classes"].split("|") else 2)
    return (tier, 0 if ds.startswith("emneuron") else 1, ds)


def decide(rows, Q, pairs, out):
    """automatic stage-5 decisions + manual overrides (dedup/overrides.json: {"exclude_datasets": {ds: reason},
    "keep_datasets": [ds, ...]})"""
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    excl, notes = {}, []
    for q, t, h in pairs:
        if q == "BENCH":
            h = [x for x in h if not _split_neighbour(x[0], x[1])]
            if len(h) >= BENCH_MIN_HITS:
                excl[t] = f"benchmark leakage: {len(h)} benchmark patches matched (e.g. {h[0][0]} -> {h[0][1]}, NCC {h[0][2]})"
            continue
        frac = len(h) / max(len(Q.get(q, [])), 1)
        notes.append({"query": q, "target": t, "hits": len(h), "query_patches": len(Q.get(q, [])), "frac": round(frac, 3)})
        if frac >= HEAVY_OVERLAP and len(h) >= MIN_HITS:
            loser = min((q, t), key=lambda d: rank(d, by))
            winner = t if loser == q else q
            excl.setdefault(loser, f"re-release of {winner}: {frac:.0%} of {q} query patches found in {t} (NCC >= {NCC_REGION})")
    ov = DATA_ROOT / "dedup/overrides.json"
    if ov.exists():
        o = json.load(open(ov))
        excl.update(o.get("exclude_datasets", {}))
        for d in o.get("keep_datasets", []):
            excl.pop(d, None)
    json.dump({"exclude_datasets": excl, "region_pairs": notes, "rule": f"benchmark hit -> exclude; overlap fraction >= {HEAVY_OVERLAP} -> "
               "exclude the lower-ranked dataset (rank: expert dense > expert organelle > silver; original > EMNeuron re-release)"},
              open(out / "decisions.json", "w"), indent=1)
    print("excluded:", excl)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=48); ap.add_argument("--per-ds", type=int, default=40)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(DATA_ROOT / "qa/images.csv")))
    out = DATA_ROOT / "dedup"; out.mkdir(exist_ok=True)
    # (a)+(b)
    exact, cand = near_dups(rows)
    with ProcessPoolExecutor(a.workers) as ex:
        conf = list(ex.map(confirm, [(i, j, rows[i], rows[j]) for i, j in cand], chunksize=8))
    near = [(i, j, s) for i, j, s in conf if s >= NCC_THUMB]
    with open(out / "duplicates.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["kind", "uid_a", "dataset_a", "uid_b", "dataset_b", "score"])
        for i, j in exact:
            w.writerow(["exact", rows[i]["uid"], rows[i]["dataset"], rows[j]["uid"], rows[j]["dataset"], 1.0])
        for i, j, s in near:
            w.writerow(["near", rows[i]["uid"], rows[i]["dataset"], rows[j]["uid"], rows[j]["dataset"], round(s, 3)])
    print("exact", len(exact), "phash candidates", len(cand), "confirmed near", len(near), flush=True)
    # (c)+(d): EM datasets only
    by = defaultdict(list)
    for r in rows:
        if r["dataset"] != "coco_panoptic":
            by[r["dataset"]].append(r)
    names = sorted(by)
    samp_q = {n: sample_rows(by[n], 12, 11) for n in names}
    samp_t = {n: sample_rows(by[n], a.per_ds, 23) for n in names}
    bench = bench_rows()
    with ProcessPoolExecutor(a.workers) as ex:
        Q = dict(ex.map(build_queries, [(n, samp_q[n]) for n in names] + [("BENCH", bench)]))
        res = [x for lst in ex.map(target_job, [(t, samp_t[t], Q) for t in names], chunksize=1) for x in lst]
    pairs = [(q, t, h) for q, t, h in res if h]
    with open(out / "region_overlap.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["query_dataset", "target_dataset", "n_hits", "n_query_images_hit", "flagged", "examples"])
        for q, t, h in sorted(pairs, key=lambda x: -len(x[2])):
            nimg = len({u.rsplit("/", 1)[0] if q != "BENCH" else u for u, _, _ in h})
            flagged = len(h) >= MIN_HITS and nimg >= 2 if q != "BENCH" else len(h) >= BENCH_MIN_HITS
            w.writerow([q, t, len(h), nimg, flagged, json.dumps(h[:5])])
    summ = {"n_images": len(rows), "exact_pairs": len(exact), "near_pairs": len(near), "phash_candidates": len(cand),
            "region_pairs_with_hits": len(pairs), "flagged_region_pairs": [(q, t, len(h)) for q, t, h in pairs if len(h) >= (BENCH_MIN_HITS if q == "BENCH" else MIN_HITS)],
            "params": {"HAMMING": HAMMING, "NCC_THUMB": NCC_THUMB, "TARGET_NM": TARGET_NM, "PATCH": PATCH, "NCC_REGION": NCC_REGION,
                       "MIN_HITS": MIN_HITS, "MAX_TARGET": MAX_TARGET, "NCC_BENCH": NCC_BENCH, "BENCH_MIN_HITS": BENCH_MIN_HITS, "query_images_per_ds": 12, "target_images_per_ds": a.per_ds, "bench_queries": len(bench)}}
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    decide(rows, Q, pairs, out)
    print(json.dumps({k: v for k, v in summ.items() if k != "params"})[:3000]); print("exit=0")


if __name__ == "__main__":
    main()
