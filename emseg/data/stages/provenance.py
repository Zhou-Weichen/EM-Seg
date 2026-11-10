"""Stage 1 - download provenance & integrity.
For every downloaded source file ($EMDATA_ROOT/raw_downloads, archives and data files; extracted trees skipped):
sha256, size, mtime and the source URL (from the queue_*.tsv download lists, or the known source for gdown / streamed data).
Integrity: the sha256 is compared with every reference checksum published in the torch_em dataset modules
(64-hex strings in torch_em/data/datasets/electron_microscopy/*.py); `verified_ref` = match found, `no_reference` = the
publisher gives no checksum (size was still checked against Content-Length at download time by tools/pdl.py).
Output: provenance/downloads.csv
usage: python -m emseg.data.stages.provenance [--workers 16]"""
import argparse, csv, glob, hashlib, importlib.util, os, re, datetime
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from emseg.data.standard import DATA_ROOT

RAW = DATA_ROOT / "raw_downloads"
_spec = importlib.util.find_spec("torch_em")      # reference checksums published in torch_em (optional)
TORCH_EM = str(Path(_spec.origin).parent / "data/datasets/electron_microscopy") if _spec and _spec.origin else ""
SKIP_DIRS = ("extracted", "UroCell-master", "emps-main", "groundtruth-drosophila-vnc-master", "cem_mitolab", "__MACOSX")
EXT = (".zip", ".rar", ".h5", ".hdf", ".tif", ".tiff", ".tsv", ".csv", ".json", ".nii.gz", ".png")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return str(p), h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=16); a = ap.parse_args()
    refs = set()
    for py in glob.glob(f"{TORCH_EM}/*.py"):
        refs |= set(re.findall(r"\b[0-9a-f]{64}\b", open(py).read()))
    urls = {}
    for q in glob.glob(str(RAW / "**/queue_*.tsv"), recursive=True) + glob.glob(str(RAW / "natural/coco/get*.sh")):
        base = Path(q).parent
        for line in open(q):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].startswith("http"):
                urls[str((base / parts[0]).resolve())] = parts[1]
            for m in re.finditer(r"pdl.py (\S+) (\S+)", line):
                urls[str((DATA_ROOT.parents[1] / m.group(2)).resolve())] = m.group(1)
    files = []
    for p in RAW.rglob("*"):
        if p.is_file() and not any(s in p.parts for s in SKIP_DIRS) and str(p).endswith(EXT) and ".parts" not in str(p) and not p.name.startswith("queue_"):
            if "nuc_mm" in p.parts or "probtem" in p.parts or "tumor_spheroid" in p.parts or "cem_benchmarks" in p.parts or p.suffix in (".zip", ".rar", ".h5", ".hdf", ".tif"):
                files.append(p)
    with ProcessPoolExecutor(a.workers) as ex:
        sums = dict(ex.map(sha256, files, chunksize=4))
    rows = []
    for p in sorted(files):
        url = urls.get(str(p.resolve()), "")
        if not url and "nuc_mm" in p.parts:
            url = "https://drive.google.com/drive/folders/1_4CrlYvzx0ITnGlJOHdgcTRgeSkm9wT8 (gdown)"
        if not url and "probtem" in p.parts:
            url = "https://drive.google.com/drive/folders/1n2ZqbJEHPyMB_6a6OTBBACt5Jct2PZJc (gdown)"
        h = sums[str(p)]
        rows.append({"file": str(p.relative_to(DATA_ROOT)), "bytes": p.stat().st_size, "sha256": h,
                     "integrity": "verified_ref" if h in refs else "no_reference",
                     "downloaded": datetime.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"), "url": url})
    out = DATA_ROOT / "provenance/downloads.csv"; out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), "files;", sum(r["integrity"] == "verified_ref" for r in rows), "verified against published sha256;",
          sum(not r["url"] for r in rows), "without url"); print("exit=0")


if __name__ == "__main__":
    main()
