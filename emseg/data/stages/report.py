"""Stage 7 - reporting: STATUS.md (live state of every source), datasets.csv (machine-readable), README.md (data card)
and checksums (sha256 of every output file). Everything is recomputed from pipeline_log.jsonl and the files on disk, so
the report can never disagree with the data.
usage: python -m emseg.data.stages.report [--checksums]"""
import argparse, ast, csv, datetime, hashlib, json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import h5py

from emseg.data.standard import DATA_ROOT, STANDARD, CLASSES
from emseg.data.registry import SOURCES, owner_of
from emseg.data.log import read

RAW = DATA_ROOT / "raw_downloads"


def src_counts():
    """per output dataset: images in the source part that was read, images written by the converter, drop reasons"""
    out = defaultdict(lambda: {"src": 0, "written": 0, "dropped": defaultdict(int)})
    for p in sorted((DATA_ROOT / "em").glob("*/*.h5")):
        try:
            with h5py.File(p, "r") as f:
                a = f.attrs; ds = a["dataset"]; kept = len(a["kept_z"]) if "kept_z" in a else f["raw"].shape[0]
                dr = ast.literal_eval(a["dropped"]) if "dropped" in a else {}
        except Exception:
            continue
        o = out[ds]; o["written"] += kept; o["src"] += kept + sum(len(v) for v in dr.values())
        for k, v in dr.items():
            o["dropped"][k] += len(v)
    for bs in sorted((DATA_ROOT / "em2d").glob("*/build_summary.json")) + sorted((DATA_ROOT / "natural/coco_panoptic").glob("*/build_summary.json")):
        b = json.load(open(bs)); ds = b.get("dataset", "coco_panoptic")
        o = out[ds]
        if "n_items_kept" in b:
            o["written"] += b["n_items_kept"]; o["src"] += b["n_items_kept"] + sum(b["dropped"].values())
            for k, v in b["dropped"].items():
                o["dropped"][k] += v
        else:                                      # COCO
            o["written"] += b["n_kept"]; o["src"] += b["n_annotations"]
            for k, v in b["filter_counts"].items():
                if k != "ok":
                    o["dropped"][k] += v
    return out


def download_state(events):
    from emseg.data.stages.download import expand
    done = {}
    for e in events:
        if e["stage"] == "download" and e["action"] in ("done", "fail", "checksum_mismatch", "incomplete"):
            done[e["file"]] = e
    st = {}
    for s in SOURCES:
        if s.method in ("stream", "local"):
            st[s.id] = {"state": s.method, "files": 0, "present": 0, "bytes": 0, "verified": 0}
            continue
        if s.method == "gdown":
            p = RAW / s.files[0][0]; fs = [q for q in p.rglob("*") if q.is_file()] if p.exists() else []
            st[s.id] = {"state": "partial" if any(e.get("action") == "incomplete" for e in events if e.get("source") == s.id) else ("present" if fs else "missing"),
                        "files": len(fs), "present": len(fs), "bytes": sum(q.stat().st_size for q in fs), "verified": 0}
            continue
        try:
            fl = expand(s)
        except Exception:
            fl = s.files
        pres = [f for f in fl if (RAW / f[0]).exists()]
        ver = sum(1 for f in pres if f[2] and done.get(f[0], {}).get("sha256") == f[2])
        st[s.id] = {"state": "complete" if len(pres) == len(fl) else ("partial" if pres else "missing"), "files": len(fl), "present": len(pres),
                    "bytes": sum((RAW / f[0]).stat().st_size for f in pres), "verified": ver, "with_ref": sum(1 for f in fl if f[2])}
    return st


def convert_state(events):
    st = defaultdict(lambda: {"done": set(), "fail": set(), "waiting": False, "last": ""})
    for e in events:
        if e["stage"] == "convert":
            s = st[e["source"]]
            if e["action"] == "done":
                s["done"].add(e.get("job")); s["fail"].discard(e.get("job"))
            elif e["action"] == "fail":
                s["fail"].add(e.get("job"))
            elif e["action"] == "waiting_download":
                s["waiting"] = True
            s["last"] = e["ts"]
    return st


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest(), str(p.relative_to(DATA_ROOT))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checksums", action="store_true"); a = ap.parse_args()
    ev = read(); now = datetime.datetime.now().isoformat(timespec="seconds")
    dls, cvs, cnt = download_state(ev), convert_state(ev), src_counts()
    qa_n = defaultdict(int)
    if (DATA_ROOT / "qa/images.csv").exists():
        for r in csv.DictReader(open(DATA_ROOT / "qa/images.csv")):
            qa_n[r["dataset"]] += 1
    summ = json.load(open(DATA_ROOT / "manifests/summary.json")) if (DATA_ROOT / "manifests/summary.json").exists() else {}
    per = {d: p for d, p in summ.get("per_dataset", {}).items() if "n_images" in p}    # ignore summaries of the old (v1) format
    filt = defaultdict(lambda: defaultdict(int))
    if (DATA_ROOT / "qa/filter_log.csv").exists():
        for r in csv.DictReader(open(DATA_ROOT / "qa/filter_log.csv")):
            filt[r["dataset"]][r["reason"].split(" ")[0]] += 1
    excluded = summ.get("excluded_datasets", {})
    tot_em = sum(p["n_images"] for d, p in per.items() if p["group"] == "em"); tot_all = sum(p["n_images"] for p in per.values())
    datasets = sorted(set(cnt) | set(per) | set(qa_n))
    rows = []
    for ds in datasets:
        s = owner_of(ds); c = cnt.get(ds, {"src": 0, "written": 0, "dropped": {}}); p = per.get(ds, {})
        rows.append({"dataset": ds, "source_id": s.id if s else "", "name": s.name if s else "", "content": s.content if s else "",
                     "classes": "|".join(p.get("classes", s.classes if s else [])), "quality": s.quality if s else "",
                     "modality": s.modality if s else "", "organism": s.organism if s else "", "res_nm": s.resolution_nm if s else "",
                     "src_images": c["src"], "converted": c["written"], "convert_dropped": json.dumps(dict(c["dropped"])),
                     "qa_images": qa_n.get(ds, 0), "filtered": json.dumps(dict(filt.get(ds, {}))), "excluded": excluded.get(ds, ""),
                     "final_images": p.get("n_images", 0), "final_train": p.get("n_train", 0), "final_val": p.get("n_val", 0),
                     "pct_of_em_images": round(100 * p.get("n_images", 0) / tot_em, 2) if tot_em and p.get("group") == "em" else "",
                     "pct_of_all_images": round(100 * p.get("n_images", 0) / tot_all, 2) if tot_all else "",
                     "mass_em_only": p.get("mass_em_only", ""), "mass_em_plus_natural": p.get("mass_em_plus_natural", ""),
                     "n_instances": json.dumps(p.get("n_inst", {})), "Mpx": p.get("Mpx", ""), "license": s.license if s else ""})
    with open(DATA_ROOT / "datasets.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # ---------------- STATUS.md ----------------
    L = [f"# Pipeline status\n\nGenerated {now} by `python -m emseg.data.stages.report` from `pipeline_log.jsonl` and the files on disk; refreshed after every stage.\n",
         "| source | method | download (files present/total, size, sha256 verified) | conversion | output datasets |", "|---|---|---|---|---|"]
    for s in SOURCES:
        d = dls.get(s.id, {}); c = cvs.get(s.id)
        conv = ("waiting for download" if c and c["waiting"] and not c["done"] else "") if c else "not started"
        if c and c["done"]:
            conv = f"done {len(c['done'])}/{len(s.converters)}"
        if c and c["fail"]:
            conv += f", failed {len(c['fail'])}"
        dl = d.get("state", "")
        if s.method in ("http", "listing"):
            dl = f"{d['state']} {d['present']}/{d['files']}, {d['bytes'] / 1e9:.2f} GB, verified {d['verified']}/{d.get('with_ref', 0)}"
        elif s.method == "gdown":
            dl = f"{d['state']} {d['files']} files, {d['bytes'] / 1e9:.2f} GB"
        outs = sorted({r["dataset"] for r in rows if r["source_id"] == s.id})
        L.append(f"| `{s.id}`{'' if s.default else ' (not public, off by default)'} | {s.method} | {dl} | {conv or '—'} | {', '.join(outs) or '—'} |")
    recent = [e for e in ev if e["action"] in ("fail", "checksum_mismatch", "watchdog_restart", "incomplete")][-15:]
    if recent:
        L += ["\n## Recent failures / retries\n"] + [f"- {e['ts']} {e['stage']} {e['action']} `{e.get('source', '')}` {e.get('file', e.get('job', ''))} {str(e.get('error', ''))[:120]}" for e in recent]
    (DATA_ROOT / "STATUS.md").write_text("\n".join(L) + "\n")
    # ---------------- checksums ----------------
    if a.checksums:
        files = [p for root in ("em", "em2d", "natural", "manifests") for p in (DATA_ROOT / root).rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
        with ProcessPoolExecutor(32) as ex:
            res = sorted(ex.map(sha_file, files, chunksize=4), key=lambda t: t[1])
        (DATA_ROOT / "checksums.sha256").write_text("".join(f"{h}  {p}\n" for h, p in res))
    # ---------------- README.md (data card) ----------------
    from emseg.data.stages.datacard import write_readme
    write_readme(rows, summ, dls, now)
    print("STATUS.md, datasets.csv, README.md written"); print("exit=0")


if __name__ == "__main__":
    main()
