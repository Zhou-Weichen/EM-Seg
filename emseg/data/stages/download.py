"""Stage 1 - download + integrity (driven entirely by emseg.data.registry).

How: every file of every registry Source is fetched by tools/pdl.py (parallel HTTP range requests, resumable through
<file>.parts/spans.json, per-file lock, final size == Content-Length); 'listing' sources first read their file list
from the server; 'gdown' sources use gdown --folder --continue; 'stream'/'local' sources have no raw download.
After each file: sha256 is computed and compared with the publisher's checksum when the registry has one
(mismatch = fail, file kept as <name>.corrupt for inspection); archives without a published checksum are tested with
their internal CRC (zip) when converted. Every action is appended to pipeline_log.jsonl and the provenance table
provenance/downloads.csv is rewritten (url, bytes, sha256, integrity, time).
Why: a corpus is only reproducible if each byte can be traced to a public URL and verified; resumable, host-aware
downloading is needed because some hosts throttle (Zenodo 403 on parallel connections, EBI refusing connections).
usage: python -m emseg.data.stages.download [source_id ...] [--retries 3]"""
import argparse, csv, hashlib, os, re, shutil, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from emseg.data.standard import DATA_ROOT
from emseg.data.registry import SOURCES, BY_ID, host_policy, EBI
from emseg.data.log import event

RAW = DATA_ROOT / "raw_downloads"
PDL = Path(__file__).resolve().parents[3] / "tools/pdl.py"
GDOWN = shutil.which("gdown") or "gdown"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def expand(src):
    """file list of a source; 'listing' sources append files read from the server"""
    files = list(src.files)
    if src.id == "cem_benchmarks":
        for sub in ("images", "masks"):
            url = f"{EBI}/10982/data/tem_benchmark/{sub}/"
            try:
                html = urllib.request.urlopen(url, timeout=120).read().decode()
                for fn in sorted(set(re.findall(r'href="([^"?/]+\.tiff?)"', html))):
                    files.append((f"em/cem_benchmarks/tem_benchmark/{sub}/{fn}", url + fn, "", ""))
            except Exception as e:  # noqa
                event("download", "listing_failed", src.id, url=url, error=str(e))
                local = RAW / f"em/cem_benchmarks/tem_benchmark/{sub}"         # server listing unavailable: the files
                for p in sorted(local.glob("*.tif*")) if local.exists() else []:  # already fetched keep their record
                    files.append((f"em/cem_benchmarks/tem_benchmark/{sub}/{p.name}", url + p.name, "", ""))
    if src.id == "tumor_spheroid":
        tsv = RAW / files[0][0]
        base = files[0][1].rsplit("/", 1)[0]
        if tsv.exists():
            names = set()
            for r in csv.DictReader(open(tsv), delimiter="\t"):
                names |= {r["Files"], r["source_image"]}
            files += [(f"em/tumor_spheroid/{n}", f"{base}/{n}", "", "") for n in sorted(names)]
    return files


_LOGGED = None


def _logged():
    """files that already have a download 'done' event with a sha256 in the log"""
    global _LOGGED
    if _LOGGED is None:
        from emseg.data.log import read
        _LOGGED = {e["file"] for e in read() if e["stage"] == "download" and e["action"] == "done" and "sha256" in e}
    return _LOGGED


def fetch(item, src_id, conn):
    dest, url, sha, flags = item
    out = RAW / dest
    if out.exists() and not Path(str(out) + ".parts").exists():
        if dest not in _logged():            # fetched earlier by another tool (pdl queue / retry loop): hash it and record it now
            h = sha256(out); ok = not sha or h == sha
            event("download", "done" if ok else "checksum_mismatch", src_id, file=dest, url=url, bytes=out.stat().st_size, sha256=h,
                  expected=sha, integrity=("verified_ref" if sha else "no_reference") if ok else "mismatch", recorded_after_download=True)
            if not ok:
                return dest, "checksum_mismatch"
        return dest, "present"
    event("download", "start", src_id, file=dest, url=url)
    r = subprocess.run([sys.executable, str(PDL), url, str(out), "--conn", str(conn)] + flags.split(), capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        event("download", "fail", src_id, file=dest, url=url, error=(r.stdout + r.stderr).strip().splitlines()[-1:] or "")
        return dest, "fail"
    h = sha256(out)
    if sha and h != sha:
        os.replace(out, str(out) + ".corrupt")
        event("download", "checksum_mismatch", src_id, file=dest, url=url, sha256=h, expected=sha)
        return dest, "checksum_mismatch"
    event("download", "done", src_id, file=dest, url=url, bytes=out.stat().st_size, sha256=h, integrity="verified_ref" if sha else "no_reference")
    return dest, "ok"


def download_source(src, retries=3):
    if src.method in ("stream", "local"):
        return {"source": src.id, "status": f"no raw download ({src.method})"}
    if src.method == "gdown":
        dest, url = src.files[0][0], src.files[0][1]
        for k in range(retries):
            event("download", "start", src.id, file=dest, url=url, method="gdown", attempt=k + 1)
            r = subprocess.run(["timeout", "3600", str(GDOWN), "--folder", "--continue", url, "-O", str(RAW / dest)], capture_output=True, text=True)
            n = sum(1 for _ in (RAW / dest).rglob("*") if _.is_file())
            event("download", "done" if r.returncode == 0 else "incomplete", src.id, file=dest, url=url, n_files=n)
            if r.returncode == 0:
                return {"source": src.id, "status": "ok", "n_files": n}
            time.sleep(600)
        return {"source": src.id, "status": "incomplete"}
    if src.method == "listing" and src.id == "tumor_spheroid":       # the list itself must be fetched first
        fetch(src.files[0], src.id, 2)
    files = expand(src)
    res = {}
    for k in range(retries):
        todo = [f for f in files if res.get(f[0]) not in ("ok", "present")]
        if not todo:
            break
        conn, par = host_policy(todo[0][1])
        with ThreadPoolExecutor(par) as ex:
            for dest, st in ex.map(lambda it: fetch(it, src.id, conn), todo):
                res[dest] = st
        if any(v not in ("ok", "present") for v in res.values()) and k < retries - 1:
            time.sleep(300 if "zenodo" in todo[0][1] else 30)
    bad = {d: s for d, s in res.items() if s not in ("ok", "present")}
    return {"source": src.id, "status": "ok" if not bad else "incomplete", "n_files": len(files), "failed": bad}


def write_provenance():
    """provenance/downloads.csv from the event log + files on disk (latest successful event per file)"""
    from emseg.data.log import read
    last = {}
    for e in read():
        if e["stage"] == "download" and e["action"] == "done" and "sha256" in e:
            last[e["file"]] = e
    rows = []
    for s in SOURCES:
        for dest, url, sha, _ in expand(s) if s.method in ("http", "listing") else []:
            p = RAW / dest; e = last.get(dest)
            rows.append({"source": s.id, "file": f"raw_downloads/{dest}", "url": url, "present": p.exists(),
                         "bytes": p.stat().st_size if p.exists() else "", "sha256": e["sha256"] if e else "",
                         "expected_sha256": sha, "integrity": (e or {}).get("integrity", "" if not p.exists() else "unlogged"),
                         "downloaded": (e or {}).get("ts", "")})
    out = DATA_ROOT / "provenance/downloads.csv"; out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sources", nargs="*"); ap.add_argument("--retries", type=int, default=3)
    a = ap.parse_args()
    srcs = [BY_ID[s] for s in a.sources] if a.sources else [s for s in SOURCES if s.default]
    groups = {}                      # host groups run concurrently, sources inside a group sequentially (host politeness)
    for s in srcs:
        u = s.files[0][1] if s.files else ""
        groups.setdefault("zenodo" if "zenodo" in u else "ebi" if "ebi.ac.uk" in u else "gdrive" if s.method == "gdown" else "other", []).append(s)

    def run_group(ss):
        for s in ss:
            r = download_source(s, a.retries)
            event("download", "source_" + r["status"].split(" ")[0], s.id, **{k: v for k, v in r.items() if k != "source"})
            print(r, flush=True)
    with ThreadPoolExecutor(len(groups)) as ex:
        list(ex.map(run_group, groups.values()))
    print("provenance ->", write_provenance()); print("exit=0")


if __name__ == "__main__":
    main()
