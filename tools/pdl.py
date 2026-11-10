#!/usr/bin/env python3
"""Parallel, resumable HTTP(S) downloader (range requests over N connections; falls back to a single stream when the
server does not support ranges). Verifies the final size against Content-Length and, if given, a checksum.

usage: python tools/pdl.py <url> <out_file> [--conn 8] [--md5 HEX | --sha256 HEX]
"""
import argparse, hashlib, json, os, sys, threading, time, urllib.request

CHUNK = 1 << 20
UA = "emseg-data/1.0 (research dataset download)"   # honest UA: zenodo.org answers 403 to a bare "Mozilla/5.0"
CTX = None   # ssl context; unverified only for hosts with expired certificates (cremi.org), via --insecure


def head(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
        return r.geturl(), int(r.headers.get("Content-Length", -1)), r.headers.get("Accept-Ranges", "") == "bytes"


def fetch_range(url, start, end, part, retries=20):
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if start + have > end:
            return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120, context=CTX) as r, open(part, "ab") as f:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
            if os.path.getsize(part) >= end - start + 1:
                return
        except Exception as e:  # noqa
            time.sleep(min(60, 2 ** attempt)); print(f"retry {part} ({e})", flush=True)
    raise RuntimeError(f"failed {part}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("url"); ap.add_argument("out"); ap.add_argument("--conn", type=int, default=8)
    ap.add_argument("--split", type=int, default=0, help="number of ranges (default = --conn)"); ap.add_argument("--md5"); ap.add_argument("--sha256"); ap.add_argument("--insecure", action="store_true"); a = ap.parse_args()
    global CTX
    if a.insecure:
        import ssl; CTX = ssl._create_unverified_context()
    if os.path.exists(a.out) and not os.path.exists(a.out + ".parts"):
        print("exists", a.out); return
    lock = a.out + ".lock"                        # one downloader per target file
    if os.path.exists(lock):
        try:
            os.kill(int(open(lock).read()), 0); raise RuntimeError(f"busy: another pdl.py is downloading {a.out}")
        except (ProcessLookupError, ValueError):
            pass
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); open(lock, "w").write(str(os.getpid()))
    url, size, ranges = head(a.url)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    if size <= 0 or not ranges or a.conn == 1:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300, context=CTX) as r, open(a.out + ".tmp", "wb") as f:
            while True:                                   # rename only when complete (no partial "exists")
                b = r.read(CHUNK)
                if not b:
                    break
                f.write(b)
        if size > 0 and os.path.getsize(a.out + ".tmp") != size:
            raise RuntimeError(f"size mismatch {os.path.getsize(a.out + '.tmp')} != {size}")
        os.replace(a.out + ".tmp", a.out)
    else:
        os.makedirs(a.out + ".parts", exist_ok=True)
        sp = a.out + ".parts/spans.json"
        if os.path.exists(sp):                       # resume with the original split, whatever --conn is now
            spans = [tuple(x) for x in json.load(open(sp))]
        else:
            legacy = [f for f in os.listdir(a.out + ".parts") if f.isdigit()]   # parts from a run before spans.json existed
            n = len(legacy) if legacy else max(1, min(a.split or a.conn, size // (8 * CHUNK) + 1)); step = size // n
            spans = [(i * step, (size - 1) if i == n - 1 else (i + 1) * step - 1) for i in range(n)]
            json.dump(spans, open(sp, "w"))
        sem = threading.Semaphore(a.conn)
        def guarded(*args):
            with sem:
                fetch_range(*args)
        th = [threading.Thread(target=guarded, args=(url, s, e, f"{a.out}.parts/{i:03d}")) for i, (s, e) in enumerate(spans)]
        [t.start() for t in th]; [t.join() for t in th]
        with open(a.out + ".tmp", "wb") as out:
            for i in range(len(spans)):
                with open(f"{a.out}.parts/{i:03d}", "rb") as f:
                    while True:
                        b = f.read(64 * CHUNK)
                        if not b:
                            break
                        out.write(b)
        got = os.path.getsize(a.out + ".tmp")
        if got != size:
            raise RuntimeError(f"size mismatch {got} != {size}")
        os.replace(a.out + ".tmp", a.out)
        for i in range(len(spans)):
            os.remove(f"{a.out}.parts/{i:03d}")
        os.remove(sp); os.rmdir(a.out + ".parts")
    for algo, ref in (("md5", a.md5), ("sha256", a.sha256)):
        if ref:
            h = hashlib.new(algo)
            with open(a.out, "rb") as f:
                for b in iter(lambda: f.read(64 * CHUNK), b""):
                    h.update(b)
            if h.hexdigest() != ref.lower():
                raise RuntimeError(f"{algo} mismatch for {a.out}")
    os.remove(lock)
    dt = time.time() - t0; sz = os.path.getsize(a.out)
    print(f"done {a.out} {sz/1e6:.1f} MB in {dt:.0f}s ({sz/1e6/max(dt,1e-6):.1f} MB/s)", flush=True)


if __name__ == "__main__":
    try:
        main(); print("exit=0")
    except Exception as e:
        print("ERROR", e); print("exit=1"); sys.exit(1)
