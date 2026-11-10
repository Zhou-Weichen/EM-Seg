"""Stage 2+3 - conversion to the unified standard (EMDATA-v2, see emseg.data.standard) for every registry Source.

How: each converter job of a Source ("module:job" in the registry) runs as its own subprocess
(isolation: one failing reader cannot corrupt another dataset); a job is skipped when the log already records it as done
and its outputs exist (resumable, idempotent); a source whose raw files are not all present is marked waiting_download.
Streaming jobs (connectome cut-outs) run under a watchdog that restarts them after 8 min without a new output file
(cloud reads can hang without timeout); they resume from the files already written (fixed seed -> same slab sequence).
What a converter does (common to all, implemented once in emseg.data.standard): read the source format; map its annotation
to the unified class vocabulary (cell / mitochondrion / nucleus / ...); per volume percentile normalisation to uint8;
per-slice 4-connected 2D instances; padding / blank / degenerate-slice removal; leakage exclusions from the registry
rules; write labels/<class>, masks/<class>, label_src/<class> + full provenance attributes.
usage: python -m emseg.data.stages.convert [source_id ...] [--force] [--include-local]"""
import argparse, subprocess, sys, time
from pathlib import Path

from emseg.data.standard import DATA_ROOT
from emseg.data.registry import SOURCES, BY_ID
from emseg.data.log import event, read

PY = sys.executable
CODE = Path(__file__).resolve().parents[3]


def command(spec):
    mod, _, job = spec.partition(":")
    if mod == "local_mount":
        return [PY, "-m", "emseg.data.converters.benchmarks", job, "--workers", "1"]
    if mod == "downloaded":
        return [PY, "-m", "emseg.data.converters.downloaded", job]
    if mod == "stream":
        return [PY, "-m", "emseg.data.converters.stream", job, "--slabs", "48", "--xy", "1024", "--nz", "8"]
    if mod == "asem":
        return [PY, "-m", "emseg.data.converters.asem"]
    if mod == "coco":
        return [PY, "-m", "emseg.data.converters.coco", job, "--workers", "32"]
    raise ValueError(spec)


def outputs_present(src):
    n = 0
    for o in src.outputs:
        for root in ("em", "em2d", "natural", "local"):
            n += sum(1 for _ in (DATA_ROOT / root).glob(o.replace("*", "*") + "/*.h5"))
    if src.id == "coco_panoptic":
        n = sum(1 for _ in (DATA_ROOT / "natural/coco_panoptic").rglob("shard_*.h5"))
    return n


def raw_present(src):
    if src.method in ("stream", "local"):
        return True
    if src.method == "gdown":
        p = DATA_ROOT / "raw_downloads" / src.files[0][0]
        return p.exists() and (any(p.rglob("*.h5")) or any(p.rglob("*.tif*")))
    from emseg.data.stages.download import expand
    return all((DATA_ROOT / "raw_downloads" / f[0]).exists() for f in expand(src))


def done_jobs():
    return {(e["source"], e.get("job")) for e in read() if e["stage"] == "convert" and e["action"] == "done"}


def run_job(src, spec, watchdog):
    cmd = command(spec); t0 = time.time()
    event("convert", "start", src.id, job=spec, cmd=" ".join(cmd))
    logf = DATA_ROOT / "_build_logs" / f"convert_{spec.replace(':', '_').replace('[', '').replace(']', '')}.log"
    logf.parent.mkdir(exist_ok=True)
    for attempt in range(30 if watchdog else 1):
        with open(logf, "a") as lf:
            p = subprocess.Popen(cmd, cwd=CODE, stdout=lf, stderr=subprocess.STDOUT)
            last, tl = outputs_present(src), time.time()
            while p.poll() is None:
                time.sleep(30)
                if watchdog:
                    n = outputs_present(src)
                    if n != last:
                        last, tl = n, time.time()
                    elif time.time() - tl > 480:
                        p.kill(); event("convert", "watchdog_restart", src.id, job=spec, n_outputs=n, attempt=attempt + 1); break
        if p.returncode == 0:
            event("convert", "done", src.id, job=spec, seconds=round(time.time() - t0), n_output_files=outputs_present(src))
            return "done"
    event("convert", "fail", src.id, job=spec, returncode=p.returncode, log=str(logf))
    return "fail"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("sources", nargs="*"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--include-local", action="store_true"); a = ap.parse_args()
    srcs = [BY_ID[s] for s in a.sources] if a.sources else [s for s in SOURCES if s.default or a.include_local]
    dj = done_jobs()
    for s in srcs:
        if not raw_present(s):
            event("convert", "waiting_download", s.id); print(s.id, "waiting_download", flush=True); continue
        for spec in s.converters:
            mod, _, job = spec.partition(":")
            if mod == "downloaded":
                from emseg.data.converters import downloaded
                if job not in downloaded.JOBS:
                    event("convert", "converter_missing", s.id, job=spec); print(s.id, spec, "converter_missing", flush=True); continue
            if not a.force and (s.id, spec) in dj and outputs_present(s) > 0:
                continue
            print(s.id, spec, run_job(s, spec, watchdog=spec.startswith(("stream", "asem"))), flush=True)
    print("exit=0")


if __name__ == "__main__":
    main()
