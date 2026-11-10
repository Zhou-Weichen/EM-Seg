"""Append-only, real-time event log of the pipeline: $EMDATA_ROOT/pipeline_log.jsonl (one JSON object per line).
Every stage writes one event per action (download start/done/fail with bytes + sha256, convert done with counts, ...),
so what was done, when, from where and with what result is always reconstructible from this file alone."""
import datetime, json, os, socket

from emseg.data.standard import DATA_ROOT

LOG = DATA_ROOT / "pipeline_log.jsonl"


def event(stage: str, action: str, source: str = "", **kw):
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(), "pid": os.getpid(),
           "stage": stage, "action": action, "source": source, **kw}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def read():
    if not LOG.exists():
        return []
    out = []
    for line in open(LOG):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out
