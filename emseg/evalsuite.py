"""Representation evaluation suite: emseg.probe on several domains for one frozen feature spec.

Default domains: snemi and cremi (sample B); add mouse30um when the in-house data is available (--domains).
Writes <out_dir>/<tag>_<domain>.json per domain and <out_dir>/<tag>_suite.json (val means per domain).

python -m emseg.evalsuite --feat enc:<ckpt> --tag <name> --out-dir <dir>
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from emseg import probe

DOMAINS = ("snemi", "cremi")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--feat", required=True); p.add_argument("--tag", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--domains", default=",".join(DOMAINS)); p.add_argument("--steps", type=int, default=2000); p.add_argument("--val-slices", type=int, default=5)
    a, rest = p.parse_known_args(argv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); suite = {"feat": a.feat}
    for d in a.domains.split(","):
        f = out / f"{a.tag}_{d}.json"
        probe.main(["--feat", a.feat, "--domain", d, "--out", str(f), "--steps", str(a.steps), "--val-slices", str(a.val_slices)] + rest)
        suite[d] = json.load(open(f))["val"]
    json.dump(suite, open(out / f"{a.tag}_suite.json", "w"), indent=2)
    print("SUITE", json.dumps({d: {k: round(v, 3) for k, v in r.items()} for d, r in suite.items() if d != "feat"})); print("exit=0", flush=True)


if __name__ == "__main__":
    main()
