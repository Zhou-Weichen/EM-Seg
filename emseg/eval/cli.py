"""Command line `emseg-eval` (= python -m emseg.eval.cli): evaluate | splits | selfcheck"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from emseg import __version__
from emseg.eval.domains import get_domain, DOMAINS
from emseg.eval.evaluate import evaluate_slices, load_prediction, write_outputs


def _domain(args):
    kw = {}
    if args.domain == "mouse30um":
        kw["mip"] = args.mip
    if args.domain == "cremi":
        kw["sample"] = args.sample
    return get_domain(args.domain, **kw)


def _scheme(args, dom):
    if args.scheme:
        return args.scheme
    return {"mouse30um": "interim", "snemi": "protocol", "cremi": "interim"}[dom.name]


def cmd_splits(args):
    dom = _domain(args)
    s = dom.splits(_scheme(args, dom))
    for k, v in s.items():
        print(f"{k}: n={len(v)} {v[:5]}{' ...' if len(v) > 5 else ''}")


def cmd_evaluate(args):
    dom = _domain(args)
    scheme = _scheme(args, dom)
    indices = dom.splits(scheme)[args.split]
    if not indices:
        sys.exit(f"split {args.split!r} of {dom.name}/{scheme} is empty (external test block not available)")
    pred_dir = Path(args.pred_dir)
    df = evaluate_slices(dom.load_label, lambda i: load_prediction(pred_dir, i), indices,
                         min_size=args.min_size, progress=print if args.verbose else None, image_fn=dom.load_image, mask_missing=args.mask_missing)
    meta = {"emseg_version": __version__, "domain": dom.name, "scheme": scheme, "split": args.split,
            "n_slices": len(indices), "min_size": args.min_size, "mask_missing": args.mask_missing, "pred_dir": str(pred_dir.resolve()),
            "mip": getattr(dom, "mip", None), "sample": getattr(dom, "sample", None)}
    write_outputs(df, Path(args.out), meta)
    s = df.mean(numeric_only=True)
    print(f"{dom.name}/{scheme}/{args.split} n={len(df)}: voi_sum={s.voi_sum:.4f} arand={s.arand:.4f} "
          f"msa={s.msa:.4f} f1_50={s.f1_50:.4f} pq50={s.pq50:.4f}")


def cmd_selfcheck(args):
    """Evaluate GT against itself (must be perfect) - an end-to-end validity check of loaders + metrics."""
    dom = _domain(args)
    scheme = _scheme(args, dom)
    indices = dom.splits(scheme)[args.split][: args.limit] if args.limit else dom.splits(scheme)[args.split]
    df = evaluate_slices(dom.load_label, dom.load_label, indices, min_size=args.min_size,
                         progress=print if args.verbose else None)
    meta = {"emseg_version": __version__, "domain": dom.name, "scheme": scheme, "split": args.split,
            "mode": "selfcheck", "n_slices": len(indices), "min_size": args.min_size}
    if args.out:
        write_outputs(df, Path(args.out), meta)
    ok = (df.voi_sum.abs() < 1e-9).all() and (df.arand.abs() < 1e-9).all() and (df.msa > 0.999999).all()
    print(f"selfcheck {dom.name}/{scheme}/{args.split} n={len(df)}: {'PASS' if ok else 'FAIL'} "
          f"(max voi_sum={df.voi_sum.max():.2e}, max arand={df.arand.max():.2e}, min msa={df.msa.min():.6f}, "
          f"mean n_true={df.n_true.mean():.1f})")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(prog="emseg-eval", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("splits", cmd_splits), ("evaluate", cmd_evaluate), ("selfcheck", cmd_selfcheck)):
        q = sub.add_parser(name)
        q.add_argument("--domain", required=True, choices=sorted(DOMAINS))
        q.add_argument("--mip", type=int, default=1, help="mouse30um mip level (default 1)")
        q.add_argument("--sample", default="A", help="cremi sample A/B/C")
        q.add_argument("--scheme", default=None, help="split scheme (default: domain default)")
        q.set_defaults(func=fn)
        if name != "splits":
            q.add_argument("--split", default="test", choices=["train", "val", "test"])
            q.add_argument("--min-size", type=int, default=0, help="GT fragment filter in px (default 0 = keep all)")
            q.add_argument("--mask-missing", type=int, default=0, help="ignore GT where the image is missing (0 px), dilated by N px (0 = off)")
            q.add_argument("--verbose", action="store_true")
        if name == "evaluate":
            q.add_argument("--pred-dir", required=True)
            q.add_argument("--out", required=True)
        if name == "selfcheck":
            q.add_argument("--limit", type=int, default=0)
            q.add_argument("--out", default=None)
    args = p.parse_args(argv)
    rc = args.func(args)
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
