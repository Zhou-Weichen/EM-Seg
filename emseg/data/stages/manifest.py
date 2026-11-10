"""Stage 6 - filtering, group-aware split, sampling weights, manifests (EMDATA-v2).

Inputs: qa/images.csv (stage 4), dedup/duplicates.csv + dedup/decisions.json (stage 5).
Outputs: manifests/em_only.csv, manifests/em_plus_natural.csv, manifests/summary.json, qa/filter_log.csv

Filters (an image is removed from the manifests, never deleted from disk; reason in qa/filter_log.csv):
  F1 small        min(H, W) < 64
  F2 saturated    sat_frac > 0.5 (more than half of the imaged, non-padding pixels clipped at 0/255)
  F3 blurred      lap_var < 0.1 x the median lap_var of its dataset (strong sharpness outlier)
  F4 degenerate   dense 'cell' labels where one instance covers > 90 % of the annotated area (and no other class)
  F5 misaligned   dense 'cell' labels whose boundaries are brighter than interiors (bnd_contrast > +10; EM membranes
                  are dark, so labels do not sit on membranes)
  F6 duplicate    lower-priority member of an exact / confirmed near-duplicate pair (stage 5a/b)
  F7 excluded     dataset excluded by a stage-5 decision (region overlap with a higher-priority dataset, benchmark
                  leakage); see dedup/decisions.json
Split (group-aware, never by random slice): datasets with >= 10 volumes -> whole volumes to 'val' (hash, ~5 %);
volume datasets with fewer volumes -> the last 5 % contiguous slices of each volume to 'val' with a 2-slice 'gap'
(excluded) before them; 2D collections -> by source group (e.g. CEM-MitoLab sub-dataset) hash, ~5 %;
COCO: val2017 -> 'val', train2017 -> 'train'. 'val' is for internal monitoring only, never a benchmark.
Weights (train rows, sum = 1 per manifest): dataset mass ∝ sqrt(labelled megapixels), x0.5 for silver labels,
spread within the dataset proportional to annotated area; em_plus_natural gives NAT_FRAC to COCO.
usage: python -m emseg.data.stages.manifest [--nat-frac 0.05]"""
import argparse, csv, hashlib, json
from collections import defaultdict

import numpy as np

from emseg.data.standard import DATA_ROOT, CLASSES

SILVER_FACTOR = 0.5
PRIORITY = {"gold_dense": 3, "gold_organelle": 2, "silver": 1}
FIELDS = ["uid", "group", "dataset", "file", "key", "z", "z_src", "H", "W", "res_xy_nm", "res_z_nm", "classes", "n_inst",
          "quality", "modality", "organism", "split", "weight"]


def tier(r):
    if r["quality"] == "silver":
        return "silver"
    return "gold_dense" if "cell" in r["classes"].split("|") else "gold_organelle"


def h01(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def fnum(x, d=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nat-frac", type=float, default=0.05); a = ap.parse_args()
    rows = list(csv.DictReader(open(DATA_ROOT / "qa/images.csv")))
    dec = json.load(open(DATA_ROOT / "dedup/decisions.json")) if (DATA_ROOT / "dedup/decisions.json").exists() else {}
    from emseg.data.registry import owner_of
    for ds in sorted({r["dataset"] for r in rows}):              # sources switched off in the registry (default=False)
        src = owner_of(ds)
        if src is not None and not src.default:
            dec.setdefault("exclude_datasets", {})[ds] = f"source disabled in registry: {src.rules[:160]}"
    excluded = set(dec.get("exclude_datasets", {}))
    drop = {}
    # F6 duplicates
    uid2 = {r["uid"]: r for r in rows}
    if (DATA_ROOT / "dedup/duplicates.csv").exists():
        for d in csv.DictReader(open(DATA_ROOT / "dedup/duplicates.csv")):
            ra, rb = uid2.get(d["uid_a"]), uid2.get(d["uid_b"])
            if ra is None or rb is None:
                continue
            loser = rb if (PRIORITY[tier(ra)], ra["dataset"]) >= (PRIORITY[tier(rb)], rb["dataset"]) else ra
            drop.setdefault(loser["uid"], f"F6 duplicate ({d['kind']}) of {(ra if loser is rb else rb)['uid']}")
    med_lap = defaultdict(list)
    for r in rows:
        med_lap[r["dataset"]].append(fnum(r["lap_var"]))
    med_lap = {k: float(np.nanmedian(v)) for k, v in med_lap.items()}
    for r in rows:
        u = r["uid"]
        cls = r["classes"].split("|")
        if r["dataset"] in excluded:
            drop.setdefault(u, f"F7 excluded dataset: {dec['exclude_datasets'][r['dataset']]}")
        elif min(int(r["H"]), int(r["W"])) < 64:
            drop.setdefault(u, "F1 small")
        elif fnum(r["sat_frac"], 0) > 0.5:
            drop.setdefault(u, "F2 saturated")
        elif fnum(r["lap_var"], 1e9) < 0.1 * med_lap[r["dataset"]]:
            drop.setdefault(u, "F3 blurred")
        elif cls == ["cell"] and fnum(r.get("maxinst_cell"), 0) > 0.9:
            drop.setdefault(u, "F4 degenerate")
        elif "cell" in cls and fnum(r.get("bnd_contrast"), -1) > 10:
            drop.setdefault(u, "F5 misaligned")
    with open(DATA_ROOT / "qa/filter_log.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["uid", "dataset", "reason"])
        for u, why in sorted(drop.items()):
            w.writerow([u, uid2[u]["dataset"], why])
    keep = [r for r in rows if r["uid"] not in drop]
    # split
    vols = defaultdict(set)
    for r in keep:
        if r["z"] != "":
            vols[r["dataset"]].add(r["volume"])
    zmax = defaultdict(int)
    for r in keep:
        if r["z"] != "":
            zmax[(r["dataset"], r["volume"])] = max(zmax[(r["dataset"], r["volume"])], int(r["z"]))
    for r in keep:
        ds = r["dataset"]
        if ds == "coco_panoptic":
            r["split"] = "val" if "/val/" in r["file"] else "train"
        elif r["z"] == "":
            grp = r["volume"] or r["key"]
            r["split"] = "val" if h01(f"{ds}/{grp}") < 0.05 else "train"
        elif len(vols[ds]) >= 10:
            r["split"] = "val" if h01(f"{ds}/{r['volume']}") < 0.05 else "train"
        else:
            Z = zmax[(ds, r["volume"])] + 1; nval = max(1, int(round(0.05 * Z))); z = int(r["z"])
            r["split"] = "val" if z >= Z - nval else ("gap" if z >= Z - nval - 2 else "train")
    # guarantee >= 1 val volume for multi-volume datasets
    for ds, vs in vols.items():
        if len(vs) >= 10 and not any(r["split"] == "val" for r in keep if r["dataset"] == ds):
            v0 = sorted(vs, key=lambda v: h01(f"{ds}/{v}"))[0]
            for r in keep:
                if r["dataset"] == ds and r["volume"] == v0:
                    r["split"] = "val"
    for r in keep:
        cls = r["classes"].split("|")
        r["n_inst"] = json.dumps({c: int(fnum(r.get(f"n_{c}"), 0)) for c in cls})
        r["group"] = "natural" if r["dataset"] == "coco_panoptic" else "em"
        r["_area"] = int(r["H"]) * int(r["W"]) * fnum(r["valid_frac"], 1.0)
        r["_lab"] = r["_area"] * max(max(fnum(r.get(f"cov_{c}"), 0) for c in cls), 1e-3)

    def weights(rs, mass):
        by = defaultdict(list)
        for r in rs:
            r["weight"] = 0.0
            if r["split"] == "train":
                by[r["dataset"]].append(r)
        dm = {d: np.sqrt(sum(r["_lab"] for r in v) / 1e6) * (SILVER_FACTOR if v[0]["quality"] == "silver" else 1.0) for d, v in by.items()}
        tot = sum(dm.values())
        for d, v in by.items():
            ar = np.array([r["_area"] for r in v]); w = mass * dm[d] / tot * ar / ar.sum()
            for r, x in zip(v, w):
                r["weight"] = float(x)
        return {d: round(mass * m / tot, 5) for d, m in dm.items()}

    em = [r for r in keep if r["group"] == "em"]; nat = [r for r in keep if r["group"] == "natural"]
    out = DATA_ROOT / "manifests"; out.mkdir(exist_ok=True)

    def write(rs, path):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore"); w.writeheader()
            for r in rs:
                r2 = dict(r); r2["weight"] = f"{r['weight']:.4e}"; w.writerow(r2)
    m_em = weights(em, 1.0); write(em, out / "em_only.csv")
    m_mix = weights(em, 1 - a.nat_frac); weights(nat, a.nat_frac); write(em + nat, out / "em_plus_natural.csv")
    per = {}
    for r in keep:
        p = per.setdefault(r["dataset"], {"group": r["group"], "classes": set(), "quality": r["quality"], "modality": r["modality"],
                                          "organism": r["organism"], "res_xy_nm": set(), "n_images": 0, "n_train": 0, "n_val": 0,
                                          "Mpx": 0.0, "n_inst": defaultdict(int), "sizes": set(), "volumes": set()})
        p["classes"] |= set(r["classes"].split("|")); p["n_images"] += 1; p["n_train"] += r["split"] == "train"; p["n_val"] += r["split"] == "val"
        p["Mpx"] += int(r["H"]) * int(r["W"]) / 1e6; p["sizes"].add(f"{r['H']}x{r['W']}"); p["volumes"].add(r["volume"] or r["key"])
        if r["res_xy_nm"] not in ("", None):
            p["res_xy_nm"].add(round(fnum(r["res_xy_nm"]), 1))
        for c, n in json.loads(r["n_inst"]).items():
            p["n_inst"][c] += n
    for d, p in per.items():
        p["classes"] = sorted(p["classes"]); p["res_xy_nm"] = sorted(p["res_xy_nm"]); p["Mpx"] = round(p["Mpx"], 1)
        s = sorted(p["sizes"]); p["sizes"] = s if len(s) <= 3 else s[:2] + [f"... ({len(s)} sizes)"]
        p["n_volumes"] = len(p.pop("volumes")); p["n_inst"] = dict(p["n_inst"])
        p["filtered"] = sum(1 for u, why in drop.items() if uid2[u]["dataset"] == d)
        p["mass_em_only"] = m_em.get(d, 0.0); p["mass_em_plus_natural"] = m_mix.get(d, 0.0) if p["group"] == "em" else a.nat_frac
    reasons = defaultdict(int)
    for why in drop.values():
        reasons[why.split(" ")[0] + " " + why.split(" ")[1]] += 1
    summ = {"em_only": {"rows": len(em), "train": sum(r["split"] == "train" for r in em), "val": sum(r["split"] == "val" for r in em),
                        "gap": sum(r["split"] == "gap" for r in em)},
            "em_plus_natural": {"rows": len(em) + len(nat), "natural_rows": len(nat), "natural_mass": a.nat_frac},
            "filtered_total": len(drop), "filtered_by_reason": dict(reasons), "excluded_datasets": dec.get("exclude_datasets", {}),
            "per_dataset": per}
    json.dump(summ, open(out / "summary.json", "w"), indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_dataset"}, default=str)); print("exit=0")


if __name__ == "__main__":
    main()
