"""Converters for SNEMI3D / CREMI / Lucchi / EMNeuron from the files downloaded from their public releases
($EMDATA_ROOT/raw_downloads/em/{snemi,cremi,lucchi,emneuron}), plus the in-house mouse_30um (not public; written under
$EMDATA_ROOT/local/, excluded from the default manifests; needs EMDATA_LOCAL_MOUNT).

Leakage rules (benchmarks of this project): SNEMI AC4 z80-99 (val) and the whole AC3 (test; = EMNeuron Kasthuri-atum/0)
are excluded; CREMI A/B/C only z0-74 (z75-124 = val/test); mouse_30um mip1 only z0-69 (z70-99 = val/test).
Duplicates: EMNeuron CREMI-sstem and Kasthuri-atum are skipped (the originals are converted from the SNEMI / CREMI
releases with the leakage rules above); EMNeuron valid/InDistribution/InDistribution/* are byte-identical copies of
valid/InDistribution/* and are skipped.
usage: python -m emseg.data.converters.benchmarks <job|all> [--workers N]
"""
from __future__ import annotations

import argparse, json, multiprocessing as mp, os
from pathlib import Path

import h5py
import numpy as np
import tifffile

from emseg.data.standard import write_volume, DATA_ROOT

MNT = Path(os.environ.get("EMDATA_LOCAL_MOUNT", DATA_ROOT / "local_src"))     # in-house data only (mouse_30um)
DL = DATA_ROOT / "raw_downloads/em"


def _emn_root():
    """extracted EMNeuron archives: <root>/labeled/<subset>, <root>/valid/{InDistribution,OutofDistribution}/<name>"""
    base = DL / "emneuron/extracted"
    for c in [base] + [p for p in base.rglob("*") if p.is_dir()]:
        if (c / "labeled").is_dir() or (c / "valid").is_dir():
            return c
    raise FileNotFoundError(base)
EMN_URL = "https://huggingface.co/datasets/yanchaoz/EMNeuron"
EMN_LIC = "MIT (EMNeuron release); original data licences apply"

# EMNeuron subsets: name -> (modality, organism, tissue, resolution [z,y,x] nm, original source). Resolutions from the
# SegNeuron README table (https://github.com/yanchaoz/SegNeuron).
EMN_META = {
    "AxonEM[H]-atum": ("ATUM-SEM", "human", "cortex (H01, AxonEM-H)", [30, 8, 8], "AxonEM (Wei et al. 2021) / H01"),
    "AxonEM[M]-sstem": ("ssTEM", "mouse", "visual cortex (MICrONS, AxonEM-M)", [40, 8, 8], "AxonEM (Wei et al. 2021) / MICrONS"),
    "Fib-25-fib": ("FIB-SEM", "drosophila", "optic lobe medulla (FIB-25)", [8, 8, 8], "Takemura et al. 2015 (FIB-25)"),
    "Hemi-brain-fib": ("FIB-SEM", "drosophila", "central brain (hemibrain)", [8, 8, 8], "Scheffer et al. 2020 (hemibrain)"),
    "J0126-sbem": ("SBF-SEM", "zebra finch", "HVC (j0126)", [20, 9, 9], "Kornfeld et al. 2017 (j0126 / ZFinch)"),
    "Mira-adwt": ("ATUM-SEM", "mouse", "cortex, Alzheimer model (vEM1)", [50, 8, 8], "SegNeuron vEM1 (adwt)"),
    "Mira-fish": ("ATUM-SEM", "zebrafish", "larval brain (vEM2)", [30, 8, 8], "SegNeuron vEM2 (zfish)"),
    "Mira-scn": ("ATUM-SEM", "mouse", "suprachiasmatic nucleus (vEM3)", [40, 8, 8], "SegNeuron vEM3 (scn)"),
}
EMN_VALID = {
    "FAFB-1": ("ssTEM", "drosophila", "whole brain (FAFB)", [40, 8, 8]), "FAFB-2": ("ssTEM", "drosophila", "whole brain (FAFB)", [40, 8, 8]),
    "FIB25-1": ("FIB-SEM", "drosophila", "medulla (FIB-25)", [8, 8, 8]), "FIB25-2": ("FIB-SEM", "drosophila", "medulla (FIB-25)", [8, 8, 8]),
    "Kstr": ("ATUM-SEM", "mouse", "Kasthuri volume (EMNeuron 'Kstr' crop)", [30, 6, 6]),
    "Harris": ("ssTEM", "rat", "hippocampus (Harris lab)", [50, 6, 6]), "Ionsem": ("ion-beam SEM", "mouse", "brain (vEM4, ion-beam etching)", [20, 8, 8]),
    "Microns[B]": ("ssTEM", "mouse", "visual cortex (MICrONS Basil)", [40, 8, 8]),
}


def _tif(p):
    return tifffile.imread(str(p))


def job_emn_labeled(name):
    mod, org, tis, res, src = EMN_META[name]; out = []
    for p in sorted((_emn_root() / "labeled" / name).glob("*_MaskIns.tif"), key=lambda q: int(q.name.split("_")[0])):
        i = p.name.split("_")[0]
        raw = _tif(p.parent / f"{i}.tif"); lab = _tif(p)
        if lab.dtype.kind == "f":
            lab = np.rint(lab).astype(np.int64)
        extra = {"source_original": src}
        out.append(write_volume(f"emneuron_{name.replace('[', '').replace(']', '').lower()}", f"vol{i}", raw, lab.astype(np.int64),
                                label_scope="dense_cells", resolution_nm=res, modality=mod, organism=org, tissue=tis,
                                source_url=EMN_URL, license=EMN_LIC, extra_attrs=extra))
    return out


def job_emn_valid(name):
    mod, org, tis, res = EMN_VALID[name]
    d = next(p for p in (_emn_root() / "valid/InDistribution" / name, _emn_root() / "valid/OutofDistribution" / name, _emn_root() / "InDistribution" / name, _emn_root() / "OutofDistribution" / name) if p.exists())
    raw = _tif(d / "raw.tif"); lab = _tif(d / "label.tif")
    if lab.dtype.kind == "f":
        lab = np.rint(lab)
    return [write_volume(f"emneuron_valid_{name.replace('[', '').replace(']', '').lower()}", "vol0", raw, lab.astype(np.int64),
                         label_scope="dense_cells", resolution_nm=res, modality=mod, organism=org, tissue=tis,
                         source_url=EMN_URL, license=EMN_LIC, extra_attrs={"source_original": f"EMNeuron validation set {name}"})]


def job_snemi():
    with h5py.File(DL / "snemi/snemi_train.h5", "r") as f:
        keys = []; f.visit(keys.append)
        raw = f["volumes/raw"][:80]; lab = f["volumes/labels/neuron_ids"][:80]
    return [write_volume("snemi3d", "ac4_z0-79", raw, lab.astype(np.int64), label_scope="dense_cells", resolution_nm=[29, 6, 6],
                         modality="ATUM-SEM", organism="mouse", tissue="somatosensory cortex (Kasthuri 2015, AC4)",
                         source_url="https://oc.embl.de/index.php/s/43iMotlXPyAB39z/download (EMBL mirror of SNEMI3D AC4, as used by torch_em)", license="SNEMI3D challenge (research use)",
                         z_index=range(80), extra_attrs={"leakage_rule": "AC4 z80-99 (benchmark val) and AC3 (benchmark test) excluded"})]


def job_cremi(sample):
    fn = {"A": "sample_A_20160501.hdf", "B": "sample_B_20160501.hdf", "C": "sample_C_20160501.hdf"}[sample]
    with h5py.File(DL / "cremi" / fn, "r") as f:
        raw = f["volumes/raw"][:75]; lab = f["volumes/labels/neuron_ids"][:75].astype(np.int64)
    # CREMI neuron_ids use 0xffffffffffffffff-style "no label" values in some versions -> treat ids >= 2**63 as 0
    lab[lab < 0] = 0
    return [write_volume("cremi", f"{sample}_z0-74", raw, lab, label_scope="dense_cells", resolution_nm=[40, 4, 4], modality="ssTEM",
                         organism="drosophila", tissue="adult brain (FAFB region)", source_url="https://cremi.org/data/",
                         license="CREMI challenge (CC-BY 4.0)", z_index=range(75),
                         extra_attrs={"leakage_rule": "z75-124 (benchmark val/test) excluded"})]


def job_lucchi():
    out = []
    for part in ("train", "test"):
        nm = {"train": "training", "test": "testing"}[part]
        raw = _tif(DL / f"lucchi/{nm}.tif"); sem = _tif(DL / f"lucchi/{nm}_groundtruth.tif")
        from scipy import ndimage as ndi
        lab = np.stack([ndi.label(s > 0)[0] for s in sem])          # semantic mito mask -> 2D instances
        out.append(write_volume("lucchi", part, raw, lab.astype(np.int64), label_scope="organelle:mito", resolution_nm=[5, 5, 5],
                                modality="FIB-SEM", organism="mouse", tissue="hippocampus CA1", source_url="https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/%20ElectronMicroscopy_Hippocampus/",
                                license="EPFL CVLab (research use)", extra_attrs={"instances_from": "connected components of the binary mito mask"}))
    return out


def job_mouse30um():
    from emseg.eval.domains import get_domain
    dom = get_domain("mouse30um", mip=1); zs = [z for z in range(70) if z != 8]
    raw = np.stack([dom.load_image(z) for z in zs]); lab = np.stack([dom.load_label(z) for z in zs]).astype(np.int64)
    return [write_volume("mouse30um_local", "mip1_z0-69", raw, lab, out_root=DATA_ROOT / "local", label_scope="dense_cells", resolution_nm=[30, 16, 16],
                         modality="ATUM-SEM (in-house)", organism="mouse", tissue="cortex (in-house mouse_30um)",
                         source_url="in-house (not public)", license="in-house, not for redistribution",
                         z_index=zs, extra_attrs={"leakage_rule": "z70-99 (benchmark val/test) and blank z8 excluded"})]


JOBS = {**{f"emn:{k}": (job_emn_labeled, k) for k in EMN_META}, **{f"emnval:{k}": (job_emn_valid, k) for k in EMN_VALID},
        "snemi": (job_snemi,), "cremi:A": (job_cremi, "A"), "cremi:B": (job_cremi, "B"), "cremi:C": (job_cremi, "C"),
        "lucchi": (job_lucchi,), "mouse30um": (job_mouse30um,)}


def _run(k):
    fn, *args = JOBS[k]
    try:
        res = fn(*args)
        for r in res:
            r.pop("slices", None)
        return k, res, None
    except Exception as e:  # noqa
        import traceback; return k, None, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("jobs", nargs="+"); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    keys = list(JOBS) if a.jobs == ["all"] else a.jobs
    log = DATA_ROOT / "_build_logs"; log.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for k, res, err in pool.imap_unordered(_run, keys):
            print(json.dumps({"job": k, "result": res, "error": err})[:3000], flush=True)
            json.dump({"job": k, "result": res, "error": err}, open(log / f"{k.replace(':', '_').replace('[', '').replace(']', '')}.json", "w"), indent=1)
    print("exit=0")


if __name__ == "__main__":
    main()
