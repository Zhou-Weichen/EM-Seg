"""Converters for the public datasets downloaded to $EMDATA_ROOT/raw_downloads/em (see README for sources).
usage: python -m emseg.data.converters.downloaded <job> [<job> ...] | all     (jobs whose raw files are missing are reported, not run)
"""
from __future__ import annotations

import argparse, glob, json, os, subprocess, sys, traceback
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from emseg.data.standard import write_volume, write_volume_mc, write_images, DATA_ROOT

RAW = DATA_ROOT / "raw_downloads/em"
S4 = ndi.generate_binary_structure(2, 1)


def _cc_of_background(boundary2d):
    """cells = 4-connected components of the non-boundary pixels."""
    return ndi.label(~boundary2d, S4)[0]


def job_vnc():
    d = RAW / "vnc/groundtruth-drosophila-vnc-master/stack1"
    raw = np.stack([np.asarray(Image.open(p)) for p in sorted(glob.glob(str(d / "raw/*.tif")))])
    mem = np.stack([np.asarray(Image.open(p)) > 0 for p in sorted(glob.glob(str(d / "membranes/*.png")))])
    # membranes png: True/255 = membrane? decide by area (membranes are the minority class)
    if mem.mean() > 0.5:
        mem = ~mem
    lab = np.stack([_cc_of_background(m) for m in mem])
    return [write_volume("vnc", "stack1", raw, lab.astype(np.int64), label_scope="dense_cells", resolution_nm=[50, 4.6, 4.6],
                         modality="ssTEM", organism="drosophila", tissue="larval ventral nerve cord (Gerhard et al. 2013)",
                         source_url="https://github.com/unidesigner/groundtruth-drosophila-vnc", license="CC BY-NC-SA 3.0",
                         extra_attrs={"instances_from": "4-connected components of the non-membrane pixels of the binary membrane GT"})]


def job_isbi2012():
    with h5py.File(RAW / "isbi2012/isbi.h5", "r") as f:
        keys = []; f.visit(keys.append)
        raw = f["raw"][:]
        lab = f["labels/gt_segmentation"][:] if "labels/gt_segmentation" in f else np.stack([_cc_of_background(m == 0) for m in f["labels/membranes"][:]])
    return [write_volume("isbi2012", "train", raw, lab.astype(np.int64), label_scope="dense_cells", resolution_nm=[50, 4, 4], modality="ssTEM",
                         organism="drosophila", tissue="first-instar larva ventral nerve cord (ISBI 2012)",
                         source_url="https://oc.embl.de/index.php/s/h0TkwqxU0PJDdMd/download (torch_em mirror of the ISBI 2012 challenge)",
                         license="ISBI 2012 challenge (research use)", extra_attrs={"h5_keys": json.dumps(keys)})]


def job_wafer4():
    with h5py.File(RAW / "wafer4/raw.h5", "r") as f:
        raw = f[list(f.keys())[0]][:]
    with h5py.File(RAW / "wafer4/labels.h5", "r") as f:
        lab = f["main"][:]
    return [write_volume("wafer4", "vol0", raw, lab.astype(np.int64), label_scope="dense_cells", resolution_nm=[35, 8, 8],
                         modality="multi-beam SEM", organism="mouse", tissue="medial entorhinal cortex layer 6 (Wafer4, CAD)",
                         source_url="https://github.com/liuxy1103/CAD", license="CC BY-NC 4.0")]


JOBS = {"vnc": job_vnc, "isbi2012": job_isbi2012, "wafer4": job_wafer4}


CEM_BENCH = {   # name: (xy nm from EMPIAR-10982 pixel_width [Å]/10, organism/tissue, modality)
    "c_elegans": (24.0, "C. elegans", "vEM (FIB-SEM / SBF-SEM; CEM benchmark)"),
    "fly_brain": (12.0, "drosophila brain", "vEM (CEM benchmark)"),
    "glycolytic_muscle": (18.0, "mouse glycolytic muscle", "vEM (CEM benchmark)"),
    "hela_cell": (15.0, "human HeLa cell", "vEM (CEM benchmark)"),
    "lucchi_pp": (5.0, "mouse hippocampus CA1 (Lucchi++ relabelled)", "FIB-SEM"),
    "salivary_gland": (15.0, "rat salivary gland", "vEM (CEM benchmark)"),
}


def job_cem_benchmarks():
    import tifffile
    out = []
    for name, (xy, org, mod) in CEM_BENCH.items():
        e, m = RAW / f"cem_benchmarks/{name}_em.tif", RAW / f"cem_benchmarks/{name}_mito.tif"
        if not (e.exists() and m.exists()):
            out.append({"dataset": f"cem_{name}", "missing": True}); continue
        raw, lab = tifffile.imread(e), tifffile.imread(m)
        if raw.ndim == 2:
            raw, lab = raw[None], lab[None]
        out.append(write_volume(f"cem_{name}", "vol0", raw, lab.astype(np.int64), label_scope="organelle:mito", resolution_nm=[xy, xy, xy],
                                modality=mod, organism=org, tissue=org, source_url="https://www.ebi.ac.uk/empiar/EMPIAR-10982/",
                                license="CC0 (EMPIAR)", extra_attrs={"note": "z spacing not given by EMPIAR; set equal to xy",
                                                                     "label_dtype_src": str(lab.dtype)}))
    return out


def _cem_pairs(root):
    """CEM-MitoLab: <subdir>/images/*.tiff + <subdir>/masks/*.tiff (same file names)."""
    for img in sorted(glob.glob(str(root / "**/images/*"), recursive=True)):
        msk = img.replace("/images/", "/masks/")
        if os.path.exists(msk):
            yield img, msk


def job_cem_mitolab():
    import tifffile, zipfile
    root = RAW / "cem/cem_mitolab"
    if not root.exists():
        with zipfile.ZipFile(RAW / "cem/cem_mitolab.zip") as z:
            z.extractall(RAW / "cem")
        root = next(p for p in (RAW / "cem").iterdir() if p.is_dir() and not p.name.endswith(".parts"))

    def items():
        for img, msk in _cem_pairs(root):
            src = os.path.relpath(img, root).split("/images/")[0]
            yield (f"{src.replace('/', '__')}__{Path(img).stem}", tifffile.imread(img), tifffile.imread(msk).astype(np.int64), None,
                   {"source_subdir": src})
    return [write_images("cem_mitolab", items(), label_scope="organelle:mito", resolution_nm=None, modality="mixed vEM / TEM (CEM-MitoLab)",
                         organism="mixed (human, mouse, rat, fly, worm, cell lines, ...)", tissue="mixed",
                         source_url="https://www.ebi.ac.uk/empiar/EMPIAR-11037/", license="CC0 (EMPIAR)",
                         extra_attrs={"note": "21,860 annotated 2D patches (mostly 224x224); pixel size heterogeneous / not given per patch"})]


def job_cem_tem():
    import tifffile
    d = RAW / "cem_benchmarks/tem_benchmark"

    def items():
        for img in sorted(glob.glob(str(d / "images/*"))):
            msk = img.replace("/images/", "/masks/")
            if os.path.exists(msk):
                yield Path(img).stem, tifffile.imread(img), tifffile.imread(msk).astype(np.int64)
    return [write_images("cem_tem_benchmark", items(), label_scope="organelle:mito", resolution_nm=None, modality="TEM",
                         organism="mixed", tissue="mixed cellular TEM", source_url="https://www.ebi.ac.uk/empiar/EMPIAR-10982/",
                         license="CC0 (EMPIAR)")]


def job_tumor_spheroid():
    """S-BIAD3263 2D manual GT: each annotated slice has a cells label and a nuclei label -> one image, two classes."""
    import csv as _csv, tifffile
    d = RAW / "tumor_spheroid"; rows = list(_csv.DictReader(open(d / "ground_truths_annotations_filelist.tsv"), delimiter="\t"))
    by = {}
    for r in rows:
        by.setdefault(r["source_image"], {})[r["target"]] = r

    def items():
        for img, t in sorted(by.items()):
            if not (d / img).exists():
                continue
            labs = {}
            for target, cls in (("cells", "cell"), ("nuclei", "nucleus")):
                if target in t and (d / t[target]["Files"]).exists():
                    labs[cls] = tifffile.imread(d / t[target]["Files"]).astype(np.int64)
            r0 = next(iter(t.values())); nm = float(r0["pixel_size_nm"])
            yield Path(img).stem, tifffile.imread(d / img), labs, None, {"resolution_nm": [nm, nm, nm], "slice_orientation": r0["slice_orientation"]}
    return [write_images("tumor_spheroid", items(), modality="FIB-SEM (gold-coated tumour spheroid)", organism="human", tissue="tumour spheroid",
                         source_url="https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD3263", license="CC BY 4.0 (BioImage Archive)",
                         classes=["cell", "nucleus"])]


def job_nuclei():
    """torch_em nuclei crops (FAFB / FlyWire nuclei; MICrONS nuclei): automatic nucleus segmentations of whole
    connectomes -> quality silver. train/val/test blocks are all public crops (none is our benchmark) -> all used."""
    out = []
    for name, org, tis, url in (("fafb_nuclei", "drosophila", "adult brain (FAFB, FlyWire nuclei)", "https://owncloud.gwdg.de/index.php/s/PDrkO02w7FWLrsh/download"),
                                ("microns_nuclei", "mouse", "visual cortex (MICrONS minnie65 nuclei)", "https://owncloud.gwdg.de/index.php/s/ToLGAzg1FAV4Sxf/download")):
        root = RAW / name / "extracted"
        if not root.exists():
            import zipfile; zipfile.ZipFile(RAW / name / f"{name}.zip").extractall(root)
        for p in sorted(root.rglob("*.h5")):
            with h5py.File(p, "r") as f:
                raw, lab = f["raw"][:], f["labels/nuclei"][:].astype(np.int64)
            r = write_volume_mc(name, f"{p.parent.name}_{p.stem}", raw, {"nucleus": lab}, resolution_nm=[-1, -1, -1], modality="ssTEM (downsampled)",
                                organism=org, tissue=tis, source_url=url, license="see original publication",
                                extra_attrs={"quality": "silver", "note": "pixel size not given by the release (downsampled connectome EM)"})
            r.pop("slices", None); out.append(r)
    return out


JOBS.update({"cem_benchmarks": job_cem_benchmarks, "cem_mitolab": job_cem_mitolab, "cem_tem": job_cem_tem, "tumor_spheroid": job_tumor_spheroid})


def job_mitoem():
    """MitoEM-R/H: labelled slices 0-499 (train 0-399, val 400-499), 4096^2 at 8x8x30 nm. Written in blocks of 25 slices
    (memory); normalisation percentiles per block. Raw = im/im%04d.png inside EM30-{R,H}-im.zip."""
    import tifffile, zipfile, io
    out = []
    for sp, org in (("R", "rat"), ("H", "human")):
        zi, zl = RAW / f"mitoem/EM30-{sp}-im.zip", RAW / f"mitoem/EM30-{sp}-mito-train-val-v2.zip"
        if not (zi.exists() and zl.exists()):
            out.append({"dataset": f"mitoem_{sp}", "missing": True}); continue
        with zipfile.ZipFile(zi) as fi, zipfile.ZipFile(zl) as fl:
            ok = lambda n: "__MACOSX" not in n and not Path(n).name.startswith("._")
            ims = {int(Path(n).stem[-4:]): n for n in fi.namelist() if n.endswith(".png") and ok(n)}
            labs = {int(Path(n).stem[-4:]): n for n in fl.namelist() if n.endswith(".tif") and ok(n)}
            zs = sorted(set(ims) & set(labs))
            for b in range(0, len(zs), 25):
                blk = zs[b:b + 25]
                if (DATA_ROOT / "em" / f"mitoem_{sp}" / f"z{blk[0]:04d}-{blk[-1]:04d}.h5").exists():
                    continue                                   # resume (e.g. R built before H finished downloading)
                raw = np.stack([np.asarray(Image.open(io.BytesIO(fi.read(ims[z])))) for z in blk])
                lab = np.stack([tifffile.imread(io.BytesIO(fl.read(labs[z]))) for z in blk]).astype(np.int64)
                r = write_volume(f"mitoem_{sp}", f"z{blk[0]:04d}-{blk[-1]:04d}", raw, lab, label_scope="organelle:mito", resolution_nm=[30, 8, 8],
                                 modality="multi-beam SEM (ATUM)", organism=org, tissue=f"{org} cortex (MitoEM-{sp})",
                                 source_url="https://mitoem.grand-challenge.org/", license="CC BY-NC 4.0", z_index=blk)
                r.pop("slices", None); out.append(r)
    return out


JOBS["mitoem"] = job_mitoem


def job_uro_cell():
    """UroCell (FIB-SEM, mouse urothelium, ~16x16x15 nm, eight 256^3 sub-volumes). One multi-class file per sub-volume:
    mitochondrion and fusiform_vesicle instance labels as given, endolysosome binary -> CC; a class is present only in
    the sub-volumes where it was annotated. Golgi (approximate regions) not used. nii (x,y,z) -> (z,y,x)."""
    import nibabel as nib
    d = RAW / "uro_cell/UroCell-master"
    if not d.exists():
        import zipfile; zipfile.ZipFile(RAW / "uro_cell/UroCell-master.zip").extractall(RAW / "uro_cell")
    load = lambda p: np.asarray(nib.load(str(p)).dataobj).transpose(2, 1, 0)
    out = []
    for p in sorted((d / "data").glob("*.nii.gz")):
        labels = {}
        for cls, sub, name in (("mito", "instance", "mitochondrion"), ("fv", "instance", "fusiform_vesicle"), ("lyso", "binary", "endolysosome")):
            q = d / cls / sub / p.name
            if q.exists():
                l = load(q); labels[name] = (ndi.label(l > 0)[0] if sub == "binary" else np.rint(l)).astype(np.int64)
        if not labels:
            continue
        r = write_volume_mc("uro_cell", p.name.replace(".nii.gz", ""), load(p), labels, resolution_nm=[15, 16, 16], modality="FIB-SEM",
                            organism="mouse", tissue="urinary bladder umbrella cells (UroCell)", source_url="https://github.com/MancaZerovnikMekuc/UroCell",
                            license="CC BY 4.0")
        r.pop("slices", None); out.append(r)
    return out


def job_emps():
    """EMPS (Yildirim & Cole 2021): 465 EM images of nanoparticles taken from publications, instance segmaps (uint16)."""
    d = RAW / "emps/emps-main"
    if not d.exists():
        import zipfile; zipfile.ZipFile(RAW / "emps/emps-main.zip").extractall(RAW / "emps")

    def items():
        for p in sorted((d / "images").glob("*.png")):
            m = d / "segmaps" / p.name
            if m.exists():
                yield p.stem, np.asarray(Image.open(p)), np.asarray(Image.open(m)).astype(np.int64)
    return [write_images("emps", items(), label_scope="organelle:nanoparticle", resolution_nm=None, modality="TEM / SEM (figures from publications)",
                         organism="none (materials science)", tissue="nanoparticles", source_url="https://github.com/by256/emps", license="MIT")]


JOBS.update({"uro_cell": job_uro_cell, "emps": job_emps, "nuclei": job_nuclei})


def job_fib25():
    """FIB-25 FFN release: groundtruth 'stack' holds supervoxel ids; 'transforms' (sv, body) maps them to bodies (applied
    here, unmapped supervoxels keep their id); tstvol-520-1 has body ids directly. 8 nm isotropic."""
    out = []
    for n in ("training_sample2", "validation_sample", "tstvol-520-1"):
        with h5py.File(RAW / f"fib25/{n}_raw.h5", "r") as f:
            raw = np.squeeze(f["raw"][:])
        with h5py.File(RAW / f"fib25/{n}_gt.h5", "r") as f:
            st = np.squeeze(f["stack"][:]).astype(np.int64)
            if "transforms" in f:
                t = f["transforms"][:].astype(np.int64)
                u, inv = np.unique(st, return_inverse=True)
                lut = dict(zip(t[:, 0].tolist(), t[:, 1].tolist()))
                st = np.array([lut.get(int(x), int(x)) for x in u], np.int64)[inv].reshape(st.shape)
        r = write_volume("fib25", n, raw, st, label_scope="dense_cells", resolution_nm=[8, 8, 8], modality="FIB-SEM", organism="drosophila",
                         tissue="optic lobe medulla (FIB-25)", source_url="https://storage.googleapis.com/ffn-flyem-fib25", license="see FlyEM",
                         extra_attrs={"note": "supervoxel -> body transforms applied"})
        r.pop("slices", None); out.append(r)
    return out


JOBS["fib25"] = job_fib25



# ---------------------------------------------------------------- micro-SAM-table datasets (Zenodo / Google Drive)
def _extract(archive: Path, dest: Path) -> Path:
    """unpack an archive once into its own directory (written to <dest>.tmp, renamed when complete -> never half-extracted);
    RAR5 archives (AimSeg) need libarchive (bsdtar), 7z/unrar from conda cannot read them."""
    if not dest.exists():
        tmp = dest.with_name(dest.name + ".tmp"); tmp.mkdir(parents=True, exist_ok=True)
        if archive.suffix == ".zip":
            import zipfile; zipfile.ZipFile(archive).extractall(tmp)
        else:
            subprocess.run(["bsdtar", "-xf", str(archive), "-C", str(tmp)], check=True)
        os.replace(tmp, dest)
    return dest


def _n5(path: Path, key: str) -> np.ndarray:
    """N5 dataset -> numpy (Z,Y,X). N5 stores dimensions fastest-first (x,y,z); tensorstore returns that order."""
    import tensorstore as ts
    a = ts.open({"driver": "n5", "kvstore": {"driver": "file", "path": str(path / key)}}).result().read().result()
    return np.transpose(a, (2, 1, 0))


def job_platynereis():
    """Vergara et al. 2021 training data (Zenodo 3675220), read as in torch_em:
    cells   : membrane/train_data_membrane_NN.n5, volumes/raw/s1 + volumes/labels/segmentation/s1 (s1 = 2x xy downsampling of
              the 10x10x25 nm SBEM -> 20x20x25 nm). The label covers a centred box (raw has context around it): raw and labels
              are cropped to the 3D bounding box of label > 0; checked aligned (boundary contrast best at shift 0 in all 9).
    nuclei  : nuclei/train_data_nuclei_NN.h5, volumes/raw + volumes/labels/nucleus_instance_labels; -1 = not annotated
              (outer slices / margin) -> annotated mask, 0 = background.
    cilia   : cilia/train_data_cilia_0[1-3].h5, volumes/raw + volumes/labels/segmentation (val_* files carry no labels)."""
    d = RAW / "platynereis"; out = []
    common = dict(modality="SBF-SEM", organism="Platynereis dumerilii", source_url="https://zenodo.org/record/3675220", license="CC BY 4.0")
    mem = _extract(d / "membrane.zip", d / "x_membrane")
    for p in sorted(mem.rglob("train_data_membrane_*.n5")):
        raw, lab = _n5(p, "volumes/raw/s1"), _n5(p, "volumes/labels/segmentation/s1").astype(np.int64)
        assert raw.shape == lab.shape, (p, raw.shape, lab.shape)
        nz = np.nonzero(lab > 0); sl = tuple(slice(int(a.min()), int(a.max()) + 1) for a in nz)
        r = write_volume_mc("platynereis_cells", p.stem, raw[sl], {"cell": lab[sl]}, resolution_nm=[25, 20, 20],
                            tissue="whole larva (6 dpf), cell membranes", extra_attrs={"crop_zyx": json.dumps([[s.start, s.stop] for s in sl]),
                                                                                       "scale": "s1"}, **common)
        r.pop("slices", None); out.append(r)
    nuc = _extract(d / "nuclei.zip", d / "x_nuclei")
    for p in sorted(nuc.rglob("train_data_nuclei_*.h5")):
        with h5py.File(p, "r") as f:
            raw, lab = f["volumes/raw"][:], f["volumes/labels/nucleus_instance_labels"][:].astype(np.int64)
        ann = lab >= 0; lab[~ann] = 0
        r = write_volume_mc("platynereis_nuclei", p.stem, raw, {"nucleus": lab}, ann_mask=ann, resolution_nm=[-1, -1, -1],
                            tissue="whole larva, nuclei", extra_attrs={"note": "pixel size not stated in the release (downsampled SBEM)"}, **common)
        r.pop("slices", None); out.append(r)
    cil = _extract(d / "cilia.zip", d / "x_cilia")
    for p in sorted(cil.rglob("train_data_cilia_*.h5")):
        with h5py.File(p, "r") as f:
            raw, lab = f["volumes/raw"][:], f["volumes/labels/segmentation"][:].astype(np.int64)
        r = write_volume_mc("platynereis_cilia", p.stem, raw, {"cilium": lab}, resolution_nm=[-1, -1, -1], tissue="whole larva, cilia",
                            extra_attrs={"note": "pixel size not stated in the release"}, **common)
        r.pop("slices", None); out.append(r)
    return out


def job_sponge_em():
    """Musser et al. 2021 (Zenodo 8150818): 3 FIB-SEM volumes of choanocyte chambers. volumes/labels/instances (-1 = not
    annotated) + volumes/labels/semantic (0 background, 1 cell, 2 cilium, 3 microvillus; order of the torch_em docstring
    'cells, cilia and microvilli', consistent with the counts: few cells / cilia, hundreds of microvilli). Each instance gets
    the majority semantic class of its pixels; instances of class 0 (one large background id in volumes 1-2) are background."""
    d = _extract(RAW / "sponge_em/sponge_em_train_data.zip", RAW / "sponge_em/x"); out = []
    for p in sorted(d.rglob("train_data_*.h5")):
        with h5py.File(p, "r") as f:
            raw, ins, sem = f["volumes/raw"][:], f["volumes/labels/instances"][:], f["volumes/labels/semantic"][:]
        ann = ins >= 0
        ids, inv = np.unique(np.where(ann, ins, 0), return_inverse=True); inv = inv.reshape(ins.shape)
        votes = np.zeros((ids.size, 4), np.int64)
        np.add.at(votes, (inv[ann], np.clip(sem[ann], 0, 3)), 1)
        cls_of = votes.argmax(1); cls_of[ids == 0] = 0
        labels = {}
        for k, name in ((1, "cell"), (2, "cilium"), (3, "microvillus")):
            labels[name] = np.where(cls_of[inv] == k, ins, 0).astype(np.int64)
        r = write_volume_mc("sponge_em", p.stem, raw, labels, ann_mask=ann, resolution_nm=[-1, -1, -1], modality="FIB-SEM",
                            organism="Spongilla lacustris", tissue="choanocyte chamber", source_url="https://zenodo.org/record/8150818",
                            license="CC BY 4.0", extra_attrs={"note": "cell = annotated cell bodies, not an exhaustive tiling (density rule decides)"})
        r.pop("slices", None); out.append(r)
    return out


def _mitoemv2_one(args):
    import nibabel as nib
    ddir, sample, split, spacing, modality = args
    raw = np.asarray(nib.load(str(ddir / "imagesTr" / f"{sample}_0000.nii.gz")).dataobj).transpose(2, 1, 0)
    lab = np.asarray(nib.load(str(ddir / "labelsTr" / f"{sample}.nii.gz")).dataobj)
    lab = lab.transpose(2, 1, 0) if lab.transpose(2, 1, 0).shape == raw.shape and sample != "me2-jurkat_train02" else lab
    assert lab.shape == raw.shape, (sample, lab.shape, raw.shape)     # jurkat_train02: labels stored (z,y,x) already (as torch_em)
    name = ddir.name.split("_ME2-")[1].lower()
    r = write_volume_mc(f"mitoemv2_{name}", sample, raw, {"mitochondrion": lab.astype(np.int64)},
                        resolution_nm=[spacing[2], spacing[1], spacing[0]], modality=modality, organism="see MitoEM v2",
                        tissue=f"MitoEM v2 {name}", source_url="https://zenodo.org/records/17635006", license="CC BY 4.0",
                        extra_attrs={"source_split": split})
    r.pop("slices", None)
    return r


def job_mitoemv2():
    """MitoEM v2 (Zenodo 17635006): 8 nnU-Net datasets (imagesTr/labelsTr/imagesTs/labelsTs + split.json). Instance labels
    (uint16). Only the train + val samples of split.json are used; the test samples (labelsTs) are held out so MitoEM v2
    stays usable as an external benchmark. NIfTI (x,y,z) -> (z,y,x); spacing from dataset.json."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for z in sorted((RAW / "mitoemv2").glob("Dataset*.zip")):
        ddir = next(_extract(z, RAW / "mitoemv2" / f"x_{z.stem}").glob("Dataset*"))
        meta = json.load(open(ddir / "dataset.json"))
        split = json.load(open(next(ddir.glob("split*.json"))))[0]
        for sp in ("train", "val"):
            for sample in split.get(sp, []):
                jobs.append((ddir, sample, sp, meta["spacing"], meta["channel_names"]["0"]))
    with ProcessPoolExecutor(8) as ex:
        return list(ex.map(_mitoemv2_one, jobs))


def job_aimseg():
    """AimSeg (Zenodo 8351731): 8 TEM images of mouse corpus callosum (Control 3 + Validation 5) with GroundTruth_Instance and
    GroundTruth_Semantic (0 background, 1 axon, 2 inner tongue, 3 myelin). Each instance gets the majority semantic class:
    1 -> axon (myelinated axon interiors; not an exhaustive cell tiling, hence its own class), 3 -> myelin; inner-tongue
    instances are left as background of both classes."""
    import tifffile

    def items():
        for rar in sorted((RAW / "aimseg").glob("*.rar")):
            d = _extract(rar, RAW / "aimseg" / f"x_{rar.stem}")
            for p in sorted(d.rglob("Images/*.tif")):
                ins = tifffile.imread(str(p).replace("/Images/", "/GroundTruth_Instance/")).astype(np.int64)
                sem = tifffile.imread(str(p).replace("/Images/", "/GroundTruth_Semantic/"))
                ids, inv = np.unique(ins, return_inverse=True); inv = inv.reshape(ins.shape)
                votes = np.zeros((ids.size, 4), np.int64); np.add.at(votes, (inv, np.clip(sem, 0, 3)), 1)
                c = votes.argmax(1); c[ids == 0] = 0
                yield f"{rar.stem}/{p.stem}", tifffile.imread(p), {"axon": np.where(c[inv] == 1, ins, 0), "myelin": np.where(c[inv] == 3, ins, 0)}
    return [write_images("aimseg", items(), modality="TEM", organism="mouse", tissue="corpus callosum (AimSeg)",
                         source_url="https://zenodo.org/records/8351731", license="CC BY 4.0", classes=["axon", "myelin"])]


def job_nucmm_z():
    """NucMM-Z (Lin et al. 2021): zebrafish whole-brain SBF-SEM, 64^3 blocks Image/{train,val}/img_*.h5 + Label/.../seg_*.h5
    (key 'main', nucleus instances). Only blocks whose label file was obtained are converted (Google Drive quota limits the
    download; the count is in the build summary). NucMM-M (micro-CT, not EM) is not used."""
    d = RAW / "nuc_mm/Zebrafish (NucMM-Z)"; out = []
    for lp in sorted(d.glob("Label/*/seg_*.h5")):
        ip = d / "Image" / lp.parent.name / lp.name.replace("seg_", "img_")
        if not ip.exists():
            continue
        with h5py.File(ip, "r") as f:
            raw = f["main"][:]
        with h5py.File(lp, "r") as f:
            lab = f["main"][:].astype(np.int64)
        r = write_volume_mc("nucmm_z", f"{lp.parent.name}_{lp.stem[4:]}", raw, {"nucleus": lab}, resolution_nm=[480, 510, 510],
                            modality="SBF-SEM (downsampled)", organism="zebrafish", tissue="larval whole brain (NucMM-Z)",
                            source_url="https://drive.google.com/drive/folders/1_4CrlYvzx0ITnGlJOHdgcTRgeSkm9wT8", license="CC BY 4.0",
                            extra_attrs={"source_split": lp.parent.name})
        r.pop("slices", None); out.append(r)
    return out


def job_probtem():
    """ProbTEM (Yoon lab; Sci. Rep. 2025): 2560^2 TEM of skeletal muscle (gray stored as RGB), binary mitochondria masks
    (PNG, anti-aliased) -> threshold >= 127 (as torch_em) -> 4-connected components = instances (touching mitochondria
    merge: semantic-to-instance conversion, noted in attrs). Only images whose mask was obtained and has the image size are
    converted."""
    d = RAW / "probtem/mitoseg_deploy/datasets/Skeletal_muscle"
    import tifffile

    def items():
        for p in sorted(d.glob("*/input/x_*.tif")):
            m = p.parent.parent / "target" / (p.stem.replace("x_", "y_") + ".png")
            if not m.exists():
                continue
            raw = tifffile.imread(p); raw = raw[..., 0] if raw.ndim == 3 else raw
            t = np.asarray(Image.open(m)); t = t[..., 0] if t.ndim == 3 else t
            if t.shape != raw.shape:            # test/y_5.png is 1280^2 for a 2560^2 image: not a verified pair -> skipped
                print("probtem: mask/image size mismatch, skipped", p.name, t.shape, raw.shape, flush=True); continue
            yield f"{p.parent.parent.name}/{p.stem}", raw, ndi.label(t >= 127, S4)[0].astype(np.int64)
    return [write_images("probtem", items(), label_scope="organelle:mito", modality="TEM", organism="mouse (skeletal muscle)",
                         tissue="skeletal muscle", source_url="https://yoonlab.unist.ac.kr/index.php/research/mitochondria-tem-dataset/",
                         license="see source", extra_attrs={"instances_from": "connected components of the binary mask"})]


JOBS.update({"platynereis": job_platynereis, "sponge_em": job_sponge_em, "mitoemv2": job_mitoemv2, "aimseg": job_aimseg,
             "nucmm_z": job_nucmm_z, "probtem": job_probtem})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("jobs", nargs="+"); a = ap.parse_args()
    keys = list(JOBS) if a.jobs == ["all"] else a.jobs
    log = DATA_ROOT / "_build_logs"; log.mkdir(parents=True, exist_ok=True); failed = False
    for k in keys:
        try:
            res = JOBS[k](); err = None
            for r in res if isinstance(res, list) else []:
                r.pop("slices", None)
        except Exception:
            res, err = None, traceback.format_exc()
        print(json.dumps({"job": k, "result": res, "error": err}, default=str)[:1500], flush=True)
        json.dump({"job": k, "result": res, "error": err}, open(log / f"dl_{k}.json", "w"), indent=1, default=str)
        failed = failed or err is not None
    print("exit=1" if failed else "exit=0")
    sys.exit(1 if failed else 0)               # a converter exception must not be logged as 'done' by emseg.data.stages.convert


if __name__ == "__main__":
    main()
