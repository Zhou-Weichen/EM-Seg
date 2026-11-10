# 📦 emseg.data — the training corpus

`emseg.data` turns 30 public EM releases into one standard: per section a uint8 image and, per annotated class, a 2D instance
map with a validity mask, plus pixel size, licence and provenance. Result (EMDATA-v2): 57 datasets, 68 k labelled sections
(gold = expert, silver = automatic connectome segmentation), quality-checked, de-duplicated and scanned for benchmark leakage.

- **🛠️ A. Build it from the public releases** — step by step below; every file is checksummed and logged.
- **📬 B. Ask us for what may be shared** — open-licence subset, pseudo-labels, checkpoints; the rest is rebuilt with route A.
  [Details](#b-get-the-processed-corpus-from-us).

---

## A. Build from scratch

### 0. Before you start

- `conda activate em2d-data && source scripts/env.sh`; set `EMDATA_ROOT` to a disk with ≥ 300 GB free.
- All sources are public and anonymous. Zenodo / EBI throttle parallel connections (handled), Google-Drive folders are
  quota-limited, cloud connectomes are read directly from public buckets.
- All stages are resumable and idempotent. The single source of truth for every dataset (URLs, checksums, licence,
  converter, leakage rules) is `emseg/data/registry.py`.

### 1. One command

```bash
tools/detach.sh DATA-corpus scripts/10_build_corpus.sh      # all EM sources, every stage; WITH_COCO=1 adds COCO (21 GB)
tools/status.sh DATA-corpus;  cat $EMDATA_ROOT/STATUS.md    # progress per source
```

### 2–8. Stages one by one

| stage | command | what it does | output |
|---|---|---|---|
| download | `python -m emseg.data.stages.download [ids]` | parallel resumable HTTP (`tools/pdl.py`), size and sha256 checks, host-aware throttling, every event logged | `raw_downloads/`, `provenance/downloads.csv`, `pipeline_log.jsonl` |
| convert | `python -m emseg.data.stages.convert [ids]` | one subprocess per converter: P0.5/P99.5 normalisation to uint8, per-section 4-connected instances, fragments < 10 px masked, padding / blank sections dropped, leakage rules | `em/<dataset>/<volume>.h5`, `em2d/<dataset>/shard_*.h5` |
| qa | `python -m emseg.data.stages.qa` | per-image quality metrics (valid area, saturation, blur, instance statistics, boundary contrast, md5, pHash) | `qa/images.csv` |
| dedup | `python -m emseg.data.stages.dedup` | exact and perceptual duplicates, cross-dataset region overlap, scan with the benchmark val / test sections as queries | `dedup/*.csv`, `dedup/decisions.json` |
| manifest | `python -m emseg.data.stages.manifest` | filters F1–F7, group-aware train / val split, sampling weights | `manifests/em_only.csv`, `manifests/em_plus_natural.csv` |
| report | `python -m emseg.data.stages.report` | status, per-dataset table, data card, checksums | `STATUS.md`, `datasets.csv`, `README.md` |
| export_hf (optional) | `python -m emseg.data.stages.export_hf` | Parquet shards with typed features in licence-tiered configs (`em_open`, `em_nc`, `em_full`, `em_plus_natural`) | `hf/emdata-v2/` |

Streaming sources (`stream_*`, `kharris15`, `asem`) have no raw download: the converter cuts random labelled slabs from the
public buckets under a watchdog. Preview montages: `python -m emseg.data.qc`.

HDF5 layout: `raw` (Z,H,W) uint8, `labels/<class>` uint32, `masks/<class>` uint8, attributes with source, licence, pixel size.
Manifest columns: `uid, group, dataset, file, key, z, z_src, H, W, res_xy_nm, res_z_nm, classes, n_inst, quality, modality,
organism, split, weight`. **Partial-label convention:** a class absent from `classes` is *unknown* on that image, not background.

### 9. Unlabelled pool

```bash
scripts/11_build_unlab_pool.sh                    # 6 built-in connectomes (1500 slabs each) + 36 from configs/unlab_sources_public.json (600 each); ~106 GB, ~1 day
scripts/11_build_unlab_pool.sh hemibrain h01      # a subset
```

`emseg/data/unlab.py` streams raw 1024² × 4 slabs (~8 nm/px) with anonymous cloud-volume reads, keeps slabs inside tissue,
and logs URL, mip, bounding box and sha256 per slab (`unlab/download_log.jsonl`). Volumes overlapping the benchmarks or the
labelled corpus are excluded.

### 10. Benchmarks

`scripts/12_get_benchmarks.sh` fetches SNEMI3D AC4 (torch_em mirror), AC3 with labels (EMNeuron release) and CREMI B
(cremi.org) and arranges them under `$EM2DSEG_DATA_ROOT`; files already in `raw_downloads/` are reused.

### 11. What to expect

| stage | time (48 cores) | result |
|---|---|---|
| download | hours to a day | 88 GB raw |
| convert | ~1.5 h + streams | 33 GB `em/`, 3 GB `em2d/` |
| qa / dedup / manifest | ~7 h | `manifests/em_only.csv`: 62 419 train / 5 672 val sections, 57 datasets |
| pool | ~1 day | 106 GB, 121 824 sections, 42 sources |

```python
import h5py, pandas as pd
m = pd.read_csv(f"{EMDATA_ROOT}/manifests/em_only.csv", low_memory=False)
r = m[(m.dataset == "cremi") & (m.split == "train")].iloc[0]
with h5py.File(f"{EMDATA_ROOT}/{r.file}") as f:
    img, lab, valid = f["raw"][r.z], f["labels/cell"][r.z], f["masks/cell"][r.z]
```
### Adding a dataset

One `Source(...)` in `emseg/data/registry.py` and one reader `job_<id>()` in `emseg/data/converters/downloaded.py`
(returning `write_volume(...)` / `write_images(...)`); every later stage picks it up.

### Sources and licences

Tiers: **open** = CC0 / CC-BY / MIT (may be re-hosted); **nc** = non-commercial; **check** = inherits the licences of the
original data; **restricted** = challenge / research-use terms. All may be *downloaded* for research; only `open` may be
*redistributed*. Cite the original publication of every dataset you use.

| source | name | fetch | tier | licence | download GB | final images | default | homepage |
|---|---|---|---|---|---|---|---|---|
| `snemi3d` | SNEMI3D (ISBI 2013), Kasthuri et al. 2015 AC4/AC3 | http | restricted | challenge terms (research use) | 0.20 | 80 | yes | https://snemi3d.grand-challenge.org/ |
| `cremi` | CREMI 2016 samples A/B/C | http | open | CC-BY 4.0 | 0.52 | 223 | yes | https://cremi.org/data/ |
| `isbi2012` | ISBI 2012 EM segmentation challenge | http | restricted | challenge terms | 0.01 | 30 | yes | https://imagej.net/events/isbi-2012-segmentation-challenge |
| `vnc` | Drosophila larva VNC (Gerhard et al. 2013) | http | nc | CC BY-NC-SA 3.0 | 0.04 | 20 | yes | https://github.com/unidesigner/groundtruth-drosophila-vnc |
| `wafer4` | Wafer4 (CAD, CVPR 2024) | http | nc | CC BY-NC 4.0 | 0.20 | 125 | yes | https://github.com/liuxy1103/CAD |
| `fib25` | FIB-25 (FFN release) | http | check | see FlyEM | 0.30 | 1290 | yes | https://github.com/google/ffn |
| `emneuron` | EMNeuron (SegNeuron, MICCAI 2024) | http | check | MIT (+ licences of the original data) | 2.94 | 9878 | yes | https://huggingface.co/datasets/yanchaoz/EMNeuron |
| `kharris15` | SynapseWeb hippocampus CA1 (Harris et al. 2015) | stream | open | CC-BY 4.0 (Open NeuroData) | stream | 213 | yes | https://neurodata.io/ |
| `tumor_spheroid` | Tumour spheroid FIB-SEM 2D GT (S-BIAD3263) | listing | open | CC BY 4.0 | 0.04 | 20 | yes | https://www.ebi.ac.uk/biostudies/BioImages/studies/S-BIAD3263 |
| `platynereis` | Platynereis EM training data (Vergara et al. 2021) | http | open | CC-BY 4.0 | 6.41 | 2380 | yes | https://zenodo.org/record/3675220 |
| `sponge_em` | Sponge EM (Musser et al. 2021) | http | open | CC-BY 4.0 | 0.16 | 192 | yes | https://zenodo.org/record/8150818 |
| `stream_hemibrain` | FlyEM hemibrain v1.2 | stream | open | CC-BY 4.0 | stream | 384 | yes | gs://neuroglancer-janelia-flyem-hemibrain |
| `stream_manc` | FlyEM MANC v1.2 | stream | open | CC-BY 4.0 | stream | 384 | yes | gs://flyem-vnc-2-26-… / gs://manc-seg-v1p2 |
| `stream_malecns` | FlyEM male CNS v0.9 | stream | open | CC-BY 4.0 | stream | 384 | yes | gs://flyem-male-cns |
| `stream_h01` | H01 human cortex (c3 segmentation) | stream | open | CC-BY 4.0 | stream | 369 | yes | gs://h01-release |
| `stream_minnie65` | MICrONS minnie65 (seg m1300) | stream | open | CC-BY 4.0 | stream | 373 | yes | bossdb iarpa_microns/minnie/minnie65 |
| `stream_flywire` | FAFB v14 + FlyWire v783 | stream | open | CC-BY 4.0 | stream | 376 | yes | gs://flywire_v141_m783 |
| `lucchi` | EPFL hippocampus mitochondria (Lucchi et al. 2012) | http | restricted | EPFL CVLab (research use) | 0.52 | 330 | yes | https://www.epfl.ch/labs/cvlab/data/data-em/ |
| `cem_benchmarks` | CEM mitochondria benchmarks (EMPIAR-10982) | listing | open | CC0 (EMPIAR) | 4.42 | 1771 | yes | https://www.ebi.ac.uk/empiar/EMPIAR-10982/ |
| `cem_mitolab` | CEM-MitoLab (EMPIAR-11037) | http | open | CC0 (EMPIAR) | 2.96 | 20385 | yes | https://www.ebi.ac.uk/empiar/EMPIAR-11037/ |
| `mitoem` | MitoEM-R / MitoEM-H (Wei et al. 2020) | http | nc | CC BY-NC 4.0 | 28.80 | 983 | yes | https://mitoem.grand-challenge.org/ |
| `mitoemv2` | MitoEM v2 (2025) | http | open | CC-BY 4.0 | 17.53 | 14170 | yes | https://zenodo.org/records/17635006 |
| `asem` | ASEM mitochondria (Gallusser et al. 2023) | stream | open | CC-BY 4.0 | stream | 442 | yes | https://open.quiltdata.com/b/asem-project |
| `uro_cell` | UroCell (Mekuč et al. 2020) | http | open | CC BY 4.0 | 0.16 | 1280 | yes | https://github.com/MancaZerovnikMekuc/UroCell |
| `nuclei` | FAFB / MICrONS nuclei crops (torch_em) | http | check | see original publications | 1.90 | 10606 | yes | https://github.com/constantinpape/torch-em |
| `nucmm_z` | NucMM-Z (Lin et al. 2021) | gdown | open | CC-BY 4.0 | 0.41 | 1138 | yes | https://drive.google.com/drive/folders/1_4CrlYvzx0ITnGlJOHdgcTRgeSkm9wT8 |
| `probtem` | Mitochondria TEM (Yoon lab, ProbTEM) | gdown | check | see source | 0.24 | 0 | no | https://yoonlab.unist.ac.kr/index.php/research/mitochondria-tem-dataset/ |
| `aimseg` | AimSeg (2023) | http | open | CC-BY 4.0 | 0.02 | 8 | yes | https://zenodo.org/records/8351731 |
| `emps` | EMPS nanoparticles (Yildirim & Cole 2021) | http | open | MIT | 0.12 | 455 | yes | https://github.com/by256/emps |
| `coco_panoptic` | COCO panoptic 2017 | http | restricted | CC-BY 4.0 (annotations), Flickr terms (images) | 21.01 | 82337 | yes (`WITH_COCO=1`) | https://cocodataset.org/#panoptic-2020 |
| `mouse30um_local` | in-house mouse cortex (target domain) | local | restricted | in-house, not for redistribution | – | 0 | no | – |

The teacher's 22 datasets (cell-labelled, gold, after the label audit) draw 3 597 sections from `open` sources, 9 221 from
`check` (EMNeuron, FIB-25), 134 from `nc` and 26 from `restricted` (ISBI 2012) — which is why the complete teacher training
set can be *rebuilt* by anyone but not *re-hosted* by us.

---

## B. Get the processed corpus from us

We can only pass on what the licence of each source allows. Everything else is reproducible with route A, which downloads it
from the original publishers in the same layout. If you would rather receive the parts that may be shared, **contact us** (open an
issue on this repository or e-mail the maintainer <!-- add your address here -->) and tell us what you need.

| item | can we share it? | how |
|---|---|---|
| `em_open` config — 41 024 train / 3 827 val sections from CC0 / CC-BY / MIT sources (13 GB Parquet) | ✅ yes | access to our private Hugging Face dataset, for research use with attribution to each source |
| `em_nc` config — 1 124 sections from CC BY-NC(-SA) sources (VNC, Wafer4, MitoEM) | ✅ non-commercial research only | same repository, separate config |
| HDF5 layout + manifests of the same open / nc subset (what this code reads) | ✅ yes | archive on request |
| `check`-tier sources — EMNeuron (9 878 sections), FIB-25 (1 290), nuclei crops (10 606) | ❌ not by us | their licences inherit from the original data; `scripts/10_build_corpus.sh` fetches and converts them in under an hour |
| `restricted` sources — SNEMI3D, ISBI 2012, Lucchi, COCO images | ❌ not by us | challenge / research-use terms; download from the publisher pages listed in the table above |
| `em_full` / `em_plus_natural` configs | ❌ no | they contain the two groups above; the pipeline builds them locally |
| unlabelled pool — 121 824 raw slabs from 42 public connectome volumes | ✅ the slab list, raw slabs on request where the volume licence allows (CC-BY FlyEM, MICrONS, H01) | `download_log.jsonl` records bucket, mip, bounding box and sha256 of every slab, so `emseg.data.unlab` re-streams an identical pool |
| teacher pseudo-labels for that pool (130 k sections, 35 GB per teacher checkpoint) | ✅ yes, research use | label maps only, keyed to the slab list |
| teacher / student checkpoints | ✅ yes, research use only | trained on data that includes non-commercial and research-use-only sources |
| the in-house target volume used for the fine-tuning numbers | ❌ never | — |

Note that the teacher's 22 gold datasets draw 3 597 sections from `open` sources but 9 221 from `check` (EMNeuron, FIB-25),
134 from `nc` and 26 from `restricted` (ISBI 2012): the exact teacher training set can be **rebuilt** by anyone with route A,
but we cannot hand it over as a whole.

Once you have access to the Hugging Face repository:

```bash
pip install -U "huggingface_hub[cli]" && hf auth login
hf download <repo-id> --repo-type dataset --include "data/em_open/*" README.md classes.json sampler.py --local-dir $EMDATA_ROOT/hf/emdata-v2
python -c "from datasets import load_dataset; ds = load_dataset('parquet', data_files='$EMDATA_ROOT/hf/emdata-v2/data/em_open/train-*.parquet'); print(ds)"
```

`sampler.py` in the repository shows weighted sampling and the decoding of the per-class instance PNGs. The training
scripts here consume the HDF5 layout (`em/`, `em2d/`, `manifests/`); ask for that layout if you want to retrain on the open
subset, and run route A to add the sources we cannot redistribute.
