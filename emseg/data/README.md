# 📦 emseg.data — the training corpus

`emseg.data` turns 30 public EM releases into **one standard**: per section a uint8 image, per annotated class a 2D instance
map plus a validity mask, with pixel size, licence and provenance attached. The result (EMDATA-v2) is 57 datasets and
68 k labelled sections (gold = expert, silver = automatic connectome segmentation), quality-checked, de-duplicated across
sources and scanned for overlap with the evaluation benchmarks. The same conventions are used for the unlabelled pool.

There are two ways to get the data:

- **🛠️ A. Build it yourself from the public releases** — step-by-step below. Every file is downloaded from its publisher,
  checksummed and logged, so the corpus is fully reproducible and you accept each source's licence yourself.
- **📬 B. Get what we are allowed to pass on** — the open-licence part of the processed corpus, the pseudo-labels and the
  checkpoints can be shared on request; sources with non-redistributable licences and the in-house volume cannot, and are
  rebuilt from the publishers with one command. [Details](#b-get-the-processed-corpus-from-us).

---

## A. Build from scratch, step by step

### 0. Before you start

- `conda activate em2d-data` (created by `scripts/00_setup_envs.sh` from `envs/data.yml`; Python 3.11 with h5py,
  tifffile, scikit-image, cloud-volume, zarr, nibabel, gdown, `bsdtar` from libarchive for RAR5 archives).
- `source scripts/env.sh` after setting `EMDATA_ROOT` to a disk with ≥ 300 GB free. The pipeline writes only under this root.
- Network: all sources are public and anonymous. Zenodo and EBI throttle parallel connections (handled), Google-Drive
  folders are quota-limited (NucMM-Z arrives partially; ProbTEM is off by default), cloud connectomes are read directly
  from the public buckets (no download, see § 3).

All stages are **resumable and idempotent**: re-running skips what is logged as done and present on disk. The single source of
truth for every dataset (URLs, checksums, licence, converter, leakage rules) is `emseg/data/registry.py`; the stages only read it.

### 1. One command for everything

```bash
tools/detach.sh DATA-corpus scripts/10_build_corpus.sh      # all EM sources, every stage; log in $RUNS_ROOT/DATA-corpus/logs/stdout.log
tools/status.sh DATA-corpus
cat $EMDATA_ROOT/STATUS.md                                   # per-source download / conversion state, refreshed after each stage
```

`WITH_COCO=1` additionally fetches COCO panoptic (21 GB) for the optional `em_plus_natural` manifest; the teacher and the student
use `em_only.csv` and do not need it. The individual stages, if you want to run or re-run them one by one:

### 2. Download — `python -m emseg.data.stages.download [source_id …]`

```bash
python -m emseg.data.stages.download                      # every default source (~88 GB, 381 files)
python -m emseg.data.stages.download cremi snemi3d emneuron            # or a subset
```

Each file is fetched by `tools/pdl.py` (parallel HTTP range requests, resumable through `<file>.parts/spans.json`, size
checked against `Content-Length`), then hashed; when the publisher gives a checksum it must match (a mismatch keeps the file
as `.corrupt`). Host policy: zenodo.org one connection and one file at a time, ftp.ebi.ac.uk two, everything else six;
EMPIAR / BioStudies file lists are read from the server at download time. Everything is appended to
`$EMDATA_ROOT/pipeline_log.jsonl` and summarised in `provenance/downloads.csv` (URL, bytes, sha256, verification, time).
Raw files land in `$EMDATA_ROOT/raw_downloads/`; they may be deleted after conversion.

### 3. Convert — `python -m emseg.data.stages.convert [source_id …]`

Every converter runs as its own subprocess (a failing reader cannot corrupt another dataset) and applies the common standard
(`emseg/data/standard.py`):

1. grey values: per volume, P0.5 / P99.5 percentiles mapped linearly to 0–255 (uint8); the membrane polarity is only recorded, never flipped;
2. labels: per section, 4-connected components of the source ids, renumbered from 1 — one 3D object cut in two on a section is two 2D instances, which is what per-section evaluation measures;
3. fragments < 10 px become *invalid* (mask 0) rather than background;
4. sections that are mostly padding (< 25 % valid), blank (std < 2) or without instances are dropped; dense classes need ≥ 30 % labelled area and ≥ 3 instances;
5. leakage rules from the registry (SNEMI only AC4 z0–79, CREMI only z0–74, EMNeuron copies of those benchmarks skipped).

Output: `$EMDATA_ROOT/em/<dataset>/<volume>.h5` (`raw` (Z,H,W), `labels/<class>`, `masks/<class>`, `label_src/<class>`,
attributes with source, licence, pixel size, kept / dropped sections) and `em2d/<dataset>/shard_*.h5` for 2D collections
(`<key>/raw`, `<key>/labels/<class>`, …). Streaming sources (`stream_*`, `kharris15`, `asem`) have no raw download: the
converter cuts random 1024² × 8 slabs with labels straight from the public buckets under a watchdog (restarts after 8 min
without a new file; the fixed seed makes the slab sequence reproducible). Preview montages: `python -m emseg.data.qc`.

### 4. Quality metrics — `python -m emseg.data.stages.qa [--workers N]`

Per image: valid fraction, intensity statistics, saturation, padding, Laplacian variance (blur), per-class instance count /
coverage / largest-instance share / median area, boundary contrast of dense labels (EM membranes are dark; bright boundaries
mean misaligned labels), md5 and a 64-bit perceptual hash. → `qa/images.csv`.

### 5. De-duplication and leakage scan — `python -m emseg.data.stages.dedup [--workers N]`

(a) exact duplicates (md5); (b) near duplicates (pHash ≤ 6 bits across volumes, confirmed by thumbnail correlation);
(c) region overlap between datasets (band-passed normalised cross-correlation of 1.5 µm patches at a common 32 nm scale —
catches the same tissue re-released at another crop or resolution); (d) the same scan with the **benchmark validation / test
sections as queries** (SNEMI AC4 z80–99 + AC3, CREMI z75–124) against every dataset. Decisions keep the higher-priority
dataset (expert dense > expert organelle > automatic). → `dedup/*.csv`, `dedup/decisions.json`.

### 6. Manifests — `python -m emseg.data.stages.manifest`

Applies the filters F1–F7 (too small, saturated, blurred, degenerate labels, labels not on membranes, duplicate, excluded
dataset; `qa/filter_log.csv` says why each image went), makes a **group-aware split** (validation = whole volumes or the last
5 % of a volume behind a 2-section gap; 2D collections by source group) and sampling weights (dataset mass ∝ √(labelled
pixels), silver × 0.5, spread uniformly over the dataset's images). → `manifests/em_only.csv`, `manifests/em_plus_natural.csv`,
`manifests/summary.json`.

| column | meaning |
|---|---|
| `uid, group, dataset, file, key, z, z_src` | identity; `file` relative to `$EMDATA_ROOT`; `key` for 2D shards; `z` section index in the HDF5, `z_src` in the source |
| `H, W, res_xy_nm, res_z_nm` | size and pixel size (NaN when the release does not state it) |
| `classes, n_inst, quality` | annotated classes (`cell|mitochondrion`), instance count, gold / silver |
| `split, weight` | train / val / gap; sampling probability of train rows (sums to 1) |

**Partial-label convention:** a class that is not listed in `classes` is *unknown* on that image, not background — the
background of a mitochondria dataset is full of unlabelled cells. The trainers only ever use `labels/cell` where `cell` is annotated.

### 7. Report — `python -m emseg.data.stages.report [--checksums]`

Recomputes `STATUS.md` (live state per source), `datasets.csv` (machine-readable composition), `README.md` (the data card with
every source, link, licence and per-dataset counts) and optionally `checksums.sha256` of every output file.

### 8. Optional: Hugging Face export — `python -m emseg.data.stages.export_hf [--configs …]`

Writes Parquet shards with typed `datasets.Features` (image + per-class instance PNGs + masks), a data card and a sampler
under `$EMDATA_ROOT/hf/emdata-v2/`, in licence-tiered configs: `em_open` (redistributable sources only), `em_nc`
(non-commercial), `em_full` (all EM rows, local use), `em_plus_natural` (+ COCO). Only `em_open` may be published.

### 9. The unlabelled pool

`scripts/11_build_unlab_pool.sh` runs `emseg/data/unlab.py`, which cuts random **raw-only** slabs (1024² × 4 sections, ~8 nm/px) from 42 public connectome volumes with
anonymous cloud-volume reads: 6 built-in sources (hemibrain, MANC, male CNS, H01, MICrONS minnie65, FlyWire; 1 500 slabs each)
and 36 BossDB / other volumes listed with their licences in `configs/unlab_sources_public.json` (600 slabs each). A slab is
kept when it is not blank and, if the source has a segmentation, lies inside tissue. Output: `unlab/unlab_<source>/slab_*.h5`,
`manifests/unlab.csv` (same columns as `em_only.csv`, empty `classes`), `unlab/download_log.jsonl` (URL, mip, bounding box,
sha256 per slab). Volumes that overlap the benchmarks or the labelled corpus (Kasthuri S1, kharris15, FIB-25, J0126) are not in the list.

```bash
scripts/11_build_unlab_pool.sh                    # everything (~106 GB, about a day; one source at a time, resumable)
scripts/11_build_unlab_pool.sh hemibrain h01      # a subset
```

### 10. Benchmarks

`scripts/12_get_benchmarks.sh` fetches SNEMI3D AC4 (train / val) from the torch_em mirror, AC3 with labels from the EMNeuron release, CREMI sample B from cremi.org,
arranged under `$EM2DSEG_DATA_ROOT` as `em2dseg` expects. Files already in `raw_downloads/` are reused.

### 11. What to expect

| stage | time on 48 cores | result |
|---|---|---|
| download | hours to a day (throttled hosts, retries) | 88 GB raw, `provenance/downloads.csv` |
| convert | ~1.5 h + streams | 33 GB `em/`, 3 GB `em2d/` (+19 GB `natural/` with COCO) |
| qa | ~6 h | `qa/images.csv` |
| dedup | ~0.5 h | `dedup/` |
| manifest + report | minutes | `manifests/em_only.csv`: 62 419 train / 5 672 val sections, 57 datasets |
| pool | ~1 day | 106 GB, 121 824 sections, 42 sources |

Read a section:

```python
import h5py, pandas as pd
m = pd.read_csv(f"{EMDATA_ROOT}/manifests/em_only.csv", low_memory=False)
r = m[(m.dataset == "cremi") & (m.split == "train")].iloc[0]
with h5py.File(f"{EMDATA_ROOT}/{r.file}") as f:
    img, lab, valid = f["raw"][r.z], f["labels/cell"][r.z], f["masks/cell"][r.z]
```

### Troubleshooting

| symptom | cause / fix |
|---|---|
| zenodo.org 403 "unusual traffic" | parallel connections or a fake browser UA; the downloader already serialises Zenodo — wait and re-run |
| `gdown` "too many accesses" | Google-Drive quota; the stage retries after 15 min, partial folders are converted as far as they go |
| cremi.org certificate error | expired certificate on the server; the registry uses `--insecure` and verifies the published sha256 |
| RAR archives fail to extract | RAR5 needs `bsdtar` (libarchive); 7z / unrar from conda cannot read them |
| a stream source makes no progress | cloud reads can hang; the watchdog restarts after 8 min, slabs already written are kept |
| `waiting_download` in STATUS.md | some raw files of that source are missing; run `python -m emseg.data.stages.download <id>` again |

### Adding a dataset

Add one `Source(...)` to `emseg/data/registry.py` (name, content, pixel size, classes, licence, redistribution tier, files with
URLs and checksums, converter job, leakage rules) and a reader `job_<id>()` in `emseg/data/converters/downloaded.py` that returns
`write_volume(...)` / `write_images(...)` calls. Every later stage, the manifests and the data card pick it up automatically.

### Sources and licences

Redistribution tiers: **open** = CC0 / CC-BY / MIT (may be re-hosted); **nc** = non-commercial licences; **check** = collections
that inherit the licences of their original data (not verified per item); **restricted** = challenge / research-use terms
or third-party image rights. All of them may be *downloaded* from the publisher for research; only `open` sources may be
*redistributed*. Please cite the original publication of every dataset you use.

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
from the original publishers in the same layout. If you want to skip the parts that may be shared, **contact us** (open an
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
