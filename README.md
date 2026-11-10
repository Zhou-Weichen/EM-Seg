# 🧠 EM-Seg

**Teacher–student pseudo-label pretraining for dense 2D instance segmentation of electron microscopy.**

A network predicts foreground + affinities (1 / 3 / 9 / 27 px); a mutex watershed turns them into per-section instances.
A large **teacher** is trained on 22 public expert-labelled datasets, labels ~130 k unlabelled sections from public
connectomes, and a **student** learns from gold and pseudo-labels together. Everything is reproducible from public data,
one command per stage.

<p align="center"><img src="docs/overview.png" width="100%" alt="overview: public EM corpus, teacher, pseudo-labels, student"></p>

## ✨ Contents

| part | role | docs |
|---|---|---|
| 📦 `emseg/data/` | corpus from 30 public releases (57 datasets, 68 k labelled sections) and the unlabelled pool (42 connectomes, 122 k sections) | [emseg/data/README.md](emseg/data/README.md) |
| 🧑‍🏫 🎓 `emseg/train/`, `emseg/models/`, `emseg/{pseudolabel,finetune,probe}.py` | teacher, pseudo-labels, student, zero-shot evaluation, fine-tuning, probes | [docs/training.md](docs/training.md) |
| 📏 `emseg/eval/`, `emseg/seg/` | metrics (VOI, ARAND, mSA, F1/PQ, error classes), benchmark domains, affinity targets, mutex-watershed decoding | [docs/evaluation.md](docs/evaluation.md) |
| 🚀 `scripts/` · 🧰 `tools/` | one resumable command per stage · detached jobs, auto-resume, downloader, figures | [tools/README.md](tools/README.md) |
| 📈 `docs/` | overview figure, results tables | [docs/results.md](docs/results.md) |

## 🗺️ Layout

```
EM-Seg/
├── emseg/                  the package  (pip install -e .  or  PYTHONPATH=<repo>)
│   ├── data/               registry · standard · pipeline · stages/ · converters/ · unlab      (data env, no torch)
│   ├── eval/               metrics · gt · evaluate · errclass · cli (emseg-eval) · domains/{snemi,cremi,mouse30um}
│   ├── seg/                targets · mws · crops · unet · lsd · prior · superpixels
│   ├── models/             pretrainnet (ConvNeXt-V2 + full-res FPN + heads) · features
│   ├── train/              data (crop sampler) · relations (pair targets, contrastive loss) · teacher_student (trainer)
│   ├── pseudolabel.py  finetune.py  probe.py  evalsuite.py       python -m emseg.<name>
│   └── paths.py            EMDATA_ROOT / RUNS_ROOT / EM2DSEG_DATA_ROOT
├── scripts/                00_setup_envs  10_build_corpus  11_build_unlab_pool  12_get_benchmarks
│                           20_train_teacher  30_pseudolabel  40_train_student  50_evaluate  60_visualize  env.sh
├── tools/                  detach.sh  status.sh  keepalive.sh  pdl.py  pseudo_stats.py  pseudo_vis.py  vis_predictions.py  label_audit.py
├── configs/                unlab_sources_public.json
├── tests/                  32 unit tests, no data needed
├── envs/                   train.yml  data.yml
└── docs/                   overview.png/.pdf  results.md  training.md  evaluation.md
```

Dependencies are one-directional: `eval` → `paths`, `seg` → `eval`, `models` → `seg`, `train` → `models` + `seg`;
`data` stands alone. `data/` and `runs/` (corpus, pool, pseudo-labels, checkpoints) live outside git.

---

## ⚙️ 1. Setup

```bash
git clone https://github.com/Zhou-Weichen/EM-Seg.git && cd EM-Seg
bash scripts/00_setup_envs.sh            # conda envs: em2d-train (torch, timm, python-elf, torch_em) · em2d-data (cloud-volume, zarr, datasets)
conda activate em2d-train && source scripts/env.sh
python -m unittest discover -s tests     # 32 tests
```

| variable (set by `scripts/env.sh`) | default | holds |
|---|---|---|
| `EMDATA_ROOT` | `<repo>/data` | corpus, pool, pseudo-labels, benchmarks |
| `RUNS_ROOT` | `<repo>/runs` | one directory per run: `config.json`, `log.jsonl`, `ckpt/`, `pred/`, `done.json` |
| `EM2DSEG_DATA_ROOT` | `$EMDATA_ROOT/benchmarks` | SNEMI / CREMI evaluation volumes |

- Export `EMDATA_ROOT` / `RUNS_ROOT` to a large disk first: ~300 GB (90 GB raw downloads, 40 GB corpus, 106 GB pool, 35 GB per pseudo-label pass).
- Reference hardware: 2 × 96 GB GPUs (batch 16 per GPU, ~43 GB for ConvNeXt-V2-L). Smaller GPUs: lower `--batch`; the learning rate scales with the global batch.
- Tested with Python 3.14, torch 2.13, timm 1.0.29, python-elf 0.9.2, torch_em 0.10.4; `pyproject.toml` lists the same dependencies as extras `[train]` / `[data]`.

---

## 📦 2. Data

**Route A — build from the public releases** (every byte traceable to a public URL):

```bash
conda activate em2d-data && source scripts/env.sh
tools/detach.sh DATA-corpus scripts/10_build_corpus.sh      # 30 releases → 57 datasets; resumable; a day or more
tools/detach.sh DATA-pool   scripts/11_build_unlab_pool.sh  # 42 connectomes → 122 k unlabelled sections; ~1 day
scripts/12_get_benchmarks.sh                                # SNEMI AC4/AC3 + CREMI B (0.9 GB)
tools/status.sh DATA-corpus                                 # progress; also $EMDATA_ROOT/STATUS.md
```

**Route B — ask us for what may be shared**: the open-licence subset (CC0 / CC-BY / MIT, 41 k sections), the pool slab list,
pseudo-labels and checkpoints, for research use. Non-redistributable sources (EMNeuron, FIB-25, SNEMI3D, ISBI 2012, Lucchi,
COCO) and the in-house target volume are not shared; route A rebuilds the former.
→ [emseg/data/README.md § B](emseg/data/README.md#b-get-the-processed-corpus-from-us)

| result | contents |
|---|---|
| `$EMDATA_ROOT/manifests/em_only.csv` | one row per labelled section: file, z, size, pixel size, classes, quality, split, weight |
| `$EMDATA_ROOT/manifests/unlab.csv` | one row per raw pool section |
| `$EMDATA_ROOT/em/*/*.h5`, `em2d/*/shard_*.h5` | uint8 `raw`, `labels/<class>`, `masks/<class>` |
| `$EMDATA_ROOT/unlab/unlab_<source>/slab_*.h5` | raw 4 × 1024² slabs |
| `$EMDATA_ROOT/benchmarks/{snemi,cremi}/` | evaluation volumes |

---

## 🧑‍🏫 3. Teacher

ConvNeXt-V2-L (ImageNet-22k) + full-resolution FPN; masked BCE + Dice on foreground + 8 affinities, supervised contrastive
loss on sampled points; 22 audited gold datasets; crops at random pixel size 4–16 nm.

```bash
conda activate em2d-train && source scripts/env.sh
GPUS=0,1 tools/detach.sh teacher scripts/20_train_teacher.sh teacher      # 40k steps, ~16 h
tools/status.sh teacher
```

Outputs in `$RUNS_ROOT/teacher/`: `ckpt/step_0{05,…,40}000.pt` (790 MB each), `ckpt/last.pt` (resume with `--resume`),
`config.json`, `log.jsonl`.

---

## 🏷️ 4. Pseudo-labels

Teacher affinities on every pool section (~8 nm/px, 1024² tiles) → mutex watershed → instances; `stats.csv` lets the student
drop empty or degenerate sections.

```bash
GPUS=0,1,2 scripts/30_pseudolabel.sh $RUNS_ROOT/teacher/ckpt/step_040000.pt teacher40k    # 3 shards, 4–5 h, 35 GB
```

Outputs in `$EMDATA_ROOT/pseudo/teacher40k/`: corpus-shaped tree with `pseudo` (uint32) and `fg` (uint8), `rows_<shard>.csv`,
`stats.csv`. Re-running resumes.

---

## 🎓 5. Student

ConvNeXt-V2-L (ImageNet-22k), 50 % gold + 50 % pseudo-labels, loss normalised per source, pseudo-label weight
ramped over 4 k steps, 60 k steps.

```bash
GPUS=0,1 tools/detach.sh student scripts/40_train_student.sh teacher40k student      # ~25 h
```


---

## 📊 6. Evaluate

Zero-shot = the network's own affinity head + mutex watershed, grid chosen on validation, metrics on test.
Fine-tune = 20 k iterations on the target training split, then the same selection and metrics.

```bash
GPUS=0 scripts/50_evaluate.sh $RUNS_ROOT/student/ckpt/step_060000.pt student60k             # zero-shot: SNEMI + CREMI B
GPUS=0 FINETUNE=1 scripts/50_evaluate.sh $RUNS_ROOT/teacher/ckpt/step_040000.pt teacher40k  # + fine-tune per domain
emseg-eval evaluate --domain cremi --sample B --split test --pred-dir <pred_XXXX.tif dir> --out <dir>  # any method, from label images
```

Per run: `done.json` (`voi_sum`, `voi_split`, `voi_merge`, `arand`, `msa`, `f1_50`, `pq50`, `M1_invisible`, `M2_chain`,
`S1_cut`, `S2_sliver`), `test_per_slice.csv`, `val_grid.json`, `pred/test/pred_XXXX.tif`.

---

## 🎨 7. Visualize

```bash
scripts/60_visualize.sh run zeroshot-student60k-cremiB   # EM | GT boundaries | GT | prediction (coloured by matching GT object)
scripts/60_visualize.sh pseudo teacher40k                      # EM | boundaries | pseudo-instances | teacher foreground
```

<p align="center"><img src="docs/example_cremiB.png" width="100%" alt="EM, GT boundaries, GT instances and student prediction on CREMI B"></p>

---

## 📈 Results (single seed, VOI sum ↓)

| model | SNEMI (AC3) | CREMI B | in-house target, zero-shot | in-house target, fine-tuned |
|---|---|---|---|---|
| teacher 40k | 0.255 | 0.340 | 0.866 | 0.389 |
| student 60k | 0.280 | 0.360 | **0.748** (17/20 sections better) | 0.391 |
| in-domain U-Net, no pretraining | 0.27 | 0.47 | – | 0.484 |

Pretraining lowers the fine-tuned target VOI from 0.48 to 0.39; pseudo-labels make the student beat its teacher zero-shot
on the unseen domain but add nothing after fine-tuning. Paired statistics, error classes, ablations: [docs/results.md](docs/results.md).

---

## 📚 Licence

Code: MIT. Data: each source keeps its publisher's licence — see the tiers in
[emseg/data/README.md](emseg/data/README.md#sources-and-licences) and cite the original publications.

## 🤝 Contact

Open an issue or e-mail the maintainer <!-- add your address here -->. Shareable data (open-licence subset, slab list,
pseudo-labels, checkpoints): [emseg/data/README.md § B](emseg/data/README.md#b-get-the-processed-corpus-from-us).
