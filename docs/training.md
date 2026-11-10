# 🧑‍🏫 🏷️ 🎓 Training

Run from the repository root after `conda activate em2d-train && source scripts/env.sh`. The scripts in `scripts/` wrap the
commands below with the recipes we used.

| entry point | role |
|---|---|
| `python -m emseg.train.teacher_student --exp gold` | teacher: ground truth of the labelled datasets |
| `python -m emseg.train.teacher_student --exp pseudo` | student: ground truth + teacher pseudo-labels |
| `python -m emseg.pseudolabel` | offline teacher pseudo-labels for unlabelled sections |
| `python -m emseg.finetune [--zero-shot]` | target-domain fine-tuning, or zero-shot evaluation of a checkpoint |
| `python -m emseg.evalsuite` | frozen-feature probe over several domains |
| `python -m emseg.data.unlab` | unlabelled pool from public connectome volumes (data environment) |

## Model (`emseg/models/pretrainnet.py`)

ConvNeXt-V2 encoder (`convnextv2_base` 89 M or `convnextv2_large` 197 M parameters, ImageNet-22k with `--pretrained`),
a full-resolution side path (so 1–2 px membranes survive the stride-4 stem), an FPN decoder to stride 1 (32 channels) and
1 × 1 heads: `aff` = foreground + 8 disaffinities at offsets ±1, ±3, ±9, ±27 px in y and x (1 = boundary), optional `conf`.
Input = `[image, 0]` (two channels). The first 7 affinity channels equal `emseg.seg.targets.OFFSETS_2D`, so any checkpoint
decodes with the standard mutex watershed.

## Loss and sampling (`emseg/train/`)

- **Affinity loss**: masked BCE + Dice on foreground and the 8 disaffinities. Targets come from two id maps per crop
  (`pos`: same id = same object, `neg`: different ids = different objects; other pairs ignored), so partial labels and
  pseudo-labels need no special cases.
- **Contrastive loss**: supervised contrastive loss on 1 024 points per image (half near membranes) at stride 4 and 1, weight 1.
- **Crops**: 512² at a random pixel size in 4–16 nm, rot90 / flips, strong photometric augmentation.
- **Sampling**: dataset mass ∝ n_rows^`--gt-power` (gold) and n_rows^`--pl-power` (pseudo-labels); `--gt-sched f0:t,f1`
  sets the share of gold samples (f0 until fraction t of the run, then f1); pseudo-labelled sections with < 8 instances
  or one instance > 60 % of the image are dropped via `stats.csv`.
- **Optimisation**: AdamW (0.9, 0.95), lr 2e-4 at global batch 32 (scaled linearly), weight decay 0.05, 1k warm-up,
  cosine to 5 %, gradient clipping 3, bf16 autocast.

## Recipes

| | command | data |
|---|---|---|
| **teacher** | `scripts/20_train_teacher.sh` → `--exp gold --arch convnextv2_large --pretrained --exclude <8 datasets> --iters 40000 --batch 16 --lr 2e-4`, 2 GPUs | 22 gold datasets, 12 978 sections |
| **student** | `scripts/40_train_student.sh` → `--exp pseudo --arch convnextv2_large --pretrained --pseudo-root … --gt-sched 0.5:1.0,0.5 --gt-power 0.5 --pl-power 0.5 --src-norm --pl-weight 1.0 --pl-ramp 4000 --iters 60000` | gold + pseudo-labels, 50 / 50 |
| smaller student | `--exp pseudo --arch convnextv2_base --init <teacher ckpt> --iters 40000 --gt-sched 1.0:0.6,0.2` | gold first, then mostly pseudo-labels |

Excluded datasets: `snemi3d` (benchmark), `tumor_spheroid` and six EMNeuron subsets that failed the label audit
(`tools/label_audit.py`). Optional pseudo-label flags: `--ignore-top q` (drop the q largest-loss pixels per image),
`--conf` (per-pixel confidence head), `--init-heads` (also load heads from `--init`).

## Run directory

```
$RUNS_ROOT/<run>/
├── config.json           arguments, datasets and rows per dataset, effective lr, parameter count
├── log.jsonl             train (loss, img_per_s, mem_gb, …), skip_step, train_done
├── logs/stdout.log       when started with tools/detach.sh
└── ckpt/last.pt          model + optimiser  →  --resume ckpt/last.pt
    ckpt/step_XXXXXX.pt   model + args every --snap-every steps
```

## Pseudo-labels (`emseg/pseudolabel.py`)

For every training row without cell labels (≤ `--per-dataset` rows per dataset): rescale to ~8 nm/px, predict with the
teacher's affinity head in 1024² tiles, mutex watershed (stride 2, foreground 0.5, min size 25), resize back, store
`pseudo` (uint32) and `fg` (uint8) in the source layout under `--out`. `--shard i --n-shards n` splits the work over GPUs;
re-running skips written sections. `tools/pseudo_stats.py <out>` writes `stats.csv`, `tools/pseudo_vis.py` renders panels.

## Zero-shot and fine-tuning (`emseg/finetune.py`)

`--zero-shot`: decode the checkpoint's own affinity head; select the mutex-watershed grid (`stride`, `fg_threshold`,
`min_size`) by mean VOI on all validation sections; score the test sections. Without `--zero-shot`: load encoder + decoder
(`--ckpt none` = random), train a fresh affinity head for 20k iterations (batch 4, 512², lr 1e-4), then the same selection
and scoring. Outputs: `done.json`, `test_per_slice.csv`, `val_grid.json`, `pred/test/pred_XXXX.tif`.

## Unlabelled pool (`emseg/data/unlab.py`)

`python -m emseg.data.unlab <source>[,…] --slabs N [--sources-json configs/unlab_sources_public.json]` streams random raw
slabs (1024² × 4 sections, ~8 nm/px) from public connectome volumes, keeps those inside tissue, writes
`unlab/unlab_<source>/slab_*.h5`, `manifests/unlab.csv` and `unlab/download_log.jsonl` (URL, mip, bounding box, sha256).
Add a volume with one JSON entry: `[em_url, em_mip, seg_url|null, seg_mip, modality, organism, tissue, licence, box|null]`.

## Compute (96 GB GPUs)

| run | GPUs | memory / GPU | wall time |
|---|---|---|---|
| teacher, batch 16 | 2 | 43 GB | 40k steps ≈ 16 h |
| student, batch 16 | 2 | 43 GB | 60k steps ≈ 25 h |
| pseudo-labels, 130k sections | 3 | < 10 GB | 4–5 h per shard |

`--workers 3 --prefetch 2` per rank is enough and keeps RAM flat (HDF5 handles in a bounded LRU). On hosts where two GPUs
hang at the first NCCL collective, `NCCL_P2P_DISABLE=1` (set by `scripts/env.sh`) fixes it.
