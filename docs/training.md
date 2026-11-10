# 🧑‍🏫 🏷️ 🎓 Training — teacher, pseudo-labels, student, evaluation

All training and evaluation code. Run from the repository root after `conda activate em2d-train && source scripts/env.sh`
(the scripts in `scripts/` wrap the commands below with the recipes we used).

| module | role | entry point |
|---|---|---|
| `emseg/train/teacher_student.py` | teacher (`--exp I2`) and student (`--exp I3`) training, single GPU or `torchrun` DDP | `python -m emseg.train.teacher_student` |
| `emseg/pseudolabel.py` | offline teacher pseudo-labels for unlabelled sections | `python -m emseg.pseudolabel` |
| `emseg/finetune.py` | zero-shot evaluation (`--zero-shot`) and target-domain fine-tuning | `python -m emseg.finetune` |
| `emseg/probe.py`, `emseg/evalsuite.py`, `emseg/models/features.py` | frozen-feature probe (how much membrane information the features carry) | `python -m emseg.evalsuite` |
| `emseg/data/unlab.py` | unlabelled pool from public connectome volumes (data environment) | `python -m emseg.data.unlab` |
| `emseg/models/pretrainnet.py` | `PretrainNet`: ConvNeXt-V2 encoder + full-resolution FPN decoder + heads | |
| `emseg/train/data.py`, `emseg/train/relations.py` | crop sampler over corpus + pseudo-labels; pair targets and contrastive loss | |
| `emseg/seg/prior.py`, `emseg/seg/superpixels.py`, `emseg/seg/mws.py` | membrane ridge prior, superpixels, mutex-watershed decoding | |

## Model

`PretrainNet(arch, feat=32, extra_heads={...})`: a timm ConvNeXt-V2 encoder (`convnextv2_base` 89 M or `convnextv2_large`
197 M parameters, ImageNet-22k weights with `--pretrained`), a full-resolution side path (stride 1 and 2 convolutions, so 1–2 px
membranes survive the stride-4 stem), an FPN decoder back to stride 1 with 32 channels, and 1 × 1 heads on top:
`aff` = 1 foreground + 8 disaffinities at offsets (−1,0) (0,−1) (−3,0) (0,−3) (−9,0) (0,−9) (−27,0) (0,−27) (torch_em
convention, 1 = boundary); optionally `conf` (1 channel). The input has two channels, `[image, 0]` (the second channel is a
legacy mask flag). The first 7 affinity channels are exactly `emseg.seg.targets.OFFSETS_2D`, so any checkpoint can be decoded with the
standard mutex watershed without retraining a head.

## Loss and sampling

- **Affinity loss**: masked BCE + Dice on foreground and the 8 disaffinities. Targets come from two id maps per crop, `pos`
  (equal id = same object) and `neg` (different ids = different objects); pairs that are neither are ignored, so partial
  labels and pseudo-labels plug in without special cases (`emseg.train.relations.pair_targets`).
- **Contrastive loss**: supervised contrastive loss on 1 024 points per image (half uniform, half drawn in proportion to the
  membrane ridge prior, i.e. hard pairs near boundaries) at two feature levels (stride 4 and stride 1), weight 1.0.
- **Crops**: 512² at a random physical pixel size in 4–16 nm (sources with unknown pixel size: random scale 0.75–1.33),
  rot90 / flips and strong photometric augmentation (gamma, blur, noise, illumination gradient, scan lines).
- **Sampling** (`emseg.train.data.InstIndex`): manifest weights within a dataset; across datasets mass ∝ n_rows^`--gt-power` for gold
  data (0.5 in recipe V2-A) and n_rows^`--pl-power` for pseudo-labelled data; `--gt-sched f0:t,f1` fixes the share of gold
  samples (f0 until fraction t of the run, then f1). Pseudo-labelled rows with < 8 instances or one instance covering > 60 %
  of the image are dropped using `stats.csv` (`tools/pseudo_stats.py`).
- Learning rate 2e-4 for a global batch of 32 (scaled linearly), AdamW (0.9, 0.95), weight decay 0.05, 1 k warm-up steps,
  cosine to 5 %, gradient clipping 3.0, bf16 autocast. Non-finite steps are skipped and logged.

## Recipes

| recipe | command | what it is |
|---|---|---|
| **teacher** | `scripts/20_train_teacher.sh` = `train_inst --exp I2 --arch convnextv2_large --pretrained --exclude <8 datasets> --iters 40000 --batch 16 --lr 2e-4` on 2 GPUs | gold labels of 22 datasets (12 978 sections) |
| **student V2-A** | `scripts/40_train_student.sh` = `train_inst --exp I3 --arch convnextv2_large --pretrained --pseudo-root $EMDATA_ROOT/pseudo/T40k --manifest em_only.csv,unlab.csv --exclude <8> --gt-sched 0.5:1.0,0.5 --gt-power 0.5 --pl-power 0.5 --src-norm --pl-weight 1.0 --pl-ramp 4000 --iters 60000 --batch 16 --lr 2e-4` | gold + pseudo-labels, 50 / 50, loss normalised per source, pseudo weight ramped over 4 k steps |
| student I3-base | `train_inst --exp I3 --arch convnextv2_base --init <teacher ckpt> --pseudo-root … --ignore-top 0.1 --gt-sched 1.0:0.6,0.2 --iters 40000` | smaller student, gold first then mostly pseudo-labels |
| superpixel pre-check | `train_inst --exp I1` | pseudo-instances from the ridge prior instead of labels (no labels needed; weak) |

The 8 excluded datasets: `snemi3d` (benchmark), `tumor_spheroid` and six EMNeuron subsets whose labels failed the
edge-alignment audit (`tools/label_audit.py`: fraction of label boundaries within 1–3 px of an image ridge, best integer
shift, background share).

Self-training options for pseudo-labelled images (all tested, none used in V2-A because they had no measurable effect):
`--ignore-top q` drops the q largest-loss pixels per image (Depth Anything V2), `--conf [--conf-norm --conf-lambda]` adds a
per-pixel confidence head, `--init-heads` also loads the heads from `--init`.

## Run directory

```
$RUNS_ROOT/<run>/
├── config.json        all arguments, datasets and rows per dataset, effective lr, parameter count, torch version
├── log.jsonl          {"event": "train", "iter", "loss", "aff", "con_p4", "con_feat", "img_per_s", "mem_gb", …}, skip_step, train_done
├── logs/stdout.log    when started with tools/detach.sh
└── ckpt/last.pt       model + optimiser (resume with --resume ckpt/last.pt; the GT schedule keeps its global clock)
    ckpt/step_XXXXXX.pt   model only + args, every --snap-every steps (default 5 000)
```

Checkpoint format: `{"model": state_dict of PretrainNet incl. extra.aff, "iter": int, "args": {... "arch", "feat", "extra_heads"}}`,
so `finetune.build(ckpt, zero_shot=True)` can rebuild the exact network.

## Pseudo-labelling (`pseudolabel.py`)

For every training row without cell labels (minus `--exclude`, optionally restricted with `--datasets`, at most `--per-dataset`
rows per dataset; `scripts/30_pseudolabel.sh` uses 6 000 and the pool + the corpus' unlabelled neural tissue): rescale the
section to ~8 nm/px (`--target-res`, factor clipped to 0.5–2), predict with the teacher's own affinity head in 1024² tiles
with 64 px overlap, mutex watershed with the default parameters (stride 2, foreground 0.5, min size 25), resize back to the
native grid and store `pseudo` (uint32) and `fg` (uint8) next to the source layout under `--out`. `--shard i --n-shards n`
splits the file list over GPUs; `rows_<shard>.csv` lists what was labelled; re-running skips written sections. Then
`tools/pseudo_stats.py <out>` writes `stats.csv` (instances, largest-instance share, foreground and boundary fractions) and
`tools/pseudo_vis.py` renders panels per dataset.

## Zero-shot evaluation and fine-tuning (`finetune.py`)

`--zero-shot`: no training; the checkpoint's affinity head is decoded with the mutex watershed, the parameter grid
(`stride ∈ {1,2}`, `fg_threshold ∈ {0, 0.5}`, `min_size ∈ {0, 25, 100}`) is selected by mean VOI on **all** validation
sections, and the test sections are scored (`emseg.eval.metrics.slice_metrics` + `emseg.eval.errclass.error_classes`).
Without `--zero-shot`: the encoder + decoder are loaded (`--ckpt none` = random init), a fresh 7-channel affinity head is
trained for 20 k iterations (batch 4, 512², lr 1e-4, AdamW, cosine, validation on 5 sections every 2 k iterations, best
checkpoint kept), then the same selection and test. `--domain snemi | cremi --sample B | mouse30um`.
Outputs: `done.json` (`test_summary`, selected parameters, checkpoint hashes), `test_per_slice.csv`, `val_grid.json`,
`pred/test/pred_XXXX.tif`.

## Frozen-feature probe (`probe.py`, `evalsuite.py`)

Freezes encoder + decoder and trains a 1 × 1 affinity head for 2 k steps on the training sections of a domain; reports VOI
with default decoding, pair-wise miss / cut rates at 1 / 3 / 9 px and a training-free pair AUC of the features. Useful to
compare checkpoints along a run without the cost of a full evaluation: `python -m emseg.evalsuite --feat enc:<ckpt> --tag <name> --out-dir <dir>`.

## Unlabelled pool (`unlab.py`, data environment)

`python -m emseg.data.unlab <source>[,<source>] --slabs N [--xy 1024 --nz 4 --sources-json configs/unlab_sources_public.json]`
streams random raw slabs from the named volumes (cloud-volume, anonymous), keeps those that are not blank and lie inside
tissue (when a segmentation exists), writes `unlab/unlab_<source>/slab_x_y_z.h5`, appends rows to `manifests/unlab.csv` and
logs every slab (URL, mip, bounding box, sha256) to `unlab/download_log.jsonl`. Add a volume by adding one entry to the JSON
(`[em_url, em_mip, seg_url|null, seg_mip, modality, organism, tissue, licence, box|null]`).

## Hardware and timing (RTX PRO 6000, 96 GB)

| run | GPUs | memory / GPU | throughput | wall time |
|---|---|---|---|---|
| teacher L, batch 16 | 2 | 43 GB | 24 img/s | 40 k steps ≈ 16 h |
| student L V2-A, batch 16 | 2 | 43 GB | 16–25 img/s | 60 k steps ≈ 25 h |
| student B, batch 16 | 2 | 22 GB | 38 img/s | 40 k steps ≈ 10 h |
| pseudo-labels, 130 k sections | 3 | < 10 GB | | 4–5 h per shard |

Data loading: `--workers 3 --prefetch 2` per rank is enough (~60 img/s) and keeps RAM flat; the pool is one HDF5 file per
section, so open handles are kept in a bounded LRU (32 per worker). On hosts where two GPUs hang at the first NCCL
collective, `NCCL_P2P_DISABLE=1` (set by `scripts/env.sh`) fixes it.
