# 📏 Evaluation (`emseg.eval`, `emseg.seg`)

## Protocol

- **Ground truth**: 2D-relabelled per section (4-connectivity); optional `--min-size` fragment removal (default 0). Predictions are relabelled, never size-filtered.
- **Primary metrics**: VOI split / merge / sum (bits) and adapted Rand error, GT label 0 ignored (CREMI definitions, `elf`).
- **Object metrics**: mSA (IoU 0.5–0.95), precision / recall / F1 / AP / PQ at IoU 0.5 and 0.75, Rand / information scores, object Dice, AJI.
- **Error classes** (`errclass.py`): the VOI split into merges across undetected membranes (`M1_invisible`), chain merges of ≥ 3 objects (`M2_chain`), true cuts (`S1_cut`) and slivers (`S2_sliver`).
- **Post-processing** parameters are selected on the validation split only (`emseg/seg/targets.py: GRIDS`).

| domain | train | val | test | file |
|---|---|---|---|---|
| `snemi` | AC4 z 0–79 | AC4 z 80–99 | AC3 z 0–99 (indices 1000 + z) | `benchmarks/snemi/snemi_train.h5`, `ac3_image.tif`, `ac3_labels.tif` |
| `cremi --sample B` | z 0–74 | z 75–99 | z 100–124 | `benchmarks/cremi/sample_B_20160501.hdf` |
| `mouse30um` (in-house, not distributed) | z 0–69 | z 70–79 | z 80–99 | — |

`scripts/12_get_benchmarks.sh` downloads and arranges the files under `$EM2DSEG_DATA_ROOT`.

## Command line

```bash
emseg-eval splits    --domain snemi                                   # emseg-eval = python -m emseg.eval.cli
emseg-eval selfcheck --domain cremi --sample B --split test           # GT against GT must be perfect
emseg-eval evaluate  --domain cremi --sample B --split test --pred-dir PRED --out OUT [--min-size N] [--mask-missing 8]
```

`PRED` holds one label image per section, `pred_{idx:04d}.tif|.npy|.png` (0 = background). `OUT/per_slice.csv` has one row
per section, `OUT/summary.json` mean / std / n per metric.

## Segmentation primitives (`emseg/seg/`)

| module | contents |
|---|---|
| `targets.py` | affinity offsets `OFFSETS_2D`, target transforms, selection grids and defaults |
| `mws.py` | `mws` (mutex watershed + size filter), `gt_of`, `voi_sum`, `SCHEME` |
| `crops.py` | in-memory random-crop dataset over a domain (fine-tuning, probe) |
| `unet.py` | torch_em `UNet2d` baseline, masked BCE + Dice loss |
| `prior.py`, `superpixels.py` | membrane ridge prior and ridge-based superpixels |

Tests: `python -m unittest discover -s tests -v` (repository root, 32 tests, no data needed).
