# 📏 Evaluation — protocol, benchmark loaders, decoding (`emseg.eval`, `emseg.seg`)

A small, dependency-light package that defines *how* a 2D dense segmentation is scored, so that every method in this
repository (and any external one) is compared on the same footing.

## Protocol

- **Ground truth** is 2D-relabelled per section (`skimage.measure.label`, 4-connectivity; optional `--min-size` fragment
  removal, default 0 = keep everything). Predictions are 2D-relabelled too but never size-filtered.
- **Primary metrics** (`metrics.py`): VOI split / merge / sum in bits and the adapted Rand error, GT label 0 ignored (CREMI
  definitions via `elf.evaluation`).
- **Object metrics**: mSA (DSB-2018 mean segmentation accuracy, IoU 0.5–0.95), precision / recall / F1 / AP / PQ at IoU 0.5
  and 0.75 from a contingency-table matching (cross-checked against `elf.matching`); Rand / information scores of
  ISBI 2012 / SNEMI3D, object Dice, AJI.
- **Error classes** (`errclass.py`): the VOI decomposed into merges between GT-adjacent objects whose shared membrane was
  not detected (`M1_invisible`), chain merges of ≥ 3 objects (`M2_chain`), true cuts of an object into ≥ 2 pieces (`S1_cut`)
  and slivers (`S2_sliver`), each as its share of the VOI; needs the two short-range affinity channels.
- **Splits** (`emseg/eval/domains/`): SNEMI — train AC4 z0–79, val AC4 z80–99, test AC3 z0–99 (indices 1000 + z); CREMI — per sample
  train z0–74, val z75–99, test z100–124 (blank sections 14 / 74 of sample C excluded); mouse30um (in-house, not
  distributed) — train z0–69, val z70–79, test z80–99.
- Post-processing parameters are always selected on the validation split only (`emseg/seg/targets.py: GRIDS`).

## Benchmark files

`scripts/12_get_benchmarks.sh` arranges them under `$EM2DSEG_DATA_ROOT` (default `$EMDATA_ROOT/benchmarks`):

```
benchmarks/snemi/snemi_train.h5           volumes/raw, volumes/labels/neuron_ids (AC4, 100 x 1024 x 1024)
benchmarks/snemi/ac3_image.tif, ac3_labels.tif   AC3 with labels (EMNeuron release, labeled/Kasthuri-atum/0.tif, 0_MaskIns.tif)
benchmarks/cremi/sample_B_20160501.hdf    volumes/raw, volumes/labels/neuron_ids (125 x 1250 x 1250)
```

## Command line

```bash
emseg-eval splits    --domain snemi                                   # emseg-eval = python -m emseg.eval.cli
emseg-eval selfcheck --domain cremi --sample B --split test           # GT against GT must be perfect (loader + metric sanity)
emseg-eval evaluate  --domain cremi --sample B --split test --pred-dir PRED --out OUT [--min-size N] [--mask-missing 8]
```

`PRED` holds one label image per section, `pred_{idx:04d}.tif|.npy|.png` (integer ids, 0 = background), indices as printed by
`splits`. `OUT/per_slice.csv` has one row per section, `OUT/summary.json` mean / std / n per metric plus the metadata.

## Segmentation primitives (`emseg/seg/`)

`mws.py`: `mws(affs, offsets, params, fg)` (torch_em mutex watershed + size filter), `gt_of`, `voi_sum`, `SCHEME`;
`targets.py`: the affinity offsets `OFFSETS_2D`, target transforms, the selection grids and defaults;
`crops.py`: in-memory random-crop dataset over a domain's sections (used by fine-tuning and the probe);
`unet.py`: the torch_em `UNet2d` baseline and the masked BCE + Dice loss; `prior.py` / `superpixels.py`: membrane ridge
prior and ridge-based superpixels. Install with `pip install -e .` (gives the `emseg-eval` command) or use the package
through `PYTHONPATH` as `scripts/env.sh` does.

## Tests

```bash
python -m unittest discover -s tests -v     # from the repository root; 32 tests, no data needed
```
