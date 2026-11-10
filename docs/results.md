# 📈 Results

One seed, test splits, **VOI sum in bits (lower is better)**; mutex-watershed parameters selected on each domain's validation
split. SNEMI = Kasthuri AC3 (100 sections), CREMI B (25 test sections), in-house target = mouse cortex ATUM-SEM (20 test
sections, not distributed). None of them is in the training corpus (leakage scan in `emseg.data.stages.dedup`).

## Zero-shot

| model | SNEMI (AC3) | CREMI B | in-house target |
|---|---|---|---|
| teacher, ConvNeXt-V2-L, 22 gold datasets, 40k steps | **0.255** | **0.340** | 0.866 |
| student, gold + pseudo-labels, 60k steps | 0.280 | 0.360 | **0.748** |
| U-Net trained on each domain | 0.27 | 0.47 | 0.48–0.50 |

Paired per-section differences, student − teacher (Wilcoxon):

| domain | n | ΔVOI | student better | p |
|---|---|---|---|---|
| in-house target | 20 | −0.118 ± 0.029 | 17 / 20 | 5e-5 |
| SNEMI (AC3) | 100 | +0.026 ± 0.008 | 31 / 100 | 3e-4 |
| CREMI B | 25 | +0.020 ± 0.005 | 4 / 25 | 1e-3 |

- On the unseen target the student beats its teacher in VOI, F1@0.5 and PQ; the gain is mostly on the merge side.
- On the public benchmarks the student is slightly worse in VOI, better in F1@0.5 / PQ at equal post-processing.
- Remaining failure mode: the nucleus is cut out of large somata (11 / 20 sections), inherited from pseudo-labels that
  label nuclei as separate instances.

## Fine-tuned on the target domain

20k iterations on the target's training sections, same selection and metrics.

| initialisation | VOI | split / merge | F1@0.5 | PQ |
|---|---|---|---|---|
| random | 0.657 | | | |
| U-Net, no pretraining | 0.484 | 0.207 / 0.282 | | |
| teacher 40k | **0.389** | 0.162 / 0.226 | 0.843 | 0.770 |
| student 60k | 0.391 | 0.160 / 0.231 | 0.844 | 0.771 |

Pretraining lowers the fine-tuned VOI from 0.48 to 0.39 on all 20 sections; teacher and student initialisations are
indistinguishable after fine-tuning (p = 0.96). Fine-tuning removes the nucleus cuts.

## Compute (RTX PRO 6000, 96 GB)

| stage | GPUs | wall time |
|---|---|---|
| teacher, 40k steps, batch 2 × 16 | 2 | ~16 h |
| pseudo-labels, 130k sections | 3 | 4–5 h per shard |
| student, 60k steps, batch 2 × 16 | 2 | ~25 h |
| zero-shot evaluation per domain | 1 | minutes |
| fine-tune per domain | 1 | 2–3 h |
