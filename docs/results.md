# 📈 Results

All numbers: one seed, test splits, **VOI sum in bits (lower is better)**, mutex-watershed parameters selected on the
validation split of each domain (`stride ∈ {1,2}`, `fg_threshold ∈ {0, 0.5}`, `min_size ∈ {0, 25, 100}`).
Domains: SNEMI = Kasthuri AC3 (100 sections, 6 nm), CREMI B (25 test sections, 4 nm), and an in-house mouse cortex ATUM-SEM
volume (20 test sections at 16 nm/px; the data are not distributed with this repository, the numbers are kept for reference).
Neither benchmark nor the in-house volume is part of the training corpus (leakage scan in `emseg.data.stages.dedup`).

## Zero-shot (affinity head of the pretrained network, no target-domain training)

| model | steps | SNEMI (AC3) | CREMI B | in-house mouse cortex |
|---|---|---|---|---|
| teacher, ConvNeXt-V2-L, 22 gold datasets | 20k | 0.228 | 0.367 | 0.852 |
| teacher | 40k | **0.255** | **0.340** | 0.866 |
| student V2-A (gold + pseudo-labels) | 20k | 0.284 | 0.405 | 0.835 |
| student V2-A | 60k | 0.280 | 0.360 | **0.748** |
| ConvNeXt-V2-B, gold only, self-supervised init (I2) | 20k | 0.394 | 0.495 | 1.41 |
| in-domain U-Net (affinity + MWS, trained on each domain) | 20k | 0.27 | 0.47 | 0.48–0.50 |

Per-section paired differences, student V2-A 60k minus teacher 40k (Wilcoxon):

| domain | n | mean ΔVOI | sections where the student is better | p |
|---|---|---|---|---|
| in-house mouse cortex | 20 | −0.118 ± 0.029 | 17 / 20 | 5e-5 |
| SNEMI (AC3) | 100 | +0.026 ± 0.008 | 31 / 100 | 3e-4 |
| CREMI B | 25 | +0.020 ± 0.005 | 4 / 25 | 1e-3 |

On the unseen target domain the student is the first model that is robustly better than its teacher (VOI, F1@0.5 and PQ
all improve; merge VOI 0.399 vs 0.45-0.48). On the two public benchmarks the student is slightly worse than the teacher in
VOI but better in F1@0.5 / PQ at equal post-processing (SNEMI +0.057 / +0.038). Its remaining failure mode on the target
domain is cutting the nucleus out of large somata (11 / 20 sections; 16 / 20 at 20k), inherited from pseudo-labels in which
nuclei are separate instances.

Error classes on the target domain (share of the VOI, `emseg.eval.errclass`): the student mainly reduces chain merges (M2)
and true cuts (S1); invisible-membrane merges (M1) stay at ~0.16.

## Fine-tuning on the target domain (20k iterations, batch 4, 512², lr 1e-4, seed 0; train 69 / val 10 / test 20 sections)

| initialisation | test VOI | split / merge | F1@0.5 | PQ |
|---|---|---|---|---|
| random | 0.657 | | | |
| in-domain U-Net baseline (no pretraining) | 0.484–0.489 | 0.207 / 0.282 | | |
| I2 (ConvNeXt-V2-B, gold only, SSL init) | 0.481 | 0.200 / 0.281 | 0.808 | 0.724 |
| teacher 40k | **0.389** | 0.162 / 0.226 | 0.843 | 0.770 |
| student V2-A 60k | 0.391 | 0.160 / 0.231 | 0.844 | 0.771 |

Multi-source supervised pretraining (teacher or student initialisation) lowers the fine-tuned VOI from 0.48 to 0.39 on all
20 test sections. The pseudo-label phase does not add anything after target-domain fine-tuning: student and teacher
initialisations are indistinguishable (paired difference +0.003 ± 0.013, p = 0.96). Fine-tuning removes the nucleus cuts.

Ablations of the student recipe (16k-step pilots): the Depth-Anything-style confidence head and the top-10 % loss
ignoring had no measurable effect (zero-shot VOI within ±0.01 on all three domains), so recipe V2-A uses neither.

## Compute (RTX PRO 6000 Blackwell, 96 GB)

| stage | GPUs | wall time | output |
|---|---|---|---|
| teacher, 40k steps, batch 2 × 16 | 2 | ~16 h (24 img/s) | 790 MB per snapshot |
| pseudo-labelling T40k, 130k sections | 3 | 4–5 h per shard | 35 GB |
| student V2-A, 60k steps, batch 2 × 16 | 2 | ~25 h | 790 MB per snapshot |
| zero-shot evaluation per domain | 1 | minutes | predictions + metrics |
| fine-tune per domain | 1 | 2–3 h | predictions + metrics |
