# ProteinMPNN-OA — ProteinMPNN with Orbital Attention

Inverse folding (backbone structure → amino-acid sequence) with
[ProteinMPNN](https://github.com/dauparas/ProteinMPNN), extended with
**Orbital Attention (OA)**: a rotation-invariant, geometry-aware attention
mechanism inside the encoder.

The project asks two questions:

1. **Does OA help?** Sequence recovery and perplexity (teacher-forced), plus
   designability of sampled sequences (re-folded with Boltz, scored by
   TM-score), against vanilla ProteinMPNN and a geometry-free attention
   baseline.
2. **What does OA learn?** Interpretability of the attention weights: score
   as a function of local geometry (d, θ, φ), residue-pair preferences,
   disulfide bonds, and ablations that delete one specific kind of
   information from the attention.

## Model variants

All variants share ProteinMPNN's featurisation and decoder. They differ only
in how an encoder layer aggregates messages from a residue's K neighbours,
selected with `--mode` (see `src/pmpnn/model.py`):

| `--mode`  | Encoder aggregation | Role |
|-----------|---------------------|------|
| `vanilla` | Original ProteinMPNN sum/scale aggregation | Baseline |
| `oa`      | `OAAttention` — scores from backbone-frame spherical harmonics (degrees ≤ `--l_max`) contracted with Bessel radial-basis filters, one sigmoid gate per degree, within a cutoff radius `--r_max_orb` (Å) | **Main model** |
| `gat`     | `GATAttention` — attention from edge features only, no explicit geometry | Ablation: attention without geometry |

Because `OAAttention` scores are built from spherical harmonics evaluated in
each residue's own backbone frame, they are invariant to global rotations of
the structure. The degree gate makes "how much angular detail to use"
learnable per edge: degree l=0 is direction-independent, so zeroing degrees
l ≥ 1 reduces the score to a function of Cα-Cα distance alone — the basis of
the *isotropic* ablation below.

The model keeps the original ProteinMPNN parameter names, so OA and GAT runs
can start from the public vanilla weights (`--from_pretrained`); only the new
attention parameters are initialised randomly. The startup line
`Loaded N original parameters. M new orbital-attention parameters randomly
initialised.` reports this.

## Run naming

Run directories under `models/<experiment>/` are named

```
<config>[_<sweep-param><value>]_seed<seed>
```

| Config     | Meaning |
|------------|---------|
| `oa1`      | `--mode oa`, 1 encoder layer |
| `oa3`      | `--mode oa`, 3 encoder layers |
| `gat3`     | `--mode gat`, 3 encoder layers |
| `vanilla3` | `--mode vanilla`, 3 encoder layers |

Examples: `oa1_seed0`; `oa3_frac0.4_seed1` (trained on 40 % of the training
clusters); `oa1_dist8_seed0` (`--r_max_orb 8.0`); `oa1_ckpt0.10_seed2`
(initialised from the vanilla checkpoint trained with 0.10 Å backbone noise,
`v_48_010.pt`). OA runs use `--r_max_orb 12.0` unless the experiment sweeps
it. The single-encoder-layer `oa1` config is the one most analyses focus on,
since with one layer the attention weights are directly interpretable.

`models/exp2_finetuning/native/` is not a model: it holds the ground-truth
sequences and structures used as the reference and self-consistency baseline.

## Layout

```
src/
  pmpnn/    model (vanilla / OA / GAT), features, data, loss, optimizer
  data/     pmpnn-to_h5   raw ProteinMPNN .pt collection → one HDF5 file
  train/    pmpnn-train   training / finetuning
  eval/     pmpnn-eval    teacher-forced metrics on the test split
  gen/      pmpnn-gen     sequence sampling; attention ablations; Boltz YAML export
  esmfold/  pmpnn-esmfold ESMFold structure prediction
  vis/      pmpnn-stats   one forward pass → stats.npz, the input to every analysis
            pmpnn-vis and friends — analyses and figures (see below)
scripts/
  <experiment>_<NN>_<stage>.sbatch   SLURM pipeline, one file per stage
  *.py                               helpers: test-set download, native-structure
                                     extraction, Boltz output sanitising, TM-score,
                                     parameter counts
models/<experiment>/<run>/           checkpoints and all per-run outputs
index/                               test-set ID lists (monomers/homomers/heteromers)
results/                             figures from an earlier iteration of the project
```

`pmpnn-stats` writes a `stats.npz` holding per-residue and per-edge
quantities (attention weights, geometry, spectral coefficients). Every
analysis reads that file instead of re-running the model, so the expensive
GPU pass happens once per run.

## Setup

```bash
make env                    # uv sync — Python 3.11, torch 2.10 + CUDA 12.8
source .venv/bin/activate   # puts the pmpnn-* commands on PATH
```

Data and starting checkpoints:

```bash
make data                                 # ProteinMPNN training set (pdb_2021aug02)
pmpnn-to_h5 data data/pdb_2021aug02.h5    # → single HDF5 file
make ckpt                                 # vanilla weights v_48_0{02,10,20,30}.pt → ckpt/
```

The SLURM scripts additionally expect, under `data/`:

| Path | Content |
|------|---------|
| `pdb_2021aug02.h5`, `list.csv`, `valid_clusters.txt`, `test_clusters.txt` | Training set and cluster splits |
| `test_{monomers,homomers,heteromers}.txt` | Held-out test IDs (same lists as `index/`) |
| `test_pdb_raw/` | Full raw PDB entries for the test sets, keeping SSBOND records — built by `scripts/download_test_monomers.py`. Needed only for the disulfide analysis, because the HDF5 training format drops header records. |

## Experiments

Each experiment has a folder under `models/` and a numbered set of scripts:

```
scripts/<experiment>_<NN>_<stage>.sbatch
```

`<experiment>` matches the `models/<experiment>/` output folder; `NN` is the
order to run them in, and scripts sharing a number are independent of each
other. Submit from the repo root:

```bash
sbatch scripts/exp2_finetuning_01_eval.sbatch
```

Every array task skips a step whose output already exists, so re-submitting a
script resumes it. Each script's header comment documents its inputs,
outputs, array layout and resource choices. `pmpnn-metrics` collects a whole
experiment into one summary table, written to `models/<experiment>/metrics.md`:

```bash
pmpnn-metrics --root models/exp1_data_fraction --out models/exp1_data_fraction/metrics.md
```

### Experiment 1 — data-fraction scaling (`exp1_data_fraction`)

Does OA need more or less data than the baselines? A grid of 4 configs
(`oa1`, `oa3`, `vanilla3`, `gat3`) × 5 training-cluster fractions (0.1, 0.2,
0.4, 0.8, 1.0) × 3 seeds = 60 runs, all trained **from scratch** so the
data-efficiency comparison is not confounded by pretrained weights.

| # | Script | Stage |
|---|--------|-------|
| 01 | `exp1_data_fraction_01_train` | Train all 60 runs |
| 02 | `exp1_data_fraction_02_gen` | Sample sequences for the test monomers |
| 03 | `exp1_data_fraction_03_stats_vis_oa1` | `stats.npz` + figures, `oa1` runs only |

### Experiment 2 — finetuning (`exp2_finetuning`)

The main comparison. 10 runs — `oa1`, `oa3`, `gat3` (3 seeds each) and
`vanilla3` — finetuned from the public vanilla checkpoint, carried all the
way through to designability and interpretability.

| # | Script | Stage |
|---|--------|-------|
| 01 | `exp2_finetuning_01_eval` | Teacher-forced test metrics → `eval/results.txt` |
| 02 | `exp2_finetuning_02_gen_monomers` | Sample sequences, test monomers |
| 02 | `exp2_finetuning_02_gen_multimers` | Same for homomers and heteromers |
| 03 | `exp2_finetuning_03_boltz` | Re-fold the sampled sequences with Boltz |
| 04 | `exp2_finetuning_04_tmscore` | Boltz predictions vs. native, via USalign |
| 05 | `exp2_finetuning_05_stats` | Forward pass → `eval/stats.npz` |
| 06 | `exp2_finetuning_06_vis` | Geometry + edge-type attention figures, per (run, layer) |
| 06 | `exp2_finetuning_06_geometry` | (d, θ, φ) score maps |
| 07 | `exp2_finetuning_07_ablation_*` | Attention ablations (below) |
| 08 | `exp2_finetuning_08_ssbond` | Attention on Cys-Cys edges vs. real SSBOND records |
| 08 | `exp2_finetuning_08_dist_corr` | Attention score vs. backbone atom-pair distances |

Steps 01–04 measure quality; 05–08 ask what the attention encodes. Step 04
needs the native structures first (`python scripts/extract_native.py`) and
the `usalign` environment module.

**Ablations (step 07).** Each re-samples sequences with one kind of
information removed at inference time via forward hooks — no retraining — and
is compared against the unablated sequences from step 02. They run on the
seed-0 runs only. The three attention ablations sweep each encoder layer
alone plus all layers at once, so the per-layer runs show *where* the
information matters; `seqdist_attn` has no per-layer sweep, because the
sequence-separation embedding is computed once before the encoder stack.

| Ablation | What it removes | Question |
|----------|-----------------|----------|
| `uniform_attn` | Flattens attention to all non-self neighbours to one uniform weight, keeping the self-weight | Is the learned attention pattern used at all? |
| `permute_attn` | Keeps the multiset of weights but randomly reassigns which neighbour gets which | Does the model need *which* neighbour, or just a non-flat profile? |
| `isotropic_attn` | Zeroes spherical-harmonic degrees l ≥ 1, leaving distance-only scores (OA runs only) | Is the angular information used, or only distance? |
| `seqdist_attn` | Overwrites every edge's true sequence separation with a constant | Is relative sequence position used? (applies to all modes, including `vanilla3`) |

### Appendices

| Folder | Sweep | Scripts |
|--------|-------|---------|
| `appA_distance_cutoff` | OA cutoff radius `r_max_orb` ∈ {8, 10, 14, 16} Å × 3 seeds (12 runs); the 12 Å default is Exp. 2 | `appA_distance_cutoff_01_gen_stats_vis` |
| `appB_ckpt_init` | Which vanilla checkpoint OA starts from: 0.02 / 0.10 / 0.30 Å backbone noise × 3 seeds (9 runs) | `appB_ckpt_init_01_gen_stats_vis` |
| `appC_randoms` | `untrained-oa`: an untrained OA model, i.e. what the geometry analyses show before any learning | — |
| `appD_others` | `ipa`, `pt`: alternative attention variants | — |

The two appendix scripts merge generation, `stats.npz` and figures into one
per-run pipeline, which is possible because every appendix run is `oa1`
(a single encoder layer, so a single layer to visualise).

## Analysis entry points

All of these read a `stats.npz`:

| Command | Output |
|---------|--------|
| `pmpnn-vis` | The main report: attention score vs. local geometry and residue-pair type, with protein-level bootstrap CIs |
| `pmpnn-geometry` | (d, θ, φ) score maps; optional volumetric export for ChimeraX |
| `pmpnn-budget` | Attention-budget distribution and neighbourhood size |
| `pmpnn-aa_heatmaps` | 20×20 amino-acid-pair score heatmaps, overall and per distance bin |
| `pmpnn-oa_spectral` | PCA over the spherical-harmonic spectral coefficients |
| `pmpnn-enrichment` | Type-specific geometric preference (γ) per residue-pair type |
| `pmpnn-ssbond`, `pmpnn-dist_corr` | Disulfide and atom-pair-distance analyses (each has `compute` then `analyze`) |
| `pmpnn-metrics` | Per-run quality table across an experiment |

Statistics convention: uncertainty comes from a **protein-level cluster
bootstrap** (resampling whole structures, not individual edges), because
edges within one protein are not independent. Analyses report CIs rather
than point estimates.

Single-run example:

```bash
pmpnn-stats data/pdb_2021aug02.h5 data/test_monomers.txt \
    models/exp2_finetuning/oa1_seed0/model_weights/best.pt \
    models/exp2_finetuning/oa1_seed0/eval/stats.npz \
    --mode oa --num_encoder_layers 1 --r_max_orb 12.0 --device cuda

pmpnn-vis models/exp2_finetuning/oa1_seed0/eval/stats.npz \
    --out_dir models/exp2_finetuning/oa1_seed0/vis_1 --layers 1
```

## Reproducing a single run by hand

Training, evaluation and sampling for one `oa1` run (`--mode vanilla` or
`--mode gat` for the baselines; drop `--r_max_orb` when not using `oa`):

```bash
pmpnn-train data/pdb_2021aug02.h5 data/list.csv \
    data/valid_clusters.txt data/test_clusters.txt \
    models/my_run \
    --from_pretrained ckpt/v_48_002.pt \
    --mode oa --num_encoder_layers 1 --r_max_orb 12.0

pmpnn-eval data/pdb_2021aug02.h5 data/list.csv \
    data/valid_clusters.txt data/test_clusters.txt \
    models/my_run/model_weights/best.pt \
    --mode oa --num_encoder_layers 1 --r_max_orb 12.0

pmpnn-gen index/test_monomers.txt models/my_run/eval/seqs \
    models/my_run/model_weights/best.pt \
    --h5_file data/pdb_2021aug02.h5 \
    --mode oa --num_encoder_layers 1 --r_max_orb 12.0

pmpnn-esmfold models/my_run/eval/seqs/*.fa    # optional: fold the designs
```

`--mode` and `--num_encoder_layers` must match the checkpoint at every later
stage, which is why the SLURM scripts derive them from the run name.

## Notes for reviewers

- **Not in version control:** `models/`, `data/` and `ckpt/` are gitignored
  (checkpoints and datasets are large); the three test-set ID lists under
  `index/` are tracked despite the ignore rule. Everything under `scripts/`
  is tracked, so a fresh clone includes the full SLURM pipelines and their
  helper scripts.
- **Training provenance.** The grid in Experiment 1 is fully reproducible from
  `exp1_data_fraction_01_train`. The Experiment 2, Appendix A and Appendix B
  checkpoints predate their analysis pipelines and have no training script in
  this repository; their hyper-parameters are recorded only in the run names
  and the appendix tables above. The checkpoints themselves store just
  `epoch`, `step`, `num_edges` and `noise_level` — not the full argument set.
- **`results/`** holds figures from an earlier iteration, under that
  iteration's naming (`pmpnn-oa-12`, `pmpnn-oa-sl-12`, `gat`, where `-12` is
  the 12 Å cutoff and `-sl` a single encoder layer). Current outputs live
  beside their checkpoints in `models/<experiment>/<run>/`.
- **Appendices C and D** contain only `stats.npz` and figures, with no
  accompanying scripts.
