## Setting up the code
```bash
make
uv sync
source .venv/bin/activate
```

## Download the datasets
```bash
make data
pmpnn-to_h5 data data/pdb_2021aug02.h5
```
## Download the vanilla checkpoints
```bash
make ckpt
```

## Train a model
### OA
```bash
pmpnn-train data/pdb_2021aug02.h5 data/list.csv data/valid_clusters.txt data/test_clusters.txt models/pmpnn-oa-12 \
    --from_pretrained ckpt/v_48_002.pt \
    --mode oa \
    --r_max_orb 12.0
```

### OA-SL
```bash
pmpnn-train data/pdb_2021aug02.h5 data/list.csv data/valid_clusters.txt data/test_clusters.txt models/pmpnn-oa-sl-12 \
    --from_pretrained ckpt/v_48_002.pt \
    --mode oa \
    --r_max_orb 12.0 \
    --num_encoder_layers 1 --debug
```

### GAT
```bash
mkdir models

pmpnn-train data/pdb_2021aug02.h5 data/list.csv data/valid_clusters.txt data/test_clusters.txt models/gat \
    --from_pretrained ckpt/v_48_002.pt \
    --mode gat
```

## Evaluate a model
### OA
```bash
pmpnn-eval data/pdb_2021aug02.h5 data/list.csv data/valid_clusters.txt data/test_clusters.txt models/pmpnn-oa-12/model_weights/best.pt \
    --mode oa \
    --r_max_orb 12.0
```

### OA-SL
```bash
pmpnn-eval data/pdb_2021aug02.h5 data/list.csv data/valid_clusters.txt data/test_clusters.txt models/pmpnn-oa-sl-12/model_weights/best.pt \
    --mode oa \
    --r_max_orb 12.0 \
    --num_encoder_layers 1
```

### GAT
```bash
pmpnn-eval data/pdb_2021aug02.h5 data/list.csv data/valid_clusters.txt data/test_clusters.txt models/gat/model_weights/best.pt \
    --mode gat
```


## Generate sequences
```bash
pmpnn-gen index/test_monomers.txt ./ models/pmpnn-oa-sl-12/model_weights/best.pt \
     --h5_file data/pdb_2021aug02.h5 \
     --mode oa \
     --r_max_orb 12 \
     --num_encoder_layers 1
```

## ESMFold
```bash
pmpnn-esmfold *.fa
```

## Analysis
### Stats
```bash
pmpnn-stats data/pdb_2021aug02.h5 index/test_monomers.txt \
    models/pmpnn-oa-sl-12/model_weights/best.pt stats.npz \
    --num_encoder_layers 1 \
    --r_max_orb 12 \
    --device cuda
``` 

### Commands:
```bash
pmpnn-budget stats.npz
pmpnn-aa_heatmaps stats.npz
pmpnn-oa_geometry stats.npz
pmpnn-oa_spectral stats.npz
```