# Diagnose Before Fine-Tuning: Support-Conditioned Operator Routing for Parameter-Efficient Vision Transformers



## Requirements

The code has been tested with the following software stack:

```text
CUDA: 11.8
Python:       3.10.0
PyTorch:      2.4.0 
TorchVision:  0.19.0
timm:         1.0.11
NumPy:        1.26.4
PyYAML:       6.0.2
```

## Installation

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate scope
pip install -e .
```

Alternatively, create it manually:

```bash
conda create -n scope python=3.10.0 -y
conda activate scope
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip install -e .
```

## Repository Structure

```text
scope/
  diagnostics/       
  routing/          
  primitives/        
  models/            
  data/              
  training/          
scripts/
  smoke_test.py          
  train_imagefolder.py  
  run_scope_vtab19.sh   

configs/
  support_diagnostics_vtab19.csv
  support_routes_vtab19.csv
  support_routes_vtab19.json
  selector_rules_v1.yaml
  imagefolder_example.yaml

results/
  tables/            
  analysis/          

docs/
  PROTOCOL.md        
  RESULTS.md         
```

## Quick Check

Run the following commands after installation:

```bash
python -m compileall scope scripts tests
python -m scope.selector --diag configs/support_diagnostics_vtab19.csv --dataset dtd
python scripts/smoke_test.py
```

Expected route output for `dtd`:

```text
redundancy_aware_evidence
```

If `torch` or `timm` is unavailable, `scripts/smoke_test.py` skips the ViT forward-pass check and still tests route selection and diagnostic-file safeguards.

## Route Selection

Query the SCOPE route for one dataset:

```bash
python -m scope.selector \
  --diag configs/support_diagnostics_vtab19.csv \
  --dataset cifar
```

Export all routes in the diagnostic file:

```bash
python -m scope.selector \
  --diag configs/support_diagnostics_vtab19.csv \
  --json-out results/scope_routes_export.json \
  --csv-out results/scope_routes_export.csv
```

The selector reads support diagnostics only. It rejects diagnostic files containing result-related columns such as `oracle`, `best_acc`, or `test_acc`.

## Training on an ImageFolder Dataset

The public training script expects a standard ImageFolder layout:

```text
/path/to/data/
  train/
    class_000/*.jpg
    class_001/*.jpg
  val/
    class_000/*.jpg
    class_001/*.jpg
```

Run SCOPE with automatic route selection:

```bash
python scripts/train_imagefolder.py \
  --data-root /path/to/data \
  --dataset-name custom_task \
  --model vit_base_patch16_224 \
  --pretrained \
  --epochs 20 \
  --batch-size 64 \
  --rank 8 \
  --layers all \
  --route auto \
  --out-dir runs/custom_task_scope
```

Run SCOPE with a manually specified route:

```bash
python scripts/train_imagefolder.py \
  --data-root /path/to/data \
  --dataset-name custom_task \
  --model vit_base_patch16_224 \
  --pretrained \
  --epochs 20 \
  --batch-size 64 \
  --rank 8 \
  --layers all \
  --route evidence_geometry \
  --out-dir runs/custom_task_scope_manual
```

Training outputs:

```text
runs/custom_task_scope/
  support_diagnostics.csv
  log.csv
  best.pt
  result.json
```

`--route auto` computes diagnostics from the training/support split before training. The validation split is used only for evaluation and checkpoint selection after the route has been selected.

## Result Files

Compact result and analysis files are provided for reference:

```text
results/tables/vtab1k_aoft_style_compact.csv
results/tables/vtab1k_group_summary.csv
results/tables/vtab1k_natural.csv
results/tables/vtab1k_specialized.csv
results/tables/vtab1k_structured.csv
results/tables/fgvc_main_comparison.csv

results/analysis/selector_ablation.csv
results/analysis/vtab_paired_significance.csv
results/analysis/vtab_average_rank.csv
results/analysis/fgvc_paired_gaps.csv
```

Read a table with Python:

```bash
python - <<'PY'
import csv
from pathlib import Path

path = Path('results/tables/vtab1k_group_summary.csv')
with path.open(newline='') as f:
    for row in csv.DictReader(f):
        print(row)
PY
```

## Reproduction Notes

For benchmark reproduction, ensure that all methods use the same:

```text
dataset split
backbone initialization
data normalization
training schedule
metric extraction rule
```

See `docs/PROTOCOL.md` for VTAB-1K and FGVC protocol notes.

## License

This repository is released under the MIT License. Datasets, pretrained weights, and third-party baseline implementations follow their respective licenses.
