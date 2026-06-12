#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DIAG=${DIAG:-$ROOT/configs/support_diagnostics_vtab19.csv}
DATASET=${1:?Usage: bash scripts/run_scope_vtab19.sh <dataset> [epochs]}
EPOCHS=${2:-100}
ROUTE=$(python3 -m scope.selector --diag "$DIAG" --dataset "$DATASET")

echo "SCOPE dataset: $DATASET"
echo "SCOPE route:   $ROUTE"
echo "Epochs:       $EPOCHS"
echo
cat <<'MSG'
This public artifact exposes the SCOPE support selector and primitive-level
configuration. Full training requires the manuscript training environment,
datasets, and ViT checkpoints. Use the selected SCOPE route above to launch the
corresponding primitive implementation in your training stack.
MSG
