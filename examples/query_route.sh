#!/usr/bin/env bash
set -euo pipefail
python -m scope.selector --diag configs/support_diagnostics_vtab19.csv --dataset dtd
