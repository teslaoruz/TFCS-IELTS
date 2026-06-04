#!/usr/bin/env bash
set -euo pipefail

python -m src.experiments.build_splits
rm -rf results/tfcs_v2/cache
python tfcs_v2_full.py
