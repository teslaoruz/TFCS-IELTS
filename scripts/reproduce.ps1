$ErrorActionPreference = "Stop"

python -m src.experiments.build_splits
if (Test-Path "results\tfcs_v2\cache") {
    Remove-Item -LiteralPath "results\tfcs_v2\cache" -Recurse -Force
}
python tfcs_v2_full.py
