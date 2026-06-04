# Contributing

Contributions should preserve paper reproducibility.

Before opening a pull request:

1. Keep changes scoped to one issue or experiment.
2. Do not commit datasets, model binaries, caches, virtual environments, or private files.
3. Run `python -m compileall src tfcs_v2_full.py`.
4. Document any change that alters preprocessing, splits, thresholds, metrics, prompts, or model hyperparameters.
5. If results change, update `REPRODUCIBILITY.md` with the reason and the new expected values.

Experimental extensions should live on a separate branch until they are promoted into the documented paper pipeline.
