# Fixes

- Resolved an issue with the YAML schema for validating configuration files required by mlrl-testbed's batch mode.
- If commands in a sbatch script use the bracket notation for specifying options of algorithmic parameters, they are now properly escaped.
- When using mlrl-testbed's read mode with the argument "--separate-folds true" (which is the default) for running experiments for individual cross validation folds, output files are now prevented from being deleted by concurrent experiments.
