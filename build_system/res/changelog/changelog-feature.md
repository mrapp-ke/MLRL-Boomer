# API Changes

- In batch mode, the argument `--base-dir` is now passed to individual experiments by default.
- The command line argument `--save-meta-data` has been added. It allows to control if meta-data should be written to output files.
- A new package [mlrl-testbed-slurm](https://pypi.org/project/mlrl-testbed-slurm/) has been added to the project. It is an extension that adds support for the Slurm Workload Manager to the package "mlrl-testbed".
