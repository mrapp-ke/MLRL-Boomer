# API Changes

- The package MLRL-Testbed does now provide a new mode for re-running experiments based on metadata.yml files that have been created by previous runs. It can be enabled via the argument `--mode read`.
- The command line argument `--exit-on-missing-input` has been added. It allows to exit the program MLRL-Testbed if any error occurs while reading input data.
- The command line argument `--print-meta-data` has been added. It allows to print the meta-data of an experiment on the console.
