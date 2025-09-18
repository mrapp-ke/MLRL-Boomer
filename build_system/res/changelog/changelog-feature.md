# API Changes

- The package MLRL-Testbed provides two new modes for re-running experiments and reading their output files, respectively. They can be enabled via the arguments `--mode run` or `--mode read` and are based on reading the metadata.yml files that have been created by previous experiments.
- The command line argument `--exit-on-missing-input` has been added. It allows to exit the program MLRL-Testbed if any error occurs while reading input data.
- The command line argument `--print-meta-data` has been added. It allows to print the meta-data of an experiment on the console.
