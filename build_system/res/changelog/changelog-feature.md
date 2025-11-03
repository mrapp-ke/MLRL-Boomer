# API Changes

- The package mlrl-testbed provides two new modes for re-running experiments and reading their output files, respectively. They can be enabled via the arguments `--mode run` or `--mode read` and are based on reading the metadata.yml files that have been created by previous experiments.
- Experiments are now canceled by default if all of their output files do already exist. This behavior can be adjusted via the newly added argument `--if-outputs-exist`.
- Default arguments can now be defined in the YAML configuration required by mlrl-testbed's batch mode.
- The command line argument `--if-input-missing` has been added. It allows to exit the program mlrl-testbed if any error occurs while reading input data. The argument `--exit-on-error` has been replaced with a similar argument `--if-output-error`.
- The command line argument `--print-meta-data` has been added. It allows to print the meta-data of an experiment on the console.
- When using mlrl-testbed's batch mode with the arguments `--separate-folds true` and `--runner slurm`, Slurm jobs that correspond to different folds of a cross validation are now automatically assigned to [job arrays](https://slurm.schedmd.com/job_array.html).
- The option `sparse` has been removed from the command line arguments `--print-label-vectors` and `--save-label-vectors`. Label vectors are now always represented in a sparse format instead of a dense one.
- The option `percentage` has been removed from the command line arguments `--save-evaluation`, `--save-data-characteristics` and `--save-prediction-characteristics`. Values stored via these arguments are now always stored as percentages, if possible.

# Quality-of-Life Improvements

- The implementations of output space statistics used by the BOOMER and SeCo algorithm have been unified and common functionality was moved into the library "libmlrlcommon".
