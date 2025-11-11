# API Changes

- The BOOMER algorithm now uses 32- rather than 64-bit statistics by default. This typically reduces the algorithm's training time and memory footprint without a significant loss of predictive accuracy.
- The package mlrl-testbed provides two new modes for re-running experiments and reading their output files, respectively. They can be enabled via the arguments `--mode run` or `--mode read` and are based on reading the metadata.yml files that have been created by previous experiments.
- Experiments are now canceled by default if all of their output files do already exist. This behavior can be adjusted via the newly added argument `--if-outputs-exist`.
- Default arguments can now be defined in the YAML configuration required by mlrl-testbed's batch mode.
- Support for datasets in the LibSVM format has been added to the package mlrl-testbed-sklearn.
- The ranking measures DCG@k and DCG@k with predefined values for k have been added to mlrl-testbed-sklearn.
- The command line argument `--dataset-format` has been added. It allows to explicitly specify the format of a dataset to be loaded.
- The command line argument `--if-input-missing` has been added. It allows to exit the program mlrl-testbed if any error occurs while reading input data. The argument `--exit-on-error` has been replaced with a similar argument `--if-output-error`.
- The command line argument `--print-meta-data` has been added. It allows to print the meta-data of an experiment on the console.
- When using mlrl-testbed's batch mode with the arguments `--separate-folds true` and `--runner slurm`, Slurm jobs that correspond to different folds of a cross validation are now automatically assigned to [job arrays](https://slurm.schedmd.com/job_array.html).
- The command line argument `--wipe-result-dir` has been removed with replacement.
- The option `sparse` has been removed from the command line arguments `--print-label-vectors` and `--save-label-vectors`. Label vectors are now always represented in a sparse format instead of a dense one.
- The option `percentage` has been removed from the command line arguments `--save-evaluation`, `--save-data-characteristics` and `--save-prediction-characteristics`. Values stored via these arguments are now always stored as percentages, if possible.

# Quality-of-Life Improvements

- The implementations of output space statistics used by the BOOMER and SeCo algorithm have been unified and common functionality was moved into the library "libmlrlcommon".
