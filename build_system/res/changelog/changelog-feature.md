# API Changes

- The command line argument `--sequential-post-optimization true` has been replaced with `--post-optimization sequential`.
- Command line arguments starting with `--store-` have been renamed to use the prefix `--save-`.
- The default value of the command line argument `--save-evaluation` (previously `--store-evaluation`) has been changed from `true` to `false`.
- The command line arguments `--output-dir`, `--create-output-dir` and `--wipe-output-dir` have been renamed to `--result-dir`, `--create-dirs` and `--wipe-result-dir`.
- The command line arguments `--result-dir` (previously `--output-dir`), `--model-load-dir`, `--model-save-dir`, `--parameter-load-dir` and `--parameter-save-dir` do now come with default values.
- The command line arguments `--save-models` and `--save-parameters` have been added for specifying whether models or parameter settings should be written to output files.
- The command line arguments `--load-models` and `--load-parameters` have been added for specifying whether models or parameter settings should be loaded from input files.
- The command line argument `--base-dir` has been added. Relative directories given for the arguments `--result-dir`, `--model-load-dir`, `--model-save-dir`, `--parameter-load-dir` and `--parameter-save-dir` are considered relative to this directory.
