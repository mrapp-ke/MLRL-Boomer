# Fixes

- An issue that caused thresholds and probabilities of isotonic regression models being swapped in output files when using the command line arguments `--store-marginal-probability-calibration-model` and `--store-joint-probability-calibration-model` has been fixed.

# Quality-of-Life Improvements

- The integration tests do now check the contents of output files.
- The build targets `format_python` and `test_format_python` now employ [autoflake](https://github.com/PyCQA/autoflake) to detect and remove unused variables and imports, as well as unnecessary `pass` statements from Python and Cython source files.
- The build targets `format_cfg` and `test_format_cfg` have been added. They enforce a consistent style for .cfg files by employing the package [config-formatter](https://github.com/Delgan/config-formatter).
- The tool [cython-lint](https://github.com/MarcoGorelli/cython-lint) is now applied to Cython source files via Continuous Integration.
