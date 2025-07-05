# Quality-of-Life Improvements

- For testing Python packages under more realistic conditions, they are now installed into isolated CI jobs, separated from the build jobs.
- [pytest](https://pytest.org/) is now used for testing the Python code.
- Python tests are now divided into blocks that are run in parallel by CI jobs

# Fixes

- Fixed an issue that caused the options `first_fold` and `last_fold` to be swapped when using the command line argument `--data-split cross-validation`.
