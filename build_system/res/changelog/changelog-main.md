# API Changes

- Models of the BOOMER and SeCo algorithm are now stored in the HDF5 file format when using Python's pickle mechanism. Models that have been pickled by previous versions cannot be loaded anymore and must be trained again.
- The library [HDF5](https://www.hdfgroup.org/solutions/hdf5/) is now a mandatory dependency for building the project from source. The pre-built packages that are available on [PyPI](https://pypi.org/) already include the library.

# Quality-of-Life Improvements

- Fitted estimators of the BOOMER and SeCo algorithm can now be saved to HDF5 files via the method `save_model` and loaded via the function `mlrl.common.load_model`. Unlike pickle files, these files do not execute any code when being loaded and can be loaded by any version of the software that supports the version of the file format they have been saved with.
- The package `mlrl-testbed` now saves the models of the BOOMER and SeCo algorithm as HDF5 files with the suffix `.h5` instead of `.pickle` files. Such files can also be loaded outside of `mlrl-testbed` via the function `mlrl.common.load_model`.
