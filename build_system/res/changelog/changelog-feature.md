# API Changes

- The executable `testbed` has been renamed to `mlrl-testbed`.
- The command line argument `--model-dir` has been replaced with the arguments `--model-load-dir` and `--model-save-dir`, which enables to save models to a different directory than the one they are loaded from. 
- The command line argument `--parameter-dir` has been replaced with the arguments `--parameter-load-dir` and `--parameter-save-dir`. The former specifies the directory, parameter settings should be loaded from, whereas the latter replaces the argument `--store-parameters`.
- The options `min_samples` and `max_samples` have been added to the values of the command line arguments `--feature-sampling` and `--instance-sampling`.
- The indices of nominal and ordinal features are now provided to a learner's `fit`-method via the keyword arguments `nominal_feature_indices` and `ordinal_feature_indices`.
- The Python API does now allow to provide custom weights for training examples to a learner's `fit`-method via the keyword argument `sample_weights`. 

# Algorithmic Enhancements
- The BOOMER algorithm can now be configured to use either 32- or 64-bit floating point values for gradients and Hessians via the command line argument `--statistic-type`. Using lower-precision values might speed up training at the risk of losing training accuracy.
- Efficient data types and data structures are now used for storing binary scores calculated by the SeCo algorithm.
- Unnecessary conversions from integer weights to floating point values are now avoided.

# Quality-of-Life Improvements

- C++ 20 is now required for compiling the project.
