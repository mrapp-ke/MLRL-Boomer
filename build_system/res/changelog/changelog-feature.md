# Algorithmic Enhancements

- The BOOMER algorithm now explicitly uses SIMD operations for parallelizing vector computations. This can be controlled via the command line argument `--simd`.
- It is now possible to control whether the BOOMER or SeCo algorithm should be allowed to learn rules with nominal conditions that use negations or not.

# mlrl-testbed

- The package `mlrl-testbed` now depends on [rich](https://github.com/Textualize/rich), which enables colorized and better structured console output.
