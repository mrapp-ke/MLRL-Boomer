# Fixes

- An issue with our CI pipeline has been fixed that resulted in some Python packages for the macOS platform being broken due to extension modules being compiled for the wrong Python version.

# Documentation

- The chapter "Algorithmic Optimizations" has been added to the documentation. It describes implementation details of our algorithms in great detail.

# Quality-of-Life Improvements

- Our build system now uses [uv](https://docs.astral.sh/uv/) as a faster replacement of the package manager pip.
- We now use [ruff](https://docs.astral.sh/ruff/) for linting and formatting Python code. This makes the dependencies [YAPF](https://github.com/google/yapf) and [pylint](https://pylint.org/) obsolete.
- To speed up GitHub workflows, the integration tests have been streamlined and redundant test cases have been removed.
- Python packages built by our CI workflows are now run in a test environment to ensure that they have been packaged correctly.
