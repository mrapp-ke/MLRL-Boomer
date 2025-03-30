# Quality-of-Life Improvements

- The build targets `format_python` and `test_format_python` now employ [autoflake](https://github.com/PyCQA/autoflake) to detect and remove unused variables and imports, as well as unnecessary `pass` statements from Python and Cython source files.
