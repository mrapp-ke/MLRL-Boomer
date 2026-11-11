(testing)=

# Automated Tests

To be able to detect problems with the project's source code early during development, it comes with unit and integration tests for the C++ and Python code it contains. If you want to execute all of these tests on your own system, you can use the following command:

````{tab} Linux
   ```text
   ./build tests
   ```
````

````{tab} macOS
   ```text
   ./build tests
   ```
````

````{tab} Windows
   ```text
   build.bat tests
   ```
````

This will result in all tests being run and their results being reported. The tests are run automatically for pull requests via {ref}`Continuous Integration <ci>` whenever relevant parts of the source code have been modified.

```{note}
If you want to execute the tests for the C++ or Python code independently, you can use the build target `tests_cpp` or `tests_python` instead of `tests`.
```

```{important}
Tests for the C++ code are only executed if the project has been compiled with testing support enabled. As described in the section {ref}`build-options`, testing support is enabled by default if the [GoogleTest](https://github.com/google/googletest) framework is available on the system.
```

## Using Multiple Workers

If the execution of Python tests should be parallelized across multiple workers, the environment variable `NUM_WORKERS` may be used:

````{tab} Linux
   ```text
   NUM_WORKERS=4 ./build tests
   ```
````

````{tab} macOS
   ```text
   FAIL_FAST=4 ./build tests
   ```
````

````{tab} Windows
   ```text
   $env:NUM_WORKERS = 4
   build.bat tests
   ```
````

```{note}
If the environment variable `NUM_WORKERS` is set to the value `auto`, the number of workers is chosen automatically based on the number of available CPU cores.
```

## Failing Fast

If the execution should be aborted as soon as a single test fails, the environment variable `FAIL_FAST` can be used as shown below:

````{tab} Linux
   ```text
   FAIL_FAST=true ./build tests
   ```
````

````{tab} macOS
   ```text
   FAIL_FAST=true ./build tests
   ```
````

````{tab} Windows
   ```text
   $env:FAIL_FAST = "true"
   build.bat tests
   ```
````

## Running Only Failed Tests

To run only the tests that failed at the last run (or all if none failed), the environment `ONLY_FAILED` can be specified:

````{tab} Linux
   ```text
   ONLY_FAILED=true ./build tests
   ```
````

````{tab} macOS
   ```text
   ONLY_FAILED=true ./build tests
   ```
````

````{tab} Windows
   ```text
   $env:ONLY_FAILED = "true"
   build.bat tests
   ```
````

## Running Only Selected Tests

It is also possible to only run the tests for certain subprojects (see {ref}`project-structure`) by providing their names as a comma-separated list via the environment variable `SUBPROJECTS`:

````{tab} Linux
   ```text
   SUBPROJECTS=boosting,seco ./build tests
   ```
````

````{tab} macOS
   ```text
   SUBPROJECTS=boosting,seco ./build tests
   ```
````

````{tab} Windows
   ```text
   $env:SUBPROJECTS = "boosting,seco"
   build.bat tests
   ```
````

Alternatively, markers of the tests that should be run can be specified via the environment variable `MARKERS`. This provides more fine-grained control, but applies only to Python tests. The following markers are available: `boosting`, `seco`, `classification`, `regression`.

````{tab} Linux
   ```text
   MARKERS=boosting,seco ./build tests
   ```
````

````{tab} macOS
   ```text
   MARKERS=boosting,seco ./build tests
   ```
````

````{tab} Windows
   ```text
   $env:MARKERS = "boosting,seco"
   build.bat tests
   ```
````

Finally, it is also possible to run tests by their name. For this purpose, the environment variable `TEST_NAME` allows to specify a substring to search for in the names of tests. Only those that contain the substring in their name are run by the following command:

````{tab} Linux
   ```text
   TEST_NAME=test_evaluation ./build tests
   ```
````

````{tab} macOS
   ```text
   TEST_NAME=test_evaluation ./build tests
   ```
````

````{tab} Windows
   ```text
   $env:TEST_NAME = "test_evaluation"
   build.bat tests
   ```
````

## Overwriting Output Files

When using the build target `tests_python`, the environment variable `OVERWRITE_OUTPUT_FILES` may be utilized to overwrite the files in the directory {repo-dir}`python/tests/res/out/` with the actual output of the corresponding test cases:

````{tab} Linux
  ```text
  OVERWRITE_OUTPUT_FILES=true ./build tests_python
  ```
````

````{tab} macOS
   ```text
   OVERWRITE_OUTPUT_FILES=true ./build tests_python
   ```
````

````{tab} Windows
   ```text
   $env:OVERWRITE_OUTPUT_FILES = "true"
   build.bat tests_python
   ```
````
