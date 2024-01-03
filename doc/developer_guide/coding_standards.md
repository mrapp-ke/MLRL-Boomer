(coding-standards)=

# Coding Standards

As it is common for Open Source projects, where everyone is invited to contribute, we rely on coding standards to ensure that new code works as expected, does not break existing functionality, and adheres to best practices we agreed on. These coding standards are described in the following.

(ci)=

## Continuous Integration

We make use of [Github Actions](https://docs.github.com/en/actions) as a [Continuous Integration](https://en.wikipedia.org/wiki/Continuous_integration) (CI) server for running predefined jobs, such as automated tests, in a controlled environment. Whenever certain parts of the project's repository have changed, relevant jobs are automatically executed.

```{tip}
A track record of past runs can be found in the [Github repository](https://github.com/mrapp-ke/MLRL-Boomer/actions).
```

The workflow definitions of individual CI jobs can be found in the directory [.github/workflows/](https://github.com/mrapp-ke/MLRL-Boomer/tree/8ed4f36af5e449c5960a4676bc0a6a22de195979/.github/workflows). Currently, the following jobs are used in the project:

- `publish.yml` is used for publishing pre-built packages on [PyPi](https://pypi.org/) (see {ref}`installation`). For this purpose, the project is built from source for each of the target platforms and architectures, using virtualization in some cases. The job is run automatically when a new release was published on [Github](https://github.com/mrapp-ke/MLRL-Boomer/releases).
- `test_build.yml` builds the project for each of the supported target plattforms, i.e., Linux, Windows, and MacOS (see {ref}`compilation`). In the Linux environment, this job does also execute all available unit and integration tests (see {ref}`testing`) and generates the latest documentation (see {ref}`documentation`). It is run whenever relevant parts of the project's source code have been modified in a branch.
- `test_format.yml` ensures that the C++ and Python code adheres to our coding style guidelines (see {ref}`code-style`). This job is run automatically whenever any changes affecting the C++ or Python source files have been pushed to a branch.

(testing)=

## Testing the Code

To be able to detect problems with the project's source code early during development, it comes with unit and integration tests for the C++ and Python code it contains. If you want to execute all of these tests on your own system, you can use the following command:

````{tab} Linux
    ```text
    ./build tests
    ```
````

````{tab} MacOS
    ```text
    ./build tests
    ```
````

````{tab} Windows
    ```text
    build.bat tests
    ```
````

```{note}
If you want to execute the tests for the C++ or Python code independently, you can use the build target `tests_cpp` or `tests_python` instead of `tests`.
```

```{warning}
Tests for the C++ code are only executed if the project has been compiled with testing support enabled. As described in the section {ref}`build-options`, testing support is enabled by default.
```

The unit and integration tests are run automatically via {ref}`ci` whenever relevant parts of the source code have been modified.

(code-style)=

## Code Style

We aim to enforce a consistent code style across the entire project. For this purpose, we employ the following tools:

- For formatting the C++ code, we use [clang-format](https://clang.llvm.org/docs/ClangFormat.html). The desired C++ code style is defined in the file [.clang-format](https://github.com/mrapp-ke/MLRL-Boomer/blob/fece21c929043d8009aab9d52f3ed2fd03d1a191/.clang-format) in the project's root directory.
- We use [YAPF](https://github.com/google/yapf) to enforce the Python code style defined in the file [.style.yapf](https://github.com/mrapp-ke/MLRL-Boomer/blob/fece21c929043d8009aab9d52f3ed2fd03d1a191/.style.yapf). In addition, [isort](https://github.com/PyCQA/isort) is used to keep the ordering of imports in Python and Cython source files consistent according to the configuration file [.isort.cfg](https://github.com/mrapp-ke/MLRL-Boomer/blob/fece21c929043d8009aab9d52f3ed2fd03d1a191/.isort.cfg) and [pylint](https://pylint.org/) is used to check for common issues in the Python code according to the configuration file [.pylintrc](https://github.com/mrapp-ke/MLRL-Boomer/blob/fece21c929043d8009aab9d52f3ed2fd03d1a191/.pylintrc).
- For applying a consistent style to Markdown files, we use [mdformat](https://github.com/executablebooks/mdformat).

If you have modified the project's source code, you can check whether it adheres to our coding standards via the following command:

````{tab} Linux
    ```text
    ./build test_format
    ```
````

````{tab} MacOS
    ```text
    ./build test_format
    ```
````

````{tab} Windows
    ```text
    build.bat test_format
    ```
````

```{note}
If you want to check for compliance with the C++ or Python code style independently, you can use the build target `test_format_cpp` or `test_format_python` instead of `test_format`. Using the build target `test_fomat_md`, results in the style of Markdown files to be checked.
```

In order to automatically format the project's source files according to our style guidelines, the following command can be used:

````{tab} Linux
    ```text
    ./build format
    ```
````

````{tab} MacOS
    ```text
    ./build format
    ```
````

````{tab} Windows
    ```
    build.bat format
    ```
````

```{note}
If you want to format only the C++ source files, you can specify the build target `format_cpp` instead of `format`. Accordingly, the target `format_python` may be used to format only the Python source files. If you want to format Markdown files, you should use the target `format_md`.
```

Whenever any source files have been modified, a {ref}`ci` job is run automatically to verify if they adhere to our code style guidelines.
