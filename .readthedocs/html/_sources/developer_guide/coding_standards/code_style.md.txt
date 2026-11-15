(code-style)=

# Code Style

We aim to enforce a consistent code style across the entire project. For this purpose, we employ the following tools.

## C++ Source Files

- For formatting the C++ code, we use [clang-format](https://clang.llvm.org/docs/ClangFormat.html). The desired C++ code style is defined in the file {repo-file}`.clang-format <build_system/targets/code_style/cpp/.clang-format>`.
- In addition, [cpplint](https://github.com/cpplint/cpplint) is used for static code analysis. It is configured according to the file {repo-file}`.cpplint.cfg <cpp/.cpplint.cfg>`.

## Python Source Files

- We use [ruff](https://docs.astral.sh/ruff/) for linting Python code and enforcing the code style defined in the file {repo-file}`.ruff.toml <build_system/targets/code_style/python/.ruff.toml>`.
- For static type checking of Python code, we use [mypy](https://www.mypy-lang.org/). It uses the configuration file {repo-file}`.mypy.ini <build_system/targets/code_style/python/.mypy.ini>`.

## Cython Source files

- [isort](https://github.com/PyCQA/isort) is used to keep the ordering of imports Cython source files consistent according to the configuration file {repo-file}`.isort.cfg <build_system/targets/code_style/cython/.isort.cfg>`
- To automatically detect and remove unused variables and imports, as well as unnecessary `pass` statements, in Cython code, we employ [autoflake](https://github.com/PyCQA/autoflake). The configuration of this tool can be found in the file {repo-file}`.autoflake.toml <build_system/targets/code_style/cython/.autoflake.toml>`.
- For linting Cython source files, we rely on [cython-lint](https://github.com/MarcoGorelli/cython-lint).

## Configuration Files

- We employ [config-formatter](https://github.com/Delgan/config-formatter) for formatting `.cfg` files.
- For applying a consistent style to Markdown files, including those used for writing the documentation, we use [mdformat](https://github.com/hukkin/mdformat).
- We apply [yamlfix](https://github.com/lyz-code/yamlfix) to YAML files to enforce the code style defined in the file {repo-file}`.yamlfix.toml <build_system/targets/code_style/yaml/.yamlfix.toml>`.
- We use [taplo](https://github.com/tamasfe/taplo) for validating and formatting TOML files according to the configuration file {repo-file}`.taplo.toml <build_system/targets/code_style/toml/.taplo.toml>`.

If you have modified the project's source code, you can check whether it adheres to our coding standards via the following command:

````{tab} Linux
   ```text
   ./build test_format
   ```
````

````{tab} macOS
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
If you want to check for compliance with the C++, Python or Cython code style independently, you can use the build target `test_format_cpp`, `test_format_python` or `test_format_cython` instead of `test_format`. Using the build target `test_format_cfg`, `test_format_md`, `test_format_yaml` or `test_format_toml` results in the style of Markdown, YAML or TOML files to be checked, respectively.
```

In order to automatically format the project's source files according to our style guidelines, the following command can be used:

````{tab} Linux
   ```text
   ./build format
   ```
````

````{tab} macOS
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
If you want to format only the C++ source files, you can specify the build target `format_cpp` instead of `format`. Accordingly, the targets `format_python` and `format_cython` may be used to format only the Python or Cython source files. If you want to format `.cfg`, Markdown, YAML or TOML files, you should use the target `format_cfg`, `format_md`, `format_yaml` or `format_toml`, respectively.
```

Whenever any source files have been modified, a {ref}`Continuous Integration <ci>` job is run automatically to verify if they adhere to our code style guidelines.
