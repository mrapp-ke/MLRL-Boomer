(coding-standards)=

# Coding Standards

As it is common for Open Source projects, where everyone is invited to contribute, we rely on coding standards to ensure that new code works as expected, does not break existing functionality, and adheres to best practices we agreed on. These coding standards are described in the following.

(ci)=

## Continuous Integration

We make use of [Github Actions](https://docs.github.com/en/actions) as a [Continuous Integration](https://en.wikipedia.org/wiki/Continuous_integration) (CI) server for running predefined jobs, such as automated tests, in a controlled environment. Whenever certain parts of the project's repository have changed, relevant jobs are automatically executed.

```{tip}
A track record of past runs can be found on Github in the [Actions](https://github.com/mrapp-ke/MLRL-Boomer/actions) tab.
```

The workflow definitions of individual CI jobs can be found in the directory [.github/workflows/](https://github.com/mrapp-ke/MLRL-Boomer/tree/8ed4f36af5e449c5960a4676bc0a6a22de195979/.github/workflows). Currently, the following jobs are used in the project:

- `publish.yml` is used for publishing pre-built packages on [PyPI](https://pypi.org/) (see {ref}`installation`). For this purpose, the project is built from source for each of the target platforms and architectures, using virtualization in some cases. The job is run automatically when a new release was published on [Github](https://github.com/mrapp-ke/MLRL-Boomer/releases). It does also increment the project's major version number and merge the release branch into its upstream branches (see {ref}`release-process`).
- `publish_development.yml` publishes development versions of packages on [Test-PyPI](https://test.pypi.org/) whenever changes to the project's source code have been pushed to the main branch. The packages built by each of these runs are also saved as [artifacts](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts) and can be downloaded as zip archives.
- `test_publish.yml` ensures that the packages to be released for different architectures and Python versions can be built. The job is run for pull requests that modify relevant parts of the source code.
- `test_build.yml` builds the project for each of the supported target platforms, i.e., Linux, Windows, and MacOS (see {ref}`compilation`). In the Linux environment, this job does also execute all available unit and integration tests (see {ref}`testing`). It is run for pull requests whenever relevant parts of the project's source code have been modified.
- `test_doc.yml` generates the latest documentation (see {ref}`documentation`) whenever relevant parts of the source code are affected by a pull request.
- `test_format.yml` ensures that all source files in the project adhere to our coding style guidelines (see {ref}`code-style`). This job is run automatically for pull requests whenever they include any changes affecting the relevant source files.
- `merge_feature.yml` and `merge_bugfix.yml` are used to merge changes that have been pushed to the feature or bugfix branch into downstream branches via pull requests (see {ref}`release-process`).
- `merge_release.yml` is used to merge all changes included in a new release published on [Github](https://github.com/mrapp-ke/MLRL-Boomer/releases) into upstream branches and update the version numbers of these branches.

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

This will result in all tests being run and their results being reported. If the execution should be aborted as soon as a single test fails, the environment variable `SKIP_EARLY` can be used as shown below:

````{tab} Linux
   ```text
   SKIP_EARLY=true ./build tests
   ```
````

````{tab} MacOS
   ```text
   SKIP_EARLY=true ./build tests
   ```
````

````{tab} Windows
   ```text
   $env:SKIP_EARLY = "true"
   build.bat tests
   ```
````

```{note}
If you want to execute the tests for the C++ or Python code independently, you can use the build target `tests_cpp` or `tests_python` instead of `tests`.
```

```{warning}
Tests for the C++ code are only executed if the project has been compiled with testing support enabled. As described in the section {ref}`build-options`, testing support is enabled by default if the [GoogleTest](https://github.com/google/googletest) framework is available on the system.
```

The unit and integration tests are run automatically via {ref}`Continuous Integration <ci>` whenever relevant parts of the source code have been modified.

(code-style)=

## Code Style

We aim to enforce a consistent code style across the entire project. For this purpose, we employ the following tools:

- For formatting the C++ code, we use [clang-format](https://clang.llvm.org/docs/ClangFormat.html). The desired C++ code style is defined in the file `.clang-format` in the project's root directory. In addition, [cpplint](https://github.com/cpplint/cpplint) is used for static code analysis. It uses the configuration file `CPPLINT.cfg`.
- We use [YAPF](https://github.com/google/yapf) to enforce the Python code style defined in the file `.style.yapf`. In addition, [isort](https://github.com/PyCQA/isort) is used to keep the ordering of imports in Python and Cython source files consistent according to the configuration file `.isort.cfg` and [pylint](https://pylint.org/) is used to check for common issues in the Python code according to the configuration file `.pylintrc`.
- For applying a consistent style to Markdown files, including those used for writing the documentation, we use [mdformat](https://github.com/executablebooks/mdformat).
- We apply [yamlfix](https://github.com/lyz-code/yamlfix) to YAML files to enforce the code style defined in the file `.yamlfix.toml`.

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
If you want to check for compliance with the C++ or Python code style independently, you can use the build target `test_format_cpp` or `test_format_python` instead of `test_format`. Using the build target `test_format_md` or `test_format_yaml` results in the style of Markdown or YAML files to be checked, respectively.
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
If you want to format only the C++ source files, you can specify the build target `format_cpp` instead of `format`. Accordingly, the target `format_python` may be used to format only the Python source files. If you want to format Markdown or YAML files, you should use the target `format_md` or `format_yaml`, respectively.
```

Whenever any source files have been modified, a {ref}`Continuous Integration <ci>` job is run automatically to verify if they adhere to our code style guidelines.

(versioning-scheme)=

## Versioning Scheme

We use [Semantic Versioning](https://semver.org/) to assign unique version numbers in the form `MAJOR.MINOR.PATCH` to the individual releases of our software packages. We refer to releases that come with an incremented major version, as *major releases*. When the minor version is increased by a release, we refer to it as a *feature release*. Updates that include bugfixes or minor improvements come with an increased patch version and are referred to as *bugfix releases*.

```{tip}
An overview of past releases, together with a description of the changes they introduced compared to the previous version, can be found in the {ref}`release notes <release-notes>`.
```

### Bugfix Releases

Bugfix releases are limited to backward-compatible changes, such as bug fixes, performance optimizations, improvements to the build system, or updates of the documentation. They are neither allowed to introduce any compatibility-breaking changes to the command line API, nor to any of the programmatic APIs in the project's Python or C++ code.

### Feature Releases

Feature releases may come with changes that do not break compatibility with the command line API or programmatic APIs provided by previous versions. As a consequence, new functionalities can be added to the algorithms provided by this project, if they do not break existing functionality. In contrast, the removal of features is only allowed for major releases.

Feature releases with the major version `0` are not obliged to maintain API compatibility, because these releases are considered to represent an early stage of development, where things may change drastically from one version to another.

### Major Releases

Increments of the major version indicate big leaps in the software's development. They are reserved for new versions of the software that introduce new functionality, fundamentally change how the software works, or come with compatibility-breaking changes. In general, major releases are not guaranteed to be compatible with past releases in any way. In particular, they may introduce compatibility-breaking API changes, affecting the command line API or programmatic APIs in the project's Python or C++ code. Moreover, models that have been trained using an older version are not guaranteed to work after updating to a new major release and must potentially be trained from scratch.

(release-process)=

## Release Process

To enable releasing new major, feature, or bugfix releases at any time, we maintain a branch for each type of release:

- `main` contains all changes that will be included in the next major release (including changes on the feature and bugfix branch).
- `feature` comes with the changes that will be part of an upcoming feature release (including changes on the bugfix branch).
- `bugfix` is restricted to minor changes that will be published as a bugfix release.

We do not allow directly pushing to the above branches. Instead, all changes must be submitted via pull requests and require certain checks to pass.

Once modifications to one of the branches have been merged, {ref}`Continuous Integration <ci>` jobs are used to automatically update downstream branches via pull requests. If all checks for such pull requests are successful, they are merged automatically. If there are any merge conflicts, they must be resolved manually. Following this procedure, changes to the feature brach are merged into the main branch (see `merge_feature.yml`), whereas changes to the bugfix branch are first merged into the feature branch and then into the main branch (see `merge_bugfix.yml`).

Whenever a new release has been published, the release branch is merged into the upstream branches (see `merge_release.yml`), i.e., major releases result in the feature and bugfix branches being updated, whereas minor releases result in the bugfix branch being updated. The version of the release branch and the affected branches are updated accordingly. The version of a branch is specified in the file `.version` in the project's root directory. Similarly, the file `.version-dev` is used to keep track of the version number used for development releases (see `publish_development.yml`).

(dependencies)=

## Dependencies

Adding dependencies to a software project always comes at a cost. Maintainers need to continuously test their software as new versions of dependencies are released and major changes in their APIs may break existing functionality. For this reason, we try to keep the number of dependencies at a minimum.

That being said, we still rely on several dependencies for compiling our source code, generating the documentation, or running the algorithms provided by this project. When using pre-built packages from [PyPI](https://pypi.org/project/mlrl-boomer/), there is no need to care about these dependencies, as they are already included in the packages. When {ref}`building from source <compilation>`, dependencies are automatically installed by the build system once they are needed, unless explicitly stated in the documentation.

The dependencies that are required by different aspects of the project, such as the build system, the Python code, or the C++ code, are defined in separate `requirements.txt` files. For dependencies that use [Semantic Versioning](https://semver.org/), we specify the earliest and latest version we support. For other dependencies, we demand for a specific version number. This strives to achieve a balance between flexibility for users and comfort for developers. On the one hand, supporting a range of versions provides more freedom to users, as our packages can more flexibly be used together with other ones, relying on the same dependencies. On the other hand, the project's maintainers must not manually update dependencies that have a minor release, while still requiring manual intervention for major updates.

To ease the life of developers, the following command provided by the project's build system may be used to check for outdated dependencies:

````{tab} Linux
   ```text
   ./build check_dependencies
   ```
````

````{tab} MacOS
   ```text
   ./build check_dependencies
   ```
````

````{tab} Windows
   ```
   build.bat check_dependencies
   ```
````
