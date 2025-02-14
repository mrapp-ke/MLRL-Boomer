(coding-standards)=

# Coding Standards

As it is common for Open Source projects, where everyone is invited to contribute, we rely on coding standards to ensure that new code works as expected, does not break existing functionality, and adheres to best practices we agreed on. These coding standards are described in the following.

(testing)=

## Testing the Code

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

This will result in all tests being run and their results being reported. If the execution should be aborted as soon as a single test fails, the environment variable `FAIL_FAST` can be used as shown below:

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

```{note}
If you want to execute the tests for the C++ or Python code independently, you can use the build target `tests_cpp` or `tests_python` instead of `tests`.
```

`````{note}
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
`````

```{warning}
Tests for the C++ code are only executed if the project has been compiled with testing support enabled. As described in the section {ref}`build-options`, testing support is enabled by default if the [GoogleTest](https://github.com/google/googletest) framework is available on the system.
```

The unit and integration tests are run automatically via {ref}`Continuous Integration <ci>` whenever relevant parts of the source code have been modified.

(code-style)=

## Code Style

We aim to enforce a consistent code style across the entire project. For this purpose, we employ the following tools:

- For formatting the C++ code, we use [clang-format](https://clang.llvm.org/docs/ClangFormat.html). The desired C++ code style is defined in the file {repo-file}`.clang-format <build_system/targets/code_style/cpp/.clang-format>`. In addition, [cpplint](https://github.com/cpplint/cpplint) is used for static code analysis. It is configured according to the file {repo-file}`.cpplint.cfg <cpp/.cpplint.cfg>`.
- We use [YAPF](https://github.com/google/yapf) to enforce the Python code style defined in the file {repo-file}`.style.yapf <build_system/targets/code_style/python/.style.yapf>`. In addition, [isort](https://github.com/PyCQA/isort) is used to keep the ordering of imports in Python and Cython source files consistent according to the configuration file {repo-file}`.isort.cfg <build_system/targets/code_style/python/.isort.cfg>` and [pylint](https://pylint.org/) is used to check for common issues in the Python code according to the configuration file {repo-file}`.pylintrc <build_system/targets/code_style/python/.pylintrc>`.
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
If you want to check for compliance with the C++ or Python code style independently, you can use the build target `test_format_cpp` or `test_format_python` instead of `test_format`. Using the build target `test_format_md`, `test_format_yaml` or `test_format_toml` results in the style of Markdown, YAML or TOML files to be checked, respectively.
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
If you want to format only the C++ source files, you can specify the build target `format_cpp` instead of `format`. Accordingly, the target `format_python` may be used to format only the Python source files. If you want to format Markdown, YAML or TOML files, you should use the target `format_md`, `format_yaml` or `format_toml`, respectively.
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

(downstream-merges)=

### Downstream Merges

Once modifications to one of the branches have been merged, {ref}`Continuous Integration <ci>` jobs are used to automatically update downstream branches via pull requests. If all checks for such pull requests are successful, they are merged automatically. If there are any merge conflicts, they must be resolved manually. Following this procedure, changes to the feature branch are merged into the main branch, whereas changes to the bugfix branch are first merged into the feature branch and then into the main branch (see description of {repo-file}`merge_feature.yml <.github/workflows/merge_feature.yml>` and {repo-file}`merge_bugfix.yml <.github/workflows/merge_bugfix.yml>` in {ref}`ci-releases`).

(triggering-releases)=

### Triggering Releases

We use a {ref}`Continuous Integration <ci>` job for triggering a new release, including the changes of one of the branches mentioned above (see description of {repo-file}`release.yml <.github/workflows/release.yml>` in {ref}`ci-releases`). Depending on the release branch, the job automatically collects the corresponding changelog entries from the files {repo-file}`.changelog-main.md`, {repo-file}`.changelog-feature.md`, and {repo-file}`.changelog-bugfix.md` and updates the file {repo-file}`CHANGELOG.md` in the project's root directory accordingly. Afterward, it will publish the new release on GitHub, which will in turn trigger the publishing of pre-built packages (see description of {repo-file}`publish.yml <.github/workflows/publish.yml>` in {ref}`ci-publishing`).

(upstream-merges)=

### Upstream Merges

Whenever a new release has been published, the release branch is merged into the upstream branches (see description of {repo-file}`merge_release.yml <.github/workflows/merge_release.yml>` in {ref}`ci-releases`), i.e., major releases result in the feature and bugfix branches being updated, whereas minor releases result in the bugfix branch being updated. The version of the release branch and the affected branches are updated accordingly. The file {repo-file}`.version` in the project's root directory specifies the version of each of these branches. Similarly, the file {repo-file}`.version-dev` keeps track of the version number used for development releases (see description of {repo-file}`publish_development.yml <.github/workflows/publish_development.yml>` in {ref}`ci-publishing`).

(dependencies)=

## Dependencies

Adding dependencies to a software project always comes at a cost. Maintainers need to continuously test their software as new versions of dependencies are released and major changes in their APIs may break existing functionality. For this reason, we try to keep the number of dependencies at a minimum.

That being said, we still rely on several dependencies for Continuous Integration, compiling our source code, generating the documentation, or running the algorithms provided by this project. When using pre-built packages from [PyPI](https://pypi.org/project/mlrl-boomer/), there is no need to care about these dependencies, as they are already included in the packages. When {ref}`building from source <compilation>`, dependencies are automatically installed by the build system once they are needed, unless explicitly stated in the documentation.

### Python Dependencies

Python dependencies that are required by different aspects of the project, such as the build system, the documentation, or our own Python code, are defined in separate `requirements.txt` and `pyproject.template.toml` files. For dependencies that use [Semantic Versioning](https://semver.org/), we specify the earliest and latest version we support. For other dependencies, we demand for a specific version number. This strives to achieve a balance between flexibility for users and comfort for developers. On the one hand, supporting a range of versions provides more freedom to users, as our packages can more flexibly be used together with other ones, relying on the same dependencies. On the other hand, the project's maintainers must not manually update dependencies that have a minor release, while still requiring manual intervention for major updates.

To ease the life of developers, the following command provided by the project's build system may be used to check for outdated dependencies:

````{tab} Linux
   ```text
   ./build check_dependencies
   ```
````

````{tab} macOS
   ```text
   ./build check_dependencies
   ```
````

````{tab} Windows
   ```
   build.bat check_dependencies
   ```
````

Alternatively, the following command may be used to update the versions of outdated dependencies automatically:

````{tab} Linux
   ```text
   ./build update_dependencies
   ```
````

````{tab} macOS
   ```text
   ./build update_dependencies
   ```
````

````{tab} Windows
   ```
   build.bat update_dependencies
   ```
````

```{note}
If you want to restrict the above commands to the build-time dependencies, required by the project's build system, or the runtime dependencies, required for running its algorithms, you can use the targets `check_build_dependencies`, `check_runtime_dependencies`, `update_build_dependencies`, and `update_runtime_dependencies` instead.
```

### GitHub Actions

Our {ref}`Continuous Integration <ci>` (CI) jobs heavily rely on so-called [Actions](https://docs.github.com/actions/sharing-automations/reusing-workflows), which are reusable building blocks provided by third-party developers. As with all dependencies, updates to these Actions may introduce breaking changes. To reduce the risk of updates breaking our CI jobs, we pin the Actions to a certain version. Usually, we only restrict the major version required by a job, rather than specifying a specific version. This allows minor updates, which are less likely to cause problems, to take effect without manual intervention.

The project's build system allows to automatically check for outdated Actions used by the project's CI jobs. The following command prints a list of all outdated Actions:

````{tab} Linux
   ```text
   ./build check_github_actions
   ```
````

````{tab} macOS
   ```text
   ./build check_github_actions
   ```
````

````{tab} Windows
   ```
   build.bat check_github_actions
   ```
````

Alternatively, the following command may be used to update the versions of outdated Actions automatically:

````{tab} Linux
   ```text
   ./build update_github_actions
   ```
````

````{tab} macOS
   ```text
   ./build update_github_actions
   ```
````

````{tab} Windows
   ```
   build.bat update_github_actions
   ```
````

```{note}
The above commands query the [GitHub API](https://docs.github.com/rest) for the latest version of relevant GitHub Actions. You can optionally specify an [API token](https://docs.github.com/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) to be used for these queries via the environment variable `GITHUB_TOKEN`. If no token is provided, repeated requests might fail due to GitHub's rate limit.
```

### GitHub Runners

For running {ref}`Continuous Integration <ci>` (CI) jobs, we use [runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners) hosted by GitHub. Runners are available for different operating systems and architectures, which is particularly relevant when building packages for the various target platforms we support. To avoid breaking the build process when GitHub updates its runners, we specify the exact version required by a particular CI job.

Our build system provides the following command to check for outdated runners used by the project:

````{tab} Linux
   ```text
   ./build check_github_runners
   ```
````

````{tab} macOS
   ```text
   ./build check_github_runners
   ```
````

````{tab} Windows
   ```
   build.bat check_github_runners
   ```
````

In addition, the command below can be used to update the versions of outdated runners automatically:

````{tab} Linux
   ```text
   ./build update_github_runners
   ```
````

````{tab} macOS
   ```text
   ./build update_github_runners
   ```
````

````{tab} Windows
   ```
   build.bat update_github_runners
   ```
````
