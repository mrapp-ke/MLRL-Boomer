(ci)=

# Continuous Integration

We make use of [GitHub Actions](https://docs.github.com/actions) as a [Continuous Integration](https://en.wikipedia.org/wiki/Continuous_integration) (CI) server for running predefined jobs, such as automated tests, in a controlled environment. Whenever certain parts of the project's repository have changed, relevant jobs are automatically executed.

```{tip}
A track record of past runs can be found on GitHub in the [Actions](https://github.com/mrapp-ke/MLRL-Boomer/actions) tab.
```

The workflow definitions of individual CI jobs can be found in YAML files located in the directory `.github/workflows/`. In the following, we discuss the different tasks dealt with by these workflows.

(ci-testing)=

## Quality Control

The following CI jobs are run automatically for all pull requests to ensure that their changes do not introduce any regressions. If one or more jobs fail, the respective pull request cannot be merged.

- `test_build.yml` builds the project for each of the supported target platforms, i.e., Linux, Windows, and macOS (see {ref}`compilation`). In the Linux environment, this job does also execute all available unit and integration tests (see {ref}`testing`). It is run for pull requests whenever relevant parts of the project's source code have been modified.
- `test_changelog.yml` ensures that all changelog files in the project adhere to the structure that is necessary to be processed automatically when publishing a new release. This job is run for pull requests if they modify one of the changelog files.
- `test_doc.yml` generates the latest documentation (see {ref}`documentation`) whenever relevant parts of the source code are affected by a pull request.
- `test_format.yml` ensures that all source files in the project adhere to our coding style guidelines (see {ref}`code-style`). This job is run automatically for pull requests whenever they include any changes affecting the relevant source files.
- `test_publish.yml` ensures that the packages to be released for different architectures and Python versions can be built. The job is run for pull requests that modify relevant parts of the source code.

(ci-releases)=

## Automated Releases

The release process described {ref}`here <release-process>` is fully automated via the CI jobs listed below. They are responsible for merging changes, which have been added to the code base via pull requests, to the correct release branches, updating the version numbers of these branches, and triggering releases.

- `merge_feature.yml` and `merge_bugfix.yml` are used to merge changes that have been pushed to the feature or bugfix branch into downstream branches via pull requests (see {ref}`downstream-merges`).
- `merge_release.yml` is used to merge all changes included in a new release published on [GitHub](https://github.com/mrapp-ke/MLRL-Boomer/releases) into upstream branches and update the version numbers of these branches (see {ref}`upstream-merges`).
- `release.yml` defines a job for releasing a new version of the software developed by this project (see {ref}`triggering-releases`). The job can be triggered manually for one of the branches mentioned in the section {ref}`release-process`. It automatically updates the project's changelog and publishes a new release on GitHub.

(ci-publishing)=

## Publishing Packages

The following CI jobs are used for building our software for different target platforms and architectures and publishing the resulting packages.

- `publish.yml` is used for publishing pre-built packages on [PyPI](https://pypi.org/) (see {ref}`installation`). For this purpose, the project is built from source for each of the target platforms and architectures, using virtualization in some cases. The job is run automatically when a new release was published on [GitHub](https://github.com/mrapp-ke/MLRL-Boomer/releases). It does also increment the project's major version number and merge the release branch into its upstream branches (see {ref}`release-process`).
- `publish_development.yml` publishes development versions of packages on [Test-PyPI](https://test.pypi.org/) whenever changes to the project's source code have been pushed to the main branch. The packages built by each of these runs are also saved as [artifacts](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts) and can be downloaded as zip archives.

## Maintenance

The CI jobs listed below are run periodically to perform different maintenance tasks, such as the automatic update of dependencies.

- `update_github_actions.yml` checks for outdated GitHub Actions used in workflow definition files. In cases where any updates are necessary, a pull request with the respective changes is opened. If the pull request passes all {ref}`necessary checks <ci-testing>`, it is merged automatically. Otherwise, manual intervention is needed.
