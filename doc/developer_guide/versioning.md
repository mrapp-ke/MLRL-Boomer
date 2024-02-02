(coding-standards)=

# Versioning Scheme

We use [Semantic Versioning](https://semver.org/) to assign unique version numbers in the form `MAJOR.MINOR.PATCH` to the individual releases of our software packages. We refer to releases that come with an incremented major version, as *major releases*. When the minor version is increased by a release, we refer to it as a *feature release*. Updates that include bugfixes or minor improvements come with an increased patch version and are referred to as *bugfix releases*.

```{tip}
An overview of past releases, together with a description of the changes they introduced compared to the previous version, can be found in the {ref}`release-notes`.
```

## Bugfix Releases

Bugfix releases are limited to backward-compatible changes, such as bug fixes, performance optimizations, improvements to the build system, or updates of the documentation. They are neither allowed to introduce any compatibility-breaking changes to the command line API, nor to any of the programmatic APIs in the project's Python or C++ code.

## Feature Releases

Feature releases may come with changes that do not break compatibility with the command line API or programmatic APIs provided by previous versions. As a consequence, new functionalities can be added to the algorithms provided by this project, if they do not break existing functionality. In contrast, the removal of features is only allowed for major releases.

Feature releases with the major version `0` are not obliged to maintain API compatibility, because these releases are considered to represent an early stage of development, where things may change drastically from one version to another.

## Major Releases

Increments of the major version indicate big leaps in the software's development. They are reserved for new versions of the software that introduce new functionality, fundamentally change how the software works, or come with compatibility-breaking changes. In general, major releases are not guaranteed to be compatible with past releases in any way. In particular, they may introduce compatibility-breaking API changes, affecting the command line API or programmatic APIs in the project's Python or C++ code. Moreover, models that have been trained using an older version are not guaranteed to work after updating to a new major release and must potentially be trained from scratch.
