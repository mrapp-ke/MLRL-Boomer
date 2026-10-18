(dependencies)=

# Dependency Management

Adding dependencies to a software project always comes at a cost. Maintainers need to continuously test their software as new versions of dependencies are released and major changes in their APIs may break existing functionality. For this reason, we try to keep the number of dependencies at a minimum.

That being said, we still rely on several dependencies for Continuous Integration, compiling our source code, generating the documentation, or running the algorithms provided by this project. When using pre-built packages from [PyPI](https://pypi.org/project/mlrl-boomer/), there is no need to care about these dependencies, as they are already included in the packages. When {ref}`building from source <compilation>`, dependencies are automatically installed by the build system once they are needed, unless explicitly stated in the documentation.

## Supported Python Versions

The packages provided by this project are built from source code that must be compiled for specific Python versions. As a consequence, they do not work with arbitrary Python releases, but can only be used with the versions they have been built for. The supported Python versions are stored in the file {repo-file}`version-python <build_system/res/versioning/version-python>` and are regularly updated as new releases of Python are published. The following command provided by the project's build system updates the supported versions in the file {repo-file}`version-python <build_system/res/versioning/version-python>` to include the latest Python release:

````{tab} Linux
   ```text
   ./build update_python_version
   ```
````

````{tab} macOS
   ```text
   ./build update_python_version
   ```
````

````{tab} Windows
   ```
   build.bat update_python_version
   ```
````

## Python Dependencies

Python dependencies that are required by different aspects of the project, such as the build system, the documentation, or our own Python code, are defined in separate `requirements.txt` and `pyproject.template.toml` files. For dependencies that use [Semantic Versioning](https://semver.org/), we specify the earliest and latest version we support. For other dependencies, we demand for a specific version number. This strives to achieve a balance between flexibility for users and comfort for developers. On the one hand, supporting a range of versions provides more freedom to users, as our packages can more flexibly be used together with other ones, relying on the same dependencies. On the other hand, the project's maintainers must not manually update dependencies that have a minor release, while still requiring manual intervention for major updates.

To ease the life of developers, the following command provided by the project's build system may be used to update the versions of outdated dependencies:

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
If you want to restrict the above commands to the build-time dependencies, required by the project's build system, or the runtime dependencies, required for running its algorithms, you can use the targets `update_build_dependencies` and `update_runtime_dependencies` instead.
```

## GitHub Actions

Our {ref}`Continuous Integration <ci>` (CI) jobs heavily rely on so-called [Actions](https://docs.github.com/actions/sharing-automations/reusing-workflows), which are reusable building blocks provided by third-party developers. As with all dependencies, updates to these Actions may introduce breaking changes. To reduce the risk of updates breaking our CI jobs, we pin the Actions to a certain version. Usually, we only restrict the major version required by a job, rather than specifying a specific version. This allows minor updates, which are less likely to cause problems, to take effect without manual intervention.

The project's build system allows to automatically check for outdated Actions used by the project's CI jobs. The following command may be used to update the versions of outdated Actions automatically:

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

## GitHub Runners

For running {ref}`Continuous Integration <ci>` (CI) jobs, we use [runners](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners) hosted by GitHub. Runners are available for different operating systems and architectures, which is particularly relevant when building packages for the various target platforms we support. To avoid breaking the build process when GitHub updates its runners, we specify the exact version required by a particular CI job.

Our build system provides the following command to update the versions of outdated runners automatically:

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
