(release-process)=

# Release Process

To enable releasing new major, feature, or bugfix releases at any time, we maintain a branch for each type of release:

- `main` contains all changes that will be included in the next major release (including changes on the feature and bugfix branch).
- `feature` comes with the changes that will be part of an upcoming feature release (including changes on the bugfix branch).
- `bugfix` is restricted to minor changes that will be published as a bugfix release.

We do not allow directly pushing to the above branches. Instead, all changes must be submitted via pull requests and require certain checks to pass.

(downstream-merges)=

## Downstream Merges

Once modifications to one of the branches have been merged, {ref}`Continuous Integration <ci>` jobs are used to automatically update downstream branches via pull requests. If all checks for such pull requests are successful, they are merged automatically. If there are any merge conflicts, they must be resolved manually. Following this procedure, changes to the feature branch are merged into the main branch, whereas changes to the bugfix branch are first merged into the feature branch and then into the main branch (see description of {repo-file}`merge_feature.yml <.github/workflows/merge_feature.yml>` and {repo-file}`merge_bugfix.yml <.github/workflows/merge_bugfix.yml>` in {ref}`ci-releases`).

(triggering-releases)=

## Triggering Releases

We use a {ref}`Continuous Integration <ci>` job for triggering a new release, including the changes of one of the branches mentioned above (see description of {repo-file}`release.yml <.github/workflows/release.yml>` in {ref}`ci-releases`). Depending on the release branch, the job automatically collects the corresponding changelog entries from the files {repo-file}`changelog-main.md <build_system/res/changelog/changelog-main.md>`, {repo-file}`changelog-feature.md <build_system/res/changelog/changelog-main.md>`, and {repo-file}`changelog-bugfix.md <build_system/res/changelog/changelog-bugfix.md>` and updates the file {repo-file}`CHANGELOG.md` in the project's root directory accordingly. Afterward, it will publish the new release on GitHub, which will in turn trigger the publishing of pre-built packages (see description of {repo-file}`publish.yml <.github/workflows/publish.yml>` in {ref}`ci-publishing`).

(upstream-merges)=

## Upstream Merges

Whenever a new release has been published, the release branch is merged into the upstream branches (see description of {repo-file}`merge_release.yml <.github/workflows/merge_release.yml>` in {ref}`ci-releases`), i.e., major releases result in the feature and bugfix branches being updated, whereas minor releases result in the bugfix branch being updated. The version of the release branch and the affected branches are updated accordingly. The file {repo-file}`version <build_system/res/versioning/version>` in the project's root directory specifies the version of each of these branches. Similarly, the file {repo-file}`version-dev <build_system/res/versioning/version-dev>` keeps track of the version number used for development releases (see description of {repo-file}`publish_development.yml <.github/workflows/publish_development.yml>` in {ref}`ci-publishing`).
