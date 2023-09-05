"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides access to directories and files belonging to different modules that are part of the project.
"""
from os import path


class BuildModule:
    """
    Provides access to directories and paths that belong to the build system.
    """

    @property
    def root_dir(self) -> str:
        return 'scons'

    @property
    def build_dir(self) -> str:
        return 'build'

    @property
    def requirements_file(self) -> str:
        return path.join(self.root_dir, 'requirements.txt')


BUILD_MODULE = BuildModule()
