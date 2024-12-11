"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for building and installing wheel packages.
"""
from typing import List

from packaging.build import Build
from packaging.modules import PythonPackageModule
from packaging.pip import PipInstallWheel
from util.files import DirectorySearch, FileType
from util.log import Log
from util.modules import Module
from util.paths import Project
from util.targets import BuildTarget
from util.units import BuildUnit

MODULE_FILTER = PythonPackageModule.Filter()


class BuildPythonWheels(BuildTarget.Runnable):
    """
    Builds Python wheel packages.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Building Python wheels for directory "%s"...', module.root_directory)
        Build(build_unit, module).run()

    def get_input_files(self, module: Module) -> List[str]:
        file_search = Project.Python.file_search() \
            .set_symlinks(False) \
            .exclude_subdirectories_by_name(Project.Python.test_directory_name) \
            .filter_by_file_type(FileType.python(), FileType.extension_module(), FileType.shared_library())
        return file_search.list(module.root_directory)

    def get_output_files(self, module: Module) -> List[str]:
        return module.wheel_directory

    def get_clean_files(self, module: Module) -> List[str]:
        clean_files = []
        Log.info('Removing Python wheels from directory "%s"...', module.root_directory)
        clean_files.append(module.wheel_directory)
        clean_files.extend(
            DirectorySearch() \
                .filter_by_name(Project.Python.build_directory_name) \
                .filter_by_substrings(ends_with=Project.Python.wheel_metadata_directory_suffix) \
                .list(module.root_directory)
        )
        return clean_files


class InstallPythonWheels(BuildTarget.Runnable):
    """
    Installs Python wheel packages.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, _: BuildUnit, module: Module):
        Log.info('Installing Python wheels for directory "%s"...', module.root_directory)
        PipInstallWheel().install_wheels(*module.find_wheels())

    def get_input_files(self, module: Module) -> List[str]:
        return module.find_wheels()
