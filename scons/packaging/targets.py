"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for building and installing wheel packages.
"""
from functools import reduce
from typing import List

from packaging.build import Build
from packaging.modules import PythonPackageModule
from packaging.pip import PipInstallWheel
from util.files import DirectorySearch, FileType
from util.modules import ModuleRegistry
from util.paths import Project
from util.targets import BuildTarget, PhonyTarget
from util.units import BuildUnit


class BuildPythonWheels(BuildTarget.Runnable):
    """
    Builds Python wheel packages.
    """

    def __init__(self, root_directory: str):
        """
        :param root_directory: The root directory of the module for which Python wheel packages should be built
        """
        self.module_filter = PythonPackageModule.Filter(root_directory)

    def run(self, build_unit: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(self.module_filter):
            print('Building Python wheels for directory "' + module.root_directory + '"...')
            Build(build_unit, module).run()

    def get_input_files(self, modules: ModuleRegistry) -> List[str]:
        package_modules = modules.lookup(self.module_filter)
        file_search = Project.Python.file_search() \
            .set_symlinks(False) \
            .exclude_subdirectories_by_name(Project.Python.test_directory_name) \
            .filter_by_file_type(FileType.python(), FileType.extension_module(), FileType.shared_library())
        return reduce(lambda aggr, module: aggr + file_search.list(module.root_directory), package_modules, [])

    def get_output_files(self, modules: ModuleRegistry) -> List[str]:
        package_modules = modules.lookup(self.module_filter)
        wheels = reduce(lambda aggr, module: module.find_wheels(), package_modules, [])
        return wheels if wheels else [module.wheel_directory for module in package_modules]

    def get_clean_files(self, modules: ModuleRegistry) -> List[str]:
        clean_files = []

        for module in modules.lookup(self.module_filter):
            print('Removing Python wheels from directory "' + module.root_directory + '"...')
            clean_files.append(module.wheel_directory)
            clean_files.extend(
                DirectorySearch() \
                    .filter_by_name(Project.Python.build_directory_name) \
                    .filter_by_substrings(ends_with=Project.Python.wheel_metadata_directory_suffix) \
                    .list(module.root_directory)
            )

        return clean_files


class InstallPythonWheels(PhonyTarget.Runnable):
    """
    Installs Python wheel packages.
    """

    def __init__(self, root_directory: str):
        """
        :param root_directory: The root directory of the module for which Python wheel packages should be installed
        """
        self.module_filter = PythonPackageModule.Filter(root_directory)

    def run(self, _: BuildUnit, modules: ModuleRegistry):
        for module in modules.lookup(self.module_filter):
            print('Installing Python wheels for directory "' + module.root_directory + '"...')
            PipInstallWheel().install_wheels(module.find_wheels())
