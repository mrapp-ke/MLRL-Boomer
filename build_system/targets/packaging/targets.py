"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for building and installing wheel packages.
"""
from typing import List

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget
from util.files import DirectorySearch, FileType
from util.log import Log
from util.toml_file import TomlFile
from util.io import TextFile

from targets.packaging.build import Build
from targets.packaging.modules import PythonPackageModule
from targets.packaging.pip import PipInstallWheel
from targets.project import Project

MODULE_FILTER = PythonPackageModule.Filter()


class GeneratePyprojectTomlFiles(BuildTarget.Runnable):
    """
    Generates pyproject.toml files.
    """

    @staticmethod
    def __set_project_version(lines: List[str]) -> List[str]:
        new_lines = []

        for line in lines:
            new_lines.append(line)

            if line.strip('\n').strip() == '[project]':
                new_lines.append('version = "' + str(Project.version()) + '"\n')

        return new_lines

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, _: BuildUnit, module: Module):
        Log.info('Generating pyproject.toml file in directory "%s"...', module.root_directory)
        template_file = TextFile(module.pyproject_toml_template_file)
        lines = self.__set_project_version(template_file.lines)
        pyproject_toml_file = TextFile(module.pyproject_toml_file)
        pyproject_toml_file.write_lines(*lines)

    def get_input_files(self, _: BuildUnit, module: Module) -> List[str]:
        return [module.pyproject_toml_template_file]

    def get_output_files(self, _: BuildUnit, module: Module) -> List[str]:
        return [module.pyproject_toml_file]

    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[str]:
        Log.info('Removing pyproject.toml file from directory "%s"', module.root_directory)
        return super().get_clean_files(build_unit, module)


class BuildPythonWheels(BuildTarget.Runnable):
    """
    Builds Python wheel packages.
    """

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Building Python wheels for directory "%s"...', module.root_directory)
        Build(build_unit, module).run()

    def get_input_files(self, _: BuildUnit, module: Module) -> List[str]:
        file_search = Project.Python.file_search() \
            .set_symlinks(False) \
            .filter_by_file_type(
                FileType.python(),
                FileType.markdown(),
                FileType.toml(),
                FileType.extension_module(),
                FileType.shared_library(),
            )
        return file_search.list(module.root_directory)

    def get_output_files(self, _: BuildUnit, module: Module) -> List[str]:
        return [module.wheel_directory]

    def get_clean_files(self, _: BuildUnit, module: Module) -> List[str]:
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
        PipInstallWheel().install_wheels(module.find_wheel())

    def get_input_files(self, _: BuildUnit, module: Module) -> List[str]:
        wheel = module.find_wheel()
        return [wheel] if wheel else []

    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[str]:
        Log.info('Uninstalling Python packages for directory "%s"...', module.root_directory)
        pyproject_toml_file = TomlFile(build_unit, module.pyproject_toml_template_file)
        package_name = pyproject_toml_file.toml_dict['project']['name']
        PipInstallWheel().uninstall_packages(package_name)
        return super().get_clean_files(build_unit, module)
