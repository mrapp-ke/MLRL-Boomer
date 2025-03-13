"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for building and installing wheel packages.
"""
from os import path
from typing import Dict, List

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget
from util.files import DirectorySearch, FileType
from util.log import Log
from util.pip import Package, Pip, Requirement, RequirementsTextFile, RequirementVersion

from targets.packaging.build import Build
from targets.packaging.modules import PythonPackageModule
from targets.packaging.pyproject_toml import PyprojectTomlFile
from targets.packaging.version_files import PythonVersionFile
from targets.project import Project

MODULE_FILTER = PythonPackageModule.Filter()


class GeneratePyprojectTomlFiles(BuildTarget.Runnable):
    """
    Generates pyproject.toml files.
    """

    @staticmethod
    def __get_requirements(template_file: PyprojectTomlFile) -> Dict[str, Requirement]:
        requirements = {}
        all_dependencies = template_file.dependencies

        if all_dependencies:
            requirements_file = RequirementsTextFile(path.join(path.dirname(template_file.file), 'requirements.txt'))

            for dependency in all_dependencies:
                package = Package(dependency)

                if dependency.startswith('mlrl-'):
                    requirement = Requirement(package, RequirementVersion.parse(str(Project.version())))
                else:
                    requirement = requirements_file.lookup_requirement(package)

                requirements[dependency] = requirement

        return requirements

    @staticmethod
    def __generate_pyproject_toml(template_file: PyprojectTomlFile) -> List[str]:
        requirements = GeneratePyprojectTomlFiles.__get_requirements(template_file)
        new_lines = []

        for line in template_file.lines:
            if line.strip('\n').strip() == '[project]':
                new_lines.append(line)
                new_lines.append('version = "' + str(Project.version()) + '"\n')
                new_lines.append('requires-python = "' + PythonVersionFile().version + '"\n')
            else:
                for dependency, requirement in requirements.items():
                    if dependency in line:
                        line = line.replace(dependency, str(requirement))
                        break

                new_lines.append(line)

        return new_lines

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Generating pyproject.toml file in directory "%s"...', module.root_directory)
        template_file = PyprojectTomlFile(build_unit, module.pyproject_toml_template_file)
        pyproject_toml_file = PyprojectTomlFile(build_unit, module.pyproject_toml_file)
        pyproject_toml_file.write_lines(*self.__generate_pyproject_toml(template_file))

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

    class InstallWheelCommand(Pip.Command):
        """
        Allows to install wheel packages via the command `pip install`.
        """

        def __init__(self, *wheels: str):
            """
            :param wheels: The paths to the wheel packages to be installed
            """
            super().__init__('install', '--force-reinstall', '--no-deps', *wheels)
            self.print_arguments(True)

    class UninstallCommand(Pip.Command):
        """
        Allows to uninstall packages via the command `pip uninstall`.
        """

        def __init__(self, *package_names: str):
            """
            :param package_names: The names of the packages to be uninstalled
            """
            super().__init__('uninstall', '--yes', *package_names)
            self.print_arguments(True)

    def __init__(self):
        super().__init__(MODULE_FILTER)

    def run(self, _: BuildUnit, module: Module):
        Log.info('Installing Python wheels for directory "%s"...', module.root_directory)
        wheel = module.find_wheel()

        if wheel:
            InstallPythonWheels.InstallWheelCommand(wheel).run()

    def get_input_files(self, _: BuildUnit, module: Module) -> List[str]:
        wheel = module.find_wheel()
        return [wheel] if wheel else []

    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[str]:
        Log.info('Uninstalling Python packages for directory "%s"...', module.root_directory)
        pyproject_toml_file = PyprojectTomlFile(build_unit, module.pyproject_toml_template_file)
        InstallPythonWheels.UninstallCommand(pyproject_toml_file.package_name).run()
        return super().get_clean_files(build_unit, module)
