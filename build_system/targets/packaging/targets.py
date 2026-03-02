"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for building and installing wheel packages.
"""
from os import environ
from pathlib import Path
from typing import Dict, List, cast, override

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget
from util.env import get_env_bool
from util.files import DirectorySearch, FileType
from util.log import Log
from util.package_manager import PackageManager
from util.requirements import Package, Requirement, RequirementsTextFile, RequirementVersion

from targets.packaging.auditwheel import Auditwheel
from targets.packaging.build import Build
from targets.packaging.modules import PythonPackageModule
from targets.packaging.pyproject_toml import PyprojectTomlFile
from targets.project import Project

MODULE_FILTER = PythonPackageModule.Filter()


class GeneratePyprojectTomlFiles(BuildTarget.Runnable):
    """
    Generates pyproject.toml files.
    """

    @staticmethod
    def __get_requirements(template_file: PyprojectTomlFile) -> Dict[str, Requirement]:
        requirements: Dict[str, Requirement] = {}
        all_dependencies = template_file.dependencies

        if all_dependencies:
            requirements_file = RequirementsTextFile(template_file.file.parent / 'requirements.txt')

            for dependency in all_dependencies:
                package = Package(dependency)

                if dependency.startswith('mlrl-'):
                    requirements[dependency] = Requirement(package, RequirementVersion.parse(str(Project.version())))
                else:
                    requirement = requirements_file.lookup_requirement(package)

                    if requirement:
                        requirements[dependency] = requirement

        return requirements

    @staticmethod
    def __generate_pyproject_toml(template_file: PyprojectTomlFile) -> List[str]:
        requirements = GeneratePyprojectTomlFiles.__get_requirements(template_file)
        lines = template_file.lines
        num_lines = len(lines)
        line_index = 0
        new_lines = []

        while line_index < num_lines:
            line = lines[line_index]
            line_stripped = line.strip('\n').strip()

            if line_stripped == '[project]':
                line_index += 1
                new_lines.append(line)
                new_lines.append('version = "' + str(Project.version()) + '"\n')
                new_lines.append('requires-python = "' + Project.Python.minimum_python_version() + '"\n')
            elif line_stripped == '[project.optional-dependencies]' \
                    or line_stripped.replace(' ', '').startswith('dependencies=['):
                while line_index < num_lines:
                    line = lines[line_index]
                    line_stripped = line.strip('\n').strip()
                    line_index += 1

                    if line_stripped == '[project.optional-dependencies]':
                        new_lines.append(line)
                    elif line_stripped.replace(' ', '').startswith('dependencies=[') and line_stripped.endswith(']') \
                            or line_stripped ==']':
                        new_lines.append(line)
                        break
                    else:
                        for dependency, requirement in requirements.items():
                            line = line.replace('"' + dependency + '"', '"' + str(requirement) + '"')

                        new_lines.append(line)
            else:
                line_index += 1
                new_lines.append(line)

        return new_lines

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        package_module = cast(PythonPackageModule, module)
        Log.info('Generating pyproject.toml file in directory "%s"...', package_module.root_directory)
        template_file = PyprojectTomlFile(build_unit, package_module.pyproject_toml_template_file)
        pyproject_toml_file = PyprojectTomlFile(build_unit, package_module.pyproject_toml_file)
        pyproject_toml_file.write_lines(*self.__generate_pyproject_toml(template_file))

    @override
    def get_input_files(self, _: BuildUnit, module: Module) -> List[Path]:
        package_module = cast(PythonPackageModule, module)
        return [package_module.pyproject_toml_template_file]

    @override
    def get_output_files(self, _: BuildUnit, module: Module) -> List[Path]:
        package_module = cast(PythonPackageModule, module)
        return [package_module.pyproject_toml_file]

    @override
    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[Path]:
        package_module = cast(PythonPackageModule, module)
        Log.info('Removing pyproject.toml file from directory "%s"', package_module.root_directory)
        return super().get_clean_files(build_unit, package_module)


class BuildPythonWheels(BuildTarget.Runnable):
    """
    Builds Python wheel packages and optionally repairs them via "auditwheel".
    """

    ENV_REPAIR_WHEELS = 'REPAIR_WHEELS'

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, build_unit: BuildUnit, module: Module):
        package_module = cast(PythonPackageModule, module)
        Log.info('Building Python wheels for directory "%s"...', package_module.root_directory)
        Build(build_unit, package_module).run()

        if get_env_bool(environ, self.ENV_REPAIR_WHEELS):
            if package_module.pure:
                Log.info('Python wheels in directory "%s" are pure and must not be repaired',
                         package_module.root_directory)
            else:
                Log.info('Repairing Python wheels in directory "%s"...', package_module.root_directory)
                wheel = package_module.find_wheel()

                if wheel:
                    Auditwheel(build_unit, wheel).run()

    @override
    def get_input_files(self, _: BuildUnit, module: Module) -> List[Path]:
        package_module = cast(PythonPackageModule, module)
        file_search = Project.Python.file_search() \
            .set_symlinks(False) \
            .filter_by_file_type(
                FileType.python(),
                FileType.markdown(),
                FileType.toml(),
                FileType.extension_module(),
                FileType.shared_library(),
            )
        return file_search.list(package_module.root_directory)

    @override
    def get_output_files(self, _: BuildUnit, module: Module) -> List[Path]:
        package_module = cast(PythonPackageModule, module)
        return [package_module.wheel_directory]

    @override
    def get_clean_files(self, _: BuildUnit, module: Module) -> List[Path]:
        package_module = cast(PythonPackageModule, module)
        clean_files = []
        Log.info('Removing Python wheels from directory "%s"...', package_module.root_directory)
        clean_files.append(package_module.wheel_directory)
        clean_files.extend(
            DirectorySearch() \
                .filter_by_name(Project.Python.build_directory_name) \
                .filter_by_substrings(ends_with=Project.Python.wheel_metadata_directory_suffix) \
                .list(package_module.root_directory)
        )
        return clean_files


class InstallPythonWheels(BuildTarget.Runnable):
    """
    Installs Python wheel packages.
    """

    class InstallWheelCommand(PackageManager.Command):
        """
        Allows to install wheel packages via the command `pip install`.
        """

        def __init__(self, *wheels: Path):
            """
            :param wheels: The paths to the wheel packages to be installed
            """
            super().__init__('install', '--force-reinstall', '--no-deps', *map(str, wheels))
            self.print_arguments(True)

    class UninstallCommand(PackageManager.Command):
        """
        Allows to uninstall packages via the command `pip uninstall`.
        """

        def __init__(self, *package_names: str):
            """
            :param package_names: The names of the packages to be uninstalled
            """
            super().__init__('uninstall', '--yes', *package_names)
            self.print_arguments(True)

    class ListCommand(PackageManager.Command):
        """
        Allows to list installed packages via the command `pip list`.
        """

        def __init__(self):
            super().__init__('list', '--format', 'freeze')
            self.print_arguments(True)

    def __init__(self):
        super().__init__(MODULE_FILTER)

    @override
    def run(self, _: BuildUnit, module: Module):
        package_module = cast(PythonPackageModule, module)
        Log.info('Installing Python wheels for directory "%s"...', package_module.root_directory)
        wheel = package_module.find_wheel()

        if wheel:
            InstallPythonWheels.InstallWheelCommand(wheel).run()

    @override
    def get_input_files(self, build_unit: BuildUnit, module: Module) -> List[Path]:
        package_module = cast(PythonPackageModule, module)
        installed_packages = InstallPythonWheels.ListCommand().capture_output().split('\n')
        pyproject_toml_file = PyprojectTomlFile(build_unit, package_module.pyproject_toml_file)
        required_package = f'{pyproject_toml_file.package_name}=={pyproject_toml_file.version}'

        if any(package == required_package for package in installed_packages):
            # If the correct version of the package is already installed, we check if the wheel has changed...
            wheel = package_module.find_wheel()
            return [wheel] if wheel else []

        # If the package is not already installed or the version differs, we force a re-installation...
        return []

    @override
    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[Path]:
        package_module = cast(PythonPackageModule, module)
        Log.info('Uninstalling Python packages for directory "%s"...', package_module.root_directory)
        pyproject_toml_file = PyprojectTomlFile(build_unit, package_module.pyproject_toml_template_file)
        InstallPythonWheels.UninstallCommand(pyproject_toml_file.package_name).run()
        return super().get_clean_files(build_unit, package_module)
