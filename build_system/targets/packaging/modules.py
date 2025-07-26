"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to Python code that can be built as wheel packages.
"""
from os import environ
from pathlib import Path
from typing import Generator, List, Optional, Set, cast

from core.build_unit import BuildUnit
from core.modules import Module, ModuleRegistry
from util.files import FileSearch
from util.format import format_iterable
from util.log import Log

from targets.modules import SubprojectModule
from targets.packaging.pyproject_toml import PyprojectTomlFile
from targets.project import Project


class PythonPackageModule(SubprojectModule):
    """
    A module that provides access to Python code that can be built as wheel packages.
    """

    class PackageNameFilter(Module.Filter):
        """
        Matches modules by their package name.
        """

        def __init__(self, package_name: str):
            """
            :param package_name: The package name to be matched
            """
            self.package_name = package_name

        def matches(self, module: 'Module', module_registry: 'ModuleRegistry') -> bool:
            build_unit = BuildUnit.for_file(Path(__file__))
            return isinstance(module, PythonPackageModule) and module.get_package_name(build_unit) == self.package_name

    class Filter(Module.Filter):
        """
        A filter that matches modules of type `PythonPackageModule`.
        """

        class TypeFilter(Module.Filter):
            """
            An internal filter that only checks the type of modules.
            """

            def matches(self, module: 'Module', _: ModuleRegistry) -> bool:
                return isinstance(module, PythonPackageModule)

        def __needs_to_be_built(self, module: 'PythonPackageModule', module_registry: ModuleRegistry) -> bool:
            return SubprojectModule.Filter.from_env(environ).matches(module, module_registry)

        def __is_dependency_to_be_built(self, module: 'PythonPackageModule', module_registry: ModuleRegistry) -> bool:
            modules_to_be_built = module_registry.lookup(PythonPackageModule.Filter.TypeFilter(),
                                                         SubprojectModule.Filter.from_env(environ))
            return any(
                self.__is_dependency_of_module(module, cast(PythonPackageModule, module_to_be_built), module_registry)
                for module_to_be_built in modules_to_be_built)

        def __is_dependency_of_module(self,
                                      module: 'PythonPackageModule',
                                      other_module: 'PythonPackageModule',
                                      module_registry: ModuleRegistry,
                                      dependencies_to_be_skipped: Optional[Set[str]] = None) -> bool:
            package_name = module.get_package_name(self.build_unit)
            dependency_names = other_module.get_dependency_names(self.build_unit)

            for dependency_name in dependency_names:
                if dependency_name == package_name:
                    return True

            dependencies_to_be_skipped = dependencies_to_be_skipped if dependencies_to_be_skipped else set()

            for dependency_module in other_module.get_dependencies(self.build_unit, module_registry):
                dependency_package_name = dependency_module.get_package_name(self.build_unit)

                if dependency_package_name not in dependencies_to_be_skipped:
                    if self.__is_dependency_of_module(module, dependency_module, module_registry,
                                                      dependencies_to_be_skipped | {dependency_package_name}):
                        return True

            return False

        def __init__(self):
            self.build_unit = BuildUnit.for_file(Path(__file__))

        def matches(self, module: Module, module_registry: ModuleRegistry) -> bool:
            return PythonPackageModule.Filter.TypeFilter().matches(module, module_registry) and (
                self.__needs_to_be_built(cast(PythonPackageModule, module), module_registry)
                or self.__is_dependency_to_be_built(cast(PythonPackageModule, module), module_registry))

    def __init__(self, root_directory: Path, wheel_directory_name: str):
        """
        :param root_directory:          The path to the module's root directory
        :param wheel_directory_name:    The name of the directory that contains wheel packages
        """
        self.root_directory = root_directory
        self.wheel_directory_name = wheel_directory_name
        self._pyproject_toml_file: Optional[PyprojectTomlFile] = None
        self._dependencies: Optional[List[PythonPackageModule]] = None

    @property
    def pyproject_toml_template_file(self) -> Path:
        """
        The path to the template file that is used for generating a pyproject.toml file.
        """
        return self.root_directory / 'pyproject.template.toml'

    @property
    def pyproject_toml_file(self) -> Path:
        """
        The path to the pyproject.toml file that specifies the meta-data of the package.
        """
        return self.root_directory / 'pyproject.toml'

    def get_package_name(self, build_unit: BuildUnit) -> str:
        """
        Retrieves and returns the name of the package from the pyproject.toml file.

        :param build_unit:  The build unit this function is called from
        :return:            The name of the package
        """
        pyproject_toml_file = self._pyproject_toml_file

        if not pyproject_toml_file:
            pyproject_toml_file = PyprojectTomlFile(build_unit, self.pyproject_toml_template_file)
            self._pyproject_toml_file = pyproject_toml_file

        return pyproject_toml_file.package_name

    def get_dependency_names(self, build_unit: BuildUnit) -> List[str]:
        """
        Retrieves and returns the names of the dependencies of the package from the pyproject.toml file.

        :param build_unit:  The build unit this function is called from
        :return:            The names of the dependencies of the package
        """
        pyproject_toml_file = self._pyproject_toml_file

        if not pyproject_toml_file:
            pyproject_toml_file = PyprojectTomlFile(build_unit, self.pyproject_toml_template_file)
            self._pyproject_toml_file = pyproject_toml_file

        return pyproject_toml_file.dependencies

    def get_dependencies(self, build_unit: BuildUnit,
                         module_registry: ModuleRegistry) -> Generator['PythonPackageModule', None, None]:
        """
        Retries and returns the modules that are dependencies of the package from the pyproject.toml file.

        :param build_unit:      The build unit this function is called from
        :param module_registry: The module registry that should be used for looking up modules
        :return:                The modules that are dependencies of the package
        """
        dependencies = self._dependencies

        if not dependencies:
            dependencies = []

            for dependency_name in self.get_dependency_names(build_unit):
                modules = module_registry.lookup(PythonPackageModule.PackageNameFilter(dependency_name))

                if modules:
                    package_module = cast(PythonPackageModule, modules[0])
                    dependencies.append(package_module)

            self._dependencies = dependencies

        yield from dependencies

    @property
    def wheel_directory(self) -> Path:
        """
        The path to the directory that contains the wheel packages that have been built for the module.
        """
        return self.root_directory / self.wheel_directory_name

    def find_wheel(self) -> Optional[Path]:
        """
        Finds and returns the wheel package that has been built for the module.

        :return: The path to the wheel package or None, if no such package has been found
        """
        wheels = FileSearch() \
            .filter_by_substrings(contains=str(Project.version(release=True)), ends_with='.whl') \
            .list(self.wheel_directory)
        if wheels and len(wheels) > 1:
            Log.error(
                'Found multiple wheel packages in directory "%s": \n\n%s\n\nRun "build_wheel --clean" to delete them.',
                self.wheel_directory, format_iterable(wheels, separator='\n'))
        return wheels[0] if wheels else None

    @property
    def pure(self) -> bool:
        """
        True, if the wheel package is a pure Pyton package without extension modules, False otherwise
        """
        return not (self.root_directory / 'setup.py').is_file()

    @property
    def subproject_name(self) -> str:
        return self.root_directory.name

    def __str__(self) -> str:
        return 'PythonPackageModule {root_directory="' + str(self.root_directory) + '"}'
