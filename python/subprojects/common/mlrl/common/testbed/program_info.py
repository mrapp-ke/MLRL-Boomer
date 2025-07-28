"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for providing the text to be shown when the "--version" flag is passed to the command line API.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, override

from tabulate import tabulate

from mlrl.common.package_info import PackageInfo

from mlrl.testbed.program_info import ProgramInfo

from mlrl.util.format import format_iterable


@dataclass
class RuleLearnerProgramInfo:
    """
    Provides information about a program that runs a rule learning algorithm.

    Attributes:
        program_info:       Information about the program
        python_packages:    A list that contains a `PackageInfo` for each Python package used by the program
    """
    program_info: ProgramInfo
    python_packages: List[PackageInfo] = field(default_factory=list)

    def __collect_python_packages(self, python_packages: Iterable[PackageInfo]) -> Set[str]:
        unique_packages = set()

        for python_package in python_packages:
            unique_packages.add(str(python_package))
            unique_packages.update(self.__collect_python_packages(python_package.python_packages))

        return unique_packages

    def __collect_dependencies(self, python_packages: Iterable[PackageInfo]) -> Dict[str, Set[str]]:
        unique_dependencies: Dict[str, Set[str]] = {}

        for python_package in python_packages:

            for dependency in python_package.dependencies:
                parent_packages = unique_dependencies.setdefault(str(dependency), set())
                parent_packages.add(python_package.package_name)

            for key, value in self.__collect_dependencies(python_package.python_packages).items():
                parent_packages = unique_dependencies.setdefault(key, set())
                parent_packages.update(value)

        return unique_dependencies

    def __collect_cpp_libraries(self, python_packages: Iterable[PackageInfo]) -> Dict[str, Set[str]]:
        unique_libraries: Dict[str, Set[str]] = {}

        for python_package in python_packages:
            for cpp_library in python_package.cpp_libraries:
                parent_packages = unique_libraries.setdefault(str(cpp_library), set())
                parent_packages.add(python_package.package_name)

            for key, value in self.__collect_cpp_libraries(python_package.python_packages).items():
                parent_packages = unique_libraries.setdefault(key, set())
                parent_packages.update(value)

        return unique_libraries

    def __collect_build_options(self, python_packages: Iterable[PackageInfo]) -> Dict[str, Set[str]]:
        unique_build_options: Dict[str, Set[str]] = {}

        for python_package in python_packages:
            for cpp_library in python_package.cpp_libraries:
                for build_option in cpp_library.build_options:
                    parent_libraries = unique_build_options.setdefault(str(build_option), set())
                    parent_libraries.add(cpp_library.library_name)

            for key, value in self.__collect_build_options(python_package.python_packages).items():
                parent_libraries = unique_build_options.setdefault(key, set())
                parent_libraries.update(value)

        return unique_build_options

    def __collect_hardware_resources(self, python_packages: Iterable[PackageInfo]) -> Dict[str, Set[str]]:
        unique_hardware_resources: Dict[str, Set[str]] = {}

        for python_package in python_packages:
            for cpp_library in python_package.cpp_libraries:
                for hardware_resource in cpp_library.hardware_resources:
                    info = unique_hardware_resources.setdefault(hardware_resource.resource, set())
                    info.add(hardware_resource.info)

            for key, value in self.__collect_hardware_resources(python_package.python_packages).items():
                info = unique_hardware_resources.setdefault(key, set())
                info.update(value)

        return unique_hardware_resources

    @staticmethod
    def __format_parent_packages(parent_packages: Set[str]) -> str:
        return 'used by ' + format_iterable(parent_packages) if parent_packages else ''

    def __get_package_info(self) -> str:
        rows = []
        python_packages = self.python_packages

        for i, python_package in enumerate(sorted(self.__collect_python_packages(python_packages))):
            rows.append(['' if i > 0 else 'Python packages:', python_package, ''])

        if python_packages:
            rows.append(['', '', ''])

        dependencies = self.__collect_dependencies(python_packages)

        for i, dependency in enumerate(sorted(dependencies.keys())):
            parent_packages = self.__format_parent_packages(dependencies[dependency])
            rows.append(['' if i > 0 else 'Dependencies:', dependency, parent_packages])

        if dependencies:
            rows.append(['', '', ''])

        cpp_libraries = self.__collect_cpp_libraries(python_packages)

        for i, cpp_library in enumerate(sorted(cpp_libraries.keys())):
            parent_packages = self.__format_parent_packages(cpp_libraries[cpp_library])
            rows.append(['' if i > 0 else 'Shared libraries:', cpp_library, parent_packages])

        if cpp_libraries:
            rows.append(['', '', ''])

        build_options = self.__collect_build_options(python_packages)

        for i, build_option in enumerate(sorted(build_options.keys())):
            parent_libraries = self.__format_parent_packages(build_options[build_option])
            rows.append(['' if i > 0 else 'Build options:', build_option, parent_libraries])

        if build_options:
            rows.append(['', '', ''])

        hardware_resources = self.__collect_hardware_resources(python_packages)

        for i, hardware_resource in enumerate(sorted(hardware_resources.keys())):
            for j, info in enumerate(sorted(hardware_resources[hardware_resource])):
                rows.append(['' if i > 0 else 'Hardware resources:', '' if j > 0 else hardware_resource, info])

        return tabulate(rows, tablefmt='plain') if rows else ''

    @override
    def __str__(self) -> str:
        program_info = str(self.program_info)
        package_info = self.__get_package_info()
        return program_info + '\n\n' + package_info if package_info else program_info
