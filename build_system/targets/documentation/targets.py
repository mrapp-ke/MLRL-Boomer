"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for generating documentations.
"""
from abc import ABC
from os import path
from typing import Dict, List, Optional

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget
from util.io import TextFile
from util.log import Log

from targets.documentation.modules import ApidocModule, SphinxModule
from targets.documentation.sphinx_build import SphinxBuild


class ApidocIndex(BuildTarget.Runnable, ABC):
    """
    An abstract base class for all targets that generate index files referencing API documentations.
    """

    @staticmethod
    def __get_template(module: ApidocModule) -> Optional[str]:
        parent_directory = path.dirname(module.output_directory)
        template = path.join(parent_directory, 'index.md.template')
        return template if path.isfile(template) else None

    @staticmethod
    def __get_templates_and_modules(modules: List[ApidocModule]) -> Dict[str, List[ApidocModule]]:
        modules_by_template = {}

        for module in modules:
            template = ApidocIndex.__get_template(module)

            if template:
                modules_in_directory = modules_by_template.setdefault(template, [])
                modules_in_directory.append(module)

        return modules_by_template

    @staticmethod
    def __index_file(template: str) -> str:
        return path.join(path.dirname(template), 'index.md')

    def __init__(self, module_filter: ApidocModule.Filter):
        """
        :param module_filter: A filter that matches the modules, the target should be applied to
        """
        super().__init__(module_filter)

    def run_all(self, _: BuildUnit, modules: List[Module]):
        for template, modules_in_directory in self.__get_templates_and_modules(modules).items():
            Log.info('Generating index file referencing API documentations from template "%s"...', template)
            references = [module.create_reference() + '\n' for module in modules_in_directory]
            new_lines = []

            for line in TextFile(template).lines:
                if line.strip() == '%s':
                    new_lines.extend(references)
                else:
                    new_lines.append(line)

            TextFile(self.__index_file(template), accept_missing=True).write_lines(*new_lines)

    def get_input_files(self, module: Module) -> List[str]:
        template = self.__get_template(module)
        return [template] if template else []

    def get_output_files(self, module: Module) -> List[str]:
        template = self.__get_template(module)
        return [self.__index_file(template)] if template else []

    def get_clean_files(self, module: Module) -> List[str]:
        Log.info('Removing index file referencing API documentation in directory "%s"', module.output_directory)
        return super().get_clean_files(module)


class BuildDocumentation(BuildTarget.Runnable):
    """
    Generates documentations.
    """

    def __init__(self):
        super().__init__(SphinxModule.Filter())

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Generating documentation for directory "%s"...', module.root_directory)
        SphinxBuild(build_unit, module).run()

    def get_input_files(self, module: Module) -> List[str]:
        return module.find_source_files()

    def get_output_files(self, module: Module) -> List[str]:
        return [module.output_directory]

    def get_clean_files(self, module: Module) -> List[str]:
        Log.info('Removing documentation generated for directory "%s"...', module.root_directory)
        return super().get_clean_files(module)
