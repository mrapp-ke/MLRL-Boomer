"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements targets for generating documentations.
"""
from abc import ABC
from os import environ, path
from typing import Dict, List, Optional

from core.build_unit import BuildUnit
from core.modules import Module
from core.targets import BuildTarget
from util.env import get_env
from util.format import format_iterable
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
        template = path.join(parent_directory, 'index.template.md')
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

    def get_input_files(self, _: BuildUnit, module: Module) -> List[str]:
        template = self.__get_template(module)
        return [template] if template else []

    def get_output_files(self, _: BuildUnit, module: Module) -> List[str]:
        template = self.__get_template(module)
        return [self.__index_file(template)] if template else []

    def get_clean_files(self, build_unit: BuildUnit, module: Module) -> List[str]:
        Log.info('Removing index file referencing API documentation in directory "%s"', module.output_directory)
        return super().get_clean_files(build_unit, module)


class BuildDocumentation(BuildTarget.Runnable):
    """
    Generates documentations.
    """

    ENV_SPHINX_BUILDER = 'SPHINX_BUILDER'

    def __init__(self):
        super().__init__(SphinxModule.Filter())
        sphinx_builder = get_env(environ, self.ENV_SPHINX_BUILDER, default=SphinxBuild.BUILDER_HTML)
        valid_builders = {SphinxBuild.BUILDER_HTML, SphinxBuild.BUILDER_LINKCHECK, SphinxBuild.BUILDER_SPELLING}

        if sphinx_builder not in valid_builders:
            Log.error('Command line argument %s must be one of {%s}, but got: "%s"', self.ENV_SPHINX_BUILDER,
                      format_iterable(valid_builders, delimiter='"'), sphinx_builder)

        self.sphinx_builder = sphinx_builder

    def run(self, build_unit: BuildUnit, module: Module):
        Log.info('Building documentation for directory "%s" (using builder "%s")"...', module.root_directory,
                 self.sphinx_builder)
        SphinxBuild(build_unit, module, builder=self.sphinx_builder).run()

    def get_input_files(self, _: BuildUnit, module: Module) -> List[str]:
        return module.find_source_files() if self.sphinx_builder == SphinxBuild.BUILDER_HTML else []

    def get_output_files(self, _: BuildUnit, module: Module) -> List[str]:
        return [module.output_directory] if self.sphinx_builder == SphinxBuild.BUILDER_HTML else []

    def get_clean_files(self, _: BuildUnit, module: Module) -> List[str]:
        Log.info('Removing documentation for directory "%s"...', module.root_directory)
        return [module.output_directory]
