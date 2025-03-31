"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to format .cfg files.
"""
from configparser import Error as ConfigParserError

from core.build_unit import BuildUnit
from util.io import ENCODING_UTF8
from util.log import Log
from util.pip import Pip

from targets.code_style.modules import CodeModule


class CfgFormatter:
    """
    Allow to format .cfg files.
    """

    def __init__(self, build_unit: BuildUnit, module: CodeModule, enforce_changes: bool = False):
        """
        :param build_unit:      The build unit from which the program should be run
        :param module:          The module, the program should be applied to
        :param enforce_changes: True, if changes should be applied to files, False otherwise
        """
        self.build_unit = build_unit
        self.module = module
        self.enforce_changes = enforce_changes

    def run(self):
        """
        Runs the formatter.
        """
        malformed_files = []

        Pip.for_build_unit(self.build_unit).install_packages('config-formatter')
        # pylint: disable=import-outside-toplevel
        from config_formatter import ConfigFormatter

        for config_file in self.module.find_source_files():
            with open(config_file, mode='r+' if self.enforce_changes else 'r', encoding=ENCODING_UTF8) as file:
                try:
                    content = file.read()
                    Log.verbose('Formatting file "%s"', config_file)
                    formatted_content = ConfigFormatter().prettify(content)

                    if content != formatted_content:
                        if self.enforce_changes:
                            Log.info('Formatting file "%s"...', config_file)
                            file.seek(0)
                            file.write(formatted_content)
                            file.truncate()
                        else:
                            Log.info('File "%s" is not properly formatted!', config_file)
                            malformed_files.append(config_file)
                except ConfigParserError as error:
                    Log.error('Failed to format file "%s"', config_file, error=error)

        if malformed_files:
            Log.error('%s %s not properly formatted!', len(malformed_files),
                      'files are' if len(malformed_files) > 1 else 'file is')
