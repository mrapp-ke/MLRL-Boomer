"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to format .cfg files.
"""
from configparser import Error as ConfigParserError

from core.build_unit import BuildUnit
from util.io import ENCODING_UTF8
from util.log import Log
from util.package_manager import PackageManager
from util.requirements import RequirementsFiles

from targets.code_style.formatter import CodeChangeDetection
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
        self.enforce_changes = enforce_changes
        self.change_detection = CodeChangeDetection(module, 'cfg_formatter')

    def run(self):
        """
        Runs the formatter.
        """
        change_detection = self.change_detection
        source_files = change_detection.find_modified_source_files()

        if source_files:
            malformed_files = []

            PackageManager.install_packages(RequirementsFiles.for_build_unit(self.build_unit), 'config-formatter')
            # pylint: disable=import-outside-toplevel
            from config_formatter import ConfigFormatter

            for source_file in source_files:
                with open(source_file, mode='r+' if self.enforce_changes else 'r', encoding=ENCODING_UTF8) as file:
                    try:
                        content = file.read()
                        formatted_content = ConfigFormatter().prettify(content)

                        if content != formatted_content:
                            if self.enforce_changes:
                                Log.info('Formatting file "%s"...', source_file)
                                file.seek(0)
                                file.write(formatted_content)
                                file.truncate()
                            else:
                                Log.info('File "%s" is not properly formatted!', source_file)
                                malformed_files.append(source_file)
                    except ConfigParserError as error:
                        Log.error('Failed to format file "%s"', source_file, error=error)

            if malformed_files:
                Log.error('%s %s not properly formatted!', len(malformed_files),
                          'files are' if len(malformed_files) > 1 else 'file is')

            change_detection.update_cache()
