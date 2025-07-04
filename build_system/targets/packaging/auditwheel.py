"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run the external program "auditwheel".
"""
import shutil

from os import path

from core.build_unit import BuildUnit
from util.files import FileSearch
from util.io import delete_files
from util.log import Log
from util.run import Program


class Auditwheel(Program):
    """
    Allows to run the external program "auditwheel".
    """

    def __init__(self, build_unit: BuildUnit, wheel: str):
        """
        :param build_unit:  The build unit from which the program should be run
        :param wheel:       The path to the wheel package to be repaired
        """
        wheel_directory = path.join(path.dirname(wheel), 'wheelhouse')
        super().__init__('auditwheel', 'repair', wheel, '--wheel-dir', wheel_directory)
        self.set_build_unit(build_unit)
        self.wheel = wheel
        self.wheel_directory = wheel_directory

    def _after(self):
        original_wheel = self.wheel
        delete_files(original_wheel)
        original_directory = path.dirname(original_wheel)
        wheel_directory = self.wheel_directory

        for wheel in FileSearch().list(wheel_directory):
            Log.info('Copying file "%s" into directory "%s"...', wheel, original_directory)
            shutil.copy(wheel, original_directory)

        delete_files(wheel_directory)
