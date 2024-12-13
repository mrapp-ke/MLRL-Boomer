"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for installing wheel packages via pip.
"""
from util.pip import Pip


class PipInstallWheel(Pip):
    """
    Allows to install wheel packages via pip.
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

    def install_wheels(self, *wheels: str):
        """
        Installs several wheel packages.
        """
        PipInstallWheel.InstallWheelCommand(*wheels).print_arguments(True).run()
