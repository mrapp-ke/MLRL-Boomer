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

    class UninstallCommand(Pip.Command):
        """
        Allows to uninstall packages via the command `pip uninstall`.
        """

        def __init__(self, *package_names: str):
            """
            :param package_names: The names of the packages to be uninstalled
            """
            super().__init__('uninstall', '--yes', *package_names)

    def install_wheels(self, *wheels: str):
        """
        Installs several wheel packages.

        :param wheels: The paths to the wheel packages to be installed
        """
        PipInstallWheel.InstallWheelCommand(*wheels).print_arguments(True).run()

    def uninstall_packages(self, *package_names: str):
        """
        Uninstalls several packages.

        :param package_names: The names of the packages to be uninstalled
        """
        PipInstallWheel.UninstallCommand(*package_names).print_arguments(True).run()
