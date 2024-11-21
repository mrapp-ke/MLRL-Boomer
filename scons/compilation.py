"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for compiling C++ and Cython code.
"""
from build_options import BuildOptions, EnvBuildOption
from modules import CPP_MODULE, PYTHON_MODULE
from util.run import Program

CPP_BUILD_OPTIONS = BuildOptions() \
        .add(EnvBuildOption(name='subprojects')) \
        .add(EnvBuildOption(name='test_support', subpackage='common')) \
        .add(EnvBuildOption(name='multi_threading_support', subpackage='common')) \
        .add(EnvBuildOption(name='gpu_support', subpackage='common'))


CYTHON_BUILD_OPTIONS = BuildOptions() \
        .add(EnvBuildOption(name='subprojects'))


class MesonSetup(Program):
    """
    Allows to run the external program "meson setup".
    """

    def __init__(self, build_directory: str, source_directory: str, build_options: BuildOptions = BuildOptions()):
        """
        :param build_directory:     The path to the build directory
        :param source_directory:    The path to the source directory
        :param build_options:       The build options to be used
        """
        super().__init__('meson', 'setup', *build_options.as_arguments(), build_directory, source_directory)
        self.print_arguments(True)


class MesonConfigure(Program):
    """
    Allows to run the external program "meson configure".
    """

    def __init__(self, build_directory: str, build_options: BuildOptions = BuildOptions()):
        """
        :param build_directory: The path to the build directory
        :param build_options:   The build options to be used
        """
        super().__init__('meson', 'configure', *build_options.as_arguments(), build_directory)
        self.print_arguments(True)
        self.build_options = build_options

    def run(self):
        if self.build_options:
            print('Configuring build options according to environment variables...')
            super().run()


class MesonCompile(Program):
    """
    Allows to run the external program "meson compile".
    """

    def __init__(self, build_directory: str):
        """
        :param build_directory: The path to the build directory
        """
        super().__init__('meson', 'compile', '-C', build_directory)
        self.print_arguments(True)


class MesonInstall(Program):
    """
    Allows to run the external program "meson install".
    """

    def __init__(self, build_directory: str):
        """
        :param build_directory: The path to the build directory
        """
        super().__init__('meson', 'install', '--no-rebuild', '--only-changed', '-C', build_directory)
        self.print_arguments(True)


def setup_cpp(**_):
    """
    Sets up the build system for compiling the C++ code.
    """
    MesonSetup(build_directory=CPP_MODULE.build_dir,
               source_directory=CPP_MODULE.root_dir,
               build_options=CPP_BUILD_OPTIONS) \
        .add_dependencies('ninja') \
        .run()


def compile_cpp(**_):
    """
    Compiles the C++ code.
    """
    MesonConfigure(CPP_MODULE.build_dir, CPP_BUILD_OPTIONS).run()
    print('Compiling C++ code...')
    MesonCompile(CPP_MODULE.build_dir).run()


def install_cpp(**_):
    """
    Installs shared libraries into the source tree.
    """
    print('Installing shared libraries into source tree...')
    MesonInstall(CPP_MODULE.build_dir).run()


def setup_cython(**_):
    """
    Sets up the build system for compiling the Cython code.
    """
    MesonSetup(build_directory=PYTHON_MODULE.build_dir,
               source_directory=PYTHON_MODULE.root_dir,
               build_options=CYTHON_BUILD_OPTIONS) \
        .add_dependencies('cython') \
        .run()


def compile_cython(**_):
    """
    Compiles the Cython code.
    """
    MesonConfigure(PYTHON_MODULE.build_dir, CYTHON_BUILD_OPTIONS)
    print('Compiling Cython code...')
    MesonCompile(PYTHON_MODULE.build_dir).run()


def install_cython(**_):
    """
    Installs extension modules into the source tree.
    """
    print('Installing extension modules into source tree...')
    MesonInstall(PYTHON_MODULE.build_dir).run()
