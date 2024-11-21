"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for compiling Cython code.
"""
from compilation.build_options import BuildOptions, EnvBuildOption
from compilation.meson import MesonCompile, MesonConfigure, MesonInstall, MesonSetup
from modules import PYTHON_MODULE

BUILD_OPTIONS = BuildOptions() \
        .add(EnvBuildOption(name='subprojects'))


def setup_cython(**_):
    """
    Sets up the build system for compiling the Cython code.
    """
    MesonSetup(build_directory=PYTHON_MODULE.build_dir,
               source_directory=PYTHON_MODULE.root_dir,
               build_options=BUILD_OPTIONS) \
        .add_dependencies('cython') \
        .run()


def compile_cython(**_):
    """
    Compiles the Cython code.
    """
    MesonConfigure(PYTHON_MODULE.build_dir, BUILD_OPTIONS)
    print('Compiling Cython code...')
    MesonCompile(PYTHON_MODULE.build_dir).run()


def install_cython(**_):
    """
    Installs extension modules into the source tree.
    """
    print('Installing extension modules into source tree...')
    MesonInstall(PYTHON_MODULE.build_dir).run()
