"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for compiling C++ and Cython code.
"""
from build_options import BuildOptions, EnvBuildOption
from meson import MesonCompile, MesonConfigure, MesonInstall, MesonSetup
from modules import CPP_MODULE, PYTHON_MODULE

CPP_BUILD_OPTIONS = BuildOptions() \
        .add(EnvBuildOption(name='subprojects')) \
        .add(EnvBuildOption(name='test_support', subpackage='common')) \
        .add(EnvBuildOption(name='multi_threading_support', subpackage='common')) \
        .add(EnvBuildOption(name='gpu_support', subpackage='common'))


CYTHON_BUILD_OPTIONS = BuildOptions() \
        .add(EnvBuildOption(name='subprojects'))


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
