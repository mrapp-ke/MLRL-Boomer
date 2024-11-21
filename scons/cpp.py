"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for compiling C++ code.
"""
from compilation.build_options import BuildOptions, EnvBuildOption
from compilation.meson import MesonCompile, MesonConfigure, MesonInstall, MesonSetup
from modules import CPP_MODULE

BUILD_OPTIONS = BuildOptions() \
        .add(EnvBuildOption(name='subprojects')) \
        .add(EnvBuildOption(name='test_support', subpackage='common')) \
        .add(EnvBuildOption(name='multi_threading_support', subpackage='common')) \
        .add(EnvBuildOption(name='gpu_support', subpackage='common'))


def setup_cpp(**_):
    """
    Sets up the build system for compiling the C++ code.
    """
    MesonSetup(build_directory=CPP_MODULE.build_dir,
               source_directory=CPP_MODULE.root_dir,
               build_options=BUILD_OPTIONS) \
        .add_dependencies('ninja') \
        .run()


def compile_cpp(**_):
    """
    Compiles the C++ code.
    """
    MesonConfigure(CPP_MODULE.build_dir, BUILD_OPTIONS).run()
    print('Compiling C++ code...')
    MesonCompile(CPP_MODULE.build_dir).run()


def install_cpp(**_):
    """
    Installs shared libraries into the source tree.
    """
    print('Installing shared libraries into source tree...')
    MesonInstall(CPP_MODULE.build_dir).run()
