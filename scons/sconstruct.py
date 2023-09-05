"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines the individual targets of the build process.
"""
from os import path

from SCons.Script.SConscript import SConsEnvironment

from modules import BUILD_MODULE

# Create temporary file ".sconsign.dblite" in the build directory...
env = SConsEnvironment()
env.SConsignFile(name=path.join(BUILD_MODULE.build_dir, '.sconsign'))
