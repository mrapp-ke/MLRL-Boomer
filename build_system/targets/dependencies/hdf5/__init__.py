"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines targets for compiling the native library dependency "HDF5" from source. This is necessary on platforms where no
suitable version of the library can be installed via a package manager, e.g., when building pre-built packages via
"cibuildwheel".
"""

from pathlib import Path

from core.build_unit import BuildUnit
from core.targets import TargetBuilder

from targets.dependencies.hdf5.targets import compile_hdf5

TARGETS = (
    TargetBuilder(BuildUnit.for_file(Path(__file__)))
    .add_phony_target('dependency_hdf5')
    .set_functions(compile_hdf5)
    .build()
)
