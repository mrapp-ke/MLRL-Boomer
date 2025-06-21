"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for retrieving information about this Python package.
"""
from mlrl.common.cython.package_info import get_cpp_library_info as get_common_cpp_library_info
from mlrl.common.package_info import PackageInfo, get_package_info as get_common_package_info

from mlrl.seco.cython.package_info import get_cpp_library_info


def get_package_info() -> PackageInfo:
    """
    Returns information about this Python package.

    :return: A `PackageInfo` that provides information about the Python package
    """
    return PackageInfo(
        package_name='mlrl-seco',
        python_packages=[get_common_package_info()],
        cpp_libraries=[get_common_cpp_library_info(), get_cpp_library_info()],
    )
