"""
This module provides common functionality for rule learning algorithms.
"""

from mlrl.common.cython.package_info import (
    get_gpu_devices,
    get_num_cpu_cores,
    get_supported_simd_extensions,
    is_gpu_available,
    is_gpu_support_enabled,
    is_multi_threading_support_enabled,
    is_simd_support_enabled,
)

__all__ = [
    'get_gpu_devices',
    'get_num_cpu_cores',
    'get_supported_simd_extensions',
    'is_gpu_available',
    'is_gpu_support_enabled',
    'is_multi_threading_support_enabled',
    'is_simd_support_enabled',
    'load_model',
]


def __getattr__(name: str):
    # The function `load_model` is imported lazily to avoid importing the module `mlrl.common.learners`, and its
    # dependencies, when importing this package
    if name == 'load_model':
        from mlrl.common.learners import load_model

        return load_model

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
