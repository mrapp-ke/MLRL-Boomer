"""
This module provides common functionality for rule learning algorithms.
"""
from mlrl.common.cython.package_info import get_gpu_devices, get_num_cpu_cores, get_supported_simd_extensions, \
    is_gpu_available, is_gpu_support_enabled, is_multi_threading_support_enabled, is_simd_support_enabled
