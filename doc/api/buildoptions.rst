.. _buildoptions:

Build Options
=============

Certain functionalities of the project can be enabled or disabled at compile-time via so-called build options. They can be specified in the configuration file `cpp/subprojects/common/meson.options`.

Multi-threading Support
-----------------------

By default, the project is built with multi-threading support enabled. This requires `OpenMP <https://www.openmp.org/>`__ to be available on the host system. In order to compile the project without multi-threading support, e.g., because OpenMP is not available, the build option ``multi_threading_support`` can be set to ``disabled`` instead of ``enabled``.

When using the :ref:`testbed`, the command ``boomer --version`` or ``boomer -v`` can be executed to check whether the program was built with multi-threading support enabled or not. It will print the build options used for compilation, as well as information about the CPU cores available on the system for multi-threading.

If you need to access this information programmatically in your own Python or C++ code, the following code snippets can be used (see :ref:`apidoc`):

.. tab:: Python

   .. code-block:: python

      from mlrl.common import get_num_cpu_cores, is_multi_threading_support_enabled

      multi_threading_support_enabled: bool = is_multi_threading_support_enabled()
      num_cpu_cores: int = get_num_cpu_cores()

.. tab:: C++

   .. code-block:: cpp

      #include "mlrl/common/info.hpp"

      bool multiThreadingSupportEnabled = isMultiThreadingSupportEnabled();
      uint32 numCpuCores = getNumCpuCores();

GPU support
-----------

GPU support via `OpenCL <https://www.khronos.org/opencl/>`__ is enabled by default when building the project. However, it can be disabled at compile-time by setting the build option ``gpu_support`` to ``disabled`` instead of ``enabled``.

An easy way to check whether the program was built with GPU support enabled or not, is to run the ``boomer --version`` or ``boomer -v`` command that is provided by the :ref:`testbed`. It will print the build options used for compiling the program, together with a list of supported GPUs available on your machine.

Alternatively, this information can be retrieved programmatically via the Python or C++ API as shown below (see :ref:`apidoc`):

.. tab:: Python

   .. code-block:: python

      from mlrl.common import get_gpu_devices, is_gpu_available, is_gpu_support_enabled
      from typing import List

      gpu_support_enabled: bool = is_gpu_support_enabled()
      gpu_available: bool = is_gpu_available()
      gpu_devices: List[str] = get_gpu_devices()

.. tab:: C++

   .. code-block:: cpp

      #include "mlrl/common/info.hpp"

      bool gpuSupportEnabled = isGpuSupportEnabled();
      bool gpuAvailable = isGpuAvailable();
      std::vector<std::string> gpuDevices = getGpuDevices();
