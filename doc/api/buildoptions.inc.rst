.. _buildoptions:

Build Options
-------------

Certain functionalities of the project can be enabled or disabled at compile-time via so-called build options. They can be specified in the configuration file `cpp/subprojects/common/meson.options`.

**Multi-threading Support**

By default, the project is built with multi-threading support enabled. This requires `OpenMP <https://www.openmp.org/>`__ to be available on the host system. In order to compile the project without multi-threading support, e.g., because OpenMP is not available, the build option ``multi_threading_support`` can be set to ``disabled`` instead of ``enabled``.

When using the :ref:`testbed`, the command ``boomer --version`` or ``boomer -v`` can be executed to check whether the program was built with multi-threading support enabled or not. It will print the build options used for compilation, as well as information about the CPU cores available on the system for multi-threading.

If you need to access this information programmatically in your own Python code, the following code snippet can be used (see :ref:`pythonapi`):

.. code-block:: python

   from mlrl.common import get_num_cpu_cores, is_multi_threading_support_enabled

   multi_threading_support_enabled: bool = is_multi_threading_support_enabled()
   num_cpu_cores: int = get_num_cpu_cores()

Similarly, the same information can also be accessed when using the C++ API programmatically (see :ref:`cppapi`):

.. code-block:: cpp

   #include "common/info.hpp"

   bool multiThreadingSupportEnabled = isMultiThreadingSupportEnabled();
   uint32 numCpuCores = getNumCpuCores();
