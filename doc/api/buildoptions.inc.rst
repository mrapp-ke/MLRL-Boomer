.. _buildoptions:

Build Options
-------------

Certain functionalities of the project can be enabled or disabled at compile-time via so-called build options. They can be specified in the configuration file `cpp/subprojects/common/meson.options`.

**Multi-threading Support**

By default, the project is built with multi-threading support enabled. This requires `OpenMP <https://www.openmp.org/>`__ to be available on the host system. In order to compile the project without multi-threading support, e.g., because OpenMP is not available, the build option ``multi_threading_support`` can be set to ``disabled`` instead of ``enabled``.
