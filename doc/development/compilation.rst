.. _compilation:

Building from Source
====================

As discussed in the previous section :ref:`project_structure`, the algorithm that is provided by this project is implemented in `C++ <https://en.wikipedia.org/wiki/C%2B%2B>`__ to ensure maximum efficiency (requires C++ 14 or newer). In addition, a `Python <https://en.wikipedia.org/wiki/Python_(programming_language)>`__ wrapper that integrates the algorithm with the `scikit-learn <https://scikit-learn.org>`__ framework is provided (requires Python 3.8 or newer). To make the underlying C++ implementation accessible from within the Python code, `Cython <https://en.wikipedia.org/wiki/Cython>`__ is used (requires Cython 3.0 or newer).

Unlike pure Python programs, the C++ and Cython source files must be compiled for a particular target platform. To ease the process of compiling the source code, the project comes with a `SCons <https://scons.org/>`__ build that automates the necessary steps. In the following, we discuss the individual steps that are necessary for building the project from scratch. This is necessary if you intend to modify the library's source code. If you want to use the algorithm without any custom modifications, the :ref:`installation` of pre-built packages is usually a better choice.

Prerequisites
-------------

As a prerequisite, a supported version of Python, a suitable C++ compiler, as well as optional libraries for multi-threading and GPU support, must be available on the host system. The installation of these software components depends on the operation system at hand. In the following, we provide installation instructions for the supported platforms.

.. tip::
    This project uses `Meson <https://mesonbuild.com/>`_ as a build system for compiling C++ code. If available on the system, Meson automatically utilizes `Ccache <https://ccache.dev/>`__ for caching previous compilations and detecting when the same compilation is being done again. Compared to the runtime without Ccache, where changes are only detected at the level of entire files, usage of this compiler cache significantly speeds up recompilation and therefore is strongly adviced.

.. tab:: Linux

   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **Python**       | Nowadays, most Linux distributions include a pre-installed version of Python 3. If this is not the case, instructions on how to install a recent Python version can be found in Python’s `Beginners Guide <https://wiki.python.org/moin/BeginnersGuide/Download>`__. As noted in this guide, Python should be installed via the distribution’s package manager if possible. |
   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **C++ compiler** | Most Linux distributions provide the `GNU Compiler Collection <https://gcc.gnu.org/>`__ (GCC), which includes a C++ compiler, as part of their software repositories. If this is the case, it can be installed via the distribution's package manager.                                                                                                                      |
   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **GoogleTest**   | The `GoogleTest <https://github.com/google/googletest>`__ framework must optionally be available in order to compile the project with :ref:`testingsupport` enabled. It should be possible to install it via the package manager of your Linux distribution.                                                                                                                |
   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **OpenMP**       | `OpenMP <https://en.wikipedia.org/wiki/OpenMP>`__, which is optionally required for :ref:`multithreadingsupport`, should be installable via your Linux distribution's package manager.                                                                                                                                                                                      |
   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **OpenCL**       | If the project should be compiled with :ref:`gpusupport`, `OpenCL <https://www.khronos.org/opencl/>`__ must be available. On Linux, it should be installable via your distribution's package manager.                                                                                                                                                                       |
   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. tab:: MacOS

   +------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **Python**       | Recent versions of MacOS do not include Python by default. A suitable Python version can manually be downloaded from the `project's website <https://www.python.org/downloads/macos/>`__. Alternatively, the package manager `Homebrew <https://en.wikipedia.org/wiki/Homebrew_(package_manager)>`__ can be used for installation via the command ``brew install python``.                                                                                   |
   +------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **C++ compiler** | MacOS relies on the `Clang <https://en.wikipedia.org/wiki/Clang>`__ compiler for building C++ code. It is part of the `Xcode <https://developer.apple.com/support/xcode/>`__ developer toolset.                                                                                                                                                                                                                                                              |
   +------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **GoogleTest**   | The `GoogleTest <https://github.com/google/googletest>`__ framework must optionally be installed in order to compile the project with :ref:`testingsupport` enabled. It can easily be installed via `Homebrew <https://en.wikipedia.org/wiki/Homebrew_(package_manager)>`__ by runnig the command ``brew install googletest``.                                                                                                                               |
   +------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **OpenMP**       | If the project should be compiled with :ref:`multithreadingsupport` enabled, the `OpenMP <https://en.wikipedia.org/wiki/OpenMP>`__ library must be installed. We recommend to install it via `Homebrew <https://en.wikipedia.org/wiki/Homebrew_(package_manager)>`__ by running the command ``brew install libomp``.                                                                                                                                         |
   +------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **OpenCL**       | The `Xcode <https://developer.apple.com/support/xcode/>`__ developer toolset should include `OpenCL <https://www.khronos.org/opencl/>`__, which is needed for :ref:`gpusupport`. However, the `OpenCL C++ headers <https://github.com/KhronosGroup/OpenCL-Headers>`__ must be installed manually. The easiest way to do so is via the `Homebrew <https://en.wikipedia.org/wiki/Homebrew_(package_manager)>`__ command ``brew install opencl-clhpp-headers``. |
   +------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. tab:: Windows

   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **Python**       | Python releases for Windows are available at the `project's website <https://www.python.org/downloads/windows/>`__, where you can download an installer.                                                                                                                                                                                            |
   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **C++ compiler** | For the compilation of the project's source code, the MSVC compiler must be used. It is included in `Visual Studio <https://visualstudio.microsoft.com/downloads/>`__.                                                                                                                                                                              |
   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **GoogleTest**   | The `GoogleTest <https://github.com/google/googletest>`__ framework must optionally be available on your system to compile the project with :ref:`testingsupport` enabled. It should already be included in recent versions of `Visual Studio <https://learn.microsoft.com/en-us/visualstudio/test/how-to-use-google-test-for-cpp?view=vs-2022>`__. |
   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **OpenMP**       | The `Build Tools for Visual Studio <https://visualstudio.microsoft.com/downloads/>`__ also include the `OpenMP <https://en.wikipedia.org/wiki/OpenMP>`__ library, which is utilized by the project for :ref:`multithreadingsupport`.                                                                                                                |
   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | **OpenCL**       | If you intend to compile the project with :ref:`gpusupport` enabled, `OpenCL <https://www.khronos.org/opencl/>`__ must be installed manually. In order to do so, we recommend to install the package ``opencl`` via the package manager `vcpkg <https://github.com/microsoft/vcpkg>`__.                                                             |
   +------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Additional build- or run-time dependencies will automatically be installed when following the instructions below and must not be installed manually.


.. tip::
    Instead of following the instructions below step by step, the following command, which automatically executes all necessary steps, can be used for simplicity.

    .. tab:: Linux

       .. code-block:: text

          ./build

    .. tab:: MacOS

       .. code-block:: text

          ./build

    .. tab:: Windows

       .. code-block:: text

          build.bat
    
    Whenever any C++, Cython or Python source files have been modified, the above command must be run again in order to rebuild modified files and install updated wheel packages into the virtual environment. If any compilation files do already exist, this will only result in the affected parts of the code to be rebuilt.

Creating a Virtual Environment
------------------------------

The build process is based on an virtual Python environment that allows to install build- and run-time dependencies in an isolated manner and independently from the host system. Once the build process was completed, the resulting Python packages are installed into the virtual environment. To create new virtual environment and install all necessarily run-time dependencies, the following command must be executed:

.. tab:: Linux

   .. code-block:: text

      ./build venv

.. tab:: MacOS

   .. code-block:: text

      ./build venv

.. tab:: Windows

   .. code-block:: text

      build.bat venv

All run-time dependencies (`numpy`, `scipy`, etc.) that are required for running the algorithms that are provided by the project should automatically be installed into the virtual environment when executing the above command. As a result, a subdirectory `venv/` should have been created in the project's root directory.

Compiling the C++ Code
----------------------

Once a new virtual environment has successfully been created, the compilation of the C++ code can be started by executing the following command:

.. tab:: Linux

   .. code-block:: text

      ./build compile_cpp

.. tab:: MacOS

   .. code-block:: text

      ./build compile_cpp

.. tab:: Windows

   .. code-block:: text

      build.bat compile_cpp

The compilation is based on the build system `Meson <https://mesonbuild.com/>`_ and uses `Ninja <https://ninja-build.org/>`_ as a backend. After the above command has terminated, a new directory `cpp/build/` should have been created. It contains the shared libraries ("libmlrlcommon", "libmlrlboosting" and possibly others) that provide the basic functionality of the project's algorithms.

Compiling the Cython Code
-------------------------

Once the compilation of the C++ code has completed, the Cython code, which allows to access the corresponding shared libraries from within Python, can be compiled in the next step. Again, Meson and Ninja are used for compilation. It can be started via the following command:

.. tab:: Linux

   .. code-block:: text

      ./build compile_cython

.. tab:: MacOS

   .. code-block:: text

      ./build compile_cython

.. tab:: Windows

   .. code-block:: text

      build.bat compile_cython

As a result of executing the above command, the directory `python/build` should have been created. It contains Python extension modules for the respective target platform.

.. note::
    Instead of performing the previous steps one after the other, the build target ``compile`` can be specfied instead of ``compile_cpp`` and ``compile_cython`` to build the C++ and Cython source files in a single step.

Installing Shared Libraries
---------------------------

The shared libraries that have been created in the previous steps from the C++ source files must afterwards be copied into the Python source tree. This can be achieved by executing the following command:

.. tab:: Linux

   .. code-block:: text

      ./build install_cpp

.. tab:: MacOS

   .. code-block:: text

      ./build install_cpp

.. tab:: Windows

   .. code-block:: text

      build.bat install_cpp

This should result in the compilation files, which were previously located in the `cpp/build/` directory, to be copied into the `cython/` subdirectories that are contained by each Python module (e.g., into the directory `python/subprojects/common/mlrl/common/cython/`).

Installing Extension Modules
----------------------------

Similar to the previous step, the Python extension modules that have been built from the project's Cython code must be copied into the Python source tree via the following command:

.. tab:: Linux

   .. code-block:: text

      ./build install_cython

.. tab:: MacOS

   .. code-block:: text

      ./build install_cython

.. tab:: Windows

   .. code-block:: text

      build.bat install_cython

As a result, the compilation files that can be found in the `python/build/` directories should have been copied into the `cython/` subdirectories of each Python module.

.. note::
    Instead of executing the above commands one after the other, the build target ``install`` can be used instead of ``install_cpp`` and ``install_cython`` to copy both, the shared libraries and the extension modules, into the source tree.

Building Wheel Packages
-----------------------

Once the compilation files have been copied into the Python source tree, wheel packages can be built for the individual Python modules via the following command:

.. tab:: Linux

   .. code-block:: text

      ./build build_wheels

.. tab:: MacOS

   .. code-block:: text

      ./build build_wheels

.. tab:: Windows

   .. code-block:: text

      build.bat build_wheels

This should result in .whl files being created in a new `dist/` subdirectory inside the directories that correspond to the individual Python modules (e.g., in the directory `python/subprojects/common/dist/`).

Installing the Wheel Packages
-----------------------------

The wheel packages that have previously been created can finally be installed into the virtual environment via the following command:

.. tab:: Linux

   .. code-block:: text

      ./build install_wheels

.. tab:: MacOS

   .. code-block:: text

      ./build install_wheels

.. tab:: Windows

   .. code-block:: text

      build.bat install_wheels

After this final step has completed, the Python packages can be used from within the virtual environment once it has been `activated <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activating-a-virtual-environment>`__. To ensure that the installation of the wheel packages was successful, check if a `mlrl/` directory has been created in the `lib/` directory of the virtual environment (depending on the Python version, it should be located at `venv/lib/python3.9/site-packages/mlrl/` or similar). If this is the case, the algorithm can be used from within your own Python code. Alternatively, the command line API can be used to start an experiment (see :ref:`testbed`).

Cleaning up Build Files
-----------------------

It is possible to delete the compilation files that result from an individual step of the build process mentioned above by using the command libe argument ``--clean`` or ``-c``. This may be useful if you want to repeat a single or multiple steps of the build process from scratch in case anything went wrong. For example, to delete the C++ compilation files, the following command can be used:

.. tab:: Linux

   .. code-block:: text

      ./build --clean compile_cpp

.. tab:: MacOS

   .. code-block:: text

      ./build --clean compile_cpp

.. tab:: Windows

   .. code-block:: text

      build.bat --clean compile_cpp

If you want to delete all compilation files that have previously been created, including the virtual environment, you should use the following command, where no build target is specified:

.. tab:: Linux

   .. code-block:: text

      ./build --clean

.. tab:: MacOS

   .. code-block:: text

      ./build --clean

.. tab:: Windows

   .. code-block:: text

      build.bat --clean
