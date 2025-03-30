(compilation)=

# Building from Source

As discussed in the previous section {ref}`project-structure`, the algorithms that are provided by this project are implemented in [C++](https://en.wikipedia.org/wiki/C%2B%2B) to ensure maximum efficiency (the minimum C++ version required for compilation is specified in the file {repo-file}`.version-cpp <cpp/.version-cpp>`). In addition, a [Python](<https://en.wikipedia.org/wiki/Python_(programming_language)>) wrapper that integrates with the [scikit-learn](https://scikit-learn.org) framework is provided (the minimum required Python version is specified in the file {repo-file}`.version-python <python/.version-python>`). To make the underlying C++ implementation accessible from within the Python code, [Cython](https://en.wikipedia.org/wiki/Cython) is used.

Unlike pure Python programs, the C++ and Cython source files must be compiled for a particular target platform. To ease the process of compiling the source code, the project comes with a build system that automates the necessary steps. In the following, we discuss the individual steps that are necessary for building the project from scratch. This is necessary if you intend to modify the library's source code. If you want to use the algorithm without any custom modifications, the {ref}`installation <installation>` of pre-built packages is usually a better choice.

## Prerequisites

As a prerequisite, a supported version of Python, a suitable C++ compiler, as well as optional libraries for multi-threading and GPU support, must be available on the host system. The installation of these software components depends on the operating system at hand. In the following, we provide installation instructions for the supported platforms.

```{tip}
This project uses [Meson](https://mesonbuild.com/) as a build system for compiling C++ code. If available on the system, Meson automatically utilizes [Ccache](https://ccache.dev/) for caching previous compilations and detecting when the same compilation is being done again. Compared to the runtime without Ccache, where changes are only detected at the level of entire files, usage of this compiler cache can significantly speed up recompilation and therefore is strongly advised.
```

````{tab} Linux
   ```{list-table}
   * - **Python**
     - Nowadays, most Linux distributions include a pre-installed version of Python 3. If this is not the case, instructions on how to install a recent Python version can be found in Python’s [Beginners Guide](https://wiki.python.org/moin/BeginnersGuide/Download). As noted in this guide, Python should be installed via the distribution’s package manager if possible.
   * - **C++ compiler**
     - Most Linux distributions provide the [GNU Compiler Collection](https://gcc.gnu.org/) (GCC), which includes a C++ compiler, as part of their software repositories. If this is the case, it can be installed via the distribution's package manager.
   * - **GoogleTest**
     - The [GoogleTest](https://github.com/google/googletest) framework must optionally be available in order to compile the project with {ref}`testing support <testing-support>` enabled. It should be possible to install it via the package manager of your Linux distribution.
   * - **OpenMP**
     - [OpenMP](https://en.wikipedia.org/wiki/OpenMP), which is optionally required for {ref}`multi-threading support <multi-threading-support>`, should be installable via your Linux distribution's package manager.         
   * - **OpenCL**
     - If the project should be compiled with {ref}`GPU support <gpu-support>`, [OpenCL](https://en.wikipedia.org/wiki/OpenCL) must be available. On Linux, it should be installable via your distribution's package manager.
   ```
````

````{tab} macOS
   ```{list-table}
   * - **Python**
     - Recent versions of macOS do not include Python by default. A suitable Python version can manually be downloaded from the [project's website](https://www.python.org/downloads/macos/). Alternatively, the package manager [Homebrew](<https://en.wikipedia.org/wiki/Homebrew_(package_manager)>) can be used for installation via the command `brew install python`.
   * - **C++ compiler**
     - macOS relies on the [Clang](https://en.wikipedia.org/wiki/Clang) compiler for building C++ code. It is part of the [Xcode](https://developer.apple.com/support/xcode/) developer toolset.
   * - **GoogleTest**
     - The [GoogleTest](https://github.com/google/googletest) framework must optionally be installed in order to compile the project with {ref}`testing support <testing-support>` enabled. It can easily be installed via [Homebrew](<https://en.wikipedia.org/wiki/Homebrew_(package_manager)>) by running the command `brew install googletest`.
   * - **OpenMP**
     - If the project should be compiled with {ref}`multi-threading support <multi-threading-support>` enabled, the [OpenMP](https://en.wikipedia.org/wiki/OpenMP) library must be installed. We recommend to install it via [Homebrew](<https://en.wikipedia.org/wiki/Homebrew_(package_manager)>) by running the command `brew install libomp`.
   * - **OpenCL**
     - The [Xcode](https://developer.apple.com/support/xcode/) developer toolset should include [OpenCL](https://en.wikipedia.org/wiki/OpenCL), which are needed for {ref}`GPU support <gpu-support>`. However, the [OpenCL C++ headers](https://github.com/KhronosGroup/OpenCL-Headers) must be installed manually. The easiest way to do so is via the [Homebrew](<https://en.wikipedia.org/wiki/Homebrew_(package_manager)>) command `brew install opencl-clhpp-headers`.
   ```
````

````{tab} Windows
   ```{list-table}
   * - **Python**
     - Python releases for Windows are available at the [project's website](https://www.python.org/downloads/windows/), where you can download an installer.
   * - **C++ compiler**
     - For the compilation of the project's source code, the MSVC compiler must be used. It is included in [Visual Studio](https://visualstudio.microsoft.com/downloads/).
   * - **GoogleTest**
     - The [GoogleTest](https://github.com/google/googletest) framework must optionally be available on your system to compile the project with {ref}`testing support <testing-support>` enabled. It should already be included in recent versions of [Visual Studio](https://learn.microsoft.com/en-us/visualstudio/test/how-to-use-google-test-for-cpp).
   * - **OpenMP**
     - The [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/) also include the [OpenMP](https://en.wikipedia.org/wiki/OpenMP) library, which is utilized by the project for {ref}`multi-threading support <multi-threading-support>`.
   * - **OpenCL**
     - If you intend to compile the project with {ref}`GPU support <gpu-support>` enabled, [OpenCL](https://en.wikipedia.org/wiki/OpenCL) must be installed manually. In order to do so, we recommend to install the package `opencl` via the package manager [vcpkg](https://github.com/microsoft/vcpkg).
   ```
````

Additional build- or run-time dependencies are automatically installed when following the instructions below and must not be installed manually.

`````{tip}
Instead of following the instructions below step by step, the following command, which automatically executes all necessary steps, can be used for simplicity.

````{tab} Linux
   ```text
   ./build
   ```
````

````{tab} macOS
   ```text
   ./build
   ```
````

````{tab} Windows
   ```text
   build.bat
   ```
````

Whenever any C++, Cython or Python source files have been modified, the above command must be run again in order to rebuild modified files and install updated wheel packages into the virtual environment. If any compilation files already exist, this does only result in the affected parts of the code to be rebuilt.
`````

`````{note}
As shown in the section {ref}`project-structure`, this project is organized in terms of several subprojects. By default, all of these subprojects are built when following the instructions below. However, the environment variable `SUBPROJECTS` may be used to restrict individual steps of the build process, such as the compilation of C++ and Cython code, the assemblage of Python packages, and the generation of API documentations, to a subset of the available subprojects. As shown below, multiple subprojects can be specified as a comma-separated list:

````{tab} Linux
   ```text
   SUBPROJECTS=common,boosting ./build
   ```
````

````{tab} macOS
   ```text
   SUBPROJECTS=common,boosting ./build
   ```
````

````{tab} Windows
   ```text
   $env:SUBPROJECTS = "common,boosting"
   build.bat
   ```
````
`````

## Creating a Virtual Environment

The build process is based on a virtual Python environment that allows to install build- and run-time dependencies in an isolated manner and independently of the host system. Once the build process was completed, the resulting Python packages are installed into the virtual environment. To create new virtual environment and install all necessarily run-time dependencies, the following command must be executed:

````{tab} Linux
   ```text
   ./build venv
   ```
````

````{tab} macOS
   ```text
   ./build venv
   ```
````

````{tab} Windows
   ```text
   build.bat venv
   ```
````

All run-time dependencies (`numpy`, `scipy`, etc.) that are required for running the algorithms that are provided by the project should automatically be installed into the virtual environment when executing the above command. As a result, a subdirectory `.venv/` should have been created in the project's root directory.

## Compiling the C++ Code

Once a new virtual environment has successfully been created, the compilation of the C++ code can be started by executing the following command:

````{tab} Linux
   ```text
   ./build compile_cpp
   ```
````

````{tab} macOS
   ```text
   ./build compile_cpp
   ```
````

````{tab} Windows
   ```text
   build.bat compile_cpp
   ```
````

The compilation is based on the build system [Meson](https://mesonbuild.com/) and uses [Ninja](https://ninja-build.org/) as a backend. After the above command has terminated, a new directory `cpp/build/` should have been created. It contains the shared libraries ("libmlrlcommon", "libmlrlboosting" and possibly others) that provide the basic functionality of the project's algorithms.

## Compiling the Cython Code

Once the compilation of the C++ code has completed, the Cython code, which allows to access the corresponding shared libraries from within Python, can be compiled in the next step. Again, Meson and Ninja are used for compilation. It can be started via the following command:

````{tab} Linux
   ```text
   ./build compile_cython
   ```
````

````{tab} macOS
   ```text
   ./build compile_cython
   ```
````

````{tab} Windows
   ```text
   build.bat compile_cython
   ```
````

As a result of executing the above command, the directory `python/build` should have been created. It contains Python extension modules for the respective target platform.

```{note}
Instead of performing the previous steps one after the other, the build target `compile` can be specified instead of `compile_cpp` and `compile_cython` to build the C++ and Cython source files in a single step.
```

## Installing Shared Libraries

The shared libraries that have been created in the previous steps from the C++ source files must afterward be copied into the Python source tree. This can be achieved by executing the following command:

````{tab} Linux
   ```text
   ./build install_cpp
   ```
````

````{tab} macOS
   ```text
   ./build install_cpp
   ```
````

````{tab} Windows
   ```text
   build.bat install_cpp
   ```
````

This should result in the compilation files, which were previously located in the `cpp/build/` directory, to be copied into the `cython/` subdirectories that are contained by each Python module (e.g., into the directory `python/subprojects/common/mlrl/common/cython/`).

## Installing Extension Modules

Similar to the previous step, the Python extension modules that have been built from the project's Cython code must be copied into the Python source tree via the following command:

````{tab} Linux
   ```text
   ./build install_cython
   ```
````

````{tab} macOS
   ```text
   ./build install_cython
   ```
````

````{tab} Windows
   ```text
   build.bat install_cython
   ```
````

As a result, the compilation files that can be found in the `python/build/` directories should have been copied into the `cython/` subdirectories of each Python module.

```{note}
Instead of executing the above commands one after the other, the build target `install` can be used instead of `install_cpp` and `install_cython` to copy both, the shared libraries and the extension modules, into the source tree.
```

## Generating `pyproject.toml` files

Once the compilation files have been copied into the Python source tree, [`pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) files that provide meta-data for the project's individual Python modules must be generated via the following command:

````{tab} Linux
   ```text
   ./build pyproject_toml
   ```
````

````{tab} macOS
   ```text
   ./build pyproject_toml
   ```
````

````{tab} Windows
   ```text
   build.bat pyproject_toml
   ```
````

As a result of running the above command, `pyproject.toml` files should have been created inside the directories that correspond to the individual Python modules (e.g., in the directory `python/subprojects/common/`).

## Building Wheel Packages

Based on the `pyproject.toml` files generated in the previous step, wheel packages can be built for the individual Python modules via the following command:

````{tab} Linux
   ```text
   ./build build_wheels
   ```
````

````{tab} macOS
   ```text
   ./build build_wheels
   ```
````

````{tab} Windows
   ```text
   build.bat build_wheels
   ```
````

This should result in .whl files being created in a new `dist/` subdirectory inside the directories that correspond to the individual Python modules (e.g., in the directory `python/subprojects/common/dist/`).

## Installing the Wheel Packages

The wheel packages that have previously been created can finally be installed into the virtual environment via the following command:

````{tab} Linux
   ```text
   ./build install_wheels
   ```
````

````{tab} macOS
   ```text
   ./build install_wheels
   ```
````

````{tab} Windows
   ```text
   build.bat install_wheels
   ```
````

After this final step has completed, the Python packages can be used from within the virtual environment once it has been [activated](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#activate-a-virtual-environment). To ensure that the installation of the wheel packages was successful, check if a `mlrl/` directory has been created in the `lib/` directory of the virtual environment (depending on the Python version and operating system, it should be located at `.venv/lib/python3.10/site-packages/mlrl/` or similar). If this is the case, the algorithm can be used from within your own Python code. Alternatively, the command line API can be used to start an experiment (see {ref}`testbed`).

## Cleaning up Build Files

It is possible to delete the compilation files that result from an individual step of the build process mentioned above by using the command line argument `--clean` or `-c`. This may be useful if you want to repeat a single or multiple steps of the build process from scratch in case anything went wrong. For example, to delete the C++ compilation files, the following command can be used:

````{tab} Linux
   ```text
   ./build --clean compile_cpp
   ```
````

````{tab} macOS
   ```text
   ./build --clean compile_cpp
   ```
````

````{tab} Windows
   ```text
   build.bat --clean compile_cpp
   ```
````

If you want to delete all compilation files that have previously been created, including the virtual environment, you should use the following command, where no build target is specified:

````{tab} Linux
   ```text
   ./build --clean
   ```
````

````{tab} macOS
   ```text
   ./build --clean
   ```
````

````{tab} Windows
   ```text
   build.bat --clean
   ```
````

(build-options)=

## Build Options

Certain functionalities of the project can be enabled or disabled at compile-time via so-called build options. They can be specified in the configuration file {repo-file}`meson.options <cpp/subprojects/common/meson.options>` or set via environment variables.

(build-type)=

### Build Type

By default, the project is compiled with compile-time optimizations enabled and debug symbols disabled. This behavior can be overruled by specifying the environment variable `BUILDTYPE`. Its value is directly passed to the [Meson](https://mesonbuild.com/) build system. Therefore, we refer to the [documentation](https://mesonbuild.com/Builtin-options.html#core-options) of this build system for a list of possible values. This build option can only be specified via the aforementioned environment variable, not via the configuration file {repo-file}`meson.options <cpp/subprojects/common/meson.options>`.

(testing-support)=

### Testing Support

This project comes with unit tests for the C++ code it contains (see {ref}`testing`). They are based on the [GoogleTest](https://github.com/google/googletest) framework. When building the project on a system where this dependency is available, the testing code is compiled and linked against the shared libraries it is supposed to test. By default, the build option `test_support` is set to `auto`, i.e., the testing code is only compiled if GoogleTest is available and no error is raised otherwise. To enforce the compilation of the testing code, the build option can be set to `enabled`. Setting it to `disabled` prevents the code from being compiled even if GoogleTest is available. Alternatively, the desired value can be specified via the environment variable `TEST_SUPPORT`.

(multi-threading-support)=

### Multi-Threading Support

By default, the project is built with multi-threading support enabled. This requires [OpenMP](https://www.openmp.org/) to be available on the host system. In order to compile the project without multi-threading support, e.g., because OpenMP is not available, the build option `multi_threading_support` can be set to `disabled` instead of `enabled`. Alternatively, the desired value can be specified via the environment variable `MULTI_THREADING_SUPPORT`.

When using the {ref}`command line API <testbed>`, the command `mlrl-testbed mlrl.boosting --version` or `mlrl-testbed mlrl.boosting -v` can be executed to check whether the program was built with multi-threading support enabled or not. It prints the build options used for compilation, as well as information about the CPU cores available on the system for multi-threading.

If you need to access this information programmatically in your own Python or C++ code, the following code snippets can be used (see {ref}`python-apidoc` and {ref}`cpp-apidoc`):

````{tab} Python
   ```python
   from mlrl.common import get_num_cpu_cores, is_multi_threading_support_enabled

   multi_threading_support_enabled: bool = is_multi_threading_support_enabled()
   num_cpu_cores: int = get_num_cpu_cores()
   ```
````

````{tab} C++
   ```cpp
   #include "mlrl/common/library_info.hpp"

   bool multiThreadingSupportEnabled = isMultiThreadingSupportEnabled();
   uint32 numCpuCores = getNumCpuCores();
   ```
````

(gpu-support)=

### GPU Support

```{warning}
So far, GPU support is still at an early stage of development. No algorithm provided by this project makes use of it yet.
```

GPU support via [OpenCL](https://en.wikipedia.org/wiki/OpenCL) is enabled by default when building the project. However, it can be disabled at compile-time by setting the build option `gpu_support` to `disabled` instead of `enabled`. Alternatively, the desired value can be specified via the environment variable `GPU_SUPPORT`.

An easy way to check whether the program was built with GPU support enabled or not, is to run the `mlrl-testbed mlrl.boosting --version` or `mlrl-testbed mlrl.boosting -v` command that is provided by the {ref}`command line API <testbed>`. It prints the build options used for compiling the program, together with a list of supported GPUs available on your machine.

Alternatively, this information can be retrieved programmatically via the Python or C++ API as shown below (see {ref}`python-apidoc` and {ref}`cpp-apidoc`):

````{tab} Python
   ```python
   from mlrl.common import get_gpu_devices, is_gpu_available, is_gpu_support_enabled
   from typing import List

   gpu_support_enabled: bool = is_gpu_support_enabled()
   gpu_available: bool = is_gpu_available()
   gpu_devices: List[str] = get_gpu_devices()
   ```
````

````{tab} C++
   ```cpp
   #include "mlrl/common/library_info.hpp"

   bool gpuSupportEnabled = isGpuSupportEnabled();
   bool gpuAvailable = isGpuAvailable();
   std::vector<std::string> gpuDevices = getGpuDevices();
   ```
````
