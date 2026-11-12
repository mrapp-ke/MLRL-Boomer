(build-options)=

# Build Options

Certain functionalities of the project can be enabled or disabled at compile-time via so-called build options. They can be specified in the configuration file {repo-file}`meson.options <cpp/subprojects/common/meson.options>` or set via environment variables.

(build-type)=

## Build Type

By default, the project is compiled with compile-time optimizations enabled and debug symbols disabled. This behavior can be overruled by specifying the environment variable `BUILDTYPE`. Its value is directly passed to the [Meson](https://mesonbuild.com/) build system. Therefore, we refer to the [documentation](https://mesonbuild.com/Builtin-options.html#core-options) of this build system for a list of possible values. This build option can only be specified via the aforementioned environment variable, not via the configuration file {repo-file}`meson.options <cpp/subprojects/common/meson.options>`.

(testing-support)=

## Testing Support

This project comes with unit tests for the C++ code it contains (see {ref}`testing`). They are based on the [GoogleTest](https://github.com/google/googletest) framework. When building the project on a system where this dependency is available, the testing code is compiled and linked against the shared libraries it is supposed to test. By default, the build option `test_support` is set to `auto`, i.e., the testing code is only compiled if GoogleTest is available and no error is raised otherwise. To enforce the compilation of the testing code, the build option can be set to `enabled`. Setting it to `disabled` prevents the code from being compiled even if GoogleTest is available. Alternatively, the desired value can be specified via the environment variable `TEST_SUPPORT`.

(multi-threading-support)=

## Multi-Threading Support

By default, the project is built with multi-threading support enabled. This requires [OpenMP](https://en.wikipedia.org/wiki/OpenMP) to be available on the host system. In order to compile the project without multi-threading support, e.g., because OpenMP is not available, the build option `multi_threading_support` can be set to `disabled` instead of `enabled`. Alternatively, the desired value can be specified via the environment variable `MULTI_THREADING_SUPPORT`.

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

(simd-support)=

## SIMD Support

```{note}
SIMD support is currently not available on Windows.
```

The algorithms provided by this project may use [single instruction, multiple data (SIMD)](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) instructions supported by modern CPUs. Support for SIMD instructions is enabled by default when compiled on a system where [xsimd](https://github.com/xtensor-stack/xsimd) is available. At compile time, the usage of SIMD instructions can be controlled explicitly by setting the build option `simd_support` to `enabled` or `disabled`. Alternatively, these values can be provided via the environment variable `SIMD_SUPPORT`.

When running the command `mlrl-testbed mlrl.boosting --version` or `mlrl-testbed mlrl.boosting -v`, the resulting output contains information about whether the program was built with SIMD support enabled or not. It also lists the instruction set extensions supported by the CPU for SIMD computations. At runtime, algorithms automatically pick an implementation best suited for the supported extensions.

Information on the support of SIMD instruction can also be retrieved programmatically via the Python or C++ API (see {ref}`python-apidoc` and {ref}`cpp-apidoc`). An example is given below:

````{tab} Python
   ```python
   from mlrl.common import is_simd_support_enabled, get_supported_simd_extensions
   from typing import List

   simd_support_enabled: bool = is_simd_support_enabled()
   simd_extensions: List[str] = get_supported_simd_extensions()
   ```
````

````{tab} C++
   ```cpp
   #include "mlrl/common/library_info.hpp"

   bool simdSupportEnabled = isSimdSupportEnabled();
   std::vector<std::string> simdExtensions = getSupportedSimdExtensions();
   ```
````

(gpu-support)=

## GPU Support

```{hint}
So far, GPU support is still at an early stage of development. No algorithm provided by this project makes use of it yet.
```

GPU support via [OpenCL](https://en.wikipedia.org/wiki/OpenCL) is enabled by default when building the project on a system where OpenCL is available. This feature can explicitly be enabled or disabled at compile-time by setting the build option `gpu_support` to `enabled` or `disabled`. Alternatively, the desired value can be specified via the environment variable `GPU_SUPPORT`.

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
