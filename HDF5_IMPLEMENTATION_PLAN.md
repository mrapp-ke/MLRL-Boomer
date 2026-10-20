# Implementation plan: HDF5-based model serialization

This document breaks down the implementation of the design described in
[HDF5_MODEL_SERIALIZATION.md](HDF5_MODEL_SERIALIZATION.md) into individual steps.

## General rules

* Each step is self-contained and can be implemented, reviewed and merged on its own (one pull request per step).
* After each step the code is in a working state:
  * the C++ libraries and Python packages compile on Linux, macOS and Windows,
  * all existing C++ tests (GoogleTest) and Python tests (unit and integration tests) pass,
  * the CI workflows succeed.
* Steps are ordered by their dependencies. A step may only rely on functionality added by earlier steps.
* Steps 1–8 do not change the behavior of the Python packages. User-visible changes are introduced in step 9 (the
  pickle format changes), step 10 (new API) and step 11 (the testbed saves models of rule learners as HDF5 files).
* Code follows the existing conventions (`doc/developer_guide/coding_standards.md`): Doxygen comments in C++ headers,
  docstrings in Python, `MLRLCOMMON_API` for exported symbols, factory functions `create…()` returning
  `std::unique_ptr`, interfaces prefixed with `I`.
* New C++ source and test files must be added to the `source_files` / `test_files` lists in
  `cpp/subprojects/common/meson.build`; new Cython modules to `cython_module_names` in
  `python/subprojects/common/meson.build`.

### Fixed decisions (from the design document)

* HDF5 is a mandatory dependency and linked dynamically.
* Only the HDF5 C API is used (no `H5Cpp`, no high-level library `hdf5_hl`).
* HDF5 types never appear in public headers.
* All rules of a model share the same score type.
* Each native model object is pickled separately as a single-group HDF5 image; additionally, whole learners can be
  saved to / loaded from a single HDF5 file.
* C++ programs do not need to make predictions from a saved file.
* Unknown hyperparameters cause loading to fail; pickles created by earlier versions do not need to remain loadable.
* `load_model` accepts files written by any version, as long as their `format_version` is supported.
* The testbed saves and loads models of rule learners only as HDF5 files; there is no option to choose the format.
  Pickle remains in use only for non-rule learners run via `testbed-sklearn`.

---

## Step 1: Make HDF5 available in development and CI environments

**Goal:** HDF5 (shared library and headers) can be installed in all environments where the code is built, without any
code using it yet.

**Result:** CI build workflows install HDF5; the developer documentation lists HDF5 as a build prerequisite.

### Tasks

1. **Choose and pin an HDF5 version** (e.g., the latest 1.14.x release). Record it in a single place in the build system
   (e.g., a constant or resource file under `build_system/res/`), so that all platforms use the same version and the
   dependency update workflows can bump it later.
2. **Build-system target `dependency_hdf5`** (new module `build_system/targets/dependencies/hdf5/`, modeled on
   `build_system/targets/dependencies/macos/targets.py` → `compile_libomp`):
   * download the source archive of the pinned version (reuse `CurlDownload` and `TarExtract`; move them from
     `targets/dependencies/macos/` to a shared location if they are needed on other platforms, too),
   * configure and build with CMake (reuse the `Cmake` helper):
     `-DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_EXAMPLES=OFF
     -DBUILD_TESTING=OFF -DHDF5_BUILD_HL_LIB=OFF -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF
     -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DCMAKE_BUILD_TYPE=Release`,
     and on macOS additionally `CMAKE_OSX_DEPLOYMENT_TARGET` / `CMAKE_OSX_ARCHITECTURES` (from the environment set by
     cibuildwheel, `MACOSX_DEPLOYMENT_TARGET` / `ARCHFLAGS`),
   * install into a fixed directory in the repository root (e.g., `hdf5/`, analogous to `libomp/`) and add it to
     `.gitignore` and to the files removed by `./build --clean` if appropriate.
   * The target must work on Linux (inside manylinux/musllinux containers), macOS and Windows (MSVC), because it is
     reused by step 3.
3. **CI build workflows** (`.github/workflows/`):
   * `template_build_linux.yml`: `sudo apt install -y libhdf5-dev` (Ubuntu provides the pkg-config module
     `hdf5-serial`, which Meson's HDF5 dependency finds automatically).
   * `template_build_macos.yml`: run `./build dependency_hdf5` (like `dependency_libomp`); extend `CPLUS_INCLUDE_PATH` /
     `LIBRARY_PATH` and set `PKG_CONFIG_PATH` to the install directory.
   * `template_build_windows.yml`: install HDF5 via the same `johnwason/vcpkg-action` that is used for OpenCL
     (`pkgs: hdf5`), or via `./build.bat dependency_hdf5`; extend `$env:INCLUDE`, `$env:LIB` and `$env:PATH` (for the
     DLL at test time) in the "Compile via MSVC" step, analogous to OpenCL.
4. **Documentation:** add HDF5 to the prerequisites tables for Linux, macOS and Windows in
   `doc/developer_guide/compilation.md` (Linux: `libhdf5-dev` / `hdf5-devel`; macOS: `brew install hdf5` for local
   development; Windows: vcpkg package `hdf5`).

### Verification

* All CI build workflows succeed on all platforms (HDF5 installed but unused).
* `./build dependency_hdf5` works locally on at least one platform.

---

## Step 2: Add HDF5 as a mandatory dependency of `libmlrlcommon`

**Goal:** `libmlrlcommon` is linked dynamically against HDF5, verified by a minimal use of the library.

**Result:** Building without HDF5 fails with a clear error message; a smoke test proves that HDF5 can be called from
`libmlrlcommon`, including from the Python packages.

### Tasks

1. **Meson** (`cpp/subprojects/common/meson.build`):
   ```meson
   hdf5_dep = dependency('hdf5', language : 'c', required : false)

   if not hdf5_dep.found()
       # Fallback for environments without pkg-config / h5cc (typically Windows): rely on INCLUDE/LIB
       hdf5_lib = cpp.find_library('hdf5', required : false)
       ...
   endif

   if not hdf5_dep.found()
       error('HDF5 is required, but could not be found. See doc/developer_guide/compilation.md')
   endif
   ```
   Add `hdf5_dep` to `dependencies`. On Windows add `-DH5_BUILT_AS_DYNAMIC_LIB` to `cpp_args` (if not already defined by
   the installed `H5pubconf.h`).
   `common_dep` (used by `boosting` / `seco`) does **not** need to export the HDF5 include directories, because public
   headers never include HDF5.
2. **Private utility** `cpp/subprojects/common/src/mlrl/common/util/hdf5.hpp` / `.cpp` with a single function
   `std::string util::getHdf5Version()` based on `H5get_libversion`.
3. **Make the dependency observable:** report the HDF5 version via `CommonLibraryInfo` in
   `src/mlrl/common/library_info.cpp`, e.g., as an additional `BuildOption("HDF5_VERSION", "HDF5 version", …)` in
   `visitBuildOptions`. This guarantees that the symbol is actually used (so that linkers with `--as-needed` keep the
   dependency) and that the Python packages load the HDF5 library at import time, which makes missing runtime
   libraries visible immediately. It is shown by the testbed's `--version` output.
4. **GoogleTest:** extend `cpp/subprojects/common/test/mlrl/common/library_info.cpp` to check that the HDF5 version
   is reported and non-empty.

### Verification

* `./build compile` and `./build tests_cpp` succeed on all platforms.
* Python integration tests pass (the extension modules can load `libmlrlcommon` and thereby HDF5).
* `--version` of the testbed shows the HDF5 version.
* Building on a machine without HDF5 fails with the error message from task 1.

---

## Step 3: Bundle HDF5 into the wheel packages

**Goal:** The pre-built wheels for all platforms contain the HDF5 shared library and work on machines without HDF5.

**Result:** The publishing workflows (cibuildwheel with auditwheel, delocate and delvewheel) produce working wheels.

### Tasks

1. **Linux** (`template_publish_platform_linux.yml`): extend `CIBW_BEFORE_ALL` to build HDF5 inside the
   manylinux/musllinux container via `./build dependency_hdf5` (distribution packages in these images are old or
   missing). Pass the install directory to the build via `CIBW_ENVIRONMENT` (`PKG_CONFIG_PATH`, `LD_LIBRARY_PATH`), so
   that Meson finds it and `auditwheel repair` can bundle it. Note that `CIBW_ENVIRONMENT` is already set for ccache;
   append to it.
2. **Linux CI build** (`template_build_linux.yml`): the step "Build Python packages" runs `./build build_wheels` with
   `REPAIR_WHEELS=true` (auditwheel). Verify that the system `libhdf5` is bundled correctly or exclude it consciously.
3. **macOS** (`template_publish_platform_macos.yml`): append `./build dependency_hdf5` to `CIBW_BEFORE_ALL` (the
   deployment target is taken from cibuildwheel's environment, see step 1), add the include/library paths to
   `CIBW_BEFORE_BUILD`, and add the HDF5 library directory to `DYLD_LIBRARY_PATH` in `CIBW_REPAIR_WHEEL_COMMAND`
   (next to libomp).
4. **Windows** (`template_publish_platform_windows.yml`): build or install HDF5 before cibuildwheel runs; pass the
   include/lib paths to the build; add `--add-path <hdf5>\bin` to `delvewheel repair`.
5. **License:** ship HDF5's license file (BSD-3-Clause-style) with the wheels, e.g., by adding it to the package data
   of `mlrl-common` or by relying on the repair tools if they copy it (check).
6. **Smoke test of wheels:** if the publishing workflows have a test step (`CIBW_TEST_COMMAND`), extend it with an
   import of `mlrl.common` and a call that touches HDF5 (e.g., reading the library info from step 2); otherwise add one.

### Verification

* `test_publish.yml` (or a manual run of the publishing workflows) produces wheels for all platforms.
* The wheels install and work in a clean virtual environment / container without HDF5 installed; the HDF5 version is
  reported.
* Optional: importing `h5py` and `mlrl.common` in the same process works on all platforms (no clash between the two
  bundled HDF5 libraries).

---

## Step 4: Private HDF5 helper module

**Goal:** Internal RAII wrappers and utilities for in-memory HDF5 files (create, open from a buffer, obtain the file
image), groups, datasets and attributes, with error handling that converts HDF5 errors into C++ exceptions.

**Result:** The helper module is covered by GoogleTest tests; nothing else uses it yet.

### Tasks

Create `cpp/subprojects/common/src/mlrl/common/serialization/hdf5/` (private, not installed), extending the utility
from step 2:

1. **Error handling** (`hdf5_error.hpp/.cpp`):
   * disable automatic error printing once (`H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr)`), e.g., in a function
     that is called before any HDF5 operation (thread-safe one-time initialization via `std::call_once`),
   * `checkHdf5(herr_t/hid_t result, const char* operation)` that throws `std::runtime_error` including the message of
     the innermost error on the HDF5 error stack (`H5Ewalk2`).
2. **RAII handle** (`hdf5_handle.hpp`): a move-only class wrapping `hid_t` with a close function (`H5Fclose`,
   `H5Gclose`, `H5Dclose`, `H5Sclose`, `H5Aclose`, `H5Pclose`, `H5Tclose`).
3. **In-memory file** (`hdf5_file.hpp/.cpp`):
   * `Hdf5File::createInMemory()`: file access property list with `H5Pset_fapl_core(fapl, 64 * 1024, false)`,
     `H5Fcreate`,
   * `Hdf5File::openFromImage(const uint8* data, std::size_t size)`: `H5Pset_fapl_core` + `H5Pset_file_image`
     (copies the buffer) + `H5Fopen(…, H5F_ACC_RDONLY, …)`,
   * `std::vector<uint8> Hdf5File::getImage()`: `H5Fflush`, `H5Fget_file_image` twice (size, then data),
   * `getRootGroup()`.
4. **Groups, datasets, attributes** (`hdf5_group.hpp/.cpp`):
   * `createGroup(name)`, `openGroup(name)`, `hasGroup(name)`, `hasDataset(name)`, group paths with `/` separators
     (create intermediate groups via a link creation property list with `H5Pset_create_intermediate_group`),
   * a type trait `Hdf5Type<T>` mapping `uint8`, `uint32`, `int32`, `int64`, `float32`, `float64` to
     `H5T_NATIVE_*`,
   * `writeDataset<T>(name, const T* data, std::size_t n)` (1-D, contiguous; zero-length datasets allowed) and
     `std::vector<T> readDataset<T>(name)` (verify rank 1 and that the stored type is convertible; HDF5 converts
     numeric types on read — decide whether to accept conversions or require exact types),
   * scalar attributes for integers, floats and strings (variable-length UTF-8, `H5Tset_cset(H5T_CSET_UTF8)`),
     string datasets (variable-length UTF-8),
   * attribute enumeration (`H5Aiterate2`) and type inspection (integer / float / string) for the generic metadata
     reader in step 5.
5. **GoogleTest** (`cpp/subprojects/common/test/mlrl/common/serialization/hdf5.cpp`; the test target already has
   `src` in its include directories): create file → write datasets/attributes of every supported type, including empty
   datasets and Unicode strings → get image → reopen from image → read and compare. Test that opening garbage data,
   reading missing datasets and type mismatches throw `std::runtime_error`.

### Verification

* `./build tests_cpp` passes on all platforms; no HDF5 messages are printed to `stderr` in the error tests.

---

## Step 5: Public model file writer and reader with metadata support

**Goal:** `IModelFileWriter` / `IModelFileReader` with the `format_version` handling and the generic metadata
functions (attributes, numeric and string arrays), but without support for model types yet.

**Result:** Round-trip tests for metadata pass; the public headers do not include HDF5.

### Tasks

1. **Public header** `cpp/subprojects/common/include/mlrl/common/serialization/model_file.hpp`:
   * `IModelFileWriter` with `setStringAttribute`, `setIntAttribute` (`int64`), `setFloatAttribute` (`float64`),
     `writeStringArray`, `writeInt64Array`, `writeFloat64Array` (each taking a group path and a name), and
     `std::vector<uint8> finish()`,
   * `IModelFileReader` with `hasGroup`, `hasDataset`, `getAttributeNames(group)`, `getAttributeType(group, name)`
     (enum `AttributeType { INT, FLOAT, STRING }`), typed getters for attributes and arrays, and
     `getArrayType(group, name)`,
   * factory functions `createModelFileWriter()` and `createModelFileReader(const uint8* data, std::size_t size)`,
   * the methods for model types are added in steps 6–8.
   * Caution: `mlrl/common/data/types.hpp` defines `int64` as `long int`, which is only 32 bits on Windows (MSVC). Use
     `std::int64_t` for the metadata API (or fix the typedef in a separate change), and map it to `H5T_NATIVE_INT64`.
2. **Implementation** `src/mlrl/common/serialization/model_file.cpp` using the helpers from step 4 (pimpl via the
   concrete classes being defined only in the `.cpp` file).
   * The writer sets the root attributes `format_version` (constant, starting at 1) and `library_version`
     (`MLRLCOMMON_LIBRARY_VERSION`).
   * The reader checks `format_version` on construction and throws a `std::runtime_error` with a clear message if it
     is missing (not an mlrl model file) or unsupported.
   * Calling any method on the writer after `finish()` throws.
3. **GoogleTest** (`test/mlrl/common/serialization/model_file.cpp`): metadata round trips; attribute enumeration;
   version mismatch and invalid buffers throw.

### Verification

* `./build tests_cpp` passes; `grep -r hdf5 cpp/subprojects/common/include` finds nothing.

---

## Step 6: Serialization of probability calibration models

**Goal:** The writer and reader support `IsotonicProbabilityCalibrationModel` and `NoProbabilityCalibrationModel`
(marginal and joint).

**Result:** C++ round-trip tests for calibration models pass.

### Tasks

1. Add to the public API:
   * `IModelFileWriter::writeMarginalProbabilityCalibrationModel(const IMarginalProbabilityCalibrationModel&)` and
     `writeJointProbabilityCalibrationModel(const IJointProbabilityCalibrationModel&)`,
   * `IModelFileReader::readMarginalProbabilityCalibrationModel()` / `readJointProbabilityCalibrationModel()`
     returning `std::unique_ptr` of the respective interface, and `hasMarginal…()` / `hasJoint…()`.
2. Private `serialization/probability_calibration_serialization.cpp`:
   * dispatch via `dynamic_cast` to `IIsotonicProbabilityCalibrationModel` or the "no calibration" class (see
     `prediction/probability_calibration_no.hpp`); throw for unknown types,
   * write group `marginal_probability_calibration_model` / `joint_probability_calibration_model` with attribute
     `type` (`"None"` / `"Isotonic"`) and, for isotonic models, attribute `num_bin_lists` and datasets `offsets`
     (`uint32`, `num_bin_lists + 1`), `thresholds`, `probabilities` (`float64`), collected with
     `IIsotonicProbabilityCalibrationModel::visit` (bins arrive grouped by list index — verify; otherwise sort or
     bucket them first),
   * read: `createNoProbabilityCalibrationModel()` or `createIsotonicProbabilityCalibrationModel(numBinLists)` +
     `addBin(listIndex, threshold, probability)`.
3. GoogleTest: round trips for both "no" models and isotonic models with empty lists, a single list and several lists;
   compare via `visit`; verify `calibrateMarginalProbability` / `calibrateJointProbability` give identical results.

### Verification

* `./build tests_cpp` passes.

---

## Step 7: Serialization of output space information

**Goal:** The writer and reader support `LabelVectorSet` and `NoOutputSpaceInfo`.

**Result:** C++ round-trip tests for output space information pass.

### Tasks

1. Add `IModelFileWriter::writeOutputSpaceInfo(const IOutputSpaceInfo&)`, `IModelFileReader::hasOutputSpaceInfo()`
   and `readOutputSpaceInfo()` (returning `std::unique_ptr<IOutputSpaceInfo>`).
2. Private `serialization/output_space_info_serialization.cpp`:
   * dispatch via `dynamic_cast` to `ILabelVectorSet` or the "no output space info" class
     (`prediction/output_space_info_no.hpp`),
   * group `output_space_info` with attribute `type` (`"None"` / `"LabelVectorSet"`) and datasets `offsets`
     (`uint32`, `num_label_vectors + 1`), `label_indices` (`uint32`), `frequencies` (`uint32`), collected via
     `ILabelVectorSet::visit`,
   * read: `createNoOutputSpaceInfo()` or `createLabelVectorSet()` + `addLabelVector(…)`, preserving order.
3. GoogleTest: empty set, set with empty label vector, several label vectors with frequencies; order is preserved.

### Verification

* `./build tests_cpp` passes.

---

## Step 8: Serialization of rule lists

**Goal:** The writer and reader support `RuleList`, using the columnar file layout, for all combinations of body types,
head types and score types, and both positions of the default rule.

**Result:** C++ round-trip tests for rule lists pass.

### Tasks

1. Add `IModelFileWriter::writeRuleModel(const IRuleModel&)`, `IModelFileReader::hasRuleModel()` and
   `readRuleModel()` (returning `std::unique_ptr<IRuleModel>`).
2. Private `serialization/rule_list_serialization.cpp`:
   * dispatch via `dynamic_cast` to `IRuleList`; throw for other `IRuleModel` implementations,
   * collect data in a single pass over `IRuleList::visit` into `std::vector`s (body type per rule; per condition
     category `numerical_leq`, `numerical_gr`, `ordinal_leq`, `ordinal_gr`, `nominal_eq`, `nominal_neq`: offsets,
     feature indices, thresholds; heads: head type, score offsets, scores, index offsets, output indices),
   * determine the score type from the first head; throw if a later head has a different score type,
   * write group `rule_model` with attributes `type = "RuleList"`, `default_rule_takes_precedence`,
     `num_used_rules`, `num_rules` and the datasets as specified in section 4 of the design document; scores are
     written with the native type (`uint8` / `float32` / `float64`) and the attribute `heads/score_type`,
   * read: `createRuleList(defaultRuleTakesPrecedence)`; for each rule, construct `ConjunctiveBody(…)` with the six
     condition counts (or no body for the default rule) and `CompleteHead<T>` / `PartialHead<T>` for the stored score
     type; fill them from slices of the datasets; call `addDefaultRule` / `addRule` in stored order; finally
     `setNumUsedRules`,
   * validate consistency on reading (offset arrays monotonic and matching dataset lengths, at most one default rule,
     `num_used_rules <= num_rules`) and throw on violation.
3. GoogleTest:
   * rule lists without default rule, with default rule first (`defaultRuleTakesPrecedence = true`) and last,
   * bodies with all six condition categories, with some categories empty,
   * complete and partial heads for `uint8`, `float32`, `float64`,
   * `numUsedRules` smaller than the number of rules,
   * compare original and loaded rule list via `visit` (all values, indices, order),
   * mixed score types throw on writing; corrupted files throw on reading.
   * Optionally: a test that trains a small model via the C++ API, serializes it and checks identical predictions.

### Verification

* `./build tests_cpp` passes.
* Rough performance check: serializing a model with ~10,000 rules takes well under a second.

---

## Step 9: Cython wrappers for the model file writer and reader, and HDF5-based pickling

**Goal:** Cython declarations and wrapper classes for `IModelFileWriter` / `IModelFileReader`. The four native model
objects are pickled as single-group HDF5 images. The old tuple-based serialization code is removed.

**Result:** Pickling of learners works as before, but uses HDF5 internally; old pickles are rejected with a clear error
message; reference `model.pickle` files of the integration tests are regenerated.

### Tasks

1. **Declarations** `python/subprojects/common/mlrl/common/cython/model_file.pxd`: `cdef extern` block for
   `mlrl/common/serialization/model_file.hpp`; every method declared with `except +`.
2. **Wrapper module** `model_file.pyx` (add to `cython_module_names` in `python/subprojects/common/meson.build`):
   * `cdef class ModelFileWriter` with `set_attribute(group, name, value)` (dispatch on `bool`/`int`/`float`/`str`),
     `write_array(group, name, values)` (NumPy integer → `int64`, float → `float64`, strings → string array),
     `write_rule_model(RuleModel)`, `write_output_space_info(OutputSpaceInfo)`,
     `write_marginal_probability_calibration_model(...)`, `write_joint_probability_calibration_model(...)`,
     `finish() -> bytes` (`PyBytes_FromStringAndSize`),
   * `cdef class ModelFileReader(data: bytes)` with `has_group`, `has_dataset`, `get_attributes(group) -> dict`,
     `get_attribute(group, name)`, `read_array(group, name) -> np.ndarray`, `has_rule_model()`,
     `read_rule_model() -> RuleModel`, `read_output_space_info()`, `read_marginal_…()`, `read_joint_…()`,
   * Factory functions to wrap native objects in the correct Cython class: reuse `create_rule_model` from
     `rule_model.pxd`; add `create_output_space_info(unique_ptr[IOutputSpaceInfo])` to `output_space_info.pxd` and
     `create_marginal_probability_calibration_model(…)` / `create_joint_probability_calibration_model(…)` to
     `probability_calibration.pxd` (choose the subclass via `dynamic_cast`). Check how the training result currently
     wraps these objects and reuse that code if it already exists.
3. **Pickling** — in `rule_model.pyx`, `output_space_info.pyx`, `probability_calibration.pyx`:
   * `__reduce__` returns `(cls, (), image_bytes)`, where the image is created with a writer containing only the
     object's group,
   * `__setstate__(self, state)`: if `state` is not `bytes`, raise a `ValueError` saying the object was serialized by an
     older version and the model must be re-trained; otherwise read the object's group and take ownership of the native
     pointer,
   * the "no" variants (`NoOutputSpaceInfo`, `NoMarginal…`, `NoJoint…`) can keep their trivial implementation or use
     the same mechanism for uniformity.
4. **Remove old code:** `SERIALIZATION_VERSION`, all `__serialize_*` / `__deserialize_*` methods and the `state`
   attributes in the `.pyx` and `.pxd` files; also the stale 4-argument `ConjunctiveBodyImpl` constructor declaration
   in `rule_model.pxd` if it becomes unused.
5. **Tests:**
   * regenerate all `python/tests/res/out/**/models/model.pickle` reference files (via the integration tests' mechanism
     for overwriting reference files),
   * add Python tests (e.g., in `python/tests/mlrl_tests/boosting/test_classifier.py` or a new module) that pickle and
     unpickle fitted learners in different configurations (partial/complete heads, isotonic calibration, example-wise /
     GFM predictors with label vector sets, regression) and compare predictions,
   * a test that unpickling a state in the old tuple format raises the expected error.

### Verification

* All Python unit and integration tests pass, including `model-persistence`.
* Wheels built by CI work (step 3) with the new module.

---

## Step 10: `save_model` / `load_model` for rule learners

**Goal:** Learners can be saved to and loaded from a single HDF5 file, including hyperparameters and fitted state
(`/learner` group), with subclass hooks for learner-specific state and restricted class resolution.

**Result:** Round-trip tests for all learners and relevant configurations pass; predictions of loaded learners are
identical to those of the original ones; unknown hyperparameters cause loading to fail.

### Tasks

1. **`RuleLearner.save_model(path)`** in `python/subprojects/common/mlrl/common/learners.py`:
   * `check_is_fitted(self)`,
   * `/learner` attributes: `class` (`f'{module}.{qualname}'`), `package` and `package_version` (from the package info
     of the learner's Python package),
   * `/learner/parameters`: all `get_params(deep=False)` entries whose value is not `None`; raise `TypeError` for
     values that are not `bool`/`int`/`float`/`str` (store `bool` explicitly, e.g., as `uint8` with an attribute
     marking it as boolean, or as a string, so that it round-trips as `bool` and not as `int`),
   * `self._write_fit_state(writer)` hook,
   * native objects: rule model (if present), output space info, both calibration models,
   * write `writer.finish()` to the path (`Path.write_bytes`).
2. **Fit-state hooks:**
   * `RuleLearner._write_fit_state` / `_read_fit_state`: `num_outputs_`, `sparse_predictions_`, `y_dtype_` (stored as
     `dtype.str`, optional),
   * `ClassificationRuleLearner` (calls `super()`): `classes_` (numeric or string array; raise for other dtypes),
     `label_encoder_` (store `classes_` of the encoder; on load, create `LabelEncoder()` and assign `classes_`),
     `constant_prediction_` (if present, the rule model is absent),
   * check `RegressionRuleLearner`, `ProbabilisticClassificationRuleLearner`, and the BOOMER/SeCo subclasses for
     further fitted attributes and handle them.
3. **`load_model(path) -> RuleLearner`** (module-level function in `mlrl/common/learners.py`, re-exported where users
   expect it, e.g., `mlrl.common`):
   * read the class name; only accept names starting with `mlrl.`; import the module, get the class and check
     `issubclass(cls, RuleLearner)`; otherwise raise `ValueError`,
   * compare stored parameter names with `cls().get_params()`; raise `ValueError` naming every unknown parameter,
   * instantiate with the stored parameters, call `_read_fit_state`, restore native objects (`model_` only if a rule
     model is present),
   * optionally a convenience classmethod `RuleLearner.load(path)` that additionally checks that the loaded learner is
     an instance of the class it was called on.
4. **Tests** (Python, in the test modules of `boosting` and `seco`):
   * for `BoomerClassifier`, `BoomerRegressor`, `SeCoClassifier`: fit → `save_model` → `load_model` → identical
     `predict`, `predict_proba` (where available), `predict_scores`, and incremental predictions,
   * string labels (`LabelEncoder`), constant labels (no rule model), sparse inputs/predictions, isotonic calibration,
     example-wise / GFM binary predictors, non-default hyperparameters,
   * error cases: unfitted learner, file with unknown hyperparameter, file with a class name outside `mlrl.*`, file
     that is not a model file.

### Verification

* All Python tests pass.
* A saved file can be opened with `h5dump` or `h5py` and shows the expected structure (manual check).

---

## Step 11: HDF5 model files in `mlrl-testbed`

**Goal:** When running experiments with rule learners, the testbed saves and loads models exclusively as HDF5 files
(via `save_model` / `load_model`). Pickle-based model files are no longer written or read for rule learners. There is
no option to choose the format.

**Result:** Integration tests for saving and loading models of rule learners use HDF5 files; their reference outputs
contain `.h5` instead of `.pickle` files.

### Tasks

`mlrl-testbed` is independent of the rule learners: models are written via the generic `PickleFileSink` /
`PickleFileSource` (`mlrl/testbed/experiments/{output/sinks,input/sources}/`), which are hard-coded in
`ModelOutputDirectoryExtension` (`output/model/extension.py`) and `ModelInputExtension` (`input/model/extension.py`).
These extensions are registered by `testbed-sklearn`'s `SkLearnRunnable.get_extensions()`
(`mlrl/testbed_sklearn/runnables.py`), which `mlrl-common`'s `RuleLearnerRunnable` extends. `testbed-sklearn` also
runs arbitrary scikit-learn estimators (e.g., the `sklearn` integration tests with reference files
`python/tests/res/out/sklearn/**/models/model.pickle`), which cannot be saved as HDF5. Therefore, pickle remains the
format for non-rule learners in `testbed-sklearn`, and HDF5 replaces it for rule learners:

1. **Make the model sink/source exchangeable in `mlrl-testbed`:** give `ModelOutputDirectoryExtension` and
   `ModelInputExtension` an overridable factory method (or constructor argument) that creates the sink / source for a
   directory; the default keeps creating `PickleFileSink` / `PickleFileSource`. No dependency on `mlrl-common` is
   introduced.
2. **HDF5 sink and source** in `mlrl-common`'s testbed integration (`python/subprojects/common/mlrl/common/testbed/`,
   e.g., `experiments/output/model/sink_hdf5.py` and `experiments/input/model/source_hdf5.py`), implementing the
   testbed's `FileSink` / `FileSource` interfaces with the suffix `h5`, calling `save_model` / `load_model`.
   `create_source()` of the sink must return the HDF5 source (needed by the testbed's "read" mode).
3. **Use them for rule learners:** in `RuleLearnerRunnable.get_extensions()` (`mlrl/common/testbed/runnables.py`),
   replace the inherited `ModelInputExtension` / `ModelOutputDirectoryExtension` with subclasses that create the HDF5
   sink/source. Check that `get_extensions()` of `SkLearnRunnable` can be overridden without duplicating the whole list
   (e.g., filter and replace by type, or add a hook in `SkLearnRunnable`).
4. **Test infrastructure:** `python/tests/mlrl_tests/comparison.py` compares output files by suffix and has a special
   case for `.pickle` files. Add handling for `.h5` files: HDF5 files are binary and can differ in bytes even for equal
   content, so compare them semantically, e.g., by loading both with `load_model` and comparing the rules via the
   model's visitor, or by comparing the outputs derived from them (the integration tests already re-run predictions in
   "read" mode).
5. **Reference outputs:** regenerate the reference outputs of all rule learner integration tests that save models
   (`python/tests/res/out/{boosting,seco}/**/models/`): `model.pickle` is replaced by `model.h5`. Reference outputs
   under `python/tests/res/out/sklearn/` remain unchanged.

### Verification

* All integration tests pass for boosting, seco and sklearn.
* `model-persistence` tests for boosting and seco write and read `.h5` files; no `.pickle` model files are created for
  rule learners anymore.

## Step 12: Documentation and changelog

**Goal:** Document `save_model` / `load_model`, the HDF5 file format, the new build prerequisite and the
incompatibility with pickles created by earlier versions.

**Result:** User guide, developer guide and changelog are up to date.

### Tasks

1. **User guide / quickstart:** add a section on saving and loading models (`doc/quickstart/usage.md` or a new page
   in `doc/user_guide/`), including:
   * `save_model` / `load_model` usage,
   * that the files are regular HDF5 files that can be inspected with `h5py` / HDFView,
   * that, unlike pickle, the format does not execute code when loading,
   * that a model file can be loaded by any version that supports its `format_version`.
2. **File format reference:** document the layout (section 4 of the design document) in the user or developer guide.
3. **Testbed documentation** (`doc/user_guide/testbed/`): models of rule learners are saved as `.h5` files
   (instead of `.pickle`) and can be loaded via `load_model` outside of the testbed.
4. **Developer guide:** verify that `compilation.md` (step 1) and `continuous_integration.md` mention HDF5 and the
   `dependency_hdf5` target.
5. **Changelog** (`CHANGELOG.md` / the changelog mechanism in `build_system/targets/changelog`): new feature
   (`save_model` / `load_model`; the testbed saves models of rule learners as `.h5` files), API change (HDF5 is a new build dependency), and the incompatibility of
   previously pickled models.
6. Add new words (e.g., "HDF5") to `doc/spelling_wordlist.txt` if the spell checker requires it.
7. Delete or archive `HDF5_MODEL_SERIALIZATION.md` and this plan from the repository root (or move them into
   `doc/developer_guide/`).

### Verification

* `./build doc` (or the documentation CI workflow) succeeds without warnings.
