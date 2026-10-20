# Using in-memory HDF5 files for model serialization

This document summarizes how models are currently serialized, and what needs to change in the C++ and Python code (and
in the build system / CI) to store models in an in-memory HDF5 file instead.

## 1. Current state

### 1.1 What gets serialized

A fitted learner (`mlrl/common/learners.py`) holds four native objects, each wrapped by a Cython class:

| Learner attribute                           | Cython classes (`python/subprojects/common/mlrl/common/cython/`)                                                                                 | C++ interface / implementation                                                                      | `SERIALIZATION_VERSION` |
|---------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-------------------------|
| `model_`                                    | `RuleList` (`rule_model.pyx`)                                                                                                                    | `IRuleList` / `RuleList` (`model/rule_list.hpp`)                                                    | 4                       |
| `output_space_info_`                        | `NoOutputSpaceInfo`, `LabelVectorSet` (`output_space_info.pyx`)                                                                                  | `IOutputSpaceInfo`, `ILabelVectorSet` / `LabelVectorSet` (`prediction/label_vector_set.hpp`)        | 3                       |
| `marginal_probability_calibration_model_`   | `NoMarginalProbabilityCalibrationModel`, `IsotonicMarginalProbabilityCalibrationModel` (`probability_calibration.pyx`)                           | `IIsotonicProbabilityCalibrationModel` (`prediction/probability_calibration_isotonic.hpp`)          | 0                       |
| `joint_probability_calibration_model_`      | `NoJointProbabilityCalibrationModel`, `IsotonicJointProbabilityCalibrationModel` (`probability_calibration.pyx`)                                 | same as above                                                                                       | 0                       |

The boosting and seco subprojects do not define model classes of their own; they only consume the ones from `common`.

### 1.2 How it works today

* **The C++ library knows nothing about serialization.** It exposes visitor functions (`IRuleList::visit`,
  `ILabelVectorSet::visit`, `IIsotonicProbabilityCalibrationModel::visit`) and mutators (`addRule`, `addDefaultRule`,
  `setNumUsedRules`, `addLabelVector`, `addBin`).
* **All serialization logic lives in Cython**, in `__reduce__` / `__setstate__`:
  * `__reduce__` visits the C++ object and builds nested Python tuples/lists of NumPy arrays (e.g., for `RuleList`:
    `(version, ([[body_state, head_state], ...], default_rule_takes_precedence, num_used_rules))`).
  * `__setstate__` checks the version (hard failure on mismatch), then rebuilds the C++ object element by element.
  * The head type (complete/partial, `uint8`/`float32`/`float64`) is inferred from the tuple length and the NumPy dtype.
  * ~350 of the 802 lines in `rule_model.pyx` (plus the corresponding `cdef` declarations in `rule_model.pxd`) are
    serialization code; `output_space_info.pyx` and `probability_calibration.pyx` contain smaller equivalents.
* Persistence of a whole learner is done by `pickle` (scikit-learn estimator pickling), e.g., by
  `mlrl-testbed`'s `sink_pickle.py` / `source_pickle.py`.
* Integration tests cover this (`model-persistence` and other test cases); reference outputs contain
  `python/tests/res/out/**/models/model.pickle` files.

### 1.3 Weaknesses this change would address

* The serialization format is Python-specific (pickle) and cannot be read or produced by C++ users of the libraries or
  by other languages/tools.
* Element-wise copying in Cython loops, many small Python objects per rule.
* Format is implicit (tuple positions, dtype sniffing) and not self-describing.

## 2. Background: in-memory HDF5

HDF5 supports files that live entirely in memory:

* **Writing:** create a file with the *core* driver and no backing store
  (`H5Pset_fapl_core(fapl, increment, /*backing_store=*/false)`, then `H5Fcreate`). After writing and `H5Fflush`, the
  raw bytes are obtained via `H5Fget_file_image(fid, nullptr, 0)` (query size) followed by a second call with a buffer.
* **Reading:** `H5Pset_fapl_core(fapl, increment, false)` + `H5Pset_file_image(fapl, buffer, size)`, then `H5Fopen`.
  Alternatively `H5LTopen_file_image` from the high-level library (`hdf5_hl`), which is an additional library; the
  low-level route above avoids that dependency.
* The resulting byte buffer **is a valid `.h5` file**. Written to disk it can be opened with `h5dump`, HDFView, or in
  Python with `h5py.File(io.BytesIO(data), "r")`.

## 3. Recommended architecture

There are two possible places to put the HDF5 code:

| Option                                                           | Pros                                                                                                                                                                           | Cons                                                                                                                                                                |
|------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **A. C++ writes/reads HDF5; Python only passes bytes** (recommended) | Single implementation; usable by C++ consumers; removes most Cython serialization code; fast (bulk dataset writes); no Python runtime dependency on `h5py`.                     | `libmlrlcommon` gains a native dependency (HDF5) that must be available at build time and bundled into wheels on all platforms.                                      |
| B. Cython writes HDF5 via `h5py`, C++ unchanged                  | No changes to the C++ build.                                                                                                                                                   | Still per-type logic in Cython; `h5py` becomes a runtime dependency; C++ users gain nothing; per-rule Python overhead remains.                                       |

The rest of this document assumes **option A**. `h5py` can still be used optionally (not as a dependency) to inspect
the produced bytes.

### Design principles

1. **Keep HDF5 out of public headers.** The public API should only expose byte buffers, e.g.
   `std::vector<uint8>` / `(const uint8* data, std::size_t size)`. Then the Cython extension modules and the
   `boosting` / `seco` libraries do not need HDF5 include paths; only `libmlrlcommon` links against the (shared) HDF5
   library.
2. **Implement serialization on top of the existing visitor and mutator APIs**, exactly like the Cython code does today.
   No changes to `RuleList`, `ConjunctiveBody`, `CompleteHead`, `PartialHead`, `LabelVectorSet` or
   `IsotonicProbabilityCalibrationModel` are strictly required.
3. **Columnar (CSR-like) layout** instead of one HDF5 group/dataset per rule. HDF5 objects carry a per-object overhead
   in the order of hundreds of bytes to kilobytes; models with thousands of rules would explode in size and be slow to
   write if every rule were its own group.
4. **Explicit, self-describing format** with type tags and a format version stored as attributes.

## 4. Proposed file layout

```
/                                            attrs: format_version (uint32), library_version (string)
/learner                                     (only in per-learner files, see 4.1)
/rule_model                                  attrs: type = "RuleList",
                                                    default_rule_takes_precedence (uint8),
                                                    num_used_rules (uint32), num_rules (uint32)
    body_type             uint8   [num_rules]        0 = EmptyBody, 1 = ConjunctiveBody
    /bodies
        /numerical_leq    offsets uint32 [num_rules + 1], feature_indices uint32 [n], thresholds float32 [n]
        /numerical_gr     (same as numerical_leq)
        /ordinal_leq      offsets, feature_indices uint32, thresholds int32
        /ordinal_gr       (same as ordinal_leq)
        /nominal_eq       offsets, feature_indices uint32, thresholds int32
        /nominal_neq      (same as nominal_eq)
    /heads                attrs: score_type = "uint8" | "float32" | "float64"
        head_type         uint8   [num_rules]        0 = complete, 1 = partial
        score_offsets     uint32  [num_rules + 1]
        scores            <score_type> [total number of scores]
        index_offsets     uint32  [num_rules + 1]    (empty range for complete heads)
        output_indices    uint32  [total number of indices of partial heads]

/output_space_info                           attrs: type = "None" | "LabelVectorSet"
    offsets               uint32  [num_label_vectors + 1]
    label_indices         uint32  [total]
    frequencies           uint32  [num_label_vectors]

/marginal_probability_calibration_model      attrs: type = "None" | "Isotonic", num_bin_lists (uint32)
    offsets               uint32  [num_bin_lists + 1]
    thresholds            float64 [total]
    probabilities         float64 [total]

/joint_probability_calibration_model         (same as marginal)
```

Notes:

* The rules are stored in the order produced by `IRuleList::visit`, which already places the default rule first or
  last depending on `isDefaultRuleTakingPrecedence()`. On reading, a rule with `body_type == 0` is passed to
  `addDefaultRule`, all others to `addRule` (this matches today's `__setstate__`).
* All rules of a model share the same score type, so a single `score_type` attribute and a single `scores` dataset per
  model suffice. The writer should still verify this while visiting the rules and throw if a head with a different
  score type is encountered.
* Zero-sized datasets are valid in HDF5 (contiguous layout), so empty condition types need no special handling.
* Compression (deflate) is possible but probably not worth it initially; it would add a zlib requirement to the HDF5
  build.

### 4.1 Per-learner model file (required)

Besides the per-object images used for pickling (each containing only the group of the respective object), a
**single HDF5 file per fitted learner** must be supported, written by `learner.save_model(path)` and read by
`load_model(path)`. It contains all four model groups plus a `/learner` group with everything that is stored in Python
today and is needed to make predictions after loading:

```
/learner                                     attrs: class = "mlrl.boosting.BoomerClassifier" (module + class name),
                                                    package = "mlrl-boosting", package_version = "0.16.0"
    /parameters                              attrs: one attribute per hyperparameter from get_params() whose value
                                                    is not None (str -> vlen UTF-8 string, int -> int64,
                                                    float -> float64, bool -> uint8)
    /fit_state                               attrs: num_outputs (uint32), sparse_predictions (uint8),
                                                    y_dtype (string, numpy dtype.str such as "<i8"; optional)
        classes                              dataset (classification only): numeric, or vlen UTF-8 strings
        label_encoder_classes                dataset (only if a LabelEncoder was used): numeric or strings
        constant_prediction                  scalar dataset (only if all labels were identical; then
                                             /rule_model is absent)
/rule_model                                  (absent if constant_prediction is present)
/output_space_info
/marginal_probability_calibration_model
/joint_probability_calibration_model
```

Why `/learner` is needed: a fitted learner is more than the four native objects. Every prediction calls
`self._create_learner()`, which builds the C++ configuration from the **hyperparameters** (e.g., `binary_predictor`,
`probability_predictor`, the calibration settings). In addition, `learners.py` relies on the fitted attributes
`num_outputs_`, `sparse_predictions_`, `y_dtype_`, `classes_`, `label_encoder_` and `constant_prediction_` (the
latter replaces `model_` if the training labels contained a single value only). Today `pickle` stores all of these
implicitly; the HDF5 file must store them explicitly.

Resulting restrictions: hyperparameters must be `None`, `str`, `int`, `float` or `bool` (true for all current
learners), and `classes_` must be numeric or string-valued. Saving a learner that violates this must raise an error
instead of silently writing a file that cannot be loaded.

The per-object images (for pickling) and the per-learner file use the same group layout and the same C++ code; a
per-object image is simply a file with a single group.

## 5. Required changes – C++

### 5.1 New, private HDF5 helper module (`cpp/subprojects/common/src/mlrl/common/serialization/`)

Not installed as public headers. Suggested contents:

* RAII wrappers around `hid_t` handles (`H5Fclose`, `H5Gclose`, `H5Dclose`, `H5Sclose`, `H5Aclose`, `H5Pclose`).
* `InMemoryHdf5File::create()`, `InMemoryHdf5File::open(const uint8* data, std::size_t size)`,
  `std::vector<uint8> getFileImage()`.
* Templated helpers `writeDataset<T>(group, name, const T* data, uint32 n)` / `readDataset<T>(...)` and attribute
  helpers, with a type mapping `uint8 → H5T_NATIVE_UINT8`, `uint32 → H5T_NATIVE_UINT32`, `int32 → H5T_NATIVE_INT32`,
  `float32 → H5T_NATIVE_FLOAT`, `float64 → H5T_NATIVE_DOUBLE`.
* Error handling: HDF5 prints its error stack to `stderr` by default. Disable this (`H5Eset_auto2(H5E_DEFAULT,
  nullptr, nullptr)`) and convert negative return values into `std::runtime_error` with a meaningful message (possibly
  walking the error stack via `H5Ewalk2`).
* Use the HDF5 **C API** (optionally with thin RAII wrappers or the header-only *HighFive* library). The official HDF5
  C++ API (`H5Cpp.h`) is a separate library that is often not built by default and is not usable with thread-safe
  HDF5 builds.

### 5.2 Public C++ API: model file writer and reader

Because a per-learner file combines several objects and Python-level metadata, the public API should be a writer and
a reader object rather than one free function per type. Both use the pimpl idiom, so that the public header does not
include HDF5 (sketch, e.g., `mlrl/common/serialization/model_file.hpp`):

```cpp
class MLRLCOMMON_API IModelFileWriter {
    public:
        virtual ~IModelFileWriter() {}

        // Each of these dispatches via dynamic_cast on the concrete type (IRuleList, ILabelVectorSet,
        // NoOutputSpaceInfo, IIsotonicProbabilityCalibrationModel, NoProbabilityCalibrationModel) and throws for
        // unsupported types.
        virtual void writeRuleModel(const IRuleModel& ruleModel) = 0;
        virtual void writeOutputSpaceInfo(const IOutputSpaceInfo& outputSpaceInfo) = 0;
        virtual void writeMarginalProbabilityCalibrationModel(const IMarginalProbabilityCalibrationModel& model) = 0;
        virtual void writeJointProbabilityCalibrationModel(const IJointProbabilityCalibrationModel& model) = 0;

        // Generic metadata, used by Python for the /learner group. "group" is a path such as "learner/parameters".
        virtual void setStringAttribute(const std::string& group, const std::string& name,
                                        const std::string& value) = 0;
        virtual void setIntAttribute(const std::string& group, const std::string& name, int64 value) = 0;
        virtual void setFloatAttribute(const std::string& group, const std::string& name, float64 value) = 0;
        virtual void writeStringArray(const std::string& group, const std::string& name,
                                      const std::vector<std::string>& values) = 0;
        virtual void writeInt64Array(const std::string& group, const std::string& name, const int64* values,
                                     std::size_t n) = 0;
        virtual void writeFloat64Array(const std::string& group, const std::string& name, const float64* values,
                                       std::size_t n) = 0;

        // Flushes the in-memory file and returns its image. The writer cannot be used afterwards.
        virtual std::vector<uint8> finish() = 0;
};

class MLRLCOMMON_API IModelFileReader {
    public:
        virtual ~IModelFileReader() {}

        virtual bool hasRuleModel() const = 0;
        virtual std::unique_ptr<IRuleModel> readRuleModel() const = 0;
        virtual std::unique_ptr<IOutputSpaceInfo> readOutputSpaceInfo() const = 0;
        virtual std::unique_ptr<IMarginalProbabilityCalibrationModel> readMarginalProbabilityCalibrationModel() const = 0;
        virtual std::unique_ptr<IJointProbabilityCalibrationModel> readJointProbabilityCalibrationModel() const = 0;

        // Generic metadata access: list attributes/datasets of a group, query their type, read them.
        virtual std::vector<std::string> getAttributeNames(const std::string& group) const = 0;
        virtual AttributeType getAttributeType(const std::string& group, const std::string& name) const = 0;
        virtual std::string getStringAttribute(const std::string& group, const std::string& name) const = 0;
        // ... int/float attributes, string/int64/float64 arrays, hasGroup(), hasDataset()
};

MLRLCOMMON_API std::unique_ptr<IModelFileWriter> createModelFileWriter();
MLRLCOMMON_API std::unique_ptr<IModelFileReader> createModelFileReader(const uint8* data, std::size_t size);
```

Design notes:

* `H5Pset_file_image` copies the buffer by default, so the Python `bytes` object does not need to outlive the reader.
* Reading and writing files **from/to disk** happens in Python (`Path.write_bytes(image)` / `read_bytes()`); the C++
  side only deals with in-memory images.
* The output space info and calibration models have no virtual "deserialize" mechanism, so the reader decides the
  concrete type to create from the `type` attribute of each group.
* Pickling of individual objects uses the same writer/reader with a single `write…` / `read…` call.
* C++ programs can read the native models from a file via `IModelFileReader`, but making predictions from a saved file
  in C++ is **out of scope**. Therefore, the predictor configuration does not need to be stored in a C++-readable form;
  the hyperparameters are stored as plain attributes that are only interpreted by Python.

Internally, each model type gets a `write…(group)` / `read…(group)` function pair in the private helper module.

Implementation notes:

* `RuleList`: two passes over `visit` (first to count conditions/scores per type to size the buffers, second to fill
  them), or a single pass appending to `std::vector`s. Writing then happens as a few bulk `H5Dwrite` calls.
* On reading, `ConjunctiveBody(numNumericalLeq, numNumericalGr, numOrdinalLeq, numOrdinalGr, numNominalEq,
  numNominalNeq)` and `CompleteHead<T>(n)` / `PartialHead<T>(n)` are allocated and filled from slices of the datasets,
  then passed to `addRule` / `addDefaultRule`; finally `setNumUsedRules`.
* A virtual `serialize()` method on the interfaces would be an alternative to the free functions, but would force a
  decision on how to dispatch deserialization by type tag (factory registry). The free-function approach mirrors how
  the Cython code already knows the concrete type and keeps the interfaces untouched.
* Thread safety: HDF5 is not thread-safe unless built with `--enable-threadsafe`. Serialization is only triggered from
  Python while holding the GIL, so this is fine, but the Cython declarations should **not** be called in `nogil`
  blocks concurrently.

### 5.3 Tests

Add GoogleTest round-trip tests (`cpp/subprojects/common/test/mlrl/common/model/…`) for every combination of body and
head type, the default rule in both positions, `numUsedRules`, label vector sets and isotonic calibration models.
Also test that corrupted/unsupported buffers raise a `std::runtime_error` instead of crashing.

## 6. Required changes – build system & packaging

### 6.1 Meson (`cpp/subprojects/common/meson.build`)

```meson
hdf5_dep = dependency('hdf5', language : 'c', required : true)
dependencies = [opencl_dep, xsimd_dep, hdf5_dep]
```

* Meson ships a dedicated HDF5 dependency handler, but it only supports the **pkg-config** and **config-tool**
  (`h5cc`) methods. This works on Linux (distribution packages) and macOS, but HDF5 builds on Windows normally ship
  CMake config files and no `.pc` file. Options for Windows: vcpkg (provides `.pc` files), conda-forge, or a fallback
  via `cpp.find_library('hdf5', dirs : …)` plus an include directory supplied through a new meson option.
* HDF5 is a **mandatory** dependency (`required : true`, no feature option).
* HDF5 is linked **dynamically**. Consequences:
  * On Windows, the define `H5_BUILT_AS_DYNAMIC_LIB` must be set when compiling `libmlrlcommon` (unless the HDF5
    version used sets it in its installed config header).
  * The HDF5 shared library must be found at runtime. For development builds this is the system/vcpkg/conda
    installation (on Windows, the DLL directory must be on `PATH` or registered via `os.add_dll_directory`). For wheels,
    the repair tools copy it into the wheel (see 6.3).
  * Another HDF5 library may be loaded into the same process by `h5py` or `pytables`. The wheel repair tools mitigate
    clashes: `auditwheel` and `delvewheel` rename the bundled libraries (hash suffix / name mangling), and macOS uses
    two-level namespaces. Still, never pass HDF5 handles between the two copies; only bytes are exchanged.
* No HDF5 wrap exists in the Meson WrapDB (HDF5 is CMake-based). Building it as a subproject is possible via Meson's
  `cmake` module, but is heavy; using a pre-installed/pre-built HDF5 is more practical.
* Since public headers do not include HDF5, `cpp/subprojects/boosting`, `cpp/subprojects/seco` and the Python meson
  files (`python/subprojects/*/meson.build`) need **no** HDF5 include paths. If HDF5 is linked as a shared library,
  the boosting/seco libraries still only link against `libmlrlcommon`.

### 6.2 Build-system targets (`build_system/`)

* There is precedent for compiling native dependencies from source: `build_system/targets/dependencies/macos`
  (`compile_libomp`), used because Homebrew libraries target the macOS version of the build machine rather than the
  wheel's deployment target. HDF5 has the same problem on macOS → add an analogous `compile_hdf5` target (CMake build
  with `-DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_EXAMPLES=OFF
  -DBUILD_TESTING=OFF -DHDF5_BUILD_HL_LIB=OFF -DHDF5_BUILD_CPP_LIB=OFF`, and the right
  `CMAKE_OSX_DEPLOYMENT_TARGET` / `CMAKE_OSX_ARCHITECTURES`). The same target can be reused on Windows and inside the
  manylinux/musllinux containers to get identical HDF5 versions everywhere.
* Pin the HDF5 version (e.g., in a wrap-like config) so that the dependency update workflows can manage it.

### 6.3 CI (`.github/workflows/`)

| Workflow                                | Required change                                                                                                                                   |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `template_build_linux.yml`              | Install `libhdf5-dev` (or build via the new target).                                                                                              |
| `template_build_macos.yml`              | Run the new `compile_hdf5` target (like `dependency_libomp`), add paths to `CPLUS_INCLUDE_PATH` / `LIBRARY_PATH` / `PKG_CONFIG_PATH`.               |
| `template_build_windows.yml`            | Install HDF5 (vcpkg/conda or the new target).                                                                                                     |
| `template_publish_platform_linux.yml`   | Extend `CIBW_BEFORE_ALL` (currently only installs `ccache`) to provide HDF5 inside manylinux (`hdf5-devel` from EPEL is too old/inconsistent → prefer building from source) and musllinux (`apk add hdf5-dev`). `auditwheel repair` bundles the shared HDF5 library automatically (the library directory must be on `LD_LIBRARY_PATH` if HDF5 is installed to a non-standard prefix). |
| `template_publish_platform_macos.yml`   | Add HDF5 to `CIBW_BEFORE_ALL`; `delocate-wheel` needs the HDF5 library directory in `DYLD_LIBRARY_PATH` (as already done for libomp).                                             |
| `template_publish_platform_windows.yml` | Provide HDF5; `delvewheel repair` needs `--add-path` pointing to the HDF5 DLL directory.                                              |

Additional considerations: wheel size increases by a few MB per platform-specific wheel, because the shared HDF5 library is bundled into each of
`mlrl-boosting`, `mlrl-seco` and `mlrl-common` (depending on which wheels contain `libmlrlcommon`); HDF5's BSD-style
license is compatible with MIT, but its license text must be shipped with the wheels.

### 6.4 Documentation

Update the developer documentation in `doc/` that lists the build prerequisites (compilers, OpenCL, libomp, …) to
include HDF5, and the changelog (format change of pickled models).

## 7. Required changes – Python / Cython

### 7.1 `.pxd` declarations

Declare the writer/reader from section 5.2 with `except +`, so that C++ exceptions become Python exceptions, e.g., in
a new `model_file.pxd`:

```cython
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "mlrl/common/serialization/model_file.hpp" nogil:

    cdef cppclass IModelFileWriter:

        void writeRuleModel(const IRuleModel& ruleModel) except +

        void writeOutputSpaceInfo(const IOutputSpaceInfo& outputSpaceInfo) except +

        void setStringAttribute(const string& group, const string& name, const string& value) except +

        # ...

        vector[uint8] finish() except +

    cdef cppclass IModelFileReader:

        bool hasRuleModel() except +

        unique_ptr[IRuleModel] readRuleModel() except +

        # ...

    unique_ptr[IModelFileWriter] createModelFileWriter() except +

    unique_ptr[IModelFileReader] createModelFileReader(const uint8* data, size_t size) except +
```

### 7.2 `__reduce__` / `__setstate__`

Replace the hand-written tuple serialization with a bytes round trip through a single-group file, e.g., for
`RuleList`:

```cython
from cpython.bytes cimport PyBytes_FromStringAndSize

def __reduce__(self):
    cdef unique_ptr[IModelFileWriter] writer_ptr = createModelFileWriter()
    writer_ptr.get().writeRuleModel(dereference(self.rule_list_ptr.get()))
    cdef vector[uint8] image = writer_ptr.get().finish()
    return (RuleList, (), PyBytes_FromStringAndSize(<char*>image.data(), image.size()))

def __setstate__(self, bytes state):
    cdef const uint8[::1] data = state
    cdef unique_ptr[IModelFileReader] reader_ptr = createModelFileReader(&data[0], data.shape[0])
    # readRuleModel() returns an IRuleModel; cast to IRuleList as done by create_rule_model() today
    ...
```

Removable code:

* `rule_model.pyx`: `SERIALIZATION_VERSION`, all `__serialize_*` and `__deserialize_*` methods, the `state`
  attribute.
* `rule_model.pxd`: the corresponding `cdef` method declarations and `cdef object state`. The visitor wrappers must stay
  (they are still used by `visit` / `visit_used`, e.g., for the testbed's model characteristics and rule output).
* `output_space_info.pyx` / `.pxd`, `probability_calibration.pyx` / `.pxd`: same pattern.
* Unrelated stale declaration noticed during the investigation: `rule_model.pxd` declares the `ConjunctiveBodyImpl`
  constructor with 4 arguments, while the C++ constructor takes 6 (`numNumericalLeq, numNumericalGr, numOrdinalLeq,
  numOrdinalGr, numNominalEq, numNominalNeq`). It currently works only because `make_unique` is declared variadic. It
  becomes irrelevant if deserialization moves to C++, but should be fixed if the declaration is kept.

Versioning moves from the Python constants into the HDF5 `format_version` attribute, checked in C++.

### 7.3 Backward compatibility

Pickles created by earlier versions do **not** need to remain loadable. The old tuple-based deserialization code is
removed completely. `__setstate__` should only check that it received `bytes` and otherwise raise an error with a
clear message (e.g., "The model was saved by an older version of the library and cannot be loaded; please re-train
it"). The `format_version` attribute takes over the role of the old `SERIALIZATION_VERSION` constants for future format
changes.

### 7.4 Public Python API: `save_model` / `load_model` (required)

A new Cython module (e.g., `mlrl/common/cython/model_file.pyx`) wraps `IModelFileWriter` / `IModelFileReader` (with
`except +` on all declarations). On top of it, pure-Python code in `mlrl/common/learners.py`:

```python
class RuleLearner(...):

    def save_model(self, path: Path | str):
        check_is_fitted(self)
        writer = ModelFileWriter()
        writer.set_string_attribute('learner', 'class', f'{type(self).__module__}.{type(self).__qualname__}')
        writer.set_string_attribute('learner', 'package_version', ...)

        for name, value in self.get_params(deep=False).items():
            if value is not None:
                writer.set_attribute('learner/parameters', name, value)  # dispatch on str/int/float/bool

        self._write_fit_state(writer)  # overridden by subclasses, e.g., classes_ for classifiers

        if getattr(self, 'model_', None) is not None:
            writer.write_rule_model(self.model_)

        writer.write_output_space_info(self.output_space_info_)
        ...
        Path(path).write_bytes(writer.finish())


def load_model(path: Path | str) -> RuleLearner:
    reader = ModelFileReader(Path(path).read_bytes())
    learner_class = _resolve_learner_class(reader.get_string_attribute('learner', 'class'))
    learner = learner_class(**reader.get_attributes('learner/parameters'))
    learner._read_fit_state(reader)
    # restore the four native objects as Cython wrapper objects
    ...
    return learner
```

Points to take care of:

* **Class resolution / security:** `load_model` imports the class named in the file. Restrict this to subclasses of
  `RuleLearner` from the `mlrl.*` packages (allow-list, or check after import), so that a file cannot trigger imports
  of arbitrary modules. Unlike `pickle`, the HDF5 format can then be loaded safely from untrusted sources; this is a
  real advantage worth documenting.
* **Subclass hooks:** the fitted attributes differ per learner type (`ClassificationRuleLearner` has `classes_`,
  `label_encoder_`, `constant_prediction_`; `RegressionRuleLearner` does not). Use overridable
  `_write_fit_state(writer)` / `_read_fit_state(reader)` methods instead of handling everything in the base class.
* **`LabelEncoder`:** not serialized itself; it is reconstructed by creating a `LabelEncoder()` and assigning
  `classes_` from the stored array.
* **Hyperparameter compatibility:** if a file contains a hyperparameter that the installed version of the learner does
  not know (e.g., because it was removed), `load_model` **fails** with an error naming the parameter. Check this
  explicitly against `get_params()` of the class before calling the constructor, so that the error message is clearer
  than a `TypeError` from `__init__`. Hyperparameters missing from the file fall back to their defaults.
* **Version checks:** `format_version` is checked in C++; `package_version` is informative and can be used for
  warnings.
* **Getting native objects back into Cython wrappers:** `create_rule_model(unique_ptr[IRuleModel])` already exists in
  `rule_model.pxd`; analogous factory functions are needed for `OutputSpaceInfo` and the calibration models (choose the
  wrapper subclass based on the concrete C++ type).
* The bytes can be inspected with `h5py.File(io.BytesIO(data), "r")`, and a saved file is a regular `.h5` file that
  `h5dump`/HDFView can open. `h5py` stays optional. Never pass HDF5 handles between `h5py` and `libmlrlcommon`.

### 7.5 `mlrl-testbed`

* Models of rule learners are saved and loaded exclusively as HDF5 files (`.h5`, via `save_model` / `load_model`).
  There is no option to choose the format; pickle-based model files are no longer used for rule learners.
* Because `mlrl-testbed` / `testbed-sklearn` also run arbitrary scikit-learn estimators, the generic pickle sink/source
  stays for those. The model input/output extensions get a hook to exchange the sink/source, which `mlrl-common`'s
  `RuleLearnerRunnable` uses to plug in HDF5.

### 7.6 Tests

* Re-generate the reference `python/tests/res/out/**/models/model.pickle` files, since the pickled byte content changes.
* The existing integration tests (`model-persistence`, predictors that load stored models) should otherwise pass
  unchanged; they are the main safety net for the Python side.
* Add a unit test that pickles/unpickles each Cython model class and compares predictions and visited contents.
* Add `save_model` / `load_model` round-trip tests for every learner (BOOMER classifier/regressor, SeCo classifier),
  covering string labels (`LabelEncoder`), constant predictions (no rule model), sparse predictions, isotonic
  calibration and label vector sets (example-wise / GFM predictors), and non-default hyperparameters. Predictions of
  the loaded learner must be identical to those of the original one.
* Regenerate the reference outputs of the rule learner integration tests that save models (`model.h5` instead of
  `model.pickle`), and compare `.h5` files semantically in `python/tests/mlrl_tests/comparison.py`.

## 8. Suggested implementation order

1. Add HDF5 to the C++ build (Meson dependency, local dev setup on all three OSs) and the private helper module.
2. Implement `IModelFileWriter` / `IModelFileReader` for the four model types and unit-test (GoogleTest) round-trips,
   starting with `IsotonicProbabilityCalibrationModel` and `LabelVectorSet` (simple, good for validating the helper
   module), then `RuleList`.
3. Switch the Cython `__reduce__` / `__setstate__` implementations, remove the old code, re-generate test fixtures.
4. Extend CI build workflows, then the wheel publishing workflows (cibuildwheel + auditwheel/delocate/delvewheel), and
   verify the produced wheels on clean machines.
5. Generic metadata support in the writer/reader, the Cython wrapper module, and `save_model` / `load_model` with
   the subclass hooks for fitted state; round-trip tests for all learners.
6. Testbed: HDF5 sink/source replace pickle for rule learners, integration tests; documentation of the file format in
   `doc/`.

## 9. Decisions

* HDF5 is a **mandatory** dependency.
* HDF5 is linked **dynamically**.
* All rules of a model share the **same score type**.
* A **per-learner HDF5 file** with `save_model` / `load_model` is required.
* C++ programs do **not** need to make predictions from a saved file; hyperparameters are only interpreted by Python.
* Pickling keeps working **per object**: each of the four native objects pickles itself as a single-group HDF5 image;
  `RuleLearner` gets no custom `__getstate__` / `__setstate__`.
* Unknown hyperparameters on load cause `load_model` to **fail**.
* Pickles created by earlier versions do **not** need to remain loadable.
* `load_model` accepts files written by any version, as long as their `format_version` is supported.
* The testbed saves models of rule learners **only as HDF5 files**; no format option. Pickle is kept only for
  non-rule learners in `testbed-sklearn`.
