(model-files)=

# Model Files

As described {ref}`here <usage-model-persistence>`, fitted estimators can be saved to files via their method `save_model` and loaded via the function `load_model`. These files use the [HDF5](https://www.hdfgroup.org/solutions/hdf5/) file format and can therefore be inspected with common tools, such as [h5py](https://www.h5py.org/) or [HDFView](https://www.hdfgroup.org/downloads/hdfview/). In the following, we describe the structure of these files.

```{note}
It is not necessary to be familiar with the structure of model files for saving and loading models. The following information is primarily intended for users who want to inspect the content of model files with external tools.
```

## Overview

A model file contains the following groups. Depending on the estimator and its configuration, some of them may be absent.

```text
/                                           Attributes: format_version, library_version
├── learner                                 The type, parameters, and fitted state of the estimator
│   ├── parameters                          The estimator's parameters
│   └── fit_state                           Attributes set when fitting the estimator
├── rule_model                              The rules that have been learned
│   ├── bodies                              The conditions of the rules
│   └── heads                               The predictions of the rules
├── output_space_info                       Information about the output space
├── marginal_probability_calibration_model  A model for the calibration of marginal probabilities
└── joint_probability_calibration_model     A model for the calibration of joint probabilities
```

All integer attributes are stored as signed 64-bit integers, all floating point attributes as 64-bit floating point values, and all strings as variable-length UTF-8 strings.

## Versioning

The root group of each file has the following attributes:

- `format_version`: The version of the file format. It is incremented whenever the structure of model files changes in a way that is not compatible with previous versions. A file can be loaded by any version of the software that supports the version of the file format it has been saved with.
- `library_version`: The version of the software that has been used for saving the file. It is only provided for informational purposes.

## The Group `learner`

This group stores everything that is needed for restoring an estimator, except for the native models stored in the remaining groups. It has the following attributes:

- `class`: The fully qualified name of the estimator's class, e.g., `mlrl.boosting.learners.BoomerClassifier`. When loading a file, only subclasses of {py:class}`mlrl.common.learners.RuleLearner` that are provided by this project are accepted.
- `package` and `package_version`: The name and version of the Python package that provides the estimator, e.g., `mlrl-boomer`.

Moreover, it contains the following datasets:

- `boolean_parameters`: The names of all parameters with boolean values. Their values are stored as integers in the group `learner/parameters`.
- `none_parameters`: The names of all parameters with value `None`.

### Parameters

The group `learner/parameters` stores the estimator's parameters, as returned by its method `get_params`, as attributes. Only parameters with string, integer, floating point, or boolean values are supported. When loading a file that contains parameters that are not supported by the estimator, an error is raised. Parameters that are not contained in a file use their default values.

### Fitted State

The group `learner/fit_state` stores the attributes of an estimator that are set when fitting it to training data. It has the following attributes:

- `num_outputs`: The number of outputs, i.e., labels or regression targets.
- `sparse_predictions`: Whether predictions are provided as sparse matrices (`1`) or not (`0`).
- `y_dtype` (optional): The data type of the ground truth, as a NumPy type string, e.g., `<i8`.
- `n_features_in` (optional): The number of features.

In addition, the following datasets may be present. For each of them, the attributes `<name>_dtype` and `<name>_ndim` specify the original NumPy data type and the number of dimensions (`0` for scalars, `1` for arrays).

- `classes`: The unique values in the ground truth of a classification problem.
- `label_encoder_classes`: The original labels, if the ground truth of a classification problem has been encoded.
- `constant_prediction`: The value that is predicted for all outputs if the training examples are all associated with the same label. In this case, the group `rule_model` is absent.
- `feature_names_in`: The names of the features, if available.

## The Group `rule_model`

This group stores the rules that have been learned in a columnar format. Rather than storing each rule separately, the conditions and predictions of all rules are concatenated and offsets are used to determine which of them belong to a particular rule. The group has the following attributes:

- `type`: The type of the model. Currently, only `RuleList` is supported.
- `default_rule_takes_precedence`: Whether the default rule, if available, is the first (`1`) or the last (`0`) rule in the model.
- `num_rules`: The total number of rules, including the default rule.
- `num_used_rules`: The number of rules that are used for making predictions.

The rules are stored in the order they are applied, i.e., the default rule is stored first or last, depending on the attribute `default_rule_takes_precedence`. The dataset `body_type` of type `uint8` specifies the type of each rule's body: `0` for the default rule, which does not have any conditions, and `1` for all other rules.

### Bodies

The group `rule_model/bodies` contains a subgroup for each type of condition: `numerical_leq` and `numerical_gr` for conditions on numerical features, `ordinal_leq` and `ordinal_gr` for conditions on ordinal features, and `nominal_eq` and `nominal_neq` for conditions on nominal features. Each of these subgroups contains the following datasets:

- `offsets` (`uint32`, length `num_rules + 1`): The conditions of the i-th rule are stored at the positions `offsets[i]` to `offsets[i + 1] - 1` in the following datasets.
- `feature_indices` (`uint32`): The indices of the features the conditions refer to.
- `thresholds` (`float32` for numerical conditions, `int32` otherwise): The thresholds of the conditions.

### Heads

The group `rule_model/heads` stores the predictions of the rules. Its attribute `score_type` specifies the data type of the scores that are predicted by all rules, i.e., `uint8`, `float32`, or `float64`. It contains the following datasets:

- `head_type` (`uint8`, length `num_rules`): The type of each rule's head: `0` for heads that predict for all outputs and `1` for heads that predict for a subset of the outputs.
- `score_offsets` (`uint32`, length `num_rules + 1`) and `scores`: The scores predicted by the i-th rule are stored at the positions `score_offsets[i]` to `score_offsets[i + 1] - 1` in the dataset `scores`.
- `index_offsets` (`uint32`, length `num_rules + 1`) and `output_indices` (`uint32`): For heads that predict for a subset of the outputs, the indices of these outputs are stored analogously. For all other heads, the range is empty.

## The Group `output_space_info`

This group stores information about the output space that may be used for making predictions. Its attribute `type` is either `None`, if no such information is available, or `LabelVectorSet`. In the latter case, it stores the unique label vectors in the training data, as well as their frequencies, by using the following datasets:

- `offsets` (`uint32`, length equal to the number of label vectors + 1): The indices of the relevant labels in the i-th label vector are stored at the positions `offsets[i]` to `offsets[i + 1] - 1` in the dataset `label_indices`.
- `label_indices` (`uint32`): The indices of the relevant labels in each label vector.
- `frequencies` (`uint32`): The number of training examples associated with each label vector.

## The Groups `marginal_probability_calibration_model` and `joint_probability_calibration_model`

These groups store models for the calibration of marginal and joint probabilities, respectively. Their attribute `type` is either `None`, if probabilities are not calibrated, or `Isotonic`, if isotonic regression is used for calibration. In the latter case, the attribute `num_bin_lists` specifies the number of lists of bins, i.e., one list per label for marginal probabilities, and one list per label vector for joint probabilities. The bins are stored by using the following datasets:

- `offsets` (`uint32`, length `num_bin_lists + 1`): The bins of the i-th list are stored at the positions `offsets[i]` to `offsets[i + 1] - 1` in the following datasets.
- `thresholds` (`float64`): The threshold of each bin.
- `probabilities` (`float64`): The probability of each bin.

## Pickled Estimators

When serializing an estimator via Python's [pickle](https://docs.python.org/3/library/pickle.html) mechanism, each of the native models it consists of is stored as a separate in-memory HDF5 file. These files use the format described above, but only contain the group of the respective model, e.g., `rule_model`.
