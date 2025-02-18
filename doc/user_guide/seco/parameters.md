(seco-parameters)=

# Overview of Parameters

The behavior of the SeCo algorithm can be controlled in a fine-grained manner via a large number of parameters. Values for these parameters may be provided as constructor arguments to the class `mlrl.seco.SeCoClassifier` as shown in the section {ref}`usage`. They can also be used to configure the algorithm when using the {ref}`command line API<testbed>`.

All parameters mentioned below are optional. If not specified manually, default settings that work well in most of the cases are used. In the following, an overview of all available parameters, as well as their default values, is provided.

## Data Format

The following parameters allow to specify the preferred format for representing the training data. Depending on the characteristics of a dataset, these parameters may help to reduce the memory footprint of the algorithm or the time it needs for training.

### `feature_format`

> *Default value = `'auto'`.*

`'auto'`
: The most suitable format for representation of the feature matrix is chosen automatically by estimating which representation requires less memory.

`'dense'`
: Enforces that the feature matrix is stored using a dense format.

`'sparse'`
: Enforces that the feature matrix is stored using a sparse format, if possible. Using a sparse format may reduce the memory footprint and/or speed up the training process on some data sets.

### `output_format`

> *Default value = `'auto'`.*

`'auto'`
: The most suitable format for representation of the ground truth matrix is chosen automatically by estimating which representation requires less memory.

`'dense'`
: Enforces that the ground truth matrix is stored using a dense format.

`'sparse'`
: Enforces that the ground truth matrix is stored using a sparse format, if possible. Using a sparse format may reduce the memory footprint on some data sets.

### `prediction_format`

> *Default value = `'auto'`.*

`'auto'`
: The most suitable format for the representation of predictions is chosen automatically based on the sparsity of the ground truth matrix supplied for training.

`'dense'`
: Enforces that predictions are stored using a dense format.

`'sparse'`
: Enforces that predictions are stored using a sparse format, if supported. Using a sparse format may reduce the memory footprint on some data sets.

## Heuristics

The following parameters may be used to control the heuristic used for guiding the training process by assessing the quality of potential rules. They should be carefully adjusted to the machine learning task one aims to solve in order to achieve optimal results in terms of predictive performance.

### `heuristic`

> *Default value = `'f-measure'`.*

`'accuracy'`

: Uses the heuristic "Accuracy" for evaluating the quality of rules. It measures the fraction of correctly predicted labels among all labels, i.e., in contrast to the heuristic "Precision", examples that are not covered by a rule are taken into account as well.

`'precision'`

: Uses the metric "Precision" for evaluating the quality of rules. It measures the fraction of correctly predicted labels among all labels that are covered by a rule.

`'recall'`

: Uses the heuristic "Recall" for evaluating the quality of rules. It measures the fraction of uncovered labels among all labels for which a rule's prediction is (or would be) correct, i.e., for which the ground truth is equal to the rule's prediction.

`'laplace'`

: Uses the heuristic "Laplace" for evaluating the quality of rules. It implements a Laplace-corrected variant of the heuristic "Precision".

`'weighted-relative-accuracy'`

: Uses the heuristic "Weighted Relative Accuracy" (WRA) for evaluating the quality of rules.

`'f-measure'`

: Uses the heuristic "F-Measure" for evaluating the quality of rules. It calculates as the (weighted) harmonic mean between the heuristics "Precision" and "Recall", where the parameter "beta" allows to trade off between both heuristics. If `beta = 1`, both heuristics are weighed equally. If `beta = 0`, this heuristic is equivalent to "Precision". As beta approaches infinity, this heuristic becomes equivalent to "Recall". The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `beta` *(Default value = `1.0`)*
  : The value of the parameter "beta".

`'m-estimate'`

: Uses the heuristic "M-Estimate" for evaluating the quality of rules. It trades off between the heuristics "Precision" and "WRA", where the "m" parameter controls the trade-off between both heuristics. If `m = 0`, this heuristic is equivalent to "Precision". As `m` approaches infinity, the isometrics of this heuristic become equivalent to those of "WRA". The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `m` *(Default value = `22.466`)*
  : The value of the parameter "m".

### `pruning_heuristic`

> *Default value `'accuracy'`.*

`'accuracy'`

: Uses the heuristic "Accuracy" for evaluating the quality of rules that should be pruned.

`'precision'`

: Uses the heuristic "Precision" for evaluating the quality of rules that should be pruned.

`'recall'`

: Uses the heuristic "Recall" for evaluating the quality of rules that should be pruned.

`'laplace'`

: Uses the heuristic "Laplace" for evaluating the quality of rules that should be pruned.

`'weighted-relative-accuracy'`

: Uses the heuristic "Weighted Relative Accuracy" (WRA) for evaluating the quality of rules that should be pruned.

`'f-measure'`

: Uses the heuristic "F-Measure" for evaluating the quality of rules that should be pruned. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `beta` *(Default value = `1.0`)*
  : The value of the parameter "beta".

`'m-estimate'`

: Uses the heuristic "M-Estimate" for evaluating the quality of rules that should be pruned. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `m` *(Default value = `22.466`)*
  : The value of the parameter "m".

### `lift_function`

> *Default value `'peak'`.*

`'none'`
: No lift function is used for evaluating the quality of multi-label rules.

`'kln'`
: The "KLN" lift function is used for evaluating the quality of multi-label rules. This lift function monotonously increases according to the natural logarithm of the number of labels for which a rule predicts.

`'peak'`
: The "Peak" lift function is used for evaluating the quality of multi-label rules. This lift function increases monotonously until a certain number of labels, where the maximum lift is reached, and monotonously decreases afterwards.

## Rule Induction

The parameters listed below allow to configure the algorithm used for the induction of individual rules and help to control the characteristics of these rules.

### `default_rule`

> *Default value = `'true'`.*

`'true'`
: A default rule that provides a default prediction for all examples is included as the first rule of a model.

`'false'`
: No default rule is used.

### `rule_induction`

> *Default value = `'top-down-greedy'`.*

`'top-down-greedy'`

: A greedy top-down search, where rules are successively refined by adding new conditions, is used for the induction of individual rules. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `max_conditions` *(Default value = `0`)*
  : The maximum number of conditions to be included in a rule's body. The given value must be at least 1 or 0, if the number of conditions should not be restricted.

  `min_coverage` *(Default value = `1`)*
  : The minimum number of training examples that must be covered by a rule. The given value must be at least 1.

  `min_support` *(Default value = `0.0`)*
  : The minimum support, i.e., the fraction of training examples that must be covered by a rule. The given value must be in the range [0, 1] or 0, if the support of rules should not be restricted.

  `max_head_refinements` *(Default value = `1`)*
  : The maximum number of times the head of a rule may be refined. The given value must be at least 1 or 0, if the number of refinements should not be restricted.

  `recalculate_predictions` *(Default value = `'true'`)*
  : `'true'`, if the predictions of rules should be recalculated on the entire training data if the parameter {ref}`instance_sampling<seco_parameters_instance_sampling>` is not set to the value `'none'`, `'false'`, if the predictions of rules should not be recalculated.

`'top-down-beam-search'`

: A top-down beam search, where rules are successively refined by adding new conditions, is used for the induction of individual rules. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `beam_width` *(Default value = `4`)*
  : The width to be used by the beam search. A larger value tends to result in more accurate rules being found, but negatively affects the training time. The given value must be at least 2.

  `resample_features` *(Default value = `'false'`)*
  : `'true'`, if a new sample of the available features should be created for each rule that is refined during a beam search, `'false'` otherwise. Does only have an effect if the parameter {ref}`feature_sampling<seco_parameters_feature_sampling>` is not set to the value `'none'`.

  `max_conditions` *(Default value = `0`)*
  : The maximum number of conditions to be included in a rule's body. The given value must be at least 2 or 0, if the number of conditions should not be restricted.

  `min_coverage` *(Default value = `1`)*
  : The minimum number of training examples that must be covered by a rule. The given value must be at least 1.

  `min_support` *(Default value = `0.0`)*
  : The minimum support, i.e., the fraction of training examples that must be covered by a rule. The given value must be in the range [0, 1] or 0, if the support of rules should not be restricted.

  `max_head_refinements` *(Default value = `1`)*
  : The maximum number of times the head of a rule may be refined. The given value must be at least 1 or 0, if the number of refinements should not be restricted.

  `recalculate_predictions` *(Default value = `'true'`)*
  : `'true'`, if the predictions of rules should be recalculated on the entire training data if the parameter {ref}`instance_sampling<seco_parameters_instance_sampling>` is not set to the value `'none'`, `'false'`, if the predictions of rules should not be recalculated.

### `head_type`

> *Default value = `'single'`.*

`'single'`
: All rules predict for a single output.

`'partial'`
: All rules predict for a subset of the available labels.

## Stopping Criteria

The following parameters can be used to configure different stopping mechanisms that may be used to terminate the induction of new rules as soon as certain criteria are met.

### `max_rules`

> *Default value = `1000`.*

The maximum number of rules to be learned (including the default rule). The given value must be at least 1 or 0, if the number of rules should not be restricted.

### `time_limit`

> *Default value = `0`.*

The duration in seconds after which the induction of rules should be canceled. The given value must be at least 1 or 0, if no time limit should be set.

## Pruning and Post-Optimization

### `holdout`

> *Default value = `'auto'`.*

`'none'`

: No holdout set is created.

`'auto'`

: The most suitable strategy for creating a holdout set is chosen automatically, depending on whether a holdout set is needed according to the parameter {ref}`rule_pruning<seco_parameters_rule_pruning>`.

`'random'`

: The available examples are randomly split into a training set and a holdout set. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `holdout_set_size` *(Default value = `0.33`)*
  : The percentage of examples to be included in the holdout set. For example, a value of 0.3 corresponds to 30% of the available examples. The given value must be in the range (0, 1).

`'stratified-output-wise'`

: The available examples are split into a training set and a holdout set according to an iterative stratified sampling method that ensures that for each label the proportion of relevant and irrelevant examples is maintained. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `holdout_set_size` *(Default value = `0.33`)*
  : The percentage of examples to be included in the holdout set. For example, a value of 0.3 corresponds to 30% of the available examples. The given value must be in the range (0, 1).

`'stratified-example-wise'`

: The available examples are split into a training set and a holdout set according to a stratified sampling method, where distinct label vectors are treated as individual classes. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `holdout_set_size` *(Default value = `0.33`)*
  : The percentage of examples to be included in the holdout set. For example, a value of 0.3 corresponds to 30% of the available examples. The given value must be in the range (0, 1).

(seco_parameters_rule_pruning)=

### `rule_pruning`

> *Default value = `'none'`.*

`'none'`
: No method for pruning individual rules is used.

`'irep'`
: Trailing conditions of rules may be pruned on a holdout set, similar to the IREP algorithm.

### `sequential_post_optimization`

> *Default value = `'false'`.*

`'false'`

: Sequential post-optimization is not used.

`'true'`

: Each rule in a previously learned model is optimized by being relearned in the context of the other rules. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `num_iterations` *(Default value = `2`)*
  : The number of times each rule should be relearned. The given value must be at least 1.

  `refine_heads` *(Default value = `'false'`)*
  : `'true'`, if the heads of rules may be refined when being relearned, `'false'`, if the relearned rules should predict for the same outputs as the original rules.

  `resample_features` *(Default value = `'true'`)*
  : `'true'`, if a new sample of the available features should be created whenever a new rule is refined, `'false'`, if the conditions of the new rule should use the same features as the original rule

## Sampling Techniques

### `random_state`

> *Default value = `1`.*

The seed to be used by random number generators. The given value must be at least 1.

### `output_sampling`

> *Default value = `'none'`.*

`'none'`

: All outputs are considered for learning a new rule.

`'round-robin'`

: A single output to be considered when learning a new rule is chosen in a round-robin fashion, i.e., the first rule is concerned with the first output, the second one with the second output, and so on. When the last output is reached, the procedure restarts at the first output.

`'without-replacement'`

: The outputs to be considered when learning a new rule are chosen randomly. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `num_samples` *(Default value = `1`)*
  : The number of outputs to be included in a sample. The given value must be at least 1.

(seco_parameters_feature_sampling)=

### `feature_sampling`

> *Default value = `'without-replacement'`.*

`'none'`

: All features are considered for learning a new rule.

`'without-replacement'`

: A random subset of the features is used to search for the refinements of rules. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `sample_size` *(Default value = `0`)*
  : The percentage of features to be included in a sample. For example, a value of 0.6 corresponds to 60% of the features. The given value must be in (0, 1\] or 0, if the sample size should be calculated as log2(A - 1) + 1), where A denotes the number of available features.

  `num_retained` *(Default value = `0`)*
  : The number of trailing features to be always included in a sample. For example, a value of 2 means that the last two features are always retained.

(seco_parameters_instance_sampling)=

### `instance_sampling`

> *Default value = `'none'`.*

`'none'`

: All training examples are considered for learning a new rule.

`'with-replacement'`

: The training examples to be considered for learning a new rule are selected randomly with replacement. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `sample_size` *(Default value = `1.0`)*
  : The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. The given value must be in the range (0, 1).

`'without-replacement'`

: The training examples to be considered for learning a new rule are selected randomly without replacement. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `sample_size` *(Default value = `0.66`)*
  : The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. The given value must be in the range (0, 1).

`'stratified-output-wise'`

: The training examples to be considered for learning a new rule are selected according to an iterative stratified sampling method that ensures that for each label the proportion of relevant and irrelevant examples is maintained. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `sample_size` *(Default value = `0.66`)*
  : The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. The given value must be in the range (0, 1).

`'stratified-example-wise'`

: The training examples to be considered for learning a new rule are selected according to stratified sampling method, where distinct label vectors are treated as individual classes. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `sample_size` *(Default value = `0.66`)*
  : The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. The given value must be in the range (0, 1).

## Approximations and Optimizations

### `feature_binning`

> *Default value = `'none'`.*

`'none'`

: No feature binning is used.

`'equal-width'`

: Examples are assigned to bins, based on their feature values, according to the equal-width binning method. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `bin_ratio` *(Default value = `0.33`)*
  : A percentage that specifies how many bins should be used. For example, a value of 0.3 means that the number of bins should be set to 30% of the total number of available training examples.

  `min_bins` *(Default value = `2`)*
  : The minimum number of bins. The given value must be at least 2.

  `max_bins` *(Default value = `0`)*
  : The maximum number of bins. The given value must be at least the value of `min_bins` or 0, if the number of bins should not be restricted.

`'equal-frequency'`

: Examples are assigned to bins, based on their feature values, according to the equal-frequency binning method. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `bin_ratio` *(Default value = `0.33`)*
  : A percentage that specifies how many bins should be used. For example, a value of 0.3 means that the number of bins should be set to 30% of the total number of available training examples.

  `min_bins` *(Default value = `2`)*
  : The minimum number of bins. The given value must be at least 2.

  `max_bins` *(Default value = `0`)*
  : The maximum number of bins. The given value must be at least the value of `min_bins` or 0, if the number of bins should not be restricted.

## Multi-Threading

The following parameters allow to specify whether multi-threading should be used for different aspects of the algorithm. Depending on your hardware, they may help to reduce the time needed for training or prediction.

```{warning}
To be able to use the algorithm's multi-threading capabilities, it must have been compiled with multi-threading support enabled, which should be the case with pre-built packages available on [PyPI](https://pypi.org/). Please refer to the section {ref}`build-options` if you intend to compile the program yourself, or if you want to check if multi-threading support is enabled for your installation.
```

### `parallel_rule_refinement`

> *Default value = `'true'`.*

`'false'`

: No multi-threading is used to search for potential refinements of rules.

`'true'`

: Multi-threading is used to search for potential refinements of rules in parallel. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `num_preferred_threads` *(Default value = `0`)*
  : The number of preferred threads. The given value must be at least 1 or 0, if the number of cores available on the machine should be used. If not enough CPU cores are available or if multi-threading support is disabled, as many threads as possible are used.

### `parallel_statistic_update`

> *Default value = `'false'`.*

`'false'`

: No multi-threading is used to assess the correctness of predictions for different examples.

`'true'`

: Multi-threading is used to assess the correctness of predictions for different examples in parallel. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `num_preferred_threads` *(Default value = `0`)*
  : The number of preferred threads. The given value must be at least 1 or 0, if the number of cores available on the machine should be used. If not enough CPU cores are available or if multi-threading support is disabled, as many threads as possible are used.

### `parallel_prediction`

> *Default value = `'true'`.*

`'false'`

: No multi-threading is used to obtain predictions for different examples.

`'true'`

: Multi-threading is used to obtain predictions for different examples in parallel. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

  `num_preferred_threads` *(Default value = `0`)*
  : The number of preferred threads. The given value must be at least 1 or 0, if the number of cores available on the machine should be used. If not enough CPU cores are available or if multi-threading support is disabled, as many threads as possible are used.
