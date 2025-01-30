(parameters)=

# Overview of Parameters

The behavior of the BOOMER algorithm can be controlled in a fine-grained manner via a large number of parameters. Values for these parameters may be provided as constructor arguments to the class {py:class}`mlrl.boosting.BoomerClassifier <mlrl.boosting.boosting_learners.BoomerClassifier>` or {py:class}`mlrl.boosting.BoomerRegressor <mlrl.boosting.boosting_learners.BoomerRegressor>` as shown in the section {ref}`usage`. They can also be used to configure the algorithm when using the {ref}`command line API<testbed>`.

All parameters mentioned below are optional. If not specified manually, default settings that work well in most of the cases are used. In the following, an overview of all available parameters, as well as their default values, is provided. Unless stated otherwise, all parameters can be used for both, classification and regression problems.

## Data Format

The following parameters allow to specify the preferred format for representing the training data. Depending on the characteristics of a dataset, these parameters may help to reduce the memory footprint of the algorithm or the time it needs for training.

### `feature_format`

> *Default value = `'auto'`.*

```{glossary}
`'auto'`
  The most suitable format for representation of the feature matrix is chosen automatically by estimating which representation requires less memory.

`'dense'`
  Enforces that the feature matrix is stored using a dense format.

`'sparse'`
  Enforces that the feature matrix is stored using a sparse format, if possible. Using a sparse format may reduce the memory footprint and/or speed up the training process on some data sets.
```

### `output_format`

> *Default value = `'auto'`.*

```{glossary}
`'auto'`
  The most suitable format for representation of the ground truth matrix is chosen automatically by estimating which representation requires less memory.

`'dense'`
  Enforces that the ground truth matrix is stored using a dense format.

`'sparse'`
  Enforces that the ground truth matrix is stored using a sparse format, if possible. Using a sparse format may reduce the memory footprint on some data sets.
```

### `prediction_format`

> *Default value = `'auto'`.*

```{glossary}
`'auto'`
  The most suitable format for the representation of predictions is chosen automatically based on the sparsity of the ground truth matrix supplied for training.

`'dense'`
  Enforces that predictions are stored using a dense format.

`'sparse'`
  Enforces that predictions are stored using a sparse format, if supported. Using a sparse format may reduce the memory footprint on some data sets.
```

## Training Objective

The following parameters may be used to control the training object and regularization settings of the algorithm. They should be carefully adjusted to the machine learning task one aims to solve in order to achieve optimal results in terms of predictive performance.

### `loss`

> *Default value = `'logistic-decomposable'` for classification problems or `'squared-error-decomposable'` for regression problems.*

```{glossary}
`'logistic-decomposable'` *(classification only)*
  A variant of the logistic loss function that is applied to each label individually. 

`'logistic-non-decomposable'` *(classification only)*
  A variant of the logistic loss function that takes all labels into account at the same time.

`'squared-hinge-decomposable'` *(classification only)*
  A variant of the squared hinge loss that is applied to each label individually.

`'squared-hinge-non-decomposable'` *(classification only)*
  A variant of the squared hinge loss that takes all labels into account at the same time.

`'squared-error-decomposable'`
  A variant of the squared error loss that is applied to each output individually.

`'squared-error-non-decomposable'`
  A variant of the squared error loss that takes all outputs into account at the same time.
```

### `shrinkage`

> *Default value = `0.3`.*

The shrinkage parameter, a.k.a. the "learning rate", that is used to shrink the weight of individual rules. The given value must be in the range (0, 1\].

### `l1_regularization_weight`

> *Default value = `0.0`.*

The weight of the L1 regularization. The given value must be at least 0. If 0 is used, the L1 regularization is turned off entirely. Increasing the value causes the model to become more conservative.

### `l2_regularization_weight`

> *Default value = `1.0`.*

The weight of the L2 regularization. The given value must be at least 0. If 0 is used, the L2 regularization is turned off entirely. Increasing the value causes the model to become more conservative.

## Rule Induction

The parameters listed below allow to configure the algorithm used for the induction of individual rules and help to control the characteristics of these rules.

### `default_rule`

> *Default value = `'auto'`.*

```{glossary}
`'auto'`
  A default rule that provides a default prediction for all examples is included as the first rule of a model unless it prevents a sparse format for the representation of gradients and Hessians from being used (see parameter {ref}`statistic_format<boosting_parameters_statistic_format>`).

`'true'`
  A default rule that provides a default prediction for all examples is included as the first rule of a model.

`'false'`
  No default rule is used.
```

### `rule_induction`

> *Default value = `'top-down-greedy'`.*

```{glossary}
`'top-down-greedy'`
  A greedy top-down search, where rules are successively refined by adding new conditions, is used for the induction of individual rules. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `max_conditions` *(Default value = `0`)*

      The maximum number of conditions to be included in a rule's body. The given value must be at least 1 or 0, if the number of conditions should not be restricted.

    - `min_coverage` *(Default value = `1`)*

      The minimum number of training examples that must be covered by a rule. The given value must be at least 1.

    - `min_support` *(Default value = `0.0`)*

      The minimum support, i.e., the fraction of training examples that must be covered by a rule. The given value must be in the range \[0, 1\] or 0, if the support of rules should not be restricted.

    - `max_head_refinements` *(Default value = `1`)*

      The maximum number of times the head of a rule may be refined. The given value must be at least 1 or 0, if the number of refinements should not be restricted.

    - `recalculate_predictions` *(Default value = `'true'`)*

      `'true'`, if the predictions of rules should be recalculated on the entire training data if the parameter {ref}`instance_sampling<boosting_parameters_instance_sampling>` is not set to the value `'none'`, `'false'`, if the predictions of rules should not be recalculated.

`'top-down-beam-search'`
  A top-down beam search, where rules are successively refined by adding new conditions, is used for the induction of individual rules. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `beam_width` *(Default value = `4`)*
    
      The width to be used by the beam search. A larger value tends to result in more accurate rules being found, but negatively affects the training time. The given value must be at least 2.

    - `resample_features` *(Default value = `'false'`)*
    
      `'true'`, if a new sample of the available features should be created for each rule that is refined during a beam search, `'false'` otherwise. Does only have an effect if the parameter {ref}`feature_sampling<boosting_parameters_feature_sampling>` is not set to the value `'none'`.

    - `max_conditions` *(Default value = `0`)*

      The maximum number of conditions to be included in a rule's body. The given value must be at least 2 or 0, if the number of conditions should not be restricted.

    - `min_coverage` *(Default value = `1`)*
    
      The minimum number of training examples that must be covered by a rule. The given value must be at least 1.

    - `min_support` *(Default value = `0.0`)*
    
      The minimum support, i.e., the fraction of training examples that must be covered by a rule. The given value must be in the range \[0, 1\] or 0, if the support of rules should not be restricted.

    - `max_head_refinements` *(Default value = `1`)*
    
      The maximum number of times the head of a rule may be refined. The given value must be at least 1 or 0, if the number of refinements should not be restricted.

    - `recalculate_predictions` *(Default value = `'true'`)*
    
      `'true'`, if the predictions of rules should be recalculated on the entire training data if the parameter {ref}`instance_sampling<boosting_parameters_instance_sampling>` is not set to the value `'none'`, `'false'`, if the predictions of rules should not be recalculated.
```

### `head_type`

> *Default value = `'auto'`.*

```{glossary}
`'auto'`
  The most suitable type of rule heads is chosen automatically, depending on the loss function.

`'single'`
  All rules predict for a single output.

`'partial-fixed'`
  All rules predict for a predefined number of outputs. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `output_ratio` *(Default value = `0.0`)*
    
      A percentage that specifies for how many outputs the rules should predict or 0, if the percentage should be set to a reasonable default value (the average label cardinality in case of classification problems). For example, a value of 0.05 means that the rules should predict for 5% of the available outputs.

    - `min_outputs` *(Default value = `2`)*
    
      The minimum number of outputs for which the rules should predict. The given value must be at least 2.

    - `max_outputs` *(Default value = `0`)*
    
      The maximum number of outputs for which the rules should predict or 0, if the number of predictions should not be restricted.

`'partial-dynamic'`
  All rules predict for a subset of the available outputs that is determined dynamically. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `threshold` *(Default value = `0.02`)*
    
      A threshold that affects for how many outputs the rules should predict. A smaller threshold results in less outputs being selected. A greater threshold results in more outputs being selected. E.g., a threshold of 0.02 means that a rule will only predict for an output if the estimated predictive quality `q` for this particular output satisfies the inequality `q^exponent > q_best^exponent * (1 - 0.02)`, where `q_best` is the best quality among all outputs. The given value must be in the range (0, 1)

    - `exponent` *(Default value = `2.0`)*
    
      An exponent that is used to weigh the estimated predictive quality for individual outputs. E.g., an exponent of 2 means that the estimated predictive quality `q` for a particular output is weighed as `q^2`. The given value must be at least 1.

`'complete'`
  All rules predict for all outputs simultaneously, potentially capturing dependencies between the outputs.
```

## Stopping Criteria

The following parameters can be used to configure different stopping mechanisms that may be used to terminate the induction of new rules as soon as certain criteria are met.

### `max_rules`

> *Default value = `1000`.*

The maximum number of rules to be learned (including the default rule). The given value must be at least 1 or 0, if the number of rules should not be restricted.

### `time_limit`

> *Default value = `0`.*

The duration in seconds after which the induction of rules should be canceled. The given value must be at least 1 or 0, if no time limit should be set.

## Pruning and Post-Optimization

The following parameters provide fine-grain control over the techniques that should be used for pruning rules or optimizing them after they have been learned. These techniques can help to prevent overfitting and may be helpful if one strives for simple models without any superfluous rules.

### `holdout`

> *Default value = `'auto'`.*

```{glossary}
`'none'`
  No holdout set is created.

`'auto'`
  The most suitable strategy for creating a holdout set is chosen automatically, depending on whether a holdout set is needed according to the parameters {ref}`rule_pruning<boosting_parameters_rule_pruning>`, {ref}`global_pruning<boosting_parameters_global_pruning>`, {ref}`marginal_probability_calibration<boosting_parameters_marginal_probability_calibration>` or {ref}`joint_probability_calibration<boosting_parameters_joint_probability_calibration>`.

`'random'`
  The available examples are randomly split into a training set and a holdout set. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `holdout_set_size` *(Default value = `0.33`)*
    
      The percentage of examples to be included in the holdout set. For example, a value of 0.3 corresponds to 30% of the available examples. The given value must be in the range (0, 1).

`'stratified-output-wise'` *(classification only)*
  The available examples are split into a training set and a holdout set according to an iterative stratified sampling method that ensures that for each label the proportion of relevant and irrelevant examples is maintained. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `holdout_set_size` *(Default value = `0.33`)*
    
      The percentage of examples to be included in the holdout set. For example, a value of 0.3 corresponds to 30% of the available examples. The given value must be in the range (0, 1).

`'stratified-example-wise'` *(classification only)*
  The available examples are split into a training set and a holdout set according to a stratified sampling method, where distinct label vectors are treated as individual classes. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `holdout_set_size` *(Default value = `0.33`)*
    
      The percentage of examples to be included in the holdout set. For example, a value of 0.3 corresponds to 30% of the available examples. The given value must be in the range (0, 1).
```

(boosting_parameters_rule_pruning)=

### `rule_pruning`

> *Default value = `'none'`.*

```{glossary}
`'none'`
  No method for pruning individual rules is used.

`'irep'`
  Trailing conditions of rules may be pruned on a holdout set, similar to the IREP algorithm. Does only have an effect if the parameter {ref}`instance_sampling<boosting_parameters_instance_sampling>` is not set to the value `'none'`.
```

(boosting_parameters_global_pruning)=

### `global_pruning`

> *Default value = `'none'`.*

```{glossary}
`'none'`
  No strategy for global pruning is used.

`'post-pruning'`
  Keeps track of the number of rules in a model that perform best on the training or holdout set according to the loss function. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `use_holdout_set` *(Default value = `'true'`)*

      `'true'`, if the quality of the current model should be measured on the holdout set, if available, `'false'`, if the training set should be used instead.

    - `remove_unused_rules` *(Default value = `'true'`)*

      `'true'`, if unused rules should be removed from the final model, `'false'` otherwise.

    - `min_rules` *(Default value = `100`)*

      The minimum number of rules that must be included in a model. The given value must be at least 1

    - `interval` *(Default value = `1`)*

      The interval to be used to check whether the current model is the best one evaluated so far. For example, a value of 10 means that the best model may contain 10, 20, ... rules. The given value must be at least 1

`'pre-pruning'`
  Stops the induction of new rules as soon as the performance of the model does not improve on the training or holdout set according to the loss function. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `use_holdout_set` *(Default value = `'true'`)*

      `'true'`, if the quality of the current model should be measured on the holdout set, if available, `'false'`, if the training set should be used instead.

    - `remove_unused_rules` *(Default value = `'true'`)*

      `'true'`, if the induction of rules should be stopped as soon as the stopping criterion is met, `'false'`, if additional rules should be included in the model without being used for prediction.

    - `min_rules` *(Default value = `100`)*

      The minimum number of rules that must be included in a model. The given value must be at least 1.

    - `update_interval` *(Default value = `1`)*

      The interval to be used to update the quality of the current model. For example, a value of 5 means that the model quality is assessed every 5 rules. The given value must be at least 1.

    - `stop_interval` *(Default value = `1`)*

      The interval to be used to decide whether the induction of rules should be stopped. For example, a value of 10 means that the rule induction might be stopped after 10, 20, ... rules. The given value must be a multiple of update_interval.

    - `num_past` *(Default value = `50`)*

      The number of quality scores of past iterations to be stored in a buffer. The given value must be at least 1.

    - `num_recent` *(Default value = `50`)*

      The number of quality scores of the most recent iterations to be stored in a buffer. The given value must be at least 1.

    - `aggregation` *(Default value = `'min'`)*

      The name of the aggregation function that should be used to aggregate the scores in both buffers. The given value must be `'min'`, `'max'` or `'avg'`.

    - `min_improvement` *(Default value = `0.005`)*

      The minimum improvement in percent that must be reached when comparing the aggregated scores in both buffers for the rule induction to be continued. The given value must be in the range \[0, 1\].
```

### `sequential_post_optimization`

> *Default value = `'false'`.*

```{glossary}
`'false'`
  Sequential post-optimization is not used.

`'true'`
  Each rule in a previously learned model is optimized by being relearned in the context of the other rules. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `num_iterations` *(Default value = `2`)*

      The number of times each rule should be relearned. The given value must be at least 1.

    - `refine_heads` *(Default value = `'false'`)*

      `'true'`, if the heads of rules may be refined when being relearned, `'false'`, if the relearned rules should be predict for the same outputs as the original rules.

    - `resample_features` *(Default value = `'true'`)*

      `'true'`, if a new sample of the available features should be created whenever a new rule is refined, `'false'`, if the conditions of the new rule should use the same features as the original rule
```

## Sampling Techniques

The following parameters allow to employ various sampling techniques that may help reducing computational costs when dealing with large datasets. Moreover, they may be used to ensure that a diverse set of rules is learned, which may lead to better generalization when dealing with large models.

### `random_state`

> *Default value = `1`.*

The seed to be used by random number generators. The given value must be at least 1.

### `output_sampling`

> *Default value = `'none'`.*

```{glossary}
`'none'`
  All outputs are considered for learning a new rule.

`'round-robin'`
  A single output to be considered when learning a new rule is chosen in a round-robin fashion, i.e., the first rule is concerned with the first output, the second one with the second output, and so on. When the last output is reached, the procedure restarts at the first output.

`'without-replacement'`
  The outputs to be considered when learning a new rule are chosen randomly. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `sample_size` *(Default value = `0.33`)*
    
      The percentage of outputs to be included in a sample. For example, a value of 0.6 corresponds to 60% of the outputs. The given value must be in (0, 1\].

    - `min_samples` *(Default value = `1`)*

      The minimum number of outputs to be included in a sample. The given value must be at least 1.

    - `max_samples` *(Default value = `1`)*

      The maximum number of outputs to be included in a sample. The given value must be at least the value of `min_samples` or 0, if the number of outputs should not be restricted.
```

(boosting_parameters_feature_sampling)=

### `feature_sampling`

> *Default value = `'without-replacement'`.*

```{glossary}
`'none'`
  All features are considered for learning a new rule.

`'without-replacement'`
  A random subset of the features is used to search for the refinements of rules. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `sample_size` *(Default value = `0`)*
    
      The percentage of features to be included in a sample. For example, a value of 0.6 corresponds to 60% of the features. The given value must be in (0, 1\] or 0, if the sample size should be calculated as log2(A - 1) + 1), where A denotes the number of available features.

    - `min_samples` *(Default value = `1`)*

      The minimum number of features to be included in a sample. The given value must be at least 1.

    - `max_samples` *(Default value = `0`)*

      The maximum number of features to be included in a sample. The given value must be at least the value of `min_samples` or 0, if the number of features should not be restricted.
    
    - `num_retained` *(Default value = `0`)*
    
      The number of trailing features to be always included in a sample. For example, a value of 2 means that the last two features are always retained.
```

(boosting_parameters_instance_sampling)=

### `instance_sampling`

> *Default value = `'none'`.*

```{glossary}
`'none'`
  All training examples are considered for learning a new rule.

`'with-replacement'`
  The training examples to be considered for learning a new rule are selected randomly with replacement. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `sample_size` *(Default value = `1.0`)*
    
      The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. The given value must be in the range (0, 1).

    - `min_samples` *(Default value = `1`)*

      The minimum number of examples to be included in a sample. The given value must be at least 1.

    - `max_samples` *(Default value = `0`)*

      The maximum number of examples to be included in a sample. The given value must be at least the value of `min_samples` or 0, if the number of examples should not be restricted.

`'without-replacement'`
  The training examples to be considered for learning a new rule are selected randomly without replacement. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `sample_size` *(Default value = `0.66`)*
    
      The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. The given value must be in the range (0, 1).

    - `min_samples` *(Default value = `1`)*

      The minimum number of examples to be included in a sample. The given value must be at least 1.

    - `max_samples` *(Default value = `0`)*

      The maximum number of examples to be included in a sample. The given value must be at least the value of `min_samples` or 0, if the number of examples should not be restricted.

`'stratified-output-wise'` *(classification only)*
  The training examples to be considered for learning a new rule are selected according to an iterative stratified sampling method that ensures that for each label the proportion of relevant and irrelevant examples is maintained. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `sample_size` *(Default value = `0.66`)*
    
      The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. The given value must be in the range (0, 1).

    - `min_samples` *(Default value = `1`)*

      The minimum number of examples to be included in a sample. The given value must be at least 1.

    - `max_samples` *(Default value = `0`)*

      The maximum number of examples to be included in a sample. The given value must be at least the value of `min_samples` or 0, if the number of examples should not be restricted.

`'stratified-example-wise'` (*classification only*)
  The training examples to be considered for learning a new rule are selected according to stratified sampling method, where distinct label vectors are treated as individual classes. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `sample_size` *(Default value = `0.66`)*
    
      The percentage of examples to be included in a sample. For example, a value of 0.6 corresponds to 60% of the available examples. The given value must be in the range (0, 1).

    - `min_samples` *(Default value = `1`)*

      The minimum number of examples to be included in a sample. The given value must be at least 1.

    - `max_samples` *(Default value = `0`)*

      The maximum number of examples to be included in a sample. The given value must be at least the value of `min_samples` or 0, if the number of examples should not be restricted.
```

## Approximations and Optimizations

The following parameters can be used to control various approximation and optimization techniques used by the algorithm. All of these aim for faster training times, possibly at the expense of predictive performance.

### `feature_binning`

> *Default value = `'none'`.*

```{glossary}
`'none'`
  No feature binning is used.

`'equal-width'`
  Examples are assigned to bins, based on their feature values, according to the equal-width binning method. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `bin_ratio` *(Default value = `0.33`)*
    
      A percentage that specifies how many bins should be used. For example, a value of 0.3 means that the number of bins should be set to 30% of the total number of available training examples.
    
    - `min_bins` *(Default value = `2`)*

      The minimum number of bins. The given value must be at least 2.
    
    - `max_bins` *(Default value = `0`)*
    
      The maximum number of bins. The given value must be at least the value of `min_bins` or 0, if the number of bins should not be restricted.

`'equal-frequency'`
  Examples are assigned to bins, based on their feature values, according to the equal-frequency binning method. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `bin_ratio` *(Default value = `0.33`)*
    
      A percentage that specifies how many bins should be used. For example, a value of 0.3 means that the number of bins should be set to 30% of the total number of available training examples.
    
    - `min_bins` *(Default value = `2`)*
    
      The minimum number of bins. The given value must be at least 2.
    
    - `max_bins` *(Default value = `0`)*
    
      The maximum number of bins. The given value must be at least the value of `min_bins` or 0, if the number of bins should not be restricted.
```

### `label_binning`

> *Default value = `'auto'`. Can only be used in classification problems.*

```{glossary}
`'none'`
  No label binning is used.

`'auto'`
  The most suitable strategy for label-binning is chosen automatically based on the loss function and the type of rule heads.

`'equal-width'`
  The labels for which a rule may predict are assigned to bins according to the equal-width binning method. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `bin_ratio` *(Default value = `0.04`)*
    
      A percentage that specifies how many bins should be used. For example, a value of 0.04 means that number of bins should be set to 4% of the number of labels.

    - `min_bins` *(Default value = `1`)*
    
      The minimum number of bins. The given value must be at least 1.

    - `max_bins` *(Default value = `0`)*
      
      The maximum number of bins or 0, if the number of bins should not be restricted.
```

(boosting_parameters_statistic_format)=

### `statistic_format`

> *Default value = `'auto'`.*

```{glossary}
`'auto'`
  The most suitable format for the representation of gradients and Hessians is chosen automatically, depending on the loss function, the type of rule heads, the characteristics of the ground truth matrix and whether a default rule is used or not.

`'dense'`
  A dense format is used for the representation of gradients and Hessians.

`'sparse'`
  A sparse format is used for the representation of gradients and Hessians, if supported by the loss function.
```

## Probability Calibration

The following parameters enable to learn calibration models that should be included in a model and may result in more accurate probability estimates being predicted.

(boosting_parameters_marginal_probability_calibration)=

### `marginal_probability_calibration`

> *Default value = `'none'`. Can only be used in classification problems.*

```{glossary}
`'none'`
  Marginal probabilities are not calibrated.

`'isotonic'`
  Marginal probabilities are calibrated via isotonic regression.

    - `'use_holdout_set'` *(Default value = `'true'`)*
    
      `'true'`, if the calibration model should be fit to the examples in the holdout set, if available, `'false'`, if the training set should be used instead.
```

(boosting_parameters_joint_probability_calibration)=

### `joint_probability_calibration`

> *Default value = `'none'`. Can only be used in classification problems.*

```{glossary}
`'none'`
  Joint probabilities are not calibrated.

`'isotonic'`
  Joint probabilities are calibrated via isotonic regression.

    - `'use_holdout_set'` *(Default value = `'true'`)*
    
      `'true'`, if the calibration model should be fit to the examples in the holdout set, if available, `'false'`, if the training set should be used instead.
```

## Prediction

The following parameters allow to configure the mechanism that is employed for obtaining different types of predictions from a model.

### `binary_predictor`

> *Default value = `'auto'`. Can only be used in classification problems.*

```{glossary}
`'auto'`
  The most suitable strategy for predicting binary labels is chosen automatically, depending on the loss function.

`'output-wise'`
  The prediction for an example is determined for each label independently. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `based_on_probabilities` *(Default value = `'false'`)*
    
      `'true'`, if binary predictions should be derived from probability estimates rather than scores if supported by the loss function, `'false'` otherwise.

    - `use_probability_calibration` *(Default value = `'true'`)*
    
      `'true'`, if a model for the calibration of probabilities should be used, if available, `'false'` otherwise. Does only have an effect if the option `based_on_probabilities` is set to the value `'true'`.

`'example-wise'`
  The label vector that is predicted for an example is chosen from the set of label vectors encountered in the training data. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `based_on_probabilities` *(Default value = `'false'`)*
    
      `'true'`, if binary predictions should be derived from probability estimates rather than scores if supported by the loss function, `'false'` otherwise.

    - `use_probability_calibration` *(Default value = `'true'`)*
    
      `'true'`, if a model for the calibration of probabilities should be used, if available, `'false'` otherwise. Does only have an effect if the option `based_on_probabilities` is set to the value `'true'`.

`'gfm'`
  The label vector that is predicted for an example is chosen according to the general F-measure maximizer (GFM).

    - `use_probability_calibration` *(Default value = `'true'`)*
    
      `'true'`, if a model for the calibration of probabilities should be used, if available, `'false'` otherwise.
```

### `probability_predictor`

> *Default value = `'auto'`. Can only be used in classification problems.*

```{glossary}
`'auto'`
  The most suitable strategy for predicting probability estimates is chosen automatically, depending on the loss function.

`'output-wise'`
  The prediction for an example is determined for each label independently

    - `use_probability_calibration` *(Default value = `'true'`)*
    
      `'true'`, if a model for the calibration of probabilities should be used, if available, `'false'` otherwise.

`'marginalized'`
  The prediction for an example is determined via marginalization over the set of label vectors encountered in the training data.

    - `use_probability_calibration` *(Default value = `'true'`)*
    
      `'true'`, if a model for the calibration of probabilities should be used, if available, `'false'` otherwise.
```

## Multi-Threading

The following parameters allow to specify whether multi-threading should be used for different aspects of the algorithm. Depending on your hardware, they may help to reduce the time needed for training or prediction.

```{warning}
To be able to use the algorithm's multi-threading capabilities, it must have been compiled with multi-threading support enabled, which should be the case with pre-built packages available on [PyPI](https://pypi.org/). Please refer to the section {ref}`build-options` if you intend to compile the program yourself, or if you want to check if multi-threading support is enabled for your installation.
```

### `parallel_rule_refinement`

> *Default value = `'auto'`.*

```{glossary}
`'auto'`
  The number of threads to be used to search for potential refinements of rules in parallel is chosen automatically, depending on the loss function.

`'false'`
  No multi-threading is used to search for potential refinements of rules.

`'true'`
  Multi-threading is used to search for potential refinements of rules in parallel. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `num_preferred_threads` *(Default value = `0`)*
    
      The number of preferred threads. The given value must be at least 1 or 0, if the number of cores available on the machine should be used. If not enough CPU cores are available or if multi-threading support is disabled, as many threads as possible are used.
```

### `parallel_statistic_update`

> *Default value = `'auto'`.*

```{glossary}
`'auto'`
  The number of threads to be used to calculate the gradients and Hessians for different examples in parallel is chosen automatically, depending on the loss function.

`'false'`
  No multi-threading is used to calculate the gradients and Hessians of different examples.

`'true'`
  Multi-threading is used to calculate the gradients and Hessians of different examples in parallel. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `num_preferred_threads` *(Default value = `0`)*

      The number of preferred threads. The given value must be at least 1 or 0, if the number of cores available on the machine should be used. If not enough CPU cores are available or if multi-threading support is disabled, as many threads as possible are used.
```

### `parallel_prediction`

> *Default value = `'true'`.*

```{glossary}
`'false'`
  No multi-threading is used to obtain predictions for different examples.

`'true'`
  Multi-threading is used to obtain predictions for different examples in parallel. The following options may be provided using the {ref}`bracket notation<bracket-notation>`:

    - `num_preferred_threads` *(Default value = `0`)*
    
      The number of preferred threads. The given value must be at least 1 or 0, if the number of cores available on the machine should be used. If not enough CPU cores are available or if multi-threading support is disabled, as many threads as possible are used.
```
