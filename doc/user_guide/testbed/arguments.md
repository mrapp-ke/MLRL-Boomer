(arguments)=

# Overview of Arguments

In addition to the mandatory arguments that must be provided to the command line API to specify the dataset to be used for training, a wide variety of optional arguments are available as well. In the following, we provide an overview of these arguments and discuss their respective purposes.

```{note}
The arguments `-h` or `--help` result in a description of all available command line arguments being printed.
```

```{note}
When running the program with the argument `-v` or `--version`, the version of the software package is printed. The output also includes information about third-party dependencies it uses, the {ref}`build options<build-options>` that have been used for building the package, as well as information about hardware resources it may utilize.
```

```{note}
The argument `--log-level` controls the level of detail used for log messages (Default value = `info`). It can be set to the values `debug`, `info`, `warn`, `warning`, `error`, `critical`, `fatal` or `notset`, where the first one provides the greatest level of detail and the last one disables logging entirely.
```

(arguments-basic-usage)=

## Basic Usage

> A more detailed description of the following arguments can be found {ref}`here<testbed>`.

The most basic command for running the program, only including mandatory arguments, is as follows:

```text
testbed <module_or_source_file> --data-dir /path/to/dataset/ --dataset dataset-name
```

### Module

The program dynamically loads a Python module or source file that provides an integration with a specific machine learning algorithm. To specify the module or source file to be used, the following mandatory arguments must be provided:

- `<module_or_source_file>` The fully qualified name of a Python module, or an absolute or relative path to a Python source file, providing a Python class that extends from {py:class}`mlrl.testbed.runnables.Runnable`. The name of the class must be `Runnable`, unless an alternative name is specified via the optional command line argument `-r` or `--runnable`.

The following optional arguments allow additional control over the loading mechanism:

- `-r` or `--runnable` (Default value = `Runnable`) The name of the class extending {py:class}`mlrl.testbed.runnables.Runnable` that resides within the module or source file specified via the argument `<module_or_source_file>`.

The arguments given above can be used to integrate any scikit-learn compatible machine learning algorithm with the comman line API. You can learn about this {ref}`here<runnables>`.

### Dataset

The following mandatory arguments must always be given to specify the dataset that should be used, as well as the location where it should be loaded from.

- `--data-dir` An absolute or relative path to the directory where the data set files are located.
- `--dataset` The name of the data set files (without suffix).

Optionally, the following arguments can be used to provide additional information about the dataset.

- `--sparse-feature-value` (Default value = `0.0`) The value that should be used for sparse elements in the feature matrix. Does only have an effect if a sparse format is used for the representation of the feature matrix, depending on the parameter `--feature-format`.

### Problem Type

The command line API can conduct experiments for classification and regression problems. When dealing with the latter, the type of the machine learning problem must explicitly be specified via the following argument:

- `--problem-type` (Default value = `classification`)

  - `classification` The dataset is considered as a classification data set.
  - `regression` The dataset is considered as a regression data set.

## Performance Evaluation

> A more detailed description of the following arguments can be found {ref}`here<evaluation>`.

One of the most important capabilities of the command line API is to train machine learning models and obtain an unbiased estimate of their predictive performance. For this purpose, the available data must be split into training and test data. The former is used to train models and the latter is used for evaluation afterwards, whereas the evaluation metrics depend on the type of predictions provided by a model.

### Strategies for Data Splitting

- `--data-split` (Default value = `train-test`)

  - `train-test` The available data is split into a single training and test set. Given that `dataset-name` is provided as the value of the argument `--dataset`, the training data must be stored in a file named `dataset-name_training.arff`, whereas the test data must be stored in a file named `dataset-name_test.arff`. If no such files are available, the program searches for a file with the name `dataset-name.arff` and splits it into training and test data automatically. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `test_size` (Default value = `0.33`) The fraction of the available data to be included in the test set, if the training and test set are not provided as separate files. Must be in (0, 1).

  - `cross-validation` A cross validation is performed. Given that `dataset-name` is provided as the value of the argument `--dataset`, the data for individual folds must be stored in files named `dataset-name_fold-1`, `dataset-name_fold-2`, etc.. If no such files are available, the program searches for a file with the name `dataset-name.arff` and splits it into training and test data for the individual folds automatically. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `num_folds` (Default value = `10`) The total number of cross validation folds to be performed. Must be at least 2.
    - `current_fold` (Default value = `0`) The cross validation fold to be performed. Must be in \[1, `num_folds`\] or 0, if all folds should be performed.

  - `none` The available data is not split into separate training and test sets, but the entire data is used for training and evaluation. This strategy should only be used for testing purposes, as the evaluation results will be highly biased and overly optimistic. Given that `dataset-name` is provided as the value of the argument `--dataset`, the data must be stored in a file named `dataset-name.arff`.

- `--evaluate-training-data` (Default value = `false`)

  - `true` The models are not only evaluated on the test data, but also on the training data.
  - `false` The models are only evaluated on the test data.

### Types of Predictions

- `--prediction-type` (Default value = `binary`)

  - `scores` The learner is instructed to predict scores. In this case, ranking measures are used for evaluation.
  - `probabilities` The learner is instructed to predict probability estimates. In this case, ranking measures are used for evaluation.
  - `binary` The learner is instructed to predict binary labels. In this case, bipartition evaluation measures are used for evaluation.

### Incremental Evaluation

- `--incremental-evaluation` (Default value = `false`)

  - `true` Ensemble models are evaluated repeatedly, using only a subset of their ensemble members with increasing size, e.g., the first 100, 200, ... rules.

    - `min_size` (Default value = `0`) The minimum number of ensemble members to be evaluated. Must be at least 0.
    - `max_size` (Default value = `0`) The maximum number of ensemble members to be evaluated. Must be greater than `min_size` or 0, if all ensemble members should be evaluated.
    - `step_size` (Default value = `1`) The number of additional ensemble members to be evaluated at each repetition. Must be at least 1.

  - `false` Models are evaluated only once as a whole.

## Data Pre-Processing

> A more detailed description of the following arguments can be found {ref}`here<pre-processing>`.

Depending on the characteristics of a dataset, it might be desirable to apply one of the following pre-processing techniques before training and evaluating machine learning models.

### One-Hot-Encoding

- `--one-hot-encoding` (Default value = `false`)

  - `true` One-hot-encoding is used to encode nominal features.
  - `false` The algorithm's ability to natively handle nominal features is used.

## Saving and Loading Models

> A more detailed description of the following arguments can be found {ref}`here<model-persistence>`.

Because the training of models can be time-consuming, it might be desirable to store them on disk for later use. This requires to specify the path of a directory where models should be saved.

- `--model-dir` (Default value = `None`)

  - An absolute or relative path to the directory where models should be stored. If such models are found in the specified directory, they are used instead of learning a new model from scratch. If no models are available, the trained models are saved in the specified directory once training has completed.

## Saving and Loading Parameters

> A more detailed description of the following arguments can be found {ref}`here<parameter-persistence>`.

As an alternative to storing the models learned by an algorithm, the algorithmic parameters used for training can be saved to disk. This may help to remember the configuration used for training a model and enables to reload the same parameter setting for additional experiments.

- `--parameter-dir` (Default value = `None`)

  - An absolute or relative path to the directory where configuration files, which specify the parameters to be used by the algorithm, are located. If such files are found in the specified directory, the specified parameter settings are used instead of the parameters that are provided via command line arguments.

- `--print-parameters` (Default value = `false`)

  - `true` Algorithmic parameters are printed on the console.
  - `false` Algorithmic parameters are not printed on the console.

- `--store-parameters` (Default value = `false`)

  - `true` Algorithmic parameters that have been set by the user are written into [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) files. Does only have an effect if the parameter `--output-dir` is specified.
  - `false` Algorithmic parameters that have been set by the user are not written into [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) files.

## Output of Experimental Results

> A more detailed description of the following arguments can be found {ref}`here<experimental-results>`.

To provide valuable insights into the models learned by an algorithm, the predictions they provide, or the data they have been derived from, a wide variety of experimental results can be written to output files or printed on the console. If the results should be written to files, it is necessary to specify an output directory:

- `--output-dir` An absolute or relative path to the directory where experimental results should be saved.

(arguments-evaluation-results)=

### Evaluation Results

- `--print-evaluation` (Default value = `true`)

  - `true` The evaluation results in terms of common metrics are printed on the console. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `decimals` (Default value = `2`) The number of decimals to be used for evaluation scores or 0, if the number of decimals should not be restricted.
    - `percentage` (Default value = `true`) `true`, if evaluation scores should be given as a percentage, if possible, `false` otherwise.
    - `enable_all` (Default value = `true`) `true`, if all supported metrics should be used unless specified otherwise, `false` if all metrics should be disabled by default.
    - `hamming_loss` (Default value = `true`) `true`, if evaluation scores according to the Hamming loss should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `hamming_accuracy` (Default value = `true`) `true`, if evaluation scores according to the Hamming accuracy metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `subset_zero_one_loss` (Default value = `true`) `true`, if evaluation scores according to the subset 0/1 loss should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `subset_accuracy` (Default value = `true`) `true`, if evaluation scores according to the subset accuracy metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `micro_precision` (Default value = `true`) `true`, if evaluation scores according to the micro-averaged precision metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `micro_recall` (Default value = `true`) `true`, if evaluation scores according to the micro-averaged recall metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `micro_f1` (Default value = `true`) `true`, if evaluation scores according to the micro-averaged F1-measure should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `micro_jaccard` (Default value = `true`) `true`, if evaluation scores according to the micro-averaged Jaccard metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `macro_precision` (Default value = `true`) `true`, if evaluation scores according to the macro-averaged precision metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `macro_recall` (Default value = `true`) `true`, if evaluation scores according to the macro-averaged recall metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `macro_f1` (Default value = `true`) `true`, if evaluation scores according to the macro-averaged F1-measure should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `macro_jaccard` (Default value = `true`) `true`, if evaluation scores according to the macro-averaged Jaccard metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `example_wise_precision` (Default value = `true`) `true`, if evaluation scores according to the example-wise precision metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `example_wise_recall` (Default value = `true`) `true`, if evaluation scores according to the example-wise recall metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `example_wise_f1` (Default value = `true`) `true`, if evaluation scores according to the example-wise F1-measure should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `example_wise_jaccard` (Default value = `true`) `true`, if evaluation scores according to the example-wise Jaccard metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `accuracy` (Default value = `true`) `true`, if evaluation scores according to the accuracy metric should be printed, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `zero_one_loss` (Default value = `true`) `true`, if evaluation scores according to the 0/1 loss should be printed, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `precision` (Default value = `true`) `true`, if evaluation scores according to the precision metric should be printed, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `recall` (Default value = `true`) `true`, if evaluation scores according to the recall metric should be printed, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `f1` (Default value = `true`) `true`, if evaluation scores according to the F1-measure should be printed, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `jaccard` (Default value = `true`) `true`, if evaluation scores according to the Jaccard metric should be printed, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `mean_absolute_error` (Default value = `true`) `true`, if evaluation scores according to the mean absolute error metric should be printed, `false` otherwise. Does only have an effect if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `mean_squared_error` (Default value = `true`) `true`, if evaluation scores according to the mean squared error metric should be printed, `false` otherwise. Does only have an effect if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `mean_absolute_error` (Default value = `true`) `true`, if evaluation scores according to the mean absolute error metric should be printed, `false` otherwise. Does only have an effect if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `mean_absolute_percentage_error` (Default value = `true`) `true`, if evaluation scores according to the mean absolute percentage error metric should be printed, `false` otherwise. Does only have an effect if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `rank_loss` (Default value = `true`) `true`, if evaluation scores according to the rank loss should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `coverage_error` (Default value = `true`) `true`, if evaluation scores according to the coverage error metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `lrap` (Default value = `true`) `true`, if evaluation scores according to the label ranking average precision metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `dcg` (Default value = `true`) `true`, if evaluation scores according to the discounted cumulative gain metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `ndcg` (Default value = `true`) `true`, if evaluation scores according to the normalized discounted cumulative gain metric should be printed, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `probabilities` or `scores`.

  - `false` The evaluation results are not printed on the console.

- `--store-evaluation` (Default value = `true`)

  - `true` The evaluation results in terms of common metrics are written into [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) files. Does only have an effect if the parameter `--output-dir` is specified.

    - `decimals` (Default value = `0`) The number of decimals to be used for evaluation scores or 0, if the number of decimals should not be restricted.
    - `percentage` (Default value = `true`) `true`, if evaluation scores should be given as a percentage, if possible, `false` otherwise.
    - `enable_all` (Default value = `true`) `true`, if all supported metrics should be used unless specified otherwise, `false` if all metrics should be disabled by default.
    - `hamming_loss` (Default value = `true`) `true`, if evaluation scores according to the Hamming loss should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `hamming_accuracy` (Default value = `true`) `true`, if evaluation scores according to the Hamming accuracy metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `subset_zero_one_loss` (Default value = `true`) `true`, if evaluation scores according to the subset 0/1 loss should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `subset_accuracy` (Default value = `true`) `true`, if evaluation scores according to the subset accuracy metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `micro_precision` (Default value = `true`) `true`, if evaluation scores according to the micro-averaged precision metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `micro_recall` (Default value = `true`) `true`, if evaluation scores according to the micro-averaged recall metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `micro_f1` (Default value = `true`) `true`, if evaluation scores according to the micro-averaged F1-measure should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `micro_jaccard` (Default value = `true`) `true`, if evaluation scores according to the micro-averaged Jaccard metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `macro_precision` (Default value = `true`) `true`, if evaluation scores according to the macro-averaged precision metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `macro_recall` (Default value = `true`) `true`, if evaluation scores according to the macro-averaged recall metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `macro_f1` (Default value = `true`) `true`, if evaluation scores according to the macro-averaged F1-measure should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `macro_jaccard` (Default value = `true`) `true`, if evaluation scores according to the macro-averaged Jaccard metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `example_wise_precision` (Default value = `true`) `true`, if evaluation scores according to the example-wise precision metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `example_wise_recall` (Default value = `true`) `true`, if evaluation scores according to the example-wise recall metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `example_wise_f1` (Default value = `true`) `true`, if evaluation scores according to the example-wise F1-measure should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `example_wise_jaccard` (Default value = `true`) `true`, if evaluation scores according to the example-wise Jaccard metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `labels`.
    - `accuracy` (Default value = `true`) `true`, if evaluation scores according to the accuracy metric should be stored, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `zero_one_loss` (Default value = `true`) `true`, if evaluation scores according to the 0/1 loss should be stored, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `precision` (Default value = `true`) `true`, if evaluation scores according to the precision metric should be stored, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `recall` (Default value = `true`) `true`, if evaluation scores according to the recall metric should be stored, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `f1` (Default value = `true`) `true`, if evaluation scores according to the F1-measure should be stored, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `jaccard` (Default value = `true`) `true`, if evaluation scores according to the Jaccard metric should be stored, `false` otherwise. Does only have an effect when dealing with single-label data and if the parameter `--prediction-type` is set to `labels`.
    - `mean_absolute_error` (Default value = `true`) `true`, if evaluation scores according to the mean absolute error metric should be stored, `false` otherwise. Does only have an effect if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `mean_squared_error` (Default value = `true`) `true`, if evaluation scores according to the mean squared error metric should be stored, `false` otherwise. Does only have an effect if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `mean_absolute_error` (Default value = `true`) `true`, if evaluation scores according to the mean absolute error metric should be stored, `false` otherwise. Does only have an effect if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `mean_absolute_percentage_error` (Default value = `true`) `true`, if evaluation scores according to the mean absolute percentage error metric should be stored, `false` otherwise. Does only have an effect if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `rank_loss` (Default value = `true`) `true`, if evaluation scores according to the rank loss should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `coverage_error` (Default value = `true`) `true`, if evaluation scores according to the coverage error metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `lrap` (Default value = `true`) `true`, if evaluation scores according to the label ranking average precision metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `dcg` (Default value = `true`) `true`, if evaluation scores according to the discounted cumulative gain metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `ndcg` (Default value = `true`) `true`, if evaluation scores according to the normalized discounted cumulative gain metric should be stored, `false` otherwise. Does only have an effect when dealing with multi-label data and if the parameter `--prediction-type` is set to `probabilities` or `scores`.
    - `training_time` (Default value = `true`) `true`, if the time that was needed for training should be stored, `false` otherwise.
    - `prediction_time` (Default value = `true`) `true`, if the time that was needed for prediction should be stored, `false` otherwise.

  - `false` The evaluation results are not written into [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) files.

(arguments-predictions)=

### Predictions

- `--print-predictions` (Default value = `false`)

  - `true` The predictions for individual examples and outputs are printed on the console.

    - `decimals` (Default value = `2`) The number of decimals to be used for real-valued predictions or 0, if the number of decimals should not be restricted.

  - `false` The predictions are not printed on the console.

- `--store-predictions` (Default value = `false`)

  - `true` The predictions for individual examples and outputs are written into .arff files. Does only have an effect if the parameter `--output-dir` is specified.

    - `decimals` (Default value = `0`) The number of decimals to be used for real-valued predictions or 0, if the number of decimals should not be restricted.

  - `false` Predictions are not written into .arff files.

(arguments-prediction-characteristics)=

### Prediction Characteristics

- `--print-prediction-characteristics` (Default value = `false`)

  - `true` The characteristics of binary predictions are printed on the console. Does only have an effect if the parameter `--predict-probabilities` is set to `false`.

    - `decimals` (Default value = `2`) The number of decimals to be used for characteristics or 0, if the number of decimals should not be restricted.
    - `percentage` (Default value = `true`) `true`, if the characteristics should be given as a percentage, if possible, `false` otherwise.
    - `outputs` (Default value = `true`) `true`, if the number of outputs should be printed, `false` otherwise.
    - `output_density` (Default value = `true`) `true`, if the density of the ground truth matrix should be printed, `false` otherwise.
    - `output_sparsity` (Default value = `true`) `true`, if the sparsity of the ground truth matrix should be printed, `false` otherwise.
    - `label_imbalance_ratio` (Default value = `true`, *classification only*) `true`, if the label imbalance ratio should be printed, `false` otherwise.
    - `label_cardinality` (Default value = `true`, *classification only*) `true`, if the average label cardinality should be printed, `false` otherwise.
    - `distinct_label_vectors` (Default value = `true`, *classification only*) `true`, if the number of distinct label vectors should be printed, `false` otherwise.

  - `false` The characteristics of predictions are not printed on the console.

- `--store-prediction-characteristics` (Default value = `false`)

  - `true` The characteristics of binary predictions are written into [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) files. Does only have an effect if the parameter `--predict-probabilities` is set to `false`.

    - `decimals` (Default value = `0`) The number of decimals to be used for characteristics or 0, if the number of decimals should not be restricted.
    - `percentage` (Default value = `true`) `true`, if the characteristics should be given as a percentage, if possible, `false` otherwise.
    - `outputs` (Default value = `true`) `true`, if the number of outputs should be stored, `false` otherwise.
    - `output_density` (Default value = `true`) `true`, if the density of the ground truth matrix should be stored, `false` otherwise.
    - `output_sparsity` (Default value = `true`) `true`, if the sparsity of the ground truth matrix should be stored, `false` otherwise.
    - `label_imbalance_ratio` (Default value = `true`, *classification only*) `true`, if the label imbalance ratio should be stored, `false` otherwise.
    - `label_cardinality` (Default value = `true`, *classification only*) `true`, if the average label cardinality should be stored, `false` otherwise.
    - `distinct_label_vectors` (Default value = `true`, *classification only*) `true`, if the number of distinct label vectors should be stored, `false` otherwise.

  - `false` The characteristics of predictions are not written into [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) files.

(arguments-data-characteristics)=

### Data Characteristics

- `--print-data-characteristics` (Default value = `false`)

  - `true` The characteristics of the training data set are printed on the console

    - `decimals` (Default value = `2`) The number of decimals to be used for characteristics or 0, if the number of decimals should not be restricted.
    - `percentage` (Default value = `true`) `true`, if the characteristics should be given as a percentage, if possible, `false` otherwise.
    - `outputs` (Default value = `true`) `true`, if the number of outputs should be printed, `false` otherwise.
    - `output_density` (Default value = `true`) `true`, if the density of the ground truth matrix should be printed, `false` otherwise.
    - `output_sparsity` (Default value = `true`) `true`, if the sparsity of the ground truth matrix should be printed, `false` otherwise.
    - `label_imbalance_ratio` (Default value = `true`, *classification only*) `true`, if the label imbalance ratio should be printed, `false` otherwise.
    - `label_cardinality` (Default value = `true`, *classification only*) `true`, if the average label cardinality should be printed, `false` otherwise.
    - `distinct_label_vectors` (Default value = `true`, *classification only*) `true`, if the number of distinct label vectors should be printed, `false` otherwise.
    - `examples` (Default value = `true`) `true`, if the number of examples should be printed, `false` otherwise.
    - `features` (Default value = `true`) `true`, if the number of features should be printed, `false` otherwise.
    - `numerical_features` (Default value = `true`) `true`, if the number of numerical features should be printed, `false` otherwise.
    - `nominal_features` (Default value = `true`) `true`, if the number of nominal features should be printed, `false` otherwise.
    - `feature_density` (Default value = `true`) `true`, if the feature density should be printed, `false` otherwise.
    - `feature_sparsity` (Default value = `true`) `true`, if the feature sparsity should be printed, `false` otherwise.

  - `false` The characteristics of the training data set are not printed on the console

- `--store-data-characteristics` (Default value = `false`)

  - `true` The characteristics of the training data set are written into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file. Does only have an effect if the parameter `--output-dir` is specified.

    - `decimals` (Default value = `0`) The number of decimals to be used for characteristics or 0, if the number of decimals should not be restricted.
    - `percentage` (Default value = `true`) `true`, if the characteristics should be given as a percentage, if possible, `false` otherwise.
    - `outputs` (Default value = `true`) `true`, if the number of outputs should be stored, `false` otherwise.
    - `output_density` (Default value = `true`) `true`, if the density of the ground truth matrix should be stored, `false` otherwise.
    - `output_sparsity` (Default value = `true`) `true`, if the sparsity of the groun dtruth matrix should be stored, `false` otherwise.
    - `label_imbalance_ratio` (Default value = `true`, *classification only*) `true`, if the label imbalance ratio should be stored, `false` otherwise.
    - `label_cardinality` (Default value = `true`, *classification only*) `true`, if the average label cardinality should be stored, `false` otherwise.
    - `distinct_label_vectors` (Default value = `true`, *classification only*) `true`, if the number of distinct label vectors should be stored, `false` otherwise.
    - `examples` (Default value = `true`) `true`, if the number of examples should be stored, `false` otherwise.
    - `features` (Default value = `true`) `true`, if the number of features should be stored, `false` otherwise.
    - `numerical_features` (Default value = `true`) `true`, if the number of numerical features should be stored, `false` otherwise.
    - `nominal_features` (Default value = `true`) `true`, if the number of nominal features should be stored, `false` otherwise.
    - `feature_density` (Default value = `true`) `true`, if the feature density should be stored, `false` otherwise.
    - `feature_sparsity` (Default value = `true`) `true`, if the feature sparsity should be stored, `false` otherwise.

  - `false` The characteristics of the training data set are not written into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file.

(arguments-label-vectors)=

### Label Vectors

- `--print-label-vectors` (Default value = `false`, *classification only*)

  - `true` The unique label vectors contained in the training data are printed on the console. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `sparse` (Default value = `false`) `true`, if a sparse representation of label vectors should be used, `false` otherwise.

  - `false` The unique label vectors contained in the training data are not printed on the console.

- `--store-label-vectors` (Default value = `false`, *classification only*)

  - `true` The unique label vectors contained in the training data are written into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file. Does only have an effect if the parameter `` `--output-dir `` is specified. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `sparse` (Default value = `false`) `true`, if a sparse representation of label vectors should be used, `false` otherwise.

  - `false` The unique label vectors contained in the training data are not written into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file.

(arguments-model-characteristics)=

### Model Characteristics

- `--print-model-characteristics` (Default value = `false`)

  - `true` The characteristics of rule models are printed on the console
  - `false` The characteristics of rule models are not printed on the console

- `--store-model-characteristics` (Default value = `false`)

  - `true` The characteristics of rule models are written into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file. Does only have an effect if the parameter `--output-dir` is specified.
  - `false` The characteristics of rule models are not written into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file.

(arguments-output-rules)=

### Rules

- `--print-rules` (Default value = `false`)

  - `true` The induced rules are printed on the console. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `print_feature_names` (Default value = `true`) `true`, if the names of features should be printed instead of their indices, `false` otherwise.
    - `print_output_names` (Default value = `true`) `true`, if the names of outputs should be printed instead of their indices, `false` otherwise.
    - `print_nominal_values` (Default value = `true`) `true`, if the names of nominal values should be printed instead of their numerical representation, `false` otherwise.
    - `print_bodies` (Default value = `true`) `true`, if the bodies of rules should be printed, `false` otherwise.
    - `print_heads` (Default value = `true`) `true`, if the heads of rules should be printed, `false` otherwise.
    - `decimals_body` (Default value = `2`) The number of decimals to be used for numerical thresholds of conditions in a rule's body or 0, if the number of decimals should not be restricted.
    - `decimals_head` (Default value = `2`) The number of decimals to be used for predictions in a rule's head or 0, if the number of decimals should not be restricted.

  - `false` The induced rules are not printed on the console.

- `--store-rules` (Default value = `false`)

  - `true` The induced rules are written into a .txt file. Does only have an effect if the parameter `--output-dir` is specified. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `print_feature_names` (Default value = `true`) `true`, if the names of features should be printed instead of their indices, `false` otherwise.
    - `print_output_names` (Default value = `true`) `true`, if the names of outputs should be printed instead of their indices, `false` otherwise.
    - `print_nominal_values` (Default value = `true`) `true`, if the names of nominal values should be printed instead of their numerical representation, `false` otherwise.
    - `print_bodies` (Default value = `true`) `true`, if the bodies of rules should be printed, `false` otherwise.
    - `print_heads` (Default value = `true`) `true`, if the heads of rules should be printed, `false` otherwise.
    - `decimals_body` (Default value = `2`) The number of decimals to be used for numerical thresholds of conditions in a rule's body or 0, if the number of decimals should not be restricted.
    - `decimals_head` (Default value = `2`) The number of decimals to be used for predictions in a rule's head or 0, if the number of decimals should not be restricted.

  - `false` The induced rules are not written into a .txt file.

(arguments-probability-calibration-models)=

### Probability Calibration Models

- `--print-marginal-probability-calibration-model` (Default value = `false`)

  - `true` The model for the calibration of marginal probabilities is printed on the console. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `decimals` (Default value = `2`) The number of decimals to be used for thresholds and probabilities or 0, if the number of decimals should not be restricted.

  - `false` The model for the calibration of marginal probabilities is not printed on the console.

- `--store-marginal-probability-calibration-model` (Default value = `false`)

  - `true` The model for the calibration of marginal probabilities is written into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file. Does only have an effect if the parameter `--output-dir` is specified. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `decimals` (Default value = `0`) The number of decimals to be used for thresholds and probabilities or 0, if the number of decimals should not be restricted.

  - `false` The model for the calibration of marginal probabilities is not written into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file.

- `--print-joint-probability-calibration-model` (Default value = `false`)

  - `true` The model for the calibration of joint probabilities is printed on the console. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `decimals` (Default value = `2`) The number of decimals to be used for thresholds and probabilities or 0, if the number of decimals should not be restricted.

  - `false` The model for the calibration of joint probabilities is not printed on the console.

- `--store-joint-probability-calibration-model` (Default value = `false`)

  - `true` The model for the calibration of joint probabilities is written into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file. Does only have an effect if the parameter `--output-dir` is specified. The following options may be specified using the {ref}`bracket notation<bracket-notation>`:

    - `decimals` (Default value = `2`) The number of decimals to be used for thresholds and probabilities or 0, if the number of decimals should not be restricted.

  - `false` The model for the calibration of joint probabilities is not written into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file.

(setting-algorithmic-parameters)=

## Setting Algorithmic Parameters

In addition to the command line arguments that are discussed above, it is often desirable to not rely on the default configuration of the BOOMER algorithm in an experiment, but to use a custom configuration. For this purpose, all of the algorithmic parameters that are discussed in the section {ref}`parameters` may be set by providing corresponding arguments to the command line API.

In accordance with the syntax that is typically used by command line programs, the parameter names must be given according to the following syntax that slightly differs from the names that are used by the programmatic Python API:

- All argument names must start with two leading dashes (`--`).
- Underscores (`_`) must be replaced with dashes (`-`).

For example, the value of the parameter `feature_binning` may be set as follows:

````{tab} BOOMER
   ```text
   testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --feature-binning equal-width
   ```
````

````{tab} SeCo
   ```text
   testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --feature-binning equal-width
   ```
````

Some algorithmic parameters, including the parameter `feature_binning`, allow to specify additional options as key-value pairs by using a {ref}`bracket notation<bracket-notation>`. This is also supported by the command line API, where the options may not contain any spaces and special characters like `{` or `}` must be escaped by using single-quotes (`'`):

````{tab} BOOMER
   ```text
   testbed mlrl.boosting\
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --feature-binning equal-width'{bin_ratio=0.33,min_bins=2,max_bins=64}'
   ```
````

````{tab} SeCo
   ```text
   testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --feature-binning equal-width'{bin_ratio=0.33,min_bins=2,max_bins=64}'
   ```
````
