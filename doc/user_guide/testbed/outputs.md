(testbed-outputs)=

# Saving and Loading Data

The {ref}`command line API <arguments>` of mlrl-testbed provides a diverse set of options for saving data that is collected during an experiment. In some cases, e.g., models or algorithmic parameters, such data can also be read from input files. Both aspects, saving and loading data, are discussed below.

```{tip}
By default, mlrl-testbed checks if the output files that should be produced by an experiments according to the given arguments do already exist. If this is the case, the experiment is canceled to avoid unnecessary computations. By providing the argument `--if-outputs-exist overwrite`, the experiment can be forced to be run anyway.
```

(meta-data)=

## Saving Meta-Data

Saving meta-data can help improving the reproducibility of experiments. Among other information, it contains the command that has been used for running an experiment and the version of mlrl-testbed used. Based on this information the experiment can be re-run later using the {ref}`run mode <testbed-run-mode>`. By default, a `metadata.yml` file is saved to the directory specified via the argument `--base-dir`, if any other output data is saved as well. To explicitly enforce saving meta-data, the argument `--save-meta-data` can be used:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --save-meta-data true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --save-meta-data true
   ```
````

Alternatively, the meta-data can be printed via the command line argument `--print-meta-data`:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-meta-data true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-meta-data true
   ```
````

(model-persistence)=

## Saving and Loading Models

Because the training of machine learning models can be time-consuming, they are usually trained once and then reused later for making predictions. For this purpose, the package mlrl-testbed provides means to store models on disk and load them from the created files later on. This requires to specify the arguments `--save-models` and `--load-models`. Optionally, the path to a directory where models should be saved, as well as a directory from which models should be loaded can be set via the command line arguments `--model-save-dir` and `--model-load-dir`. If not specified manually, the default `models` is used for both.

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --model-save-dir /path/to/models \
       --model-load-dir /path/to/models \
       --save-models true \
       --load-models true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --model-save-dir /path/to/models \
       --model-load-dir /path/to/models \
       --save-models true \
       --load-models true
   ```
````

```{note}
The paths of the directories specified via the arguments `--model-save-dir` and `--model-load-dir` can be either absolute or relative to the working directory. They must not refer to the same directory, which allows saving models to a different directory than the one they are loaded from. 
```

If {ref}`train-test splits<train-test-split>` are used for evaluating the predictive performance of models, a single model is fit to the training data and stored in a file:

- `model.pickle`

If a {ref}`cross validation<cross-validation>` is performed instead, one model is trained per cross validation fold and all of these models are stored in the specified directory. For example, a 5-fold cross validation results in the following files:

- `model_fold-1.pickle`
- `model_fold-2.pickle`
- `model_fold-3.pickle`
- `model_fold-4.pickle`
- `model_fold-5.pickle`

When executing the aforementioned command again, the program recognizes the previously stored models in the specified directory. Instead of training them from scratch, the models are then loaded from the respective files, which should be much faster than training them again.

(parameter-persistence)=

## Saving and Loading Parameters

To remember the parameters that have been used for training a model, it might be useful to save them to disk. Similar to {ref}`saving models<model-persistence>`, keeping the resulting files allows to load a previously used configuration and reuse it at a later point in time.

This requires to specify the argument `--save-parameters`. Optionally, the command line argument `--parameter-save-dir`, can be used to specify a directory where algorithmic parameters (see {ref}`setting-algorithmic-parameters`) should be saved. If this argument is not specified, the default value `parameters` is used. For example, the following command sets a custom value for a parameter, which is stored in an output file:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --parameter-save-dir /path/to/parameters \
       --save-parameters true \
       --shrinkage 0.5
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --parameter-save-dir /path/to/parameters \
       --save-parameters true \
       --heuristic precision
   ```
````

If {ref}`train-test splits<train-test-split>` are used for splitting the available data into training and test sets, a single model is trained and its configuration is saved to a file:

- `parameters.csv`

If a {ref}`cross validation<cross-validation>` is performed instead, one model is trained per cross validation fold and the configurations of all of these models are stored in the specified directory. For example, a 5-fold cross validation results in the following files:

- `parameters_fold-1.csv`
- `parameters_fold-2.csv`
- `parameters_fold-3.csv`
- `parameters_fold-4.csv`
- `parameters_fold-5.csv`

```{note}
Only parameters with custom values are included in the output files. Parameters for which the default value is used are not included.
```

If you want to print all custom parameters that are used by a learning algorithm on the console, you can specify the argument `--print-parameters true`:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-parameters true \
       --shrinkage 0.5
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-parameters true \
       --heuristic precision
   ```
````

Once parameters have been saved to a directory, they can be loaded in subsequent experiments by using the command line argument `--load-parameters`. The path to the directory, where the input files are located, can be set via the argument `--parameter-load-dir`. If no path is given, `parameters` is used as the default. Loading parameters from input files allows to omit the respective parameters from the command line. If a parameter is included in both, the loaded file and the command line arguments, the latter takes precedence.

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --parameter-load-dir /path/to/parameters \
       --load-parameters true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --parameter-load-dir /path/to/parameters \
       --load-parameters true
   ```
````

```{note}
The paths of the directories that are specified via the arguments `--parameter-save-dir` and `--parameter-load-dir` can be either absolute or relative to the working directory.
```

(experimental-results)=

## Saving Experimental Results

One of the most important features provided by the package mlrl-testbed is the ability to output a wide variety of experimental results that provide valuable insights into the models learned by a machine learning algorithm, the predictions it provides, and the data it has been trained on.

Each of these information can either be printed to the console or saved to output files. The latter requires to provide a directory, where the output files should be saved. As shown in the examples below, the path to this directory can be specified via the argument `--result-dir`. However, this argument is optional. If not given explicitly, the default value `results` is used.

```{note}
The path to the directory, where experimental results should be saved, can be either absolute or relative to the working directory.
```

```{tip}
By default, the directory specified via the argument `--result-dir` is created automatically if it does not exist. This functionality can be turned off via the argument `--create-dirs false`.
```

```{tip}
By providing the argument `--save-all true`, the program can be instructed to write all available output data to files. Similarly, the argument `--print-all true` results in all output data being printed on the console.  
```

(output-evaluation-results)=

### Evaluation Results

By default, the predictive performance of all models trained during an experiment is evaluated in terms of commonly used evaluation metrics and the evaluation results are printed to the console. In addition, the evaluation results can also be written to output files. The command line argument `--print-evaluation` can be used to explicitly enable or disable printing the evaluation results:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-evaluation true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-evaluation true
   ```
````

Accordingly, the argument `--save-evaluation` allows to enable or disable saving the evaluation results to [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) files:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-evaluation true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-evaluation true
   ```
````

```{tip}
The command line arguments ``--print-evaluation`` and ``--save-evaluation`` come with several options for customization described {ref}`here<arguments-evaluation-results>`. It is possible to specify the performance metrics that should be used for evaluation by providing a black- or whitelist. Moreover, one can specify whether performance scores should be given as percentages and the number of decimals used for these scores can be chosen freely.
```

The number of models evaluated during an experiment varies depending on the strategy used for splitting the available data into training and test sets. When using {ref}`train-test splits<train-test-split>`, only a single model is evaluated. The performance scores according to different metrics that assess the quality of the model's predictions are saved to a single output file. In addition, when {ref}`evaluating on the training data<evaluating-training-data>`, the performance scores for the model's predictions on the training set are also evaluated and written to a file. As shown below, the names of the output files specify whether predictions for the training or test set have been evaluated:

- `evaluation_train.csv`
- `evaluation_test.csv`

When using a {ref}`cross validation<cross-validation>`, a model is trained and evaluated for each fold. Again, the names of the output files specify whether predictions for the training or test data have been evaluated:

- `evaluation_train_fold-1.csv`
- `evaluation_test_fold-1.csv`
- `evaluation_train_fold-2.csv`
- `evaluation_test_fold-2.csv`
- `evaluation_train_fold-3.csv`
- `evaluation_test_fold-3.csv`
- `evaluation_train_fold-4.csv`
- `evaluation_test_fold-4.csv`
- `evaluation_train_fold-5.csv`
- `evaluation_test_fold-5.csv`

(output-predictions)=

### Predictions

In cases where the {ref}`evaluation results<output-evaluation-results>` obtained via the arguments `--print-evaluation` or `--save-evaluation` are not sufficient for a detailed analysis, it may be desired to directly inspect the predictions provided by the evaluated models. They can be printed on the console by proving the argument `--print-predictions`. If the ground truth should also be printed, the argument `--print-ground-truth` must be provided as well:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-predictions true \
       --print-ground-truth true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-predictions true \
       --print-ground-truth true
   ```
````

Alternatively, the argument `--save-predictions` and `--save-ground-truth` can be used to save a dataset with the predictions or the ground truth, respectively, to [ARFF](https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/) files:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-predictions true \
       --save-ground-truth true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-predictions true \
       --save-ground-truth true
   ```
````

```{tip}
Depending on the {ref}`type of predictions<prediction-types>`, the machine learning models used in an experiment are supposed to provide, the predictions stored in the resulting output files are either binary values (if binary predictions are provided), or real values (if scores or probability estimates are provided). When working with real-valued ground truth or predictions, the option ``decimals`` may be supplied to the arguments ``--print-predictions``, ``--save-predictions``, ``--print-ground-truth` and `--save-ground-truth` to specify the number of decimals that should be included in the output (see {ref}`here<arguments-predictions>` for more information).
```

When using {ref}`train-test splits<train-test-split>`, a single model is trained and queried for predictions for the test set. These predictions are written to a single output file. When {ref}`evaluating on the training data<evaluating-training-data>`, predictions are also obtained for the training set and written to an additional output file. The names of the output files indicate whether the predictions have been obtained for the training or test set, respectively:

- `predictions_train.arff`
- `predictions_test.arff`

When using a {ref}`cross validation<cross-validation>` for performance evaluation, a model is trained for each fold. Similar to before, the names of the output files indicate whether the predictions correspond to the training or test data:

- `predictions_train_fold-1.arff`
- `predictions_test_fold-1.arff`
- `predictions_train_fold-2.arff`
- `predictions_test_fold-2.arff`
- `predictions_train_fold-3.arff`
- `predictions_test_fold-3.arff`
- `predictions_train_fold-4.arff`
- `predictions_test_fold-4.arff`
- `predictions_train_fold-5.arff`
- `predictions_test_fold-5.arff`

(output-prediction-characteristics)=

### Prediction Characteristics

By using the command line argument `--print-prediction-characteristics`, characteristics regarding a model's predictions can be printed:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-prediction-characteristics true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-prediction-characteristics true
   ```
````

Alternatively, they statistics can be written to a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file by using the argument `--save-prediction-characteristics`:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-prediction-characteristics true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-prediction-characteristics true
   ```
````

```{tip}
The output produced by the arguments ``--print-data-characteristics`` and ``--save-data-characteristics`` can be customized via several options described {ref}`here<arguments-prediction-characteristics>`. It is possible to exclude certain statistics from the output, to specify whether they should be given as percentages, and how many decimal places should be used.
```

The statistics obtained via the arguments given above correspond to the test data for which predictions are obtained from the model. Consequently, they depend on the strategy used for splitting a dataset into training and test sets. When using {ref}`train-test splits<train-test-split>`, predictions for a single test set are obtained and their characteristics are written to a file. In addition, statistics for the training data are written to an additional output file when {ref}`evaluating on the training data<evaluating-training-data>`:

- `prediction_characteristics_train.csv`
- `prediction_characteristics_test.csv`

When using a {ref}`cross validation<cross-validation>`, the data is split into several parts of which each one is used once for prediction. Multiple output files are needed to save the statistics for different cross validation folds. For example, a 5-fold cross validation results in the following files:

- `prediction_characteristics_fold-1.csv`
- `prediction_characteristics_fold-2.csv`
- `prediction_characteristics_fold-3.csv`
- `prediction_characteristics_fold-4.csv`
- `prediction_characteristics_fold-5.csv`

The statistics obtained via the previous commands include the following:

- **The number of outputs:** Indicates for how many outputs predictions have been obtained.
- **The sparsity of the prediction matrix:** The percentage of elements in the prediction matrix that are zero.

When dealing with classification problems, the following statistics are given as well:

- **The average label cardinality:** The average number of labels predicted as relevant for each example.
- **The number of distinct label vectors:** The number of unique label combinations predicted for different examples.
- **The label imbalance ratio:** A measure for the imbalance between labels predicted as relevant and irrelevant, respectively. [^charte2013]

(output-data-characteristics)=

### Data Characteristics

To obtain insightful statistics regarding the characteristics of a dataset, the command line argument `--print-data-characteristics` may be helpful:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-data-characteristics true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-data-characteristics true
   ```
````

If you prefer to write the statistics into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file, the argument `--save-data-characteristics` can be used:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-data-characteristics true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-data-characteristics true
   ```
````

```{tip}
As shown {ref}`here<arguments-data-characteristics>`, the arguments ``--print-data-characteristics`` and ``--save-data-characteristics`` come with several options that allow to exclude specific statistics from the respective output. It is also possible to specify whether percentages should be preferred for presenting the statistics. Additionally, the number of decimals to be included in the output can be limited.
```

The statistics provided by the previous commands are obtained on the training data and therefore depend on the strategy used for splitting a dataset into training and test sets. If {ref}`train-test splits<train-test-split>` are used, a single training set is used and its characteristics are saved to a file:

- `data_characteristics.csv`

In contrast, when using a {ref}`cross validation<cross-validation>`, the data is split into several parts of which each one is used once for training. As a result, multiple output files are created in a such a scenario. For example, a 5-fold cross validation results in the following files:

- `data_characteristics_fold-1.csv`
- `data_characteristics_fold-2.csv`
- `data_characteristics_fold-3.csv`
- `data_characteristics_fold-4.csv`
- `data_characteristics_fold-5.csv`

The output produced by the previous commands includes the following information regarding a dataset's features:

- **The number of examples contained in a dataset:** Besides the total number, the number of examples per type of feature (numerical, ordinal, or nominal) is also given.
- **The sparsity of the feature matrix:** This statistic calculates as the percentage of elements in the feature matrix that are equal to zero.

In addition, the following statistics regarding the ground truth in a dataset are provided:

- **The total number of available outputs**
- **The sparsity of the ground truth matrix:** This statistic calculates as the percentage of elements in the ground truth matrix that are zero.

When dealing with classification problems, the following statistics are given as well:

- **The average label cardinality:** The average number of relevant labels per example.
- **The number of distinct label vectors:** The number of unique label combinations present in a dataset.
- **The label imbalance ratio:** An important metric in multi-label classification measuring to which degree the distribution of relevant and irrelevant labels is unbalanced. [^charte2013]

(output-label-vectors)=

### Label Vectors

We refer to the unique labels combinations present for different examples in a classification dataset as label vectors. They can be printed by using the command line argument `--print-label-vectors`:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-label-vectors true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-label-vectors true
   ```
````

If you prefer writing the label vectors into an output file, the argument `--save-label-vectors` can be used:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --save-label-vectors true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --save-label-vectors true
   ```
````

When using {ref}`train-test splits<train-test-split>` for splitting the available data into distinct training and test sets, a single output file is created. It stores the label vectors present in the training data:

- `label_vectors.csv`

When using a {ref}`cross validation<cross-validation>`, several models are trained on different parts of the dataset. The label vectors present in each of these training sets are written to separate output files. For example, the following files result from a 5-fold cross validation:

- `label_vectors_fold-1.csv`
- `label_vectors_fold-2.csv`
- `label_vectors_fold-3.csv`
- `label_vectors_fold-4.csv`
- `label_vectors_fold-5.csv`

The above commands output each label vector present in a dataset, as well as their frequency, i.e., the number of examples they are associated with. Moreover, each label vector is assigned a unique index. By default, feature vectors are given in the following sparse format:

```text
[2 3 4]
```

This notation specifies the indices of all relevant labels in a label vector (counting from zero and sorted in increasing order), while all irrelevant ones are omitted. The example above corresponds to the binary label vector `[0 0 1 1 1 0]`. Due to its compactness, the sparse representation is particularly well-suited when dealing with a large number of labels.

(output-rule-specific)=

### Rule-specific Results

The rule learning algorithms developed by this project extend the functionality of mlrl-testbed with rule-specific options. In the following, we discuss options for saving experiment results that are specific to this particular type of algorithms. Unless noted otherwise, the following options are available when using the packages [mlrl-boomer](https://pypi.org/project/mlrl-boomer/) and [mlrl-seco](https://pypi.org/project/mlrl-seco/) with mlrl-testbed.

(output-model-characteristics)=

#### Model Characteristics

To obtain a quick overview of some statistics that characterize a rule-based model learned by one of the algorithms provided by this project, the command line argument `--print-model-characteristics` can be useful:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-model-characteristics true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-model-characteristics true
   ```
````

The above command results in a tabular representation of the characteristics being printed on the console. If one intends to write them into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file instead, the argument `--save-model-characteristics` may be used:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-model-characteristics true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-model-characteristics true
   ```
````

Model characteristics are obtained for each model training during an experiment. This means that a single output file is created when using on {ref}`train-test splits<train-test-split>`:

- `model_characteristics.csv`

When using a {ref}`cross validation<cross-validation>`, several models are trained on different parts of the available data, resulting in multiple output files being saved to the output directory. For example, the following files are created when conducting a 5-fold cross validation:

- `model_characteristics_fold-1.csv`
- `model_characteristics_fold-2.csv`
- `model_characteristics_fold-3.csv`
- `model_characteristics_fold-4.csv`
- `model_characteristics_fold-5.csv`

The statistics captured by the previous commands include the following:

- **Statistics about conditions:** Information about the number of rules in a model, as well as the different types of conditions contained in their bodies.
- **Statistics about predictions:** The distribution of positive and negative predictions provided by the rules in a model.
- **Statistics per local rule:** The minimum, average, and maximum number of conditions and predictions the rules in a model entail in their bodies and heads, respectively.

(output-rules)=

#### Rules

It is considered one of the advantages of rule-based machine learning models that they capture patterns found in the training data in a human-comprehensible form. This enables to manually inspect the models and reason about their predictive behavior. To help with this task, the package mlrl-testbed allows to output the rules in a model using a textual representation. If the text should be printed on the console, the following command specifying the argument `--print-rules` can be used:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-rules true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-rules true
   ```
````

Alternatively, by using the argument `--save-rules`, a textual representation of models can be written to a text file in the specified output directory:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-rules true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --result-dir /path/to/results/ \
       --save-rules true
   ```
````

```{tip}
Both, the ``--print-rules`` and ``--save-rules`` arguments, come with several options that allow to customize the textual representation of models. An overview of these options is provided {ref}`here<arguments-output-rules>`.
```

When using {ref}`train-test splits<train-test-split>`, only a single model is trained. Consequently, the above command results in a single output file being created:

- `rules.csv`

A {ref}`cross validation<cross-validation>` results in multiple output files, each one corresponding to one of the models trained for an individual fold, being written. For example, a 5-fold cross validation produces the following files:

- `rules_fold-1.csv`
- `rules_fold-2.csv`
- `rules_fold-3.csv`
- `rules_fold-4.csv`
- `rules_fold-5.csv`

Each rule in a model consists of a *body* and a *head* (we use the notation `body => head`). The body specifies to which examples a rule applies. It consists of one or several conditions that compare the feature values of given examples to thresholds derived from the training data. The head of a rule consists of the predictions it provides for individual outputs. The predictions provided by a head may be restricted to a subset of the available output or even a single one.

If not configured otherwise, the first rule in a model is a *default rule*. Unlike the other rules, it does not contain any conditions in its body and therefore applies to any given example. As shown in the following example, it always provides predictions for all available labels:

```text
{} => (output1 = -1.45, output2 = 1.45, output3 = -1.89, output4 = -1.94)
```

In regression models, the predictions of individual rules sum up to the regression scores predicted by the overall model. In classification models, a rule's prediction for a particular label is positive, if most examples it covers are associated with the respective label, otherwise it is negative. The ratio between the number of examples that are associated with a label, and those that are not, affects the absolute size of the default prediction. Large values indicate a strong preference towards predicting a particular label as relevant or irrelevant, depending on the sign.

The remaining rules only apply to examples that satisfy all the conditions in their bodies. For example, the body of the following rule consists of two conditions:

```text
{feature1 <= 1.53 & feature2 > 7.935} => (output1 = -0.31)
```

Examples that satisfy all conditions in a rule's body are said to be "covered" by the rule. If this is the case, the rule assigns a positive or negative value to one or several outputs. Similar to the default rule, the absolute size of the value corresponds to the weight of the rule's prediction. The larger the value, the stronger the impact of the respective rule, compared to the other ones.

(output-probability-calibration-models)=

#### Probability Calibration Models

```{note}
Probability calibration models are only supported by the gradient boosting algorithm [mlrl-boomer](https://pypi.org/project/mlrl-boomer/).
```

The gradient boosting algorithms provided by this project allow to obtain probabilistic predictions. These predictions can optionally be fine-tuned via calibration models to improve the reliability of the probability estimates. We support two types of calibration models for tuning marginal and joint probabilities, respectively. If one needs to inspect these calibration models, the command line arguments `--print-marginal-probability-calibration-model` and `--print-joint-probability-calibration-model` may be helpful:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-marginal-probability-calibration-model true \
       --print-joint-probabiliy-calibration-model true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --print-marginal-probability-calibration-model true \
       --print-joint-probabiliy-calibration-model true
   ```
````

Alternatively, a representations of the calibration models can be written to [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) files by using the arguments `--save-marginal-probability-calibration-model` and `--save-joint-probability-calibration-model`

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --save-marginal-probability-calibration-model true \
       --save-joint-probabiliy-calibration-model true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --save-marginal-probability-calibration-model true \
       --save-joint-probabiliy-calibration-model true
   ```
````

```{tip}
All of the above commands come with options for customizing the textual representation of models. A more detailed description of these options is available {ref}`here<arguments-probability-calibration-models>`.
```

Calibration models are learned during training and depend on the training data. {ref}`train-test splits<train-test-split>`, where only a single model is trained, result in a single file being created for each type of calibration model:

- `marginal_probability_calibration_model.csv`
- `joint_probability_calibration_model.csv`

In contrast, a {ref}`cross validation<cross-validation>` produces multiple output files. Each one corresponds to a calibration model learned on the training data for an individual fold. For example, the following files are created when using a 5-fold cross validation:

- `marginal_probability_calibration_model_fold-1.csv`
- `marginal_probability_calibration_model_fold-2.csv`
- `marginal_probability_calibration_model_fold-3.csv`
- `marginal_probability_calibration_model_fold-4.csv`
- `marginal_probability_calibration_model_fold-5.csv`
- `joint_probability_calibration_model_fold-1.csv`
- `joint_probability_calibration_model_fold-2.csv`
- `joint_probability_calibration_model_fold-3.csv`
- `joint_probability_calibration_model_fold-4.csv`
- `joint_probability_calibration_model_fold-5.csv`

[^charte2013]: Charte, Francisco, Antonio J. Rivera, María José del Jesus, and Francisco Herrera (2019). ‘REMEDIAL-HwR: Tackling multilabel imbalance through label decoupling and data resampling hybridization’. In: *Neurocomputing* 326-327, pp. 110–122.
