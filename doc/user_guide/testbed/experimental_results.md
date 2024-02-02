(experimental-results)=

# Output of Experimental Results

One of the most important features provided by the command line API is the ability to output a wide variety of experimental results that provide valuable insights into the models learned by a machine learning algorithm, the predictions it provides, and the data it has been trained on.

Each of these information can either be printed to the console or saved to output files. The latter requires to provide a directory, where the output files should be saved. As shown in the examples below, the path to this directory must be specified via the argument `--output-dir`.

```{note}
The path of the directory, where experimental results should be saved, can be either absolute or relative to the working directory.
```

(output-evaluation-results)=

## Evaluation Results

TODO

(output-predictions)=

## Predictions

TODO

(output-prediction-characteristics)=

## Prediction Characteristics

TODO

(output-data-characteristics)=

## Data Characteristics

To obtain insightful statistics regarding the characteristics of a data set, the command line argument ``--print-data-characteristics`` may be helpful:

```text
boomer --data-dir /path/to/datsets/ --dataset dataset-name --print-data-characteristics true
```

If you prefer to write the statistics into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file, the argument ``--store-data-characteristics`` can be used:

```text
boomer --data-dir /path/to/datsets/ --dataset dataset-name --output-dir /path/to/results/ --store-data-characteristics true
```

```{tip}
As shown {ref}`here<arguments-data-characteristics>`, the arguments ``--print-data-characteristics`` and ``--store-data-characteristics`` come with several options that allow to exclude specific statistics from the respective output. It is also possible to specify whether percentages should be prefered for presenting the statistics. Additionally, the number of decimals to be included in the output can be limited.
```

The statistics provided by the previous commands are obtained on the training data and therefore depend on the strategy used for splitting a dataset into training and test sets. If {ref}`train-test-split` are used, a single training set is used and its characteristics will be saved to a file:

- `data_characteristics_overall.csv`

In contrast, when using a {ref}`cross-validation`, the data is split into several parts of which each one is used once for training. As a result, multiple output files will be created in a such a scenario. For example, a 5-fold cross validation will result in the following files:

- `data_characteristics_fold-1.csv`
- `data_characteristics_fold-2.csv`
- `data_characteristics_fold-3.csv`
- `data_characteristics_fold-4.csv`
- `data_characteristics_fold-5.csv`

The output produced by the previous commands includes the following information regarding a dataset's features:

- The total number of examples contained in a dataset, as well as the number of examples per type of feature (numerical, ordinal, or nominal).
- The sparsity among the feature values of all examples, calculated as the percentage of elements in the feature matrix that are equal to zero.

In addition, the following statistics regarding the labels in a dataset are provided:

- The total number of available labels.
- The percentage of irrelevant labels among all examples, corresponding to the sparsity of the label matrix.
- The average label cardinality, i.e., the average number of relevant labels per example.
- The number of distinct label vectors, i.e., the number of unique label combinations that are present in a dataset.
- The *label imbalance ratio* [^charte2013], which is an important metric in multi-label classification. It measures to which degree the distribution of relevant and irrelevant labels is unbalanced.

(output-label-vectors)=

## Label Vectors

TODO

(output-model-characteristics)=

## Model Characteristics

To obtain a quick overview of some statistics that characterize a rule-based model learned by one of the algorithms provided by this project, the command line argument ``--print-model-characteristics`` can be useful:

```text
boomer --data-dir /path/to/datsets/ --dataset dataset-name --print-model-characteristics true
```

The above command results in a tabular representation of the characteristics being printed on the console. If one intends to write them into a [.csv](https://en.wikipedia.org/wiki/Comma-separated_values) file instead, the argument ``--store-model-characteristics`` may be used:

```text
boomer --data-dir /path/to/datsets/ --dataset dataset-name --output-dir /path/to/results/ --store-model-characteristics true
```

Model characteristics are obtained for each model training during an experiment. This means that a single output file will be created when using on {ref}`train-test-split`:

- `model_characteristics_overall.csv`

When using a {ref}`cross-validation`, several models are trained on different parts of the available data, resulting in multiple output files being saved to the output directory. For example, the following files will be created when conducting a 5-fold cross validation:

- `model_characteristics_fold-1.csv`
- `model_characteristics_fold-2.csv`
- `model_characteristics_fold-3.csv`
- `model_characteristics_fold-4.csv`
- `model_characteristics_fold-5.csv`

The statistics captured by the previous commands include the following:

- **Statistics about conditions:** Information about the number of rules in a model, as well as the different types of conditons contained in their bodies.
- **Statistics about predictions:** The distribution of positive and negative predictions provided by the rules in a model.
- **Statistics per local rule:** The minimum, average, and maximum number of conditions and predictions the rules in a model entail in their bodies and heads, respectively.

(output-rules)=

## Rules

It is considered one of the advantages of rule-based machine learning models that they capture patterns found in the training data in a human-comprehensible form. This enables to manually inspect the models and reason about their predictive behavior. To help with this task, the command line API allows to output the rules in a model using a textual representation. If the text should be printed on the console, the following command specifying the argument ``--print-rules`` can be used:

```text
boomer --data-dir /path/to/datsets/ --dataset dataset-name --print-rules true
```

Alternatively, by using the argument ``--store-rules``, a textual representation of models can be written into a text file in the specifed output directory: 

```text
boomer --data-dir /path/to/datsets/ --dataset dataset-name --output-dir /path/to/results/ --store-rules true
```

```{tip}
Both, the ``--print-rules`` and ``--store-rules`` arguments, come with several options that allow to customize the textual representation of models. An overview of these options is provided {ref}`here<arguments-output-rules>`.
```

When using {ref}`train-test-split`, only a single model is trained. Consequently, the above command will result in a single output file being created:

- `rules_overall.csv`

A {ref}`cross-validation` results in multiple output files, each one corresponding to one of the models trained for an individual fold, being written. For example, a 5-fold cross validation produces the following files:

- `rules_fold-1.csv`
- `rules_fold-2.csv`
- `rules_fold-3.csv`
- `rules_fold-4.csv`
- `rules_fold-5.csv`

Each rule in a model consists of a *body* and a *head* (we use the notation ``body => head``). The body specifies to which examples a rule applies. It consist of one or several conditions that compare the feature values of given examples to thresholds derived from the training data. The head of a rule consists of the predictions it provides for individual labels. The predictions provided by a head may be restricted to a subset of the available labels or even a single one.

If not configured otherwise, the first rule in a model is a *default rule*. Unlike the other rules, it does not contain any conditions in its body and therefore applies to any given example. As shown in the following example, it always provides predictions for all available labels:

```text
{} => (label1 = -1.45, label2 = 1.45, label3 = -1.89, label4 = -1.94)
```

The prediction for a particular label is positive, if most examples are associated with the respective label, otherwise it is negative. The ratio between the number of examples that are associated with a label, and those that are not, affects the absolute size of the default prediction. Large values indicate a stong preference towards predicting a particular label as relevant or irrelevant, depending on the sign.

The remaining rules only apply to examples that satisfy all of the conditions in their bodies. For example, the body of the following rule consists of two conditions:

```text
{feature1 <= 1.53 & feature2 > 7.935} => (label1 = -0.31)
```

Examples that satisfy all conditions in a rule's body are said to be "covered" by the rule. If this is the case, the rule assigns a positive or negative value to one or several labels. Similar to the default rule, a positive value expresses a preference towards predicting the corresponding label as relevant. A negative value contributes towards predicting the label as irrelevant. The absolute size of the value corresponds to the weight of the rule's prediction. The larger the value, the stronger the impact of the respective rule, compared to the other ones.

(output-probability-calibration-models)=

## Probability Calibration Models

TODO

[^charte2013]: Charte, Francisco, Antonio J. Rivera, María José del Jesus, and Francisco Herrera (2019). ‘REMEDIAL-HwR: Tackling multilabel imbalance through label decoupling and data resampling hybridization’. In: *Neurocomputing* 326-327, pp. 110–122.
