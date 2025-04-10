(testbed)=

# Using the Command Line API

As an alternative to using algorithms provided by this project in your own Python program (see {ref}`usage`), the command line API that is provided by the package [mlrl-testbed](https://pypi.org/project/mlrl-testbed/) (see {ref}`installation`) can be used to run experiments without the need to write code. Currently, it provides the following functionalities:

- The predictive performance in terms of commonly used evaluation measures can be assessed by using predefined splits of a dataset into training and test data or via [cross validation](<https://en.wikipedia.org/wiki/Cross-validation_(statistics)>).
- Experimental results can be written into output files. This includes evaluation scores, the predictions of a model, textual representations of rules, as well as the characteristics of models or datasets.
- Models can be stored on disk and reloaded for later use.

## Running Experiments

```{tip}
The command line API can also be used to conduct experiments with any machine learning algorithm of your choice. All that is needed for this are few lines of Python code as described {ref}`here<runnables>`.
```

Depending on the capabilities of an algorithm, the command line API supports both, classification and regression problems. In the following, we provide examples for both scenarios.

### Classification Problems

The following example illustrates use the command line API for applying the BOOMER algorithm, or the SeCO algorithm, to a particular classification dataset:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name
   ```
````

Both arguments that are included in the above command are mandatory:

- `--data-dir` An absolute or relative path to the directory where the data set files are located.
- `--dataset` The name of the data set files (without suffix).

The program expects the data set files to be provided in the [Mulan format](https://mulan.sourceforge.net/format.html). It requires two files to be present in the specified directory:

1. An [ARFF](https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/) file that specifies the feature values and ground truth of the training examples.
2. An XML file that specifies the names of the outputs.

The Mulan dataset format is commonly used for benchmark datasets that allow to compare the performance of different machine learning approaches in empirical studies. A collection of publicly available benchmark datasets is available [here](https://github.com/mrapp-ke/Boomer-Datasets).

If an XML file is not provided, the program tries to parse the number of outputs from the `@relation` declaration that is contained in the ARFF file, as it is intended by the [MEKA project's dataset format](https://waikato.github.io/meka/datasets/). According to the MEKA format, the number of outputs may be specified by including the substring "-C L" in the `@relation` name, where "L" is the number of leading features in the dataset that should be treated as outputs.

### Regression Problems

In addition to classification problems, the BOOMER algorithm can also be used for solving regression problems. As shown below, the argument `--problem-type` instructs the command line API to consider the given dataset as a regression dataset:

```text
mlrl-testbed mlrl.boosting \
    --data-dir /path/to/datasets/ \
    --dataset dataset-name \
    --problem-type regression
```

The semantic of the mandatory arguments `--data-dir` and `--dataset` is the same as for classification problems.

## Optional Arguments

In addition to the mandatory arguments that must be provided to the command line API for specifying the dataset used for training, a wide variety of optional arguments for customizing the program's behavior are available as well. An overview of all available command line arguments is provided in the section {ref}`arguments`. For example, they can be used to specify an output directory, where experimental results should be stored:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --output-dir /path/to/output/
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --output-dir /path/to/output/
   ```
````

Moreover, algorithmic parameters that control the behavior of the machine learning algorithm can be set via command line arguments as well. For example, as shown in the section {ref}`setting-algorithmic-parameters`, the value of the parameter `feature_binning` can be specified as follows:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --feature-binning equal-width
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --feature-binning equal-width
   ```
````

Some algorithmic parameters, including the parameter `feature_binning`, come with additional options in the form of key-value pairs. They can be specified by using a {ref}`bracket notation<bracket-notation>` as shown below:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --feature-binning equal-width'{bin_ratio=0.33,min_bins=2,max_bins=64}'
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --feature-binning equal-width'{bin_ratio=0.33,min_bins=2,max_bins=64}'
   ```
````

(bracket-notation)=

## Bracket Notation

Each algorithmic parameter is identified by a unique name. Depending on the type of parameter, it either accepts numbers as possible values or allows to specify a string that corresponds to a predefined set of possible values (boolean values are also represented as strings).

In addition to the specified value, some parameters allow to provide additional options as key-value pairs. These options must be provided by using the following bracket notation:

```text
'value{key1=value1,key2=value2}'
```

For example, the parameter `feature_binning` allows to provide additional options and may be configured as follows:

```text
'equal-width{bin_ratio=0.33,min_bins=2,max_bins=64}'
```
