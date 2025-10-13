(testbed-datasets)=

# Supported Dataset Formats

The package mlrl-testbed is build in a modular fashion. This means that extensions can be used to extend its functionality, including the support for different dataset formats. In the following, the dataset formats supported by these extensions are discussed in detail.

(dataset-format-arff)=

## Attribute-Relation File Format (ARFF)

The [Attribute-Relation File Format] has been proposed by researchers from the University of Waikato, New Zealand. It is used by the [WEKA](https://ml.cms.waikato.ac.nz/weka) machine learning software developed by the same people. Support for this file format is brought to mlrl-testbed by the package [mlrl-testbed-arff](https://pypi.org/project/mlrl-testbed-arff/).

```{note}
Currently, the package [mlrl-testbed-arff](https://pypi.org/project/mlrl-testbed-arff/) is a hard dependency of [mlrl-testbed](https://pypi.org/project/mlrl-testbed/) and is therefore installed alongside it automatically. In the future, this behavior might change and the dependency might become optional.
```

### Mulan Format

By default, mlrl-testbed checks if the dataset files are present in the variant used by the [Mulan](https://github.com/tsoumakas/mulan) project. It requires two files to be present in a given directory:

1. An [ARFF](https://waikato.github.io/weka-wiki/formats_and_processing/arff_stable/) file that specifies the feature values and ground truth of the training examples.
2. An XML file that specifies the names of the outputs.

For example, the ARFF file could look like this:

```
@relation MultiLabelExample

@attribute feature_1 numeric
@attribute feature_2 numeric
@attribute feature_3 numeric
@attribute label_1 {0, 1}
@attribute label_2 {0, 1}
@attribute label_3 {0, 1}
@attribute label_4 {0, 1}
@attribute label_5 {0, 1}
```

The XML file corresponding to the ARFF file above would look like this:

```xml
<labels xmlns="http://mulan.sourceforge.net/labels">
  <label name="label_1"/>
  <label name="label_2"/>
  <label name="label_3"/>
  <label name="label_4"/>
  <label name="label_5"/>
</labels>
```

In contrast to the MEKA format discussed below, the Mulan format allows to treat any attribute in an ARFF file as an output and does not require them to be located at the start or end.

### MEKA Format

If an XML file is not provided, the program tries to parse the number of outputs from the `@relation` declaration that is contained in the ARFF file, as it is intended by the [MEKA](https://waikato.github.io/meka/) project's [dataset format](https://waikato.github.io/meka/datasets/). According to this format, the number of outputs must be specified by including the substring "-C L" in the `@relation` name, where "L" is the number of leading features in the dataset that should be treated as outputs.
