(pre-processing)=

# Data Pre-Processing

Depending on the dataset at hand, it might be desirable to apply pre-processing techniques to the data before training a machine learning model. The pre-processing techniques that are supported are discussed in the following. When using such a technique, it is applied to the training and test sets (see {ref}`evaluation`), before training a model and querying it for predictions, respectively.

## One-Hot-Encoding

```{warning}
When using the algorithms provided by this project, the use of one-hot-encoding is typically not advised, as they can deal with nominal and binary features in a more efficient way. However, as argued below, it might still be useful for a fair comparison with machine learning approaches that cannot deal with such features.
```

Not all machine learning methods can deal with nominal or binary features out-of-the-box. In such cases, it is necessary to pre-process the available data in order to convert these features into numerical ones. The most commonly used technique for this purpose is referred to as [one-hot-encoding](https://en.wikipedia.org/wiki/One-hot). It replaces each feature that comes with a predefined set of discrete values, with several numerical features corresponding to each of the potential values. The values for these newly added features are set to `1`, if an original data point was associated with the corresponding nominal value, or `0` otherwise. Because the resulting dataset typically entails more features than the original one, the use of one-hot-encoding often increases the computational costs and time needed for training a machine learning model.

Even though nominal and binary features are natively supported in an efficient way by all algorithms provided by this project, it might still be useful to use one-hot-encoding if one seeks for a fair comparison with machine learning approaches that cannot deal with such features. In such cases, you can provide the argument `--one-hot-encoding true` to the command line API:

````{tab} BOOMER
   ```text
   mlrl-testbed mlrl.boosting \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --one-hot-encoding true
   ```
````

````{tab} SeCo
   ```text
   mlrl-testbed mlrl.seco \
       --data-dir /path/to/datasets/ \
       --dataset dataset-name \
       --one-hot-encoding true
   ```
````

Under the hood, the program makes use of scikit-learn's {py:class}`sklearn.preprocessing.OneHotEncoder` for pre-processing the data.
