(usage)=

# Using the Python API

The BOOMER algorithm and the SeCo algorithm provided by this project are published as the packages [mlrl-boomer](https://pypi.org/project/mlrl-boomer/) and [mlrl-seco](https://pypi.org/project/mlrl-seco/), respectively (see {ref}`installation`). The former is implemented by the classes {py:class}`mlrl.boosting.BoomerClassifier <mlrl.boosting.boosting_learners.BoomerClassifier>` and {py:class}`mlrl.boosting.BoomerRegressor <mlrl.boosting.boosting_learners.BoomerRegressor>`, whereas the latter is implemented by the class {py:class}`mlrl.seco.SeCoClassifier <mlrl.seco.seco_learners.SeCoClassifier>`. All of these classes follow the conventions of a scikit-learn [estimator](https://scikit-learn.org/stable/glossary.html#term-estimators). Therefore, they can be used similarly to other machine learning methods that are included in this popular framework. The [getting started guide](https://scikit-learn.org/stable/getting_started.html) that is provided by the scikit-learn developers is a good starting point for learning about the framework's functionalities and how to use them.

## Fitting an Estimator

Whereas the SeCo algorithm is restricted to [classification](https://en.wikipedia.org/wiki/Multi-label_classification) problems, the BOOMER algorithm can also be used for solving [regression](https://en.wikipedia.org/wiki/Regression_analysis) problems. In the following, we demonstrate the use of these algorithms.

### Classification Problems

An illustration of how the classification algorithms can be fit to exemplary training data is shown in the following:

````{tab} BOOMER
   ```python
   from mlrl.boosting import BoomerClassifier

   clf = BoomerClassifier()  # Create a new estimator
   x = [[  1,  2,  3],  # Two training examples with three features
        [ 11, 12, 13]]
   y = [[1, 0],  # Ground truth labels of each training example
        [0, 1]]
   clf.fit(x, y)
   ```
````

````{tab} SeCo
   ```python
   from mlrl.seco import SeCoClassifier

   clf = SeCoClassifier()  # Create a new estimator
   x = [[  1,  2,  3],  # Two training examples with three features
        [ 11, 12, 13]]
   y = [[1, 0],  # Ground truth labels of each training example
        [0, 1]]
   clf.fit(x, y)
   ```
````

The `fit` method accepts two inputs, `x` and `y`:

- A two-dimensional feature matrix `x`, where each row corresponds to a training example and each column corresponds to a particular feature.
- An one- or two-dimensional binary feature matrix `y`, where each row corresponds to a training example and each column corresponds to a label. If an element in the matrix is unlike zero, it indicates that the respective label is relevant to an example. Elements that are equal to zero denote irrelevant labels. In multi-label classification, where each example may be associated with several labels, the label matrix is two-dimensional. However, the algorithms are also capable of dealing with traditional binary classification problems, where an one-dimensional vector of ground truth labels is provided to the learning algorithm.

Both, `x` and `y`, are expected to be [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html) or equivalent [array-like](https://scikit-learn.org/stable/glossary.html#term-array-like) data types.

### Regression Problems

For solving regression problems rather than classification problems, as shown above, the BOOMER algorithm can be used as follows:

```python
from mlrl.boosting import BoomerRegressor

clf = BoomerRegressor()  # Create a new estimator
x = [[  1,  2,  3],  # Two training examples with three features
     [ 11, 12, 13]]
y = [[0.34, -1.20],  # Ground truth scores of each training example
     [1.43,  0.78]]
clf.fit(x, y)
```

The arguments that must be passed to the `fit` method are similar to the ones used in classification problems and are expected to be [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html) or equivalent [array-like](https://scikit-learn.org/stable/glossary.html#term-array-like) data types: `x` is a feature matrix and `y` is a one- or two-dimensional ground truth matrix, where each row corresponds to a training example and each column corresponds to a numerical output variable to predict for. The algorithm supports to predict for a single output variable or multiple ones.

### Using Sparse Matrices

In addition to dense matrices like [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html), the algorithms also support to use [scipy sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html). If certain cases, where the feature matrices consists mostly of zeros (or any other value), this can require significantly less amounts of memory and may speed up training. Sparse matrices can be provided to the `fit` method via the arguments `x` and `y` just as before. Optionally, the value that should be used for sparse elements in the feature matrix `x` can be specified via the keyword argument `sparse_feature_value`:

```python
clf.fit(x, y, sparse_feature_value = 0.0)
```

### Setting Algorithmic Parameters

In the previous example the algorithms' default configurations are used. However, in many cases it is desirable to adjust the algorithmic behavior by providing custom values for one or several of the algorithm's parameters. This can be achieved by passing the names and values of the respective parameters as constructor arguments:

````{tab} BOOMER
   ```python
   clf = BoomerClassifier(max_rules=100, loss='logistic_example_wise')
   clf = BoomerRegressor(max_rules=100, loss='squared_error_example_wise')
   ```
````

````{tab} SeCo
   ```python
   clf = SeCoClassifier(max_rules=100, heuristic='m-estimate')
   ```
````

A description of all available parameters is available for both, the {ref}`BOOMER<parameters>` and the {ref}`SeCo<seco-parameters>` algorithm.

## Making Predictions

Once an estimator has been fitted to the training data, its `predict` method can be used to obtain predictions for previously unseen examples:

```python
pred = clf.predict(x)
print(pred)
```

In this example, we use the estimator to predict for the same data that has previously been used for training. In case of the classification problem shown above, this results in the original ground truth labels to be printed:

```python
[[1 0]
 [0 1]]
```

In practice, one usually retrieves the data from files rather than manually specifying the values of the feature and ground truth matrices. A collection of benchmark datasets can be found [here](https://github.com/mrapp-ke/Boomer-Datasets).

The argument `x` that must be passed to the `predict` method, has the same semantics as for the `fit` method. It can either be a [numpy array](https://numpy.org/doc/stable/reference/generated/numpy.array.html), an equivalent [array-like](https://scikit-learn.org/stable/glossary.html#term-array-like) data type, or a [scipy sparse matrix](https://docs.scipy.org/doc/scipy/reference/sparse.html). In the latter case, the value that should be used for sparse elements in the feature matrix `x` can be specified via the keyword argument `sparse_feature_value`:

```python
pred = clf.predict(x, sparse_feature_value = 0.0)
```
