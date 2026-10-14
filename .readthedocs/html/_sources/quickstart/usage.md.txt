(usage)=

# Using the Python API

The BOOMER algorithm and the SeCo algorithm provided by this project are published as the packages [mlrl-boomer](https://pypi.org/project/mlrl-boomer/) and [mlrl-seco](https://pypi.org/project/mlrl-seco/), respectively (see {ref}`installation`). The former is implemented by the classes {py:class}`mlrl.boosting.BoomerClassifier <mlrl.boosting.learners.BoomerClassifier>` and {py:class}`mlrl.boosting.BoomerRegressor <mlrl.boosting.learners.BoomerRegressor>`, whereas the latter is implemented by the class {py:class}`mlrl.seco.SeCoClassifier <mlrl.seco.learners.SeCoClassifier>`. All of these classes follow the conventions of a scikit-learn [estimator](https://scikit-learn.org/stable/glossary.html#term-estimators). Therefore, they can be used similarly to other machine learning methods that are included in this popular framework. The [getting started guide](https://scikit-learn.org/stable/getting_started.html) that is provided by the scikit-learn developers is a good starting point for learning about the framework's functionalities and how to use them.

## Fitting an Estimator

Whereas the SeCo algorithm is restricted to {ref}`classification problems <user-guide-classification>`, the BOOMER algorithm can also be used for solving {ref}`regression problems <user-guide-regression>`. In the following, we demonstrate the use of these algorithms.

For simplicity, the following examples use small, hard-coded data matrices as inputs to the algorithms. In practice, one usually retrieves the data from files rather than manually specifying the values of the feature and ground truth matrices. Information about supported datasets can be found {ref}`here <testbed-datasets>`.

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
- A one- or two-dimensional, binary label matrix `y`, where each row corresponds to a training example and each column corresponds to a label. If an element in the matrix is unlike zero, it indicates that the respective label is relevant to an example. Elements that are equal to zero denote irrelevant labels. In multi-label classification, where each example may be associated with several labels, the label matrix is two-dimensional. However, the algorithms are also capable of dealing with traditional binary classification problems, where a one-dimensional vector of ground truth labels is provided to the learning algorithm.

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

In addition to dense matrices like [numpy arrays](https://numpy.org/doc/stable/reference/generated/numpy.array.html), the algorithms also support to use [scipy sparse matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html). If certain cases, where the feature matrices consists mostly of zeros (or any other value), this can require significantly fewer amounts of memory and may speed up training. Sparse matrices can be provided to the `fit` method via the arguments `x` and `y` just as before. Optionally, the value that should be used for sparse elements in the feature matrix `x` can be specified via the keyword argument `sparse_feature_value`:

```python
clf.fit(x, y, sparse_feature_value = 0.0)
```

### Nominal and Ordinal Features

The algorithms provided by this project are capable of dealing with nominal and ordinal features. In both cases, the corresponding feature values are expected to be integers. Unlike ordinal and numerical (real-valued) feature values, nominal feature values (including binary ones) cannot be sorted. If nominal or ordinal features are present in a dataset, it is necessary to inform the algorithms about these features. Otherwise, they will be treated as numerical ones. As can be seen in the following, the keyword arguments `ordinal_feature_indices` and `nominal_feature_indices` are meant to be used for specifying the indices of ordinal and nominal features, respectively:

```python
clf.fit(x, y, nominal_feature_indices=[0, 2], ordinal_feature_indices=[1])
```

### Custom Weights for Training Examples

By default, all training examples have identical weights. This means that incorrect predictions for each of these examples are penalized in the same way by the training algorithm. However, in some use cases, e.g., when dealing with imbalanced data, it might be desirable to penalize incorrect predictions for some examples more heavily than for others. For this reason, it is possible to provide arbitrary (positive) integer- or real-valued weights to an algorithm's `fit`-method via the keyword argument `sample_weight`:

```python
clf.fit(x, y, sample_weight=[1.5, 1])
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

(usage-prediction)=

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

The argument `x` that must be passed to the `predict` method, has the same semantics as for the `fit` method. It can either be a [numpy array](https://numpy.org/doc/stable/reference/generated/numpy.array.html), an equivalent [array-like](https://scikit-learn.org/stable/glossary.html#term-array-like) data type, or a [scipy sparse matrix](https://docs.scipy.org/doc/scipy/reference/sparse.html). In the latter case, the value that should be used for sparse elements in the feature matrix `x` can be specified via the keyword argument `sparse_feature_value`:

```python
pred = clf.predict(x, sparse_feature_value = 0.0)
```

By default, the data type of the ground truth is also used for the predictions. If a different type should be used, it can be specified via the keyword argument `dtype`:

```python
import numpy as np

pred = clf.predict(x, dtype=np.float32)
```

### Predicting Probabilities

As a probabilistic machine learning method, the {py:class}`mlrl.boosting.BoomerClassifier <mlrl.boosting.learners.BoomerClassifier>` is capable of predicting probability estimates. These probabilities can be obtained by invoking a previously fitted estimator's `predict_proba` method:

```python
pred = clf.predict_proba(x)
print(pred)
```

In case of a multi-label classification problem, the probabilities are given as a matrix, where each row and column corresponds to a query example and label, respectively. Each value in the matrix specifies the probability of the respective label being relevant to an example. Furthermore, as shown in the following example, all values are in the range \[0, 1\]:

```python
[[0.98 0.23]
 [0.19 0.84]]
```

Following scikit-learn conventions, when dealing with single-label problems, the output format differs from the format used above. In the single-label case, a matrix with two columns is returned. The values in these columns correspond to the probability of the positive or negative class being correct for an example, respectively.

Finally, it should be noted that the `predict_proba` function supports the same keyword arguments as {ref}`previously described <usage-prediction>` with regard to the `predict` function.

### Predicting Scores

In addition to probabilities, the {py:class}`mlrl.boosting.BoomerClassifier <mlrl.boosting.learners.BoomerClassifier>` may also provide real-valued scores as predictions. These scores are the raw predictions from which binary predictions and probabilities are derived. As shown in the example below, scores can be obtained from a fitted estimator via the function `decision_function`:

```python
pred = clf.decision_function(x)
print(pred)
```

The scores are given as a matrix, where each row and column corresponds to a query example and label. As shown in the example below, a single score is in the range \[$-\infty$, $+\infty$\]. It indicates whether the corresponding label is likely to be relevant to an example, if it is positive, or irrelevant, if it is negative. Moreover, the absolute value of a score corresponds to the confidence of the model being correct.

```python
[[ 4.62 -2.48]
 [-1.92  3.34]]
```

The function `decision_function` supports the same keyword arguments as the `predict` function discussed {ref}`earlier <usage-prediction>`.

## Accessing the Rules in a Model

In some cases it might be desirable to access the rule in a model that has been learned via the `fit` method. For this purpose, we provide a convenient API that is illustrated in the following example:

```python
clf = clf.fit(x, y)

for rule in clf.model_:
    for condition in rule.body:
        print(f'{condition.feature_index}, {condition.comparator} {condition.threshold}')
    
    for prediction in rule.head:
        print(f'{prediction.output_index} {prediction.value}')
```

For details, we refer to the API reference of the classes {py:class}`RuleModel <mlrl.common.cython.rule_model.RuleModel>`, {py:class}`Rule <mlrl.common.cython.rule_model.Rule>`, {py:class}`Condition <mlrl.common.cython.rule_model.Condition>` and {py:class}`Prediction <mlrl.common.cython.rule_model.Prediction>`.
