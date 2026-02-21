(user-guide-nominal-features)=

# Nominal Features

An advantage of rule learning algorithms is their ability to naturally deal with nominal features by using operators like $=$ or $\neq$ for the conditions in a rule's body. This is in contrast to many statistical machine learning methods that cannot deal with nominal features directly. Instead, they require to apply preprocessing techniques to the data before training. Most commonly, [one-hot-encoding](https://en.wikipedia.org/wiki/One-hot) is used to convert nominal features to numerical ones. It replaces a single nominal feature with a fixed number of discrete values with several binary features that specify for each of the original values whether it applies to an example or not. Such a conversion may drastically increase the number of features in a dataset and therefore can negatively affect the complexity of a learning task.

```{image} ../../_static/feature_vector_nominal_coverage_light.svg
---
align: center
width: 18%
alt: Coverage of nominal conditions that can be created from a single threshold 
  0 using the $=$ or $\neq$ operator
class: only-light
---
```

```{image} ../../_static/feature_vector_nominal_coverage_dark.svg
---
align: center
width: 18%
alt: Coverage of nominal conditions that can be created from a single threshold 
  0 using the $=$ or $\neq$ operator
class: only-dark
---
```

To deal with nominal features, we rely on the same principles used by the {ref}`pre-sorted search algorithm <user-guide-pre-sorted>` in the case of numerical features, including the ability to use sparse representations of feature values. In the case of a nominal feature, the feature values associated with the individual training examples are not arbitrary real numbers but are limited to a predefined set of discrete values that do not necessarily correspond to a continuous range and possibly include negative values. As a result, the thresholds that potential conditions may use are not formed by averaging adjacent feature values but correspond to the discrete values associated with the available training examples. Two conditions must be evaluated for each of the values encountered by the algorithm in a sorted vector of feature values. As shown in the previous image, they use the operator $=$ and $\neq$, respectively. Whereas a condition that uses the former operator covers neighboring examples with the same value, the examples that satisfy a condition with the latter operator do not correspond to a continuous range in a sorted vector of feature values. This requires algorithmic adjustments when it comes to the aggregating statistics that correspond to examples covered by nominal conditions.

## Aggregation of Statistics (Phase I and II)

The algorithm follows the same order for processing the sorted feature values as outlined in the section {ref}`user-guide-feature-sparsity` to facilitate the use of sparse feature representations when dealing with nominal features. At first, it processes the examples associated with negative feature values. Afterward, it evaluates the conditions that result from positive feature values, and finally, in a third phase, potential conditions with zero thresholds are considered. During the first and second phase, the statistics of examples with the same feature value are aggregated individually. We denote the aggregated statistics for different feature values as $\boldsymbol{s}', \boldsymbol{s}'', \dots$. This is in contrast to the aggregation of statistics in the case of numerical features, where the statistics of all examples with negative and positive feature values are aggregated. As illustrated in the image below, the globally aggregated statistics ($\boldsymbol{s}$), which are provided to the algorithm beforehand, are used to obtain the statistics corresponding to examples that satisfy conditions using the $\neq$ operator. This requires to compute the difference between the globally aggregated statistics and the aggregated statistics of all examples associated with a particular discrete value.

```{image} ../../_static/aggregation_nominal_1_light.svg
---
align: center
width: 32%
alt: Evaluation of nominal conditions that separate examples with a particular 
  value from the remaining ones
class: only-light
---
```

```{image} ../../_static/aggregation_nominal_1_dark.svg
---
align: center
width: 32%
alt: Evaluation of nominal conditions that separate examples with a particular 
  value from the remaining ones
class: only-dark
---
```

## Aggregation of Statistics (Phase III)

During the third phase of the algorithm, special treatment is required to evaluate conditions with zero thresholds if any examples with zero feature values are available. To determine whether such examples exist, the sum of the weights of all examples that have previously been processed in the first and second phases of the algorithm is compared to the weights of all examples in a dataset or a sample thereof, as described earlier. To obtain the aggregated statistics that correspond to the examples with zero feature values, the statistics $\boldsymbol{s}', \boldsymbol{s}'', \dots$ that have been computed during the previous phases must be aggregated. We denote the resulting accumulated statistics as $\boldsymbol{s}^{*} = \boldsymbol{s}' + \boldsymbol{s}'' + \dots$. As shown in the following image, they correspond to the examples with non-zero feature values covered by a condition that uses the $\neq$ operator. To evaluate a condition that uses the $=$ operator and covers all examples with zero feature values, inaccessible by the algorithm when using a sparse feature representation, the difference between the globally aggregated statistics ($\boldsymbol{s}$) and the accumulated ones are computed.

```{image} ../../_static/aggregation_nominal_2_light.svg
---
align: center
width: 32%
alt: Evaluation of nominal conditions that separate examples with zero values 
  from the remaining ones
class: only-light
---
```

```{image} ../../_static/aggregation_nominal_2_dark.svg
---
align: center
width: 32%
alt: Evaluation of nominal conditions that separate examples with zero values 
  from the remaining ones
class: only-dark
---
```
