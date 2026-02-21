(user-guide-feature-sparsity)=

# Exploiting Feature Sparsity

When dealing with training data where most feature values are equal to zero (or another predominant value), using a sparse representation of the feature matrix reduces the amount of memory required for storing its elements and facilitates the implementation of algorithms that can deal with such data in a computationally efficient way. In the following, we discuss a variant of the previously introduced {ref}`pre-sorted search algorithm <user-guide-pre-sorted>` that allows to search for potential refinements of rules using both dense and sparse feature matrices. The use of sparse input data, where feature values are provided in the [compressed sparse column (CSC)](<https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)>) format, may drastically reduce training times.

## Enumeration of Thresholds

When dealing with a sparse representation of the feature matrix, the sorted column vector for each feature, which serves as a basis for enumerating possible thresholds, only contains non-zero feature values.

On the one hand, because only non-zero values must be processed, this reduces the algorithm's computational complexity. However, on the other hand, the algorithm cannot identify the examples with zero feature values. Therefore, to enumerate all thresholds that result from a sparse vector, including those that result from examples with zero feature values, a "sparsity-aware" search algorithm must follow three phases:

- **Phase I:** It starts by processing the sorted feature values in increasing order as before. Traversal of the feature values must be stopped as soon as a positive value or a value equal to zero is encountered.
- **Phase II:** Afterward, it processes the sorted feature values in decreasing order until a negative value or a value equal to zero is encountered.
- **Phase III:** After all elements in a given vector have been processed, it is possible to deal with the thresholds that eventually result from examples with zero feature values. To determine whether such examples exist, the number of elements with non-zero weights processed so far can be compared to the total number of examples in a dataset or a sample thereof. If all available examples have already been processed, a single threshold can be formed by averaging the largest negative feature value and the smallest positive one that has been encountered in the previous phases. Otherwise, two thresholds between zero and each of these values can be made up.

An example is given in the following. It shows a sparse representation of a numerical feature vector, where values that are equal to zero are omitted. The thresholds that result from averaging adjacent feature values are shown to the right. The numbers in parentheses (I, II, III) specify the phases of the sparsity-aware search algorithm that are responsible for the evaluation of individual thresholds.

```{image} ../../_static/feature_vector_sparse_light.svg
---
align: center
width: 22%
alt: Sparse representation of a numerical feature vector, where values that are 
  equal to zero are omitted
class: only-light
---
```

```{image} ../../_static/feature_vector_sparse_dark.svg
---
align: center
width: 22%
alt: Sparse representation of a numerical feature vector, where values that are 
  equal to zero are omitted
class: only-dark
---
```

An algorithm that follows the aforementioned procedure is not only able to deal with dense and sparse vector representations, but also enumerates all thresholds that are considered by the {ref}`pre-sorted search algorithm <user-guide-pre-sorted>`. However, if a large fraction of the feature values are equal to zero, it involves far less computational steps. The step-wise procedure that we use for the enumeration of thresholds resembles ideas that are used by [XGBoost](https://github.com/dmlc/xgboost)[^chen2016] for the construction of decision trees from sparse feature values. However, the latter requires to traverse the values that correspond to each feature twice, whereas our approach processes the values only once. In general, the proposed approach is not restricted to the space-efficient representation of zero values, but can omit explicit storage of any value that is predominant in the data. For simplicity, we restrict ourselves to the former.

## Aggregation of Statistics (Phase I)

The first phase of the sparsity-aware search algorithm, where examples with negative feature values are processed, is identical to the {ref}`pre-sorted search algorithm <user-guide-pre-sorted>`. During this initial phase, the aggregated statistics of examples that satisfy potential conditions are obtained as illustrated in the image above. If a condition uses the $\leq$ operator, the statistics of the examples it covers correspond to the previously aggregated statistics ($\boldsymbol{s}'$) of already processed examples. To obtain the statistics of examples covered by a condition that uses the $>$ operator, the difference ($\boldsymbol{s} - \boldsymbol{s}'$) between the globally aggregated statistics ($\boldsymbol{s}$) and the previously aggregated ones are computed. The first phase ends as soon as an example with a positive or zero feature value is encountered. The statistics that have been aggregated until this point ($\boldsymbol{s}'$) include all examples with negative feature values and are retained for later use during the third phase of the algorithm.

## Aggregation of Statistics (Phase II)

The second phase, where examples with positive feature values are considered, follows the same principles as the previous phase. However, the examples are processed in decreasing order of their respective feature values. Consequently, the incrementally aggregated statistics (denoted as $\boldsymbol{s}''$ to distinguish them from the variables used in the first phase) updated at each step correspond to the examples that cover a condition using the $>$ operator. To obtain the aggregated statistics of examples that satisfy a condition that uses the $\leq$ operator, the difference ($\boldsymbol{s} - \boldsymbol{s}''$) between the globally aggregated statistics and the previously aggregated ones must be computed. The end of the second phase is reached as soon as an example with a negative or zero feature value is encountered. The statistics aggregated during this phase ($\boldsymbol{s}''$) include the statistics of all examples with positive feature values. They are kept in memory for use during the third and final phase.

## Aggregation of Statistics (Phase III)

After the second phase has finished, the algorithm is able to decide whether any examples with zero feature values, which are neither stored by a sparse representation of the feature values nor can explicitly be accessed by the rule induction algorithm, are available. This is the case if the sum of the weights of all examples processed until this point is smaller than the total sum of weights of all examples in a dataset or a sample thereof. In any case, it is necessary to evaluate potential conditions that separate the examples with positive feature values from the remaining ones (possibly including examples with zero feature values). As shown on the left side of the image below, this requires considering two conditions with the operator $\leq$ and $>$, respectively. The statistics of examples that satisfy the latter correspond to the statistics aggregated during the algorithm's second phase ($\boldsymbol{s}''$). To obtain the statistics of examples that are covered by the former, the difference ($\boldsymbol{s} - \boldsymbol{s}''$) between the statistics aggregated during the second phase and the globally aggregated ones must be computed. In addition, if any examples with zero feature values are available, additional conditions using the operators $\leq$ and $>$ that separate the examples with negative feature values from the remaining ones must be considered. The statistics of examples that are covered by the former correspond to the statistics that have been aggregated during the first phase of the algorithm ($\boldsymbol{s}'$). In contrast, the statistics of examples that satisfy the latter must again be computed by taking the globally aggregated statistics into account. They calculate as the difference ($\boldsymbol{s} - \boldsymbol{s}'$) between the statistics corresponding to examples with negative feature values, which have been processed during the first phase, and the globally aggregated ones that have been computed beforehand. An example of how the aggregated statistics for the evaluation of these conditions are obtained is given on the right side of the image below. If no examples with zero feature values are available, the evaluation of additional conditions, as depicted in this image, can be omitted.

```{image} ../../_static/aggregation_sparse_light.svg
---
align: center
width: 80%
alt: Evaluation of conditions that separate examples with positive (left) or 
  negative (right) values from the remaining ones
class: only-light
---
```

```{image} ../../_static/aggregation_sparse_dark.svg
---
align: center
width: 80%
alt: Evaluation of conditions that separate examples with positive (left) or 
  negative (right) values from the remaining ones
class: only-dark
---
```

## Keeping Track of Covered Examples

Once the best condition among all available candidates has been added to a rule, it is necessary to keep track of the examples that are covered by the modified rule. This is crucial because additional conditions that may be added during later refinement iterations must only be created from the feature values of examples that satisfy the existing conditions. When dealing with a dense representation of feature values, the feature values of all examples can directly be accessed. Keeping track of the covered examples is straightforward in this case. However, given a sparse representation, it becomes a non-trivial task since the algorithm does not know the examples that come with zero feature values. To overcome this problem, we utilize a data structure suited to keep track of the covered examples in both cases, regardless of the feature representation used. It maintains a vector that stores a value for each example in a dataset, as well as an *indicator value*. If the value that corresponds to a certain example is equal to the indicator value, it is considered to be covered. This enables to answer queries to the indicator function $I ( x )$, as defined {ref}`here <user-guide-indicator-function>`\`, in constant time by comparing the value of the $n$-th example to the indicator value. An example of such a data structure for nine examples is shown in the image below. Initially, when a new rule does not contain any conditions yet, the indicator value and the values in the vector are all set to zero, i.e., all examples are considered to be covered by the rule. When a new condition is added to the rule, the data structure is updated by following one of the following strategies:

1. If the examples that satisfy the condition do not have zero feature values, the corresponding elements and the indicator value are both set to the total number of conditions.
2. Otherwise, the elements that correspond to uncovered examples are updated, whereas the indicator value remains unchanged.

```{image} ../../_static/coverage_mask_light.svg
---
align: center
width: 60%
alt: Visualization of the data structure that is used to keep track of the 
  examples that are covered by a rule
class: only-light
---
```

```{image} ../../_static/coverage_mask_dark.svg
---
align: center
width: 60%
alt: Visualization of the data structure that is used to keep track of the 
  examples that are covered by a rule
class: only-dark
---
```

Updating the data structure after a new condition has been found requires to take the range of examples it covers into account. If only examples with negative (or positive) feature values satisfy the respective condition, i.e., if the condition's threshold is less than (greater than or equal to) zero, and it uses the $\leq$ ($>$) operator, the corresponding values of the proposed data structure can be updated directly. In such a case, the values of covered examples and the indicator value are set to the number of conditions that are currently contained in the rule's body, marking them as covered. If a condition is satisfied by examples with zero feature values, for which the corresponding indices are unknown, the values that correspond to the uncovered examples are updated instead by setting them to the current number of conditions. However, the indicator value remains unchanged, which renders the examples that correspond to the updated values uncovered, whereas examples with unmodified values remain covered if they already satisfy the previous conditions.

[^chen2016]: Tianqi Chen and Carlos Guestrin (2016). ‘XGBoost: A Scalable Tree Boosting System’. In: *Proc. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794.
