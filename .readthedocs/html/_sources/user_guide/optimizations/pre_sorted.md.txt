(user-guide-pre-sorted)=

# Pre-Sorted Search

For the efficient {ref}`induction of single rules <user-guide-rule-induction>`, an efficient evaluation of a rule's possible refinements is crucial. Instead of evaluating each possible refinement in isolation, this can often be sped up by integrating the evaluation of multiple refinements into a single pass through the data. In this section, we discuss pre-sorting of examples as a way that works particularly well for numeric data. We will first discuss the base algorithm and subsequently show how it can be extended to deal with sparse, nominal, and missing feature values.

A pre-sorted search algorithm sorts the available training examples by their values for individual features before training starts. Afterward, the examples are repeatedly processed in this predetermined order to build a model. This idea originates from early work on the efficient construction of decision trees[^mehta1996][^shafer1996]. Due to the conceptual similarities between tree- and rule-based models, it can easily be generalized to rule learning methods. For example, it is used by *JRip*, an implementation of RIPPER[^cohen1995] that is part of the [WEKA](https://ml.cms.waikato.ac.nz/weka)[^hall2009] project, or the implementations of SLIPPER[^cohen1999] and RuleFit[^friedman2008] included in the [imodels](https://github.com/csinva/imodels)[^singh2021] package for interpretable models. Both rule- and tree-based learning approaches require enumerating the thresholds that may be used to make up nodes in a decision tree or conditions in a rule, respectively. These thresholds result from the feature values of the training examples, given in the form of a feature matrix as previously defined {ref}`here <user-guide-features>`. For each training example, it assigns a feature value to each of the available features. In the following, we restrict ourselves to numerical features before discussing means to deal with nominal features or missing feature values.

## Memory Layout

When searching for the best condition that may be added to a rule, the available features are dealt with independently. For each feature $A_{l}$ to be considered by the algorithm, the thresholds that may be used by the first condition of a rule result from a vector of feature values $( x_{1l}, \dots, x_{nl} )$ that corresponds to the $l$-th column of the feature matrix. As different features are dealt with in isolation, we usually omit the index of the respective feature for brevity. To facilitate column-wise access to the feature matrix, it should be given in the *Fortran-contiguous* memory layout, also known as *[column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order)*. The following illustration show representations of a $3 \times 3$ matrix in the Fortran-contiguous and [compressed sparse column (CSC)](<https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)>) format. The former uses a single one-dimensional array to store all values in column-major order, whereas the latter, which is better suited for sparse data, uses the three arrays:

1. The array *values* stores all non-zero values in column-major order.
2. For each value in *values*, *row_indices* stores the index of the corresponding row, starting at zero.
3. The $i$-th element in *column_indices* specifies the index of the first element in *values* and *row_indices* that belongs to the $i$-th column.

```{image} ../../_static/user_guide/optimizations/memory_layout_fortran_csc_light.svg
---
align: center
width: 60%
alt: Representation of a 3×3 matrix in the Fortran-contiguous and compressed 
  sparse column (CSC) format
class: only-light
---
```

```{image} ../../_static/user_guide/optimizations/memory_layout_fortran_csc_dark.svg
---
align: center
width: 60%
alt: Representation of a 3×3 matrix in the Fortran-contiguous and compressed 
  sparse column (CSC) format
class: only-dark
---
```

## Feature Permutation

To enumerate the thresholds for a particular feature, the elements in the corresponding column vector must be sorted in increasing order. For this purpose, we use a bijective permutation function $\tau: \mathbb{N}^{+} \rightarrow \mathbb{N}^{+}$, where $\tau ( i )$ specifies the index of the example that corresponds to the $i$-th element in the sorted vector

```{math}
---
label: sorted_feature_vector
---
( x_{\tau ( 1 )}, \dots, x_{\tau ( N )} ) \text{ with } \tau ( i ) \leq \tau ( i + 1 ), \forall i \in [ 1, N ).
```

An exemplary vector of sorted feature values, together with the corresponding thresholds, is shown below. Each of the thresholds is typically computed by averaging two adjacent feature values. Because these values do not change as additional conditions or rules should be learned, sorting the values that correspond to a particular feature is necessary only once during training, and previously sorted vectors can be kept in memory for repeated access.

```{image} ../../_static/user_guide/optimizations/feature_vector_numerical_light.svg
---
align: center
width: 18%
alt: An exemplary vector of sorted feature values, together with the 
  corresponding thresholds
class: only-light
---
```

```{image} ../../_static/user_guide/optimizations/feature_vector_numerical_dark.svg
---
align: center
width: 18%
alt: An exemplary vector of sorted feature values, together with the 
  corresponding thresholds
class: only-dark
---
```

Note, however, that this representation must not necessarily be used as the memory layout for storing the values of a particular feature. For storing ordinal and nominal feature values, our algorithm use a sparse memory layout, similar to the CSC layout presented above, for maximum efficiency.

(user-guide-indicator-function)=

## Indicator Function

If an existing rule should be refined by adding a condition to its body, only a subset of the feature values must be considered to make up potential thresholds. The subset corresponds to the examples that satisfy the existing conditions. We use an indicator function $I: \mathbb{N}^{+} \rightarrow \{ 0, 1 \}$ to check whether individual examples should be taken into account by the search algorithm:

```{math}
---
label: indicator_function
---
I ( n ) = \begin{cases}
  1 & \text{if example } \boldsymbol{x}_{n} \text{ is currently covered} \\
  0 & \text{otherwise}.
\end{cases}
```

If an example is not covered, its feature value may not be used to make up thresholds for additional conditions. A data structure that helps to keep track of the covered examples efficiently, rather than comparing the feature values of each example to the existing conditions, is presented in the section {ref}`user-guide-feature-sparsity`. This data structure also facilitates dealing with sparse feature values.

## Enumeration of Conditions

As shown in the pseudocode below, the feature values that correspond to a particular feature are processed in sorted order to enumerate the thresholds of potential conditions. When dealing with numerical features, the thresholds result from averaging adjacent feature values (cf. line 9). The calculation of thresholds is restricted to the feature values of examples that are covered according to the indicator function $I$ and have non-zero weights according to a weight vector $\boldsymbol{w}$. The weights result from the application of an (optional) sampling method (cf. lines 3 and 7). When dealing with numerical features, each threshold can be used to form two distinct conditions, using the relational operator $\leq$ or $>$, respectively.

```{math}
\textbf{in:}\quad & \text{Vector of sorted feature values } (x_{\tau( n )})_n^N, \\
& \text{quality of the current rule } q, \text{ indicator function } I, \\
& \text{weights of training examples } \boldsymbol{w}, \\
& \text{statistics } S = \{ \boldsymbol{s}_n \}_n^N, \text{ globally aggregated statistics } \boldsymbol{s}, \\
\\
\text{1:} \quad & \text{best refinement } r^* = \emptyset, \text{ best quality } q^* = q \\
\text{2:} \quad & \textbf{for } i = 1 \textbf{ to } N \textbf{ do} \\
\text{3:} \quad & \quad \textbf{if } I ( \tau (i)) = 1 \textbf{ and } w_{\tau ( i )} > 0 \textbf{ then} \\
\text{4:} \quad & \quad \quad \textbf{break} \\
\text{5:} \quad & \text{initialize sum of statistics } \boldsymbol{s}' = \boldsymbol{s}_{\tau ( i )} \\
\text{6:} \quad & \textbf{for } j = i + 1 \textbf{ to } N \textbf{ do} \\
\text{7:} \quad & \quad \textbf{if } I ( \tau (j)) = 1 \textbf{ and } w_{\tau ( j )} > 0 \textbf{ then} \\
\text{8:} \quad & \quad \quad \text{update sum } \boldsymbol{s}' = \boldsymbol{s}' + \boldsymbol{s}_{\tau (j)} \\
\text{9:} \quad & \quad \quad \text{threshold } t = \text{avg} ( x_{\tau (i)}, x_{\tau (j)} ) \\
\text{10:} \quad & \quad \quad \text{updated head } \boldsymbol{\hat{p}}', \text{ quality } q' = \texttt{FIND\_HEAD} ( \boldsymbol{s}' ) \\
\text{11:} \quad & \quad \quad \textbf{if } q' \text{ is better than } q^* \textbf{ then} \\
\text{12:} \quad & \quad \quad \quad \text{update refinement } r^* = \{ t, \leq, \boldsymbol{\hat{p}}' \} \text{ and its quality } q^* = r' \\
\text{13:} \quad & \quad \quad \boldsymbol{\hat{p}}', q' = \texttt{FIND\_HEAD} ( \boldsymbol{s} - \boldsymbol{s}' ) \\
\text{14:} \quad & \quad \quad \textbf{if } q' \text{ is better than } q^* \textbf{ then} \\
\text{15:} \quad & \quad \quad \quad \text{update refinement } r^* = \{ t, >, \boldsymbol{\hat{p}}' \} \text{ and its quality } q^* = r' \\
\text{16:} \quad & \quad \quad i = j \\
\text{17:} \quad & \textbf{return} \text{ best refinement } r^*, \text{ its quality } q^*
```

As can be seen in the image below, when a condition that uses the former operator is added to a rule, it results in all examples that correspond to the previously processed feature values being covered. In contrast, a condition that uses the latter operator covers all the other examples.

```{image} ../../_static/user_guide/optimizations/feature_vector_numerical_coverage_light.svg
---
align: center
width: 17%
alt: Coverage of numerical conditions that can be created from a single 
  threshold 1.0 using the $\leq$ or $>$ operator
class: only-light
---
```

```{image} ../../_static/user_guide/optimizations/feature_vector_numerical_coverage_dark.svg
---
align: center
width: 17%
alt: Coverage of numerical conditions that can be created from a single 
  threshold 1.0 using the $\leq$ or $>$ operator
class: only-dark
---
```

## Aggregation of Statistics

As can be seen in the lines 10 and 13 of the pseudocode given above, it is necessary to construct a head for each candidate rule that results from adding a new condition to a rule. In addition, the quality of the resulting rule must be assessed in terms of a numerical score. Both the predictions provided by a head and the estimated quality depend on the aggregated label space statistics of the covered examples. We exploit the fact that conditions using the $\leq$ operator, when evaluated in sorted order by increasing thresholds, are satisfied by a superset of the examples covered by the previous condition using the same operator but a smaller threshold. Processing the possible conditions in the aforementioned order enables the pre-sorted search algorithm to compute the aggregated statistics (corresponding to a vector of gradients and Hessians in case of the {ref}`BOOMER algorithm <user-guide-boomer>` and confusion matrices in case of the {ref}`SeCo algorithm <user-guide-seco>`) incrementally (cf. line 8). For the efficient evaluation of conditions that use the $>$ operator, the search algorithm is provided with the statistics that result from the aggregation over all training examples that are currently covered and have a non-zero weight (denoted as $\boldsymbol{s}$). The difference between the globally aggregated statistics and the previously aggregated ones ($\boldsymbol{s} - \boldsymbol{s}'$) yields the aggregated statistics of the examples covered by such a condition (cf. line 13). As the global aggregation of statistics does not depend on a particular feature, this operation must be performed only once per rule, even when searching for a rule's best refinement across multiple features. The following image provides an example of how the aggregated statistics are computed for the conditions that can be created from a single threshold.

```{image} ../../_static/user_guide/optimizations/aggregation_numerical_light.svg
---
align: center
width: 28%
alt: Aggregation of statistics depending on the coverage of conditions that use 
  the $\leq$ or $>$ operator and a single threshold 1.0
class: only-light
---
```

```{image} ../../_static/user_guide/optimizations/aggregation_numerical_dark.svg
---
align: center
width: 28%
alt: Aggregation of statistics depending on the coverage of conditions that use 
  the $\leq$ or $>$ operator and a single threshold 1.0
class: only-dark
---
```

[^mehta1996]: Manish Mehta, Rakesh Agrawal, and Jorma Rissanen (1996). ‘SLIQ: A Fast Scalable Classifier for Data Mining’. In: *Proc. International Conference on Extending Database Technology*, pp. 18-32.

[^shafer1996]: John C. Shafer, Rakesh Agrawal, and Manish Mehta (1996). ‘SPRINT: A Scalable Parallel Classifier for Data Mining’. In: *Proc. International Conference on Very Large Data Bases*, 96, pp. 544-555.

[^cohen1995]: William W. Cohen (1995). ‘Fast Effective Rule Induction’. In: *Proc. International Conference on Machine Learning (ICML)*, pp. 115–123.

[^hall2009]: Mark Hall, Eibe Frank, Geoffrey Holmes, Bernhard Pfahringer, Peter Reutemann, and Ian H. Witten (2009). ‘The WEKA Data Mining Software: An Update’. In: *SIGKDD Explorations*, 11.1, pp. 10-18.

[^cohen1999]: William W. Cohen and Yoram Singer (1999). ‘A Simple, Fast, and Effective Rule Learner’. In: *Proc. AAAI Conference on Artificial Intelligence*, pp. 335–342.

[^friedman2008]: Jerome H. Friedman and Bogdan E. Popescu (2008). ‘Predictive learning via rule ensembles’. In: *The Annals of Applied Statistics*, 2.3, pp. 916-954.

[^singh2021]: Chandan Singh, Keyan Nasseri, Yan Shuo Tan, Tiffany Tang, and Bin Yu (2021). ‘imodels: a python package for fitting interpretable models’. In: *Journal of Open Source Software*, 6.61, p. 3192.
