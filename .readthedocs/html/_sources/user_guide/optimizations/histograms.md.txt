(user-guide-histograms)=

# Histogram-based Search

The {ref}`exploitation of feature sparsity <user-guide-feature-sparsity>` helps reduce training times on many benchmark datasets, as they often come with high feature sparsity. However, it does not provide significant advantages on datasets with low feature sparsity. Our algorithms provide an alternative to the {ref}`pre-sorted search algorithm <user-guide-pre-sorted>` to efficiently deal with the latter type of datasets. It is based on assigning examples with similar values for a particular feature to a predefined number of bins and using an aggregated representation of their corresponding label space statistics, referred to as *histograms*. Depending on how many bins are used, this approach drastically reduces the number of candidates the rule induction algorithm must consider. Histogram-based approaches have previously been used to deal with complex classification tasks in modern implementations of gradient boosted decision trees, such as [XGBoost](https://github.com/dmlc/xgboost)[^chen2016] or [LightGBM](https://github.com/microsoft/LightGBM)[^ke2017]. In the following, we discuss a generalization of the underlying concept, which has evolved from prior research on decision tree learning[^alsabti1998][^jin2003][^li2007][^kamath2002], to rule learning methods.

## Assigning Examples to Bins

A histogram-based rule induction algorithm requires grouping the available training examples into a predefined number of bins. Different approaches can principally be used to determine such a mapping[^kotsiantis2006]. We restrict ourselves to unsupervised binning methods, where the assignment is solely based on the feature values of the training examples. This is in contrast to supervised methods, such as the *weighted quantile sketch* approach that originates from the [XGBoost](https://github.com/dmlc/xgboost) algorithm[^chen2016], where information about the true class labels of individual examples, or even their label space statistics, are taken into account. Compared to approaches that utilize the label space statistics map from examples to bins, unsupervised binning methods can usually be implemented more efficiently. This is because a mapping solely based on feature values remains unchanged for the entire training process, whereas the statistics for individual examples are subject to change and require adjusting the mapping whenever a model is refined.

### Equal-width Feature Binning

The first binning method that we consider for our experiments is referred to as *equal-width binning*. This method, which is commonly used to discretize numerical feature values, is based on dividing the range of values for a particular feature into equally-sized intervals, such that the absolute difference between the smallest and largest value in each bin are the same. Given a predefined number of bins $B$, the maximum difference between the values that are assigned to an interval calculates as

```{math}
w = \frac{\textit{max} - \textit{min}}{B},
```

where $\textit{min}$ and $\textit{max}$ denote the largest and smallest value in a bin, respectively. Based on the value $w$, a mapping $\sigma: \mathbb{R} \rightarrow \mathbb{N}^{+}$ from individual feature values $x_{n}$ to the index of the corresponding bin can be obtained as

```{math}
\sigma_{\textit{eq.-width}} ( x_{n} ) = \text{min} ( \lfloor \frac{x_{n} - \textit{min}}{w} \rfloor + 1, B ).
```

### Equal-frequency Feature Binning

Another well-known method to discretize numerical features is *equal-frequency binning*. Unlike equal-width binning, which is supposed to result in bins with values close to each other, this particular discretization method aims to obtain bins that contain approximately the same number of values. The available examples are first sorted in ascending order by their respective feature values to determine the bins for a particular feature. This results in a sorted vector of feature values $( x_{\tau ( 1 )}, \dots, x_{\tau ( N )} )$, where the permutation function $\tau ( i )$ specifies the index of the example that corresponds to the $i$-th element in the sorted vector. Afterward, the sorted values are divided into a predefined number of intervals, such that each bin contains the same number of values. Given an individual feature value $x_{n}$, the index of the corresponding bin calculates as

```{math}
\sigma_{\textit{eq.-freq.}} ( x_{n} ) = \lfloor \tau ( n ) - 1 \rfloor + 1.
```

In practice, examples with identical feature values should be prevented from being assigned to different bins. However, for reasons of brevity, this is omitted from the above formula.

### Assigning Discrete Values to Bins

To handle datasets that do not only include numerical feature values, but also come with nominal features, we use an appropriate binning method to deal with the latter. It creates a bin for each discrete value encountered in the training data and assigns examples with identical values to the same bin.

## Enumeration of Thresholds

We denote the set of example indices that have been assigned to the $b$-th bin via a mapping function $\sigma$ as

```{math}
\mathcal{B}_{b} = \{ n \in \{ 1, \dots, N \} \rvert \sigma ( x_{n} ) = b \}.
```

Given $B$ bins previously created for a particular feature, one can obtain $B - 1$ thresholds that the conditions of potential candidate rules may use. Depending on whether the $\leq$ or $>$ operator is used by a condition, the $b$-th threshold separates the examples that correspond to the bins $\mathcal{B}_{1}, \dots, \mathcal{B}_{b}$ from the examples that have been assigned to the bins $\mathcal{B}_{b + 1}, \dots, \mathcal{B}_{B}$. The individual thresholds $t_{1}, \dots, t_{B - 1}$ calculate as the average of the largest and smallest feature value in two neighboring bins $\mathcal{B}_{b}$ and $\mathcal{B}_{b + 1}$. Depending on the characteristics of the binning method at hand, some bins may remain empty. For the enumeration of potential thresholds, bins that are not associated with any examples should be ignored. When dealing with bins that have been created from nominal feature values, all examples in a particular bin have the same feature value. In such a case, the conditions in a rule's body may test for presence or absence of these $B$ feature values.

## Creation of Histograms

When using unsupervised binning methods, the mapping of examples to bins and the thresholds resulting from individual bins must only be determined once during training. They should be obtained when a particular feature is considered by the rule induction algorithm for the first time and should be kept in memory for repeated access. In contrast, the histograms that serve as a basis for evaluating candidate rules must be created from scratch whenever a rule should be refined. As shown in the pseudocode below, they result from aggregating the label space statistics of examples that have been assigned to the same bin. Examples that do not satisfy the conditions that have previously been added to the body of a rule must be ignored. We use an {ref}`indicator function <user-guide-indicator-function>` $I$ to keep track of the examples that are covered by a rule. In addition, the extent to which the statistics of individual training examples contribute to a histogram depends on their respective weights. This enables the histogram-based search algorithm to use different samples of the available training examples to induce individual rules.

```{math}
\textbf{in:}\quad & \text{Bins } (\mathcal{B}_{b})_b^B, \text{ statistics } S = \{ \boldsymbol{s} \}_n^N, \\
& \text{ indicator function } I, \text{ weights of training examples } \boldsymbol{w} \\
\\
\text{1:} \quad & \text{initialize empty histogram } S' = \{ \boldsymbol{s}'_{b} \}_b^B, \text{ where all elements of } \boldsymbol{s}'_b \text{ are set to zero} \\
\text{2:} \quad & \textbf{for } n = 1 \textbf{ to } N \textbf{ do} \\
\text{3:} \quad & \quad \textbf{if } I ( n ) = 1 \textbf{ and } w_{n} > 0 \textbf{ then} \\
\text{4:} \quad & \quad \quad \text{obtain bin index } b = \sigma ( x_n ) \\
\text{5:} \quad & \quad \quad \text{update } S' \text{ by setting } \boldsymbol{s}'_b = \boldsymbol{s}'_b + w_n \boldsymbol{s}_n = \sigma ( x_n ) \\
\text{6:} \quad & \textbf{return} \text{ histogram } S'
```

## Evaluation of Refinements

When using a histogram-based search algorithm, evaluating candidate rules in terms of a given loss function, in case of the {ref}`BOOMER algorithm <user-guide-boomer>`, or heuristic, in case of the {ref}`SeCo algorithm <user-guide-seco>`, follows the same principles as its {ref}`pre-sorted counterpart <user-guide-pre-sorted>`. However, instead of taking the feature values of individual training examples into account for making up conditions that can be added to a rule's body, the conditions to be considered by the histogram-based algorithm result from the predetermined thresholds that correspond to the bins for a particular feature. Even when an existing rule should be refined, i.e., when an existing rule covers only a subset of the training examples, the thresholds remain unchanged to increase the algorithm's efficiency. Similar to the pre-sorted rule induction algorithm, the histogram-based approach is based on incrementally aggregating the statistics of training examples that are covered by the considered refinements. However, instead of aggregating statistics at the level of individual training examples, it relies on the statistics that correspond to the individual bins of a histogram. For the efficient evaluation of conditions that use the $>$ operator in case of numerical features, or the $\neq$ operator in case of nominal features, the algorithm is provided with globally aggregated statistics that are determined beforehand and computes the difference between previously processed statistics that correspond to individual bins and the globally aggregated ones as previously discussed {ref}`here <user-guide-feature-sparsity>`. The statistics of examples with missing feature values are excluded from the globally aggregated statistics, as previously described {ref}`here <user-guide-missing-features>`. In addition, the respective examples are ignored when determining the mapping to individual bins. Consequently, the histogram-based rule induction method can handle missing feature values.

[^chen2016]: Tianqi Chen and Carlos Guestrin (2016). ‘XGBoost: A Scalable Tree Boosting System’. In: *Proc. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794.

[^ke2017]: Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu (2017). ‘LightGBM: A Highly Efficient Gradient Boosting Decision Tree’. In: *Proc. Advances in Neural Information Processing Systems* 30, pp. 3146–3154.

[^alsabti1998]: Khaled Alsabti, Sanjay Ranka, and Vineet Singh (1998). ‘CLOUDS: A Decision Tree Classifier for Large Datasets’. In: *Proc. International Conference on Knowledge Discovery and Data Mining*, pp. 2-8.

[^jin2003]: Ruoming Jin and Gagan Agrawal (2003). ‘Communication and Memory Efficient Parallel Decision Tree Construction’. In: *Proc. SIAM International Conference on Data Mining*, pp. 119-129.

[^li2007]: Ping Li, Qiang Wu, and Christopher Burges. ‘McRank: Learning to Rank Using Multiple Classification and Gradient Boosting’. In: *Advances in Neural Information Processing Systems*, 20.

[^kamath2002]: Chandrika Kamath, Erick Cantú-Paz, and David Littau (2002). ‘Approximate Splitting for Ensembles of Trees using Histograms’. In: *Proc. SIAM International Conference on Data Mining*, pp. 370--383.

[^kotsiantis2006]: Sotiris B. Kotsiantis and Dimitris Kanellopoulos (2006). ‘Discretization Techniques: A recent survey’. In: *GESTS International Transactions on Computer Science and Engineering*, 32.1, pp. 47-58.
