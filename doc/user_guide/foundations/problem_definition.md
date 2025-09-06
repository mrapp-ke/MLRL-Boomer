(user-guide-problem-definition)=

# Problem Definition

In the following subsections, we explore the problem domain addressed by the algorithms developed by this project. While discussing the different machine learning settings, these algorithms can be used in, many terms and mathematical notations used throughout this documentation are introduced.

(user-guide-classification)=

## Classification

In [machine learning](https://en.wikipedia.org/wiki/Machine_learning), [classification](https://en.wikipedia.org/wiki/Statistical_classification) refers to tasks that require the automatic assignment of objects to classes.

### Binary and Multi-class

Among the machine learning methods that deal with classification problems, [binary](https://en.wikipedia.org/wiki/Binary_classification) and [multi-class classification](https://en.wikipedia.org/wiki/Multiclass_classification) have the longest history of active research. In this documentation, we refer to these traditional classification settings as *single-label* or *single-output* problems. Algorithms that are aimed at these particular types of classification problems should be capable of assigning objects, usually referred to as *examples* or *instances*, to one out of two or several mutually exclusive classes. E.g., the examples could be text documents, which a classifier should automatically assign to one out of several predefined topics. Our classification algorithms are based on [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning), i.e., they aim to deduce a predictive model from a limited set of labeled training examples for which the true classes are known. A good model should generalize well beyond the provided observations, such that it can be used to make predictions for unseen examples. This learning paradigm is in contrast to [unsupervised](https://en.wikipedia.org/wiki/Unsupervised_learning) and [semi-supervised learning](https://en.wikipedia.org/wiki/Weak_supervision), where no labeled examples are provided to the classifier beforehand.

(user-guide-mlc)=

### Multi-label

Unlike in binary or multi-class classification, where an example always corresponds to a single class, in [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification), a single example may be associated with several class labels at the same time. For example, in the field of text classification, a text document can often not be assigned to a single category unambiguously but may belong to multiple topics, such as "Politics" and "Economy", simultaneously. Hence, the goal of a multi-label classifier is to assign a particular example to an arbitrary number of labels $\lambda_k$ out of a predefined and finite labelset $\mathcal{L} = \{\lambda_1, \dots, \lambda_K \}$. Because binary classification can be considered as a special case of multi-label classification where only a single label is available, i.e., where $\lvert \mathcal{L} \rvert = 1$, we restrict ourselves, without loss of generality, to a formal definition of multi-label classification. We specify the labels that are associated with an example in the form of a binary label vector

```{math}
\boldsymbol{𝒚} = ( 𝑦_1, \dots, 𝑦_K ) \in \mathcal{Y},
```

where each element $y_k \in \{ 0, 1 \}$ indicates whether the $k$-th label is irrelevant ($y_k = 0$) or relevant ($y_k = 1$) to the example. The label space $\mathcal{Y} = \{ 0, 1 \}^K$ denotes all possible labelings.

(user-guide-regression)=

## Regression

When dealing with [regression problems](https://en.wikipedia.org/wiki/Regression_analysis), the goal is to estimate the relationship between one or more dependent variables, which we refer to as *outputs*, and several independent variables, referred to as *features*. If only a single output is involved in a regression task, we characterize it as a *single-output* problem. In contrast, in *multi-output* regression problems, two or more outputs should be estimated simultaneously. Whereas the {ref}`SeCo algorithm <user-guide-seco>` is currently restricted to {ref}`classification problems <user-guide-classification>`, the {ref}`BOOMER algorithm <user-guide-boomer>` also supports regression problems. As it is based on [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning), it deduces a predictive model from a given set of training examples associated with regression scores to be modeled. The model can afterward be used to estimate scores for unseen examples. For simplicity, we stick with the notation introduced with regard to {ref}`(multi-label) classification problems <user-guide-mlc>`. However, in the case of regression problems, individual examples are not associated with binary label vectors but with vectors of real-valued regression scores. In this case, one has to deal with the output space $\mathcal{Y} = \mathbb{R}^K$.

(user-guide-features)=

## Feature Space

Our algorithms deal with structured tabular data, where each example can be represented by a feature vector

```{math}
\boldsymbol{x} = ( x_1, \dots, x_L ) \in \mathcal{X}
```

that assigns constant values to *numerical*, *ordinal*, or *nominal* attributes or features $A_l$ out of a predefined set $\mathcal{A} = \{ A_1, \dots, A_L \}$ inherent to the application domain at hand. $\mathcal{X} \in \mathcal{R}^L$ denotes the features space that consists of all possible feature vectors. A feature value that corresponds to a numerical attribute may be any positive or negative real number, e.g., a temperature. In the case of ordinal attributes, the values are restricted to a predefined finite set. Such categorical values, e.g., temperatures specified as either "cold", "warm" or "hot", are usually encoded by enumerating the available categories in a meaningful order. In contrast, the categorical values of nominal attributes are not subject to any order. E.g., no meaningful order can be imposed on boolean values like "true" and "false". Our algorithms also implement means to deal with missing feature values, i.e., datasets where individual elements in an example’s feature vector are unspecified.

(user-guide-models)=

## Predictive Models

The methods implemented by this project treat single- and multi-label classification, as well as single- and multi-output regression, as a supervised learning problem, i.e., a model is fit to the examples in a given training dataset

```{math}
\mathcal{D} = \{ ( x_n, y_n ) | 1 \leq n \leq N \} \subset \mathcal{X} \times \mathcal{Y}
```

for which the true labels, in case of classification problems, or target scores, in the case of regression problems, are known. In general, we refer to this information as the *ground truth*. The goal is to learn a model $f : \mathcal{X} \rightarrow \mathcal{Y}$ that maps from the feature space $\mathcal{X}$ to the output space $\mathcal{Y}$. A model of this kind can be considered as a predictive function that provides a prediction $\boldsymbol{\hat{y}} = f ( x )$ for any given example. In this documentation, we denote the binary label vector or vector of real-valued regression scores that is predicted by a model as

```{math}
\boldsymbol{\hat{y}} = ( \hat{y}_1, \dots, \hat{y}_K ) \in \mathcal{Y}.
```

Instead of providing a prediction in the form of a binary label vector, multi-label classifiers may also deliver a ranking of the available labels, where the labels are ordered by their relevance for a particular example. The {ref}`BOOMER algorithm <user-guide-boomer>` achieves this by predicting a vector of real-valued scores $\boldsymbol{\hat{y}} \in \mathbb{R}^K$, where elements with larger values represent labels with greater relevance. Alternatively, this algorithm does also allow to assess the relevance or irrelevance of individual labels in terms of probabilities $\boldsymbol{\hat{y}} \in [0, 1]^K$. These real-valued representations can subsequently be turned into binary predictions, e.g., by applying a threshold that separates relevant labels from irrelevant ones.
