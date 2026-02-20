(boomer-methodology)=

# Methodology

```{seealso}
Most of the content in this chapter has been taken from the publication {ref}`‘Learning Gradient-boosted Multi-label Classification Rules’, Michael Rapp et al. (2020) <references-rapp2020boomer>`.
```

In the following, we discuss the methodology used by BOOMER - an algorithm for learning gradient boosted multi-output classification and regression rules. It is based on a generalization of the popular and well-researched gradient boosting framework to multivariate problems that enables minimizing decomposable and non-decomposable loss functions. Moreover, the algorithm can be considered an instantiation of the framework presented {ref}`here <user-guide-framework>`. Due to the modularity of the underlying framework, individual aspects of the algorithm are interchangeable and can be tailored to different applications and datasets by choosing a suitable implementation.

## Boosted Multi-output Rules

The BOOMER algorithm, which is discussed in the following, is concerned with learning ensembles of {ref}`probabilistic rules <user-guide-predictions>`. Like other [ensemble methods](https://en.wikipedia.org/wiki/Ensemble_learning), it focuses on the predictive performance of models at the expense of their simplicity and interpretability. We denote an ensemble that consists of $T$ additive classification functions $f_t \in \mathcal{F}$, referred to as *ensemble members*, as

```{math}
F = ( f_1, \dots, f_T ).
```

By $\mathcal{F}$ we denote the set of potential classification functions. Given an example $\boldsymbol{x}_n$, all the ensemble members provide probabilistic predictions, given as a vector of real-valued confidence scores

```{math}
\boldsymbol{\hat{p}}_n^t = f_t ( \boldsymbol{x}_n ) = ( \hat{p}_{n1}^t, \dots, \hat{p}_{nK}^t ) \in \mathbb{R}^K.
```

In classification problems, each score expresses a preference for predicting the label $\lambda_k$ as irrelevant, if $\hat{p}_k < 0$, or relevant, if $\hat{p}_k > 0$. In regression problems, the scores of all rules covering a query example are summed up to obtain the regression score to be predicted. The scores provided by individual members of an ensemble can be aggregated into a single vector of confidence scores by calculating the element-wise vector sum

```{math}
\boldsymbol{\hat{p}}_n = F ( \boldsymbol{x}_n ) = \boldsymbol{\hat{p}}_n^1 + \dots + \boldsymbol{\hat{p}}_n^T \in \mathbb{R}^K.
```

In regression problems, this vector is returned as the final prediction of the ensemble. When dealing with a classification task, this vector can subsequently be turned into a binary label vector or a vector of probability estimates.

As individual ensemble members, we use {ref}`conjunctive rules <user-guide-rules>`. Such rules can be viewed as predictive functions, where the body is a mathematical function $b : X \rightarrow \{ 0, 1 \}$ that evaluates to $1$ if a given example satisfies all conditions in the body or to $0$ if at least one of its conditions is not met. An individual condition compares the value of the $l$ -th attribute of an example to a constant, by using a relational operator, such as $=$ and $\neq$, if the attribute $A_l$ is nominal, or $\leq$ and $>$, if $A_l$ is numerical or ordinal. The head of a rule assigns a numerical score to each output. If a given example $\boldsymbol{x}$ belongs to the axis-parallel region in the feature space $\mathcal{X}$ covered by the rule, i.e., if it satisfies all conditions in the rule’s body, a vector $\boldsymbol{\hat{p}}$ is predicted. If the example is not covered, a null vector is predicted instead. Consequently, a probabilistic rule of this kind can be considered a mathematical function $f : \mathcal{X} \rightarrow \mathbb{R}^K$, defined as

```{math}
f ( \boldsymbol{x} ) = b ( \boldsymbol{x} ) \boldsymbol{\hat{p}}.
```

This is similar to the notation used by Dembczyński, Kotłowski, and Słowiński[^dembczynski2010] in the context of single-label classification. However, in a multi-output setting, we consider the head as a vector rather than a scalar to enable rules to predict for several outputs. The BOOMER algorithm is capable of producing all types of rules discussed here, including probabilistic rules with single- or multi-output heads. The latter can either be partial or complete, i.e., they assign a non-zero confidence score to several, or even all, outputs. The vector predicted by the former provides a non-zero prediction for exactly one output and assigns zeros to the others.

## Multivariate Boosting

In the following subsections, we discuss the gradient boosting framework as used by the BOOMER algorithm.

### Surrogate Loss Functions

An ensemble of additive functions $F = ( f_1, \dots, f_t )$, as previously introduced, should be trained such that the expected empirical risk with respect to a certain (surrogate) *loss function* $\ell$ is minimized. As the members of an ensemble predict numerical confidence scores rather than binary label vectors, as usually desired in {ref}`multi-label classification <user-guide-mlc>`, discrete functions, such as commonly used evaluation measures, are not suited to assess the quality of potential ensemble members during training. Instead, continuous loss functions that can be minimized in place of the actual target measure should be used as surrogates. For this purpose, in case of classification problems, we use multivariate loss functions $\ell : \{ -1, +1 \}^K \times \mathbb{R}^K \rightarrow \mathbb{R}$. In regression problems, loss functions $\ell : \mathbb{R}^K \times \mathbb{R}^K \rightarrow \mathbb{R}$ are used. These loss functions take two vectors $\boldsymbol{y}_n$ and $\boldsymbol{\hat{p}}_n$ as arguments. The former corresponds to the ground truth, whereas the latter represents the predictions of the ensemble members. In classification problems, $\boldsymbol{y}_n$ corresponds to the true labeling of an example $\boldsymbol{x}_n$. It specifies whether individual labels $\lambda_k$ are relevant ($y_{nk} = +1$) or irrelevant ($y_{nk} = -1$) to the respective example. In regression problems, it specifies the ground truth regression scores to be modeled.

Note that binary ground truth labels are often given in the form $y_{nk} \in \{ 0 , 1 \}$. Such a representation can easily be transformed into the form $y_{nk} \in \{ −1, +1 \}$ required here.

### Stage-wise Additive Modeling

Given a specific loss function $\ell$ to be optimized, we are concerned with the minimization of the regularized training objective

```{math}
---
label: objective
---
R ( F ) = \sum_{n=1}^N \ell ( \boldsymbol{y}_n, \boldsymbol{\hat{p}}_n ) + \sum_{t=1}^T \Omega ( f_t ),
```

where $\Omega$ denotes an (optional) regularization term that may be used to penalize the complexity of the individual ensemble members. It may help to avoid overfitting and ensure the convergence towards a global optimum if $\ell$ is not convex. Unfortunately, constructing an ensemble of additive functions that minimizes the objective given above is a hard optimization problem. In gradient boosting, this problem is tackled by training the model in a stage-wise procedure, where the individual ensemble members are added one after the other, as originally proposed by Friedman, Hastie, and Tibshirani.[^friedman2000] At each iteration $t$, the vector of scores that the existing ensemble members predict for an example $\boldsymbol{x}_n$ can be calculated based on the predictions of the previous iteration. We denote it as

```{math}
F_t ( \boldsymbol{x}_n ) = F_{t-1} ( \boldsymbol{x}_n ) + f_t ( \boldsymbol{x}_n ) = ( \boldsymbol{\hat{p}}_n^1 + \dots + \boldsymbol{\hat{p}}_n^{t-1} ) + \boldsymbol{\hat{p}}_n^t.
```

Substituting the additive calculation of the predictions into the objective function in {eq}`objective` yields the objective to be minimized by the ensemble member added at the $t$-th iteration. It calculates as

```{math}
---
label: iterative_objective
---
R ( f_t ) = \sum_{n=1}^N \ell ( \boldsymbol{y}_n, F_{t-1} ( \boldsymbol{x}_n ) ) + \Omega ( f_t ).
```

#### Taylor Approximation

To efficiently minimize the training objective when about to be adding a new ensemble member $f_t$, we rewrite {eq}`iterative_objective` in terms of the second-order multivariate Taylor approximation

```{math}
---
label: taylor_approximation
---
R ( f_t ) \approx \sum_{n=1}^N \left( \ell \left( \boldsymbol{y}_n, F_{t-1} ( \boldsymbol{x}_n ) \right) + \boldsymbol{g}_n \boldsymbol{\hat{p}}_n^t + \frac{1}{2} \boldsymbol{\hat{p}}_n^t H_n \boldsymbol{\hat{p}}_n^t \right) + \Omega ( f_t ),
```

where $\boldsymbol{g}_n = ( g_{n1}, \dots, g_{nK} )$ denotes the vector of first or partial derivatives of the loss function $\ell$ with respect to the existing ensemble members' predictions for a particular example $\boldsymbol{x}_n$ and individual outputs $\lambda_{k}$. Accordingly, the Hessian matrix $H_n = ((h_{n11}, \dots, h_{n1K}), \dots, (h_{nK1}, \dots, h_{nKK}))$ consists of second-order partial derivatives corresponding to pairs of outputs. We compute individual gradients and Hessians as

```{math}
g_{ni} = \frac{\partial \ell}{\partial \hat{p}_{ni}} ( \boldsymbol{y}_n, F_{t-1} ( \boldsymbol{x}_n ) ) \quad \text{and} \quad h_{nij} = \frac{\partial \ell}{\partial \hat{p}_{ni} \partial \hat{p}_{nj}} ( \boldsymbol{y}_n, F_{t-1} ( \boldsymbol{x}_n ) ).
```

#### Training Objective

By removing constant terms, {eq}`taylor_approximation` can be further simplified, resulting in the approximated training objective

```{math}
---
label: training_objective
---
\widetilde{R} ( f_t ) = \sum_{n=1}^N \left( \boldsymbol{g}_n \boldsymbol{\hat{p}}_n^t + \frac{1}{2} \boldsymbol{\hat{p}}_n^t H_n \boldsymbol{\hat{p}}_n^t \right) + \Omega ( f_t ).
```

At each training iteration, the objective function $\widetilde{R}$ can be used as a quality measure to decide which of the potential ensemble members improves the current model the most. This requires the predictions of these ensemble members for examples $\boldsymbol{x}_n$ to be known. How the predictions can be found depends on the type of ensemble members and the loss function at hand. The following subsection presents solutions to this problem when using rules as the additive functions of an ensemble.

## Induction of Rules

In the following, we outline the algorithm used by BOOMER for learning an ensemble of gradient boosted single- or multi-output rules that minimize a given loss function in expectation. It is based on the mathematical foundations introduced in the previous section and is implemented in adherence to the modular framework presented {ref}`here <user-guide-framework>`. The basic structure of the iterative procedure used for assembling an ensemble is illustrated by the following pseudocode:

```{math}
\textbf{in:}\quad & \text{Training examples } D = \{ \boldsymbol{x}_n, \boldsymbol{y}_n \}_n^N, \\
& \text{first and second derivative } \ell' \text{ and } \ell'' \text{ of the loss function,} \\
& \text{number of rules } T, \\
& \text{shrinkage parameter } \eta \\
\textbf{out:}\quad & \text{Ensemble of rules } F \\
\\
\text{1:} \quad & S = \{( \boldsymbol{g}_n, H_n ) \}_n^N = \text{calculate gradients and Hessians w.r.t.} \ell' \text{and} \ell'' \\
\text{2:} \quad & \boldsymbol{w}_1 = \text{set weights for each example to 1} \\
\text{3:} \quad & f_1 : \boldsymbol{\hat{y}}_1 \leftarrow b_1 \text{ with } b_1 ( \boldsymbol{x} ) = 1, \forall \boldsymbol{x} \text{ and } \boldsymbol{\hat{y}}_1 = \texttt{FIND\_HEAD} ( D, \boldsymbol{w}_1, S, b_1 ) \\ 
\text{4:} \quad & \textbf{for } t = 2 \textbf{ to } T \textbf{ do} \\
\text{5:} \quad & \quad S = \text{ update gradients and Hessians of examples covered by } f_{t-1} \\
\text{6:} \quad & \quad \boldsymbol{w}_t = \text{ obtain a weight for each example via instance sampling} \\
\text{7:} \quad & \quad f_t : \boldsymbol{\hat{y}}_t \leftarrow b_t = \texttt{REFINE\_RULE} ( D, \boldsymbol{w}_t, S) \\
\text{8:} \quad & \quad \boldsymbol{\hat{y}} = \texttt{FIND\_HEAD} ( D, \boldsymbol{w}_t, S, b_t ) \\
\text{9:} \quad & \quad \boldsymbol{\hat{y}} = \eta \cdot \boldsymbol{\hat{y}}_t \\
\text{10:} \quad & \textbf{return} \text{ ensemble of rules } F = \{ f_1, \dots, f_T \}
```

As discussed {ref}`here <user-guide-model-assemblage>`, rules only provide predictions for examples they cover. The first rule $f_1 : \boldsymbol{\hat{y}}1 \leftarrow b_1$ in the ensemble is a {ref}`default rule <user-guide-default-rule>` covering all examples, i.e., $b_1 ( \boldsymbol{x} ) = 1, \forall \boldsymbol{x}$. In subsequent iterations of the algorithm, more specific rules are added. All rules $f_t$, including the default rule, contribute to the final predictions of the ensemble according to the probabilistic scores assigned to individual outputs by their heads $\boldsymbol{\hat{p}}_t$. The scores are chosen such that the objective function in {eq}`training_objective` is minimized. At each iteration $t$, this requires the {ref}`output space statistics <user-guide-statistics>` $S$, consisting of gradients and Hessians, to be (re-)calculated based on the scores $\hat{p}_{nk}$ predicted by the current model for each example $\boldsymbol{x}_n$ and output $\lambda_k$, as well as the corresponding ground truth $y_{nk}$. At the first iteration, the predictions for all examples and outputs are zero, i.e., $\hat{p}_{nk} = 0, \forall n, k$ if $t = 1$. While the default rule always provides confidence scores for each output, the remaining rules may either predict for a single output, a subset of the available outputs, or all of them. The computations that are necessary to obtain loss-minimizing predictions for the default rule and each of the remaining rules are presented in the section {ref}`boomer-rule-evaluation` below.

To learn the rules $f_2, \dots, f_T$, we use a greedy top-down search, where the body is iteratively refined by adding new conditions, and the head is adjusted accordingly at each step. The algorithm used for the refinement of rules is outlined in the section {ref}`boomer-rule-refinement`. As discussed {ref}`here <user-guide-instance-sampling>`, instance sampling can be used to learn each rule on a different sample of the training examples. As more diversified and less correlated rules are learned, this reduces the variance of the ensemble members. However, once the construction of a rule has finished, its predictions are recomputed with respect to the entire training data, which we have found to be an effective countermeasure against overfitting the sample used for constructing the rule. As an additional measure to reduce the risk of fitting noise in the data, the scores predicted by a rule may be multiplied by a shrinkage parameter $\eta \in ( 0, 1 ]$. Small values for $\eta$ reduce the impact of individual rules on the overall model.

(boomer-rule-evaluation)=

### Computation of Predictions

As can be seen in the pseudocode above, the function $\texttt{FIND\_HEAD}$ is used to find optimal confidence scores to be predicted by a particular rule $f_t$, i.e., scores that minimize the objective function $\widetilde{R}$ introduced in {eq}`training_objective`. In addition, it provides an estimate of their quality. Because rules provide the same predictions for all examples they cover and abstain for others, the objective function in {eq}`training_objective` can further be simplified. We can sum up the gradient vectors and Hessian matrices that correspond to the covered examples, resulting in the objective

```{math}
---
label: simplified_objective
---
\widetilde{R} ( f_t ) = \boldsymbol{g} \boldsymbol{\hat{p}} + \frac{1}{2} \boldsymbol{\hat{p}} H \boldsymbol{\hat{p}} + \Omega ( f_t ),
```

where $\boldsymbol{g} = \sum_n ( b ( \boldsymbol{x}_n ) w_n \boldsymbol{g}_n )$ denotes the element-wise weighted sum of the gradient vectors and $H = \sum_n ( b ( \boldsymbol{x}_n ) w_n \boldsymbol{g}_n )$ corresponds to the sum of the Hessian matrices. As shown in the pseudocode below, the sums of gradients and Hessians are provided to the function $\texttt{FIND\_HEAD}$ to determine a rule's predictions and a corresponding estimate of its quality.

```{math}
\textbf{in:}\quad & \text{Sums of gradients } \boldsymbol{g} = \sum\nolimits_n ( b ( \boldsymbol{x}_n ) w_n \boldsymbol{g}_n ), \\
& \text{sums of Hessians } H = \sum\nolimits_n ( b ( \boldsymbol{x}_n ) w_n H_n ) \\
\textbf{out:}\quad & \text{Single- or multi-output head } \boldsymbol{\hat{p}}, \\
& \text{quality } q \\
\\
\text{1:} \quad & \text{Initialize } \boldsymbol{g} = \sum\nolimits_n b ( \boldsymbol{x}_n ) w_n \boldsymbol{g}_n \text{ and } H = \sum\nolimits_n b ( \boldsymbol{x}_n ) w_n H_n \\ 
\text{2:} \quad & \textbf{if} \text{ loss is decomposable } \textbf{or} \text{ searching for a single-output head } \textbf{then} \\
\text{3:} \quad & \quad \boldsymbol{\hat{p}} = \text{ obtain } \hat{p}_k \text{ w.r.t. } \boldsymbol{g} \text{ and } H \text{ for each output} \\
\text{4:} \quad & \quad \textbf{if} \text{ searching for a single-output head } \textbf{then} \\
\text{5:} \quad & \quad\quad \boldsymbol{\hat{p}} = \text{ find best single-output prediction } \hat{p}_k \in \boldsymbol{\hat{p}} \\
\text{6:} \quad & \textbf{else} \\
\text{7:} \quad & \quad\quad \boldsymbol{\hat{p}} = \text{ obtain } ( \hat{p}_1, \dots, \hat{p}_K ) \text{ w.r.t. } \boldsymbol{g} \text{ and } H \\
\text{8:} \quad & q = \text{evaluate w.r.t. } \boldsymbol{g}, H \text{ and } \boldsymbol{\hat{p}} \\
\text{9:} \quad & \textbf{return} \text{ head } \boldsymbol{\hat{p}}, \text{ quality } q
```

#### Regularization

To penalize extreme predictions, BOOMER can (optionally) use the $L_2$ regularization term

```{math}
\Omega_{L2} ( f_t ) = \frac{1}{2} \delta || \boldsymbol{\hat{p}}^t ||_2^2,
```

where $|| \boldsymbol{x} ||_2 = \sum_i x_i^2$ denotes the [Euclidean norm](<https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm>) and $\delta \geq 0$ is a hyperparameter that controls the weight of the regularization term.

In addition, $L_1$ regularization is also supported. It uses the term

```{math}
\Omega_{L1} ( f_t ) = \gamma || \boldsymbol{\hat{p}}^t ||_1,
```

where $|| \boldsymbol{x} || = \sum_i | x_i |$ denotes the [Taxicab or Manhattan norm](<https://en.wikipedia.org/wiki/Norm_(mathematics)#Taxicab_norm_or_Manhattan_norm>) and $\gamma \geq 0$ controls the weight of the term.

#### System of Linear Equations

To ensure that predictions $\boldsymbol{\hat{p}}$ minimize the regularized training objective $\widetilde{R}$, we equate the first partial derivative of {eq}`simplified_objective` with respect to $\boldsymbol{\hat{p}}$ with zero:

```{math}
---
label: linear_system
---
\frac{\partial \widetilde{R}}{\partial \boldsymbol{\hat{p}}} ( f_t ) = & (\boldsymbol{g} + \boldsymbol{r}) + ( H + R) \boldsymbol{\hat{p}} = 0 \\
\Longleftrightarrow & ( H + R ) \boldsymbol{\hat{p}} = - ( \boldsymbol{g} + \boldsymbol{r} ),
```

where $\boldsymbol{r} = ( r_1, \dots, r_K )$ is a vector with $r_k = -\gamma$ if $g_k > \gamma$, $\gamma$ if $g_k < -\gamma$, or $0$ otherwise. $R = \text{diag} ( \delta )$ is a diagonal matrix with the elements on the diagonal set to the value $\delta$. Equation {eq}`linear_system` can be considered a system of $K$ linear equations, where $H + R$ is a matrix of coefficients, $-( \boldsymbol{g} + \boldsymbol{r} )$ is a vector of ordinates and $\boldsymbol{\hat{p}}$ is the vector of unknowns to be determined. For commonly used loss functions, the sums of Hessians $h_{ij}$ and $h_{ji}$ are equal. Consequently, the matrix of coefficients is symmetrical.

#### Closed-form Solution

In the general case, i.e., if the loss function is non-decomposable, the linear system in {eq}`linear_system` must be solved to determine the optimal multi-output head $\boldsymbol{\hat{p}}$. However, when dealing with a decomposable loss function, the first and second derivative with respect to a particular element $\hat{p}_i \in \boldsymbol{\hat{p}}$ is independent of any other element $\hat{p}_j \in \boldsymbol{\hat{p}}$. This causes the sums of Hessians $H_{ij}$ that do not exclusively depend on $\hat{p}_i$, i.e., if $i \neq j$, to become zero. In such a case, the linear system reduces to compute the optimal prediction $\hat{p}_i$ for the $i$-th output via the closed-form solution

```{math}
---
label: closed_form
---
\hat{p}_i = - \frac{g_i + r_i}{h_{ii} + \delta},
```

where $r_i = -\gamma$ if $g_i > \gamma$, $\gamma$ if $g_i < -\gamma$, or $0$ otherwise.

Similarly, when interested in single-output rules that predict for the $i$-th output, the predictions $\hat{p}_j$ with $j \neq i$ are known to be zero because the rule will abstain for the corresponding outputs. Consequently, {eq}`closed_form` can be used to determine the predictions of single-output rules event if the loss function is non-decomposable.

(boomer-rule-refinement)=

### Refinement of Rules

To learn a new rule, we use a greedy top-down search, also referred to as top-down hill climbing[^fuernkranz2012], as previously mentioned in the section {ref}`user-guide-rule-induction`. The pseudocode below is meant to outline the general procedure of such a search algorithm. For brevity, it does not include any algorithmic optimizations that can drastically improve the computational efficiency in practice.

```{math}
\textbf{in:}\quad & \text{Training examles } D = \{ ( \boldsymbol{x}_n, \boldsymbol{y}_n ) \}_n^N, \\
& \text{weights } \boldsymbol{w}, \\
& \text{statistics } S = \{ ( \boldsymbol{g}_n, H_n ) \}_n^N, \\
& \text{current rule} f \text{ and its quality } q \text{ (both optional)} \\
\textbf{out:}\quad & \text{Best rule } f^* \\
\\
\text{1:} \quad & \text{Initialize best rule } f^* = f \text{ and best quality } q^* = q \\
\text{2:} \quad & A' = \text{ select a random subset of features from } D \\ 
\text{3:} \quad & \textbf{foreach} \text{ possible condition } c \text{ on features } A' \text{ and examples } D \textbf{ do} \\
\text{4:} \quad & \quad f' : b' \rightarrow \boldsymbol{\hat{p}} = \text{ copy of current rule } f \\
\text{5:} \quad & \quad \text{add condition } c \text{ to body } b' \\
\text{6:} \quad & \quad \text{Calculate } \boldsymbol{g} = \sum\nolimits_n ( b ( \boldsymbol{x}_n ) w_n \boldsymbol{g}_n ) \text{ and } H = \sum\nolimits_n ( b ( \boldsymbol{x}_n ) w_n H_n ) \\
\text{7:} \quad & \quad \text{head } \boldsymbol{\hat{p}}, \text{ quality } q' = \texttt{FIND\_HEAD} ( \boldsymbol{g}, H ) \\
\text{8:} \quad & \quad \textbf{if } q' < q^* \textbf{ then} \\
\text{9:} \quad & \quad\quad \text{update best rule } f^* = f' and its quality q^* = q' \\
\text{10:} \quad & \textbf{if } f^* \neq f \textbf{ then} \\
\text{11:} \quad & \quad D' = \text{ subset of } D \text{ covered by } f^* \\
\text{12:} \quad & \quad \textbf{return } \texttt{REFINE\_RULE} ( D', \boldsymbol{w}, S, f^*, q^* ) \\
\text{13:} \quad & \textbf{return} \text{ best rule } f^*
```

The search for a new rule starts with an empty body that is successively refined by adding additional conditions. Adding conditions to its body causes a rule to become more specific and results in fewer examples being covered. The conditions, which may be used to refine an existing body, result from the feature values of the training examples in the case of nominal features or from averaging adjacent values in the case of numerical attributes. In addition to instance sampling, we use {ref}`feature sampling <user-guide-feature-sampling>` to select a subset of the available features whenever a new condition should be added. This leads to more diverse ensembles of rules and reduces the computational costs by limiting the number of potential candidate rules. For each condition that may be added to the current body at a particular iteration, the head of the rule is updated via the function $\texttt{FIND\_HEAD}$ discussed in the previous section. When learning single-label rules and if not configured otherwise, each refinement of the current rule is obliged to predict for the same label (omitted in the code above for brevity). Among all refinements, the one that minimizes the regularized objective in {eq}`training_objective` is chosen. If no refinement results in an improvement according to said objective, the refinement process stops. By default, no additional stopping criteria are used, and therefore they are omitted from the code above.

[^dembczynski2010]: Krzysztof Dembczyński, Wojciech Kotłowski, and Roman Słowiński (2010). ‘ENDER: A statistical framework for boosting decision rules’. In: *Data Mining and Knowledge Discovery* 21.1, pp. 52–90.

[^friedman2000]: Jerome H. Friedman, Trevor Hastie, and Robert Tibshirani (2000). ‘Additive Logistic Regression: A Statistical View of Boosting’. In: *The Annals of Statistics* 28.2, pp. 337–407.

[^fuernkranz2012]: Johannes Fürnkranz, Dragan Gamberger, Nada Lavrač (2012). ‘Foundations of Rule Learning’. *Springer Science & Business Media*.
