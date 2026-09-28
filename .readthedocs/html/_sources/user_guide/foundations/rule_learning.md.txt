(user-guide-rule-learning)=

# Rule Learning Algorithms

Rule-based machine learning models are a popular approach in symbolic learning with a long history of active research. For example, Fürnkranz, Gamberger, and Lavrač[^fuernkranz2012] provide a broad overview of the topic. In the following, we discuss the fundamentals of rule-based approaches that are important for understanding the algorithms provided by this project.

## Rule-based Models

Rule-based models express domain knowledge in terms of conditional clauses that refer to attributes present in the data. Each rule consists of a *body* and a *head*. Whereas the former consists of a set of *conditions*, the latter provides predictions. In accordance with existing work on the topic, we use the notation

```{math}
Head \leftarrow Body
```

for the representation of rules. We mostly use conjunctive rules, where the body is given as a conjunction of several conditions. However, there are also approaches that make use of both, logical OR (∨) and AND (∧) operators, for the concatenation of conditions in a rule's body.[^michalski1980][^theron1996] The following table illustrates the typical structure of a rule-based classification model (in this example for predicting whether mushrooms are poisonous or edible) that consists of several conjunctive rules.

```{list-table}
* - **poisonous**
  - ←
  - odor = *foul*
* - **poisonous**
  - ←
  - gill-size = *narrow* ∧ gill-color = *buff*
* - **poisonous**
  - ←
  - gill-size = *narrow* ∧ odor = *pungent*
* - **poisonous**
  - ←
  - odor = *creosote*
* - **poisonous**
  - ←
  - spore-sprint-color = *green*
* - **poisonous**
  - ←
  - stalk-surface-above-ring = *silky* ∧ gill-spacing = *close*
* - **poisonous**
  - ←
  - habitat = *leaves* ∧ cap-color = *white*
* - **poisonous**
  - ←
  - stalk-color-above-ring = *yellow*
* - **edible**
  - ←
  - ∅
```

(user-guide-trees)=

## Rules vs. Decision Trees

Rule models and [decision trees](https://en.wikipedia.org/wiki/Decision_tree) are closely related, as both use logical clauses to test for the properties of given examples and to determine a prediction. In fact, each decision tree can be transformed into an equivalent rule-based model by viewing the paths in a tree, from the root node to each one of its leaves, as individual rules. However, unlike in decision trees, the individual rules in a rule-based model must not necessarily be non-overlapping. If an example satisfies the conditions in a rule's body, it is said to be *covered* by the rule. In such a case, the prediction provided by the rule's head applies to the example. Depending on its conditions, a rule covers an axis-aligned, hyper-rectangular region of the *feature space*, i.e., the space of all possible examples. In contrast to decision trees, which are global models that provide a prediction for each given example, a single rule is a local model that only applies to examples it covers. For this reason, many rule learning approaches use a *default rule* that does not contain any conditions in its body and therefore applied to all examples. It is meant to provide a default prediction, usually corresponding to the majority class, i.e., the class that occurs most often in the training data, for examples not covered by another rule. As a result of these conceptual similarities and differences, rules can be considered a more general concept class than the more commonly used decision trees. Compared to tree-based models, they provide additional flexibility when it comes to the selection of rules to be included in a model.

## Learning Approaches

When it comes to techniques and algorithms for rule induction, one needs to distinguish between *descriptive* and *predictive rule learning*. The former is used to describe learning techniques aimed at discovering interesting patterns in unlabeled data. It allows extracting frequent patterns of co-occurring feature values, also referred to as [association rules](https://en.wikipedia.org/wiki/Association_rule_learning), from a given dataset. In contrast, predictive rule learning methods, which this project focuses on, aim to derive a model from labeled training data which can afterward be used for obtaining predictions. As discussed below, most of these approaches are based on the ideas of so-called *covering algorithms*.

(user-guide-covering)=

### Covering Algorithms

One of the most commonly used strategies for the induction of predictive rules is the separate-and-conquer (SeCo) paradigm as described by Fürnkranz.[^fuernkranz1999] Methods that are based on this particular covering algorithm, such as the {ref}`SeCo algorithm <user-guide-seco>` developed by this project, learn a set of rules by following an iterative procedure. At first, a new rule that covers a subset of the given training examples is induced. Afterward, the examples it covers are removed from the training set, and the algorithm proceeds by learning the next rule. Ultimately, the training procedure finishes as soon as no examples are left or if a certain stopping criterion is met. The SeCo strategy results in an ordered list of rules, commonly referred to as a *decision list*. Because it was learned on a subset of the training data, where examples covered by previously induced rules have already been removed, each rule in the list depends on its predecessors. Given an example to predict for, the dependence between rules is taken into account by processing the rules in a decision list in the order of their induction.

Similar to the construction of decision trees, individual rules are usually built by a covering algorithm in a top-down fashion, where the rule is successively refined by adding new conditions to its initially empty body. As a result of adding a new condition, the rule covers fewer examples and becomes more specific. To strive for a balance between too general and overly specific rules, it is essential to avoid the problem of under- or overfitting the data.[^janssen2008] It is crucial to use a suitable evaluation measure for comparing the quality of potential refinements to achieve a good trade-off between the *coverage* and *consistency* of a rule. In the rule learning literature, measures that guide the rule refinement process are often referred to as *heuristics*. Traditionally, the development, analysis, and empirical evaluation of reliable heuristics have played an important role in research on predictive rule learning and a large number of different heuristics have consequently been proposed in the past.[^fuernkranz2005][^fuernkranz2012][^janssen2010]

(user-guide-boosting)=

### Boosting Algorithms

Decision trees and rule learners are generally considered as learning approaches with high *variance*. This means that they are very sensitive to small changes in the training data, which may result in drastic changes in the constructed model. On the one hand, this flexibility in model structure causes the mentioned classification methods to be unstable, as their predictions can easily be affected by minor perturbations of the training inputs. On the other hand, it results in low *bias*, i.e., it ensures that the learner can flexibly adjust to different data distributions and can model complex decision boundaries. The [bias and variance](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) of a learning method, as defined by Breiman[^breiman1996], are closely related to the risk of under- or overfitting the data. Methods that balance these aspects well are expected to deliver more reliable yet accurate predictions. A well-known technique that helps to reduce the variance of a classifier, while at the same time increasing its bias, is the use of ensemble methods that combine the predictions of several unstable models.

A popular approach, which is used by our {ref}`BOOMER algorithm <user-guide-boomer>` and is based on the idea to incorporate several unstable models in an ensemble, is commonly referred to as [boosting](<https://en.wikipedia.org/wiki/Boosting_(machine_learning)>). Its success in the machine learning community originates from the highly influential [AdaBoost](https://en.wikipedia.org/wiki/AdaBoost) algorithm.[^freund1997] The basic idea of boosting is to adjust the weights of the training examples depending on whether the existing ensemble members accurately deal with them. Examples for which the existing members already provide accurate predictions are assigned smaller weights, whereas more emphasis is placed on mispredicted examples. The ensemble member constructed at a particular iteration of the training algorithm is obliged to focus on examples that are insufficiently handled by its predecessors and therefore is meant to rectify their predictions. There is a close connection between the basic principles of boosting and the adjustment of weights as performed by rule learning algorithms based on *weighted covering*.[^weiss2000][^gamberger2000] This generalization of the SeCo paradigm does not remove examples entirely once they have been covered, but reduces their weights whenever an additional rule covers them. As a consequence, examples that one or more rules have already covered have a smaller impact on the evaluation of new rules, while uncovered examples have a greater chance to be covered in subsequent iterations of the algorithm instead. An advantage of weighted covering is the reduced impact of rules learned during the early stages of the covering algorithm. In contrast, boosting algorithms follow a statistically well-justified framework for computing the updates of weights. They employ a [loss function](https://en.wikipedia.org/wiki/Loss_function) to measure the overall quality of an ensemble’s predictions at each training iteration and weigh the individual training examples according to their impact on the overall performance. This enables to guide the construction of new ensemble members such that the resulting model ultimately optimizes the given loss function.

Even though boosting techniques have received early attention in the rule learning community, they are nowadays more commonly used to construct tree-based models. Implementations of gradient boosted decision trees, such as [XGBoost](https://github.com/dmlc/xgboost)[^chen2016], [LightGBM](https://github.com/microsoft/LightGBM)[^ke2017], [CatBoost](https://github.com/catboost/catboost)[^prokhorenkova2018], or [ThunderGBM](https://github.com/Xtra-Computing/thundergbm)[^wen2020], use decision trees as ensemble members. They are among the strongest classification methods available today and are used successfully in numerous application domains.

[^fuernkranz2012]: Johannes Fürnkranz, Dragan Gamberger, Nada Lavrač (2012). ‘Foundations of Rule Learning’. *Springer Science & Business Media*.

[^michalski1980]: Ryszard S. Michalski (1980). ‘Pattern Recognition as Rule-Guided Inductive Inference’. In: *IEEE Transactions on Pattern Analysis and Machine Intelligence* 2.4, pp. 349–361.

[^theron1996]: Hendrik Theron and Ian Cloete (1996). ‘BEXA: A Covering Algorithm for Learning Propositional Concept Descriptions’. In: *Machine Learning* 24.1, pp. 5–40.

[^fuernkranz1999]: Johannes Fürnkranz (1999). ‘Separate-and-Conquer Rule Learning’. In: *Artificial Intelligence Review* 13.1, pp. 3–54.

[^janssen2008]: Frederik Janssen and Johannes Fürnkranz (2008). ‘An Empirical Investigation of the Trade-Off Between Consistency and Coverage in Rule Learning Heuristics’. In: *Proc. International Conference on Discovery Science*, pp. 40–51.

[^fuernkranz2005]: Johannes Fürnkranz and Peter A. Flach (2005). ‘ROC ’n’ Rule Learning — Towards a Better Understanding of Covering Algorithms’. In: *Machine learning* 58.1, pp. 39–77.

[^janssen2010]: Frederik Janssen and Johannes Fürnkranz (2010). ‘On the quest for optimal rule learning heuristics’. In: *Machine Learning* 78.3, pp. 343–379.

[^breiman1996]: Leo Breiman (1996). ‘Bias, Variance, and Arcing Classifiers’. In: *Technical Report 460, Statistics Department, University of California, Berkeley*.

[^freund1997]: Yoav Freund and Robert E. Schapire (1997). ‘A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting’. In: *Journal of Computer and System Sciences* 55.1, pp. 119–139.

[^weiss2000]: Sholom M. Weiss and Nitin Indurkhya (2000). ‘Lightweight Rule Induction’. In: Proc. *International Conference on Machine Learning (ICML)*, pp. 1135–1142.

[^gamberger2000]: Dragan Gamberger and Nada Lavrač (2000). ‘Confirmation Rule Sets’. In: *Proc. European Conference on Principles of Data Mining and Knowledge Discovery (PKDD)*, pp. 34–43.

[^chen2016]: Tianqi Chen and Carlos Guestrin (2016). ‘XGBoost: A Scalable Tree Boosting System’. In: *Proc. ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794.

[^ke2017]: Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu (2017). ‘LightGBM: A Highly Efficient Gradient Boosting Decision Tree’. In: *Proc. Advances in Neural Information Processing Systems* 30, pp. 3146–3154.

[^prokhorenkova2018]: Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, Andrey Gulin (2018). ‘CatBoost: unbiased boosting with categorical features’. In: *Proc. Advances in Neural Information Processing Systems*.

[^wen2020]: Zeyi Wen, Hanfeng Liu, Jiashuai Shi, Qinbin Li, Bingsheng He, and Jian Chen (2020). ‘ThunderGBM: Fast GBDTs and Random Forests on GPUs’. In: *Journal of Machine Learning Research* 21.108, pp. 1−5.
