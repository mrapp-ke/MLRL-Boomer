(user-guide-multi-threading)=

# Multi-Threading

To utilize the multi-core architecture or hyper-threading capabilities of today's CPUs, our algorithms speed up the training of rule-based models by executing certain algorithmic aspects in parallel rather than sequentially.

## Parallelized Training

Unlike ensemble methods, where individual members are independent of each other, e.g., in Random Forests[^breiman2001], most rule learning methods do not allow to construct individual rules in parallel due to the sequential nature of their training procedure, where each rule is built in the context of its predecessors. Instead, the following possibilities exist to parallelize computational steps that are involved in the induction of a single rule:

- **Multi-Threaded Evaluation of Refinement Candidates:** The evaluation of conditions that can possibly be added to a rule's body requires enumerating the feature values of the training examples for each available feature, aggregating the label space statistics of examples they cover, and computing the predictions and quality of the resulting candidates. Multi-threading may be used to evaluate the refinement candidates for different features in parallel.
- **Parallel Computation of Predictions and Quality Scores:** For each candidate considered during the construction of a single rule, the predictions for different class labels and an estimate of their quality must be obtained. These operations are particularly costly if interactions between labels should be considered as is often the case in multi-label classification. In such a scenario, the parallelization of these operations across several labels may help reduce training times.
- **Distributed Update of Label Space Statistics:** After a rule has been learned, the label space statistics of all examples it covers must be updated. The complexity of this operation depends on how many examples are covered and is affected by the number of labels for which a rule predicts. Moreover, the update becomes more costly if statistics are not only provided for individual labels but also for pairs of labels or even entire label sets. Depending on the methodology used by a particular rule learning approach, training times may be reduced by updating the statistics for different examples in parallel.

The benefits of using the aforementioned possibilities for parallelization heavily depend on the characteristics of a particular dataset and the learning algorithm. In some cases, the overhead of managing and synchronizing multiple threads outweighs the speedup that the parallel execution of computations may achieve. Consequently, the use of multi-threading may even have a negative effect on the time that is needed for training.

## Parallelized Prediction

In addition to the use of multi-threading to speed up training, parallelization can also be used when predictions for several examples should be obtained. Delivering predictions for a given set of examples requires first enumerating the rules in a model and identifying those rules that cover each of the provided examples. Second, the predictions provided by the heads of these rules must be aggregated to obtain an overall prediction. As both of these steps may be carried out independently for each example, multi-threading can be used to predict for different examples in parallel.

[^breiman2001]: Leo Breiman (2001). ‘Random Forests’. In: *Machine Learning* 45.1, pp. 5–32.
