/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include "boosting/data/statistic_vector_example_wise_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"
#include <vector>
#include <queue>


namespace boosting {

    /**
     * The type of a priority queue which is used to identify the labels for which a rule is able to predict most
     * accurately.
     */
    typedef std::priority_queue<IndexedValue<float64>, std::vector<IndexedValue<float64>>, IndexedValue<float64>::Compare> PriorityQueue;

    /**
     * Calculates scores that asses the quality of optimal predictions for each label and adds them to a priority queue
     * with fixed capacity.
     *
     * @tparam IndexIterator            The type of the iterator that provides access to the index of each label
     * @param priorityQueue             A reference to an object of type `PriorityQueue`, which should be used for
     *                                  sorting
     * @param maxCapacity               The maximum capacity of the given priority queue
     * @param statisticIterator         An iterator that provides access to the gradients and Hessians for each label
     * @param indexIterator             An iterator that provides access to the index of each label
     * @param numElements               The number of elements
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     */
    template<typename IndexIterator>
    static inline void sortLabelWiseCriteria(
            PriorityQueue& priorityQueue, uint32 maxCapacity,
            DenseExampleWiseStatisticVector::gradient_const_iterator gradientIterator,
            DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator hessianIterator,
            IndexIterator indexIterator, uint32 numElements, float64 l1RegularizationWeight,
            float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numElements; i++) {
            float64 score = calculateLabelWiseScore(gradientIterator[i], hessianIterator[i], l1RegularizationWeight,
                                                    l2RegularizationWeight);

            if (priorityQueue.size() < maxCapacity) {
                priorityQueue.emplace(indexIterator[i], score);
            } else if (priorityQueue.top().value > score) {
                priorityQueue.pop();
                priorityQueue.emplace(indexIterator[i], score);
            }
        }
    }

}