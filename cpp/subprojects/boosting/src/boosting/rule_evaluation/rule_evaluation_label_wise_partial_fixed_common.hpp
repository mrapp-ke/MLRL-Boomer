/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include "boosting/data/statistic_vector_label_wise_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"
#include <vector>
#include <queue>


namespace boosting {

    /**
     * Allows to compare two objects of type `IndexedValue` that store the optimal prediction for a labels, as well as
     * its index, according to the following strict weak ordering: If the absolute value of the first object is greater,
     * it goes before the second one.
     */
    struct CompareLabelWiseCriteria {

        /**
         * Returns whether the a given object of type `IndexedValue` that stores the optimal prediction for a label, as
         * well as its index, should go before a second one.
         *
         * @param lhs   A reference to a first object of type `IndexedValue`
         * @param rhs   A reference to a second object of type `IndexedValue`
         * @return      True, if the first object should go before the second one, false otherwise
         */
        inline bool operator()(const IndexedValue<float64>& lhs, const IndexedValue<float64>& rhs) const {
            return std::abs(lhs.value) > std::abs(rhs.value);
        }

    };

    /**
     * The type of a priority queue which is used to identify the labels for which a rule is able to predict most
     * accurately.
     */
    typedef std::priority_queue<IndexedValue<float64>, std::vector<IndexedValue<float64>>, CompareLabelWiseCriteria> PriorityQueue;

    /**
     * Calculates scores that assess the quality of optimal predictions for each label and adds them to a priority queue
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
    static inline void sortLabelWiseQualityScores(
            PriorityQueue& priorityQueue, uint32 maxCapacity,
            const DenseLabelWiseStatisticVector::const_iterator& statisticIterator, IndexIterator indexIterator,
            uint32 numElements, float64 l1RegularizationWeight, float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numElements; i++) {
            const Tuple<float64>& tuple = statisticIterator[i];
            float64 score = calculateLabelWiseScore(tuple.first, tuple.second, l1RegularizationWeight,
                                                    l2RegularizationWeight);

            if (priorityQueue.size() < maxCapacity) {
                priorityQueue.emplace(indexIterator[i], score);
            } else if (std::abs(priorityQueue.top().value) < std::abs(score)) {
                priorityQueue.pop();
                priorityQueue.emplace(indexIterator[i], score);
            }
        }
    }

}