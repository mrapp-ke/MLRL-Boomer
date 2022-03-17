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
     * A priority queue with fixed capacity that stores quality scores and corresponding label indices. If the maximum
     * capacity of the queue is exceeded, the entry with the lowest quality score is discarded.
     */
    class FixedPriorityQueue final {

        private:

            typedef IndexedValue<float64> T;

            std::priority_queue<T, std::vector<T>, T::Compare> priorityQueue_;

            uint32 maxCapacity_;

        public:

            /**
             * @param maxCapacity The maximum capacity
             */
            FixedPriorityQueue(uint32 maxCapacity)
                : maxCapacity_(maxCapacity) {

            }

            /**
             * Adds a new quality score and corresponding label index to the priority queue.
             *
             * @param index         The index to be added
             * @param qualityScore  The quality score to be added
             */
            void emplace(uint32 index, float64 qualityScore) {
                if (priorityQueue_.size() < maxCapacity_) {
                    priorityQueue_.emplace(index, qualityScore);
                } else if (priorityQueue_.top().value > qualityScore) {
                    priorityQueue_.pop();
                    priorityQueue_.emplace(index, qualityScore);
                }
            }

            /**
             * Returns the entry with the lowest quality score.
             *
             * @return A reference to an object of type `IndexedValue` that stores the quality score and corresponding
             *         index
             */
            const IndexedValue<float64>& top() const {
                return priorityQueue_.top();
            }

            /**
             * Removes the entry with the lowest quality score.
             */
            void pop() {
                priorityQueue_.pop();
            }

    };

    /**
     * Calculates scores that asses the quality of optimal predictions for each label and adds them to a priority queue
     * with fixed capacity.
     *
     * @tparam IndexIterator            The type of the iterator that provides access to the index of each label
     * @param priorityQueue             A reference to a priority queue, which should be used for sorting
     * @param statisticIterator         An iterator that provides access to the gradients and Hessians for each label
     * @param indexIterator             An iterator that provides access to the index of each label
     * @param numElements               The number of elements
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     */
    template<typename IndexIterator>
    static inline void sortLabelWiseQualityScores(
            FixedPriorityQueue& priorityQueue, const DenseLabelWiseStatisticVector::const_iterator& statisticIterator,
            IndexIterator indexIterator, uint32 numElements, float64 l1RegularizationWeight,
            float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numElements; i++) {
            const Tuple<float64>& tuple = statisticIterator[i];
            float64 qualityScore = calculateLabelWiseQualityScore(tuple.first, tuple.second, l1RegularizationWeight,
                                                                  l2RegularizationWeight);
            priorityQueue.emplace(indexIterator[i], qualityScore);
        }
    }

}