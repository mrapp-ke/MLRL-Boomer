/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/indexed_value.hpp"
#include <vector>
#include <queue>


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
         * @return A reference to an object of type `IndexedValue` that stores the quality score and corresponding index
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
