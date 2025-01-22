/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"
#include "mlrl/common/data/vector_sparse_array.hpp"
#include "mlrl/common/util/iterators.hpp"
#include "rule_evaluation_decomposable_common.hpp"

#include <algorithm>

namespace boosting {

    /**
     * Allows to compare two objects of type `IndexedValue` that store the optimal prediction for an output, as well as
     * its index, according to the following strict weak ordering: If the absolute value of the first object is greater,
     * it goes before the second one.
     *
     * @tparam T The type of the objects to be compared
     */
    template<typename T>
    struct CompareOutputWiseCriteria final {
        public:

            /**
             * Returns whether the a given object of template type `T` that stores the optimal prediction for an output,
             * as well as its index, should go before a second one.
             *
             * @param lhs   A reference to a first object of template type `T`
             * @param rhs   A reference to a second object of template type `T`
             * @return      True, if the first object should go before the second one, false otherwise
             */
            inline bool operator()(const T& lhs, const T& rhs) const {
                return std::abs(lhs.value) > std::abs(rhs.value);
            }
    };

    /**
     * Calculates the scores to be predicted for individual outputs and sorts them by their quality, such that the first
     * `numPredictions` elements are the best-rated ones.
     *
     * @tparam StatisticIterator        The type of the iterator that provides access to the gradients and Hessians
     * @param scoreIterator             An iterator, the calculated scores and their corresponding indices should be
     *                                  written to
     * @param statisticIterator         An iterator that provides access to the gradients and Hessians for each output
     * @param numOutputs                The total number of available outputs
     * @param numPredictions            The number of the best-rated predictions to be determined
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     */
    template<typename ScoreIterator, typename StatisticIterator>
    static inline void sortOutputWiseScores(ScoreIterator scoreIterator, StatisticIterator& statisticIterator,
                                            uint32 numOutputs, uint32 numPredictions, float32 l1RegularizationWeight,
                                            float32 l2RegularizationWeight) {
        for (uint32 i = 0; i < numOutputs; i++) {
            const typename util::iterator_value<StatisticIterator>& statistic = statisticIterator[i];
            typename util::iterator_value<ScoreIterator>& entry = scoreIterator[i];
            entry.index = i;
            entry.value = calculateOutputWiseScore(statistic.gradient, statistic.hessian, l1RegularizationWeight,
                                                   l2RegularizationWeight);
        }

        std::partial_sort(scoreIterator, &scoreIterator[numPredictions], &scoreIterator[numOutputs],
                          CompareOutputWiseCriteria<typename util::iterator_value<ScoreIterator>>());
    }

}
