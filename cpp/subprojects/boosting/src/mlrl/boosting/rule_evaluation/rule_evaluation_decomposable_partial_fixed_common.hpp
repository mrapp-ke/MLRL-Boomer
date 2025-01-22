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
     */
    struct CompareOutputWiseCriteria final {
        public:

            /**
             * Returns whether the a given object of type `IndexedValue` that stores the optimal prediction for an
             * output, as well as its index, should go before a second one.
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
     * Calculates the scores to be predicted for individual outputs and sorts them by their quality, such that the first
     * `numPredictions` elements are the best-rated ones.
     *
     * @tparam StatisticIterator        The type of the iterator that provides access to the gradients and Hessians
     * @param tmpIterator               An iterator that provides random access to a temporary array, which should be
     *                                  used to store the sorted scores and their original indices
     * @param statisticIterator         An iterator that provides access to the gradients and Hessians for each output
     * @param numOutputs                The total number of available outputs
     * @param numPredictions            The number of the best-rated predictions to be determined
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     */
    template<typename StatisticIterator>
    static inline void sortOutputWiseScores(SparseArrayVector<float64>::iterator tmpIterator,
                                            StatisticIterator& statisticIterator, uint32 numOutputs,
                                            uint32 numPredictions, float32 l1RegularizationWeight,
                                            float32 l2RegularizationWeight) {
        for (uint32 i = 0; i < numOutputs; i++) {
            const typename util::iterator_value<StatisticIterator>& statistic = statisticIterator[i];
            IndexedValue<float64>& entry = tmpIterator[i];
            entry.index = i;
            entry.value = calculateOutputWiseScore(statistic.gradient, statistic.hessian, l1RegularizationWeight,
                                                   l2RegularizationWeight);
        }

        std::partial_sort(tmpIterator, &tmpIterator[numPredictions], &tmpIterator[numOutputs],
                          CompareOutputWiseCriteria());
    }

}
