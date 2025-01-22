/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/vector_statistic_non_decomposable_dense.hpp"
#include "rule_evaluation_decomposable_partial_fixed_common.hpp"

namespace boosting {

    /**
     * Calculates scores that assess the quality of optimal predictions for each output and sorts them, such that the
     * first `numPredictions` elements are the best-rated ones.
     *
     * @tparam GradientIterator         The type of the iterator that provides access to the gradients
     * @tparam HessianIterator          The type of the iterator that provides access to the Hessians
     * @param tmpIterator               An iterator that provides random access to a temporary array, which should be
     *                                  used to store the sorted scores and their original indices
     * @param gradientIterator          An iterator that provides access to the gradient for each output
     * @param hessianIterator           An iterator that provides access to the Hessian for each output
     * @param numOutputs                The total number of available outputs
     * @param numPredictions            The number of the best-rated predictions to be determined
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     */
    template<typename GradientIterator, typename HessianIterator>
    static inline void sortOutputWiseCriteria(SparseArrayVector<float64>::iterator tmpIterator,
                                              GradientIterator gradientIterator, HessianIterator hessianIterator,
                                              uint32 numOutputs, uint32 numPredictions, float32 l1RegularizationWeight,
                                              float32 l2RegularizationWeight) {
        for (uint32 i = 0; i < numOutputs; i++) {
            IndexedValue<float64>& entry = tmpIterator[i];
            entry.index = i;
            entry.value = calculateOutputWiseScore(gradientIterator[i], hessianIterator[i], l1RegularizationWeight,
                                                   l2RegularizationWeight);
        }

        std::partial_sort(tmpIterator, &tmpIterator[numPredictions], &tmpIterator[numOutputs],
                          CompareOutputWiseCriteria<IndexedValue<float64>>());
    }

}
