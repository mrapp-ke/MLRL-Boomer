/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/vector_statistic_non_decomposable_dense.hpp"
#include "rule_evaluation_decomposable_partial_dynamic_common.hpp"

#include <utility>

namespace boosting {

    /**
     * Determines and returns the minimum and maximum absolute score to be predicted for an output. The scores to be
     * predicted for individual outputs are also written to a given iterator.
     *
     * @tparam ScoreIterator            The type of the iterator, the scores should be written to
     * @tparam GradientIterator         The type of the iterator that provides access to the gradients
     * @tparam HessianIterator          The type of the iterator that provides access to the Hessians
     * @param scoreIterator             An iterator, the scores should be written to
     * @param gradientIterator          An iterator that provides access to the gradient for each output
     * @param hessianIterator           An iterator that provides access to the Hessian for each output
     * @param numOutputs                The total number of available outputs
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     * @return                          A `std::pair` that stores the minimum and maximum absolute score
     */
    template<typename ScoreIterator, typename GradientIterator, typename HessianIterator>
    static inline std::pair<float64, float64> getMinAndMaxScore(ScoreIterator scoreIterator,
                                                                GradientIterator gradientIterator,
                                                                HessianIterator hessianIterator, uint32 numOutputs,
                                                                float32 l1RegularizationWeight,
                                                                float32 l2RegularizationWeight) {
        float64 score = calculateOutputWiseScore(gradientIterator[0], hessianIterator[0], l1RegularizationWeight,
                                                 l2RegularizationWeight);
        scoreIterator[0] = score;
        float64 maxAbsScore = std::abs(score);
        float64 minAbsScore = maxAbsScore;

        for (uint32 i = 1; i < numOutputs; i++) {
            score = calculateOutputWiseScore(gradientIterator[i], hessianIterator[i], l1RegularizationWeight,
                                             l2RegularizationWeight);
            scoreIterator[i] = score;
            score = std::abs(score);

            if (score > maxAbsScore) {
                maxAbsScore = score;
            } else if (score < minAbsScore) {
                minAbsScore = score;
            }
        }

        return std::make_pair(minAbsScore, maxAbsScore);
    }
}
