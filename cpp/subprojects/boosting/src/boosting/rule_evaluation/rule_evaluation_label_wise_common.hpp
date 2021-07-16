/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/math/math.hpp"
#include "boosting/data/statistic_vector_dense_label_wise.hpp"


namespace boosting {

    /**
     * Calculates and returns the optimal score to be predicted for a single label, based on the corresponding gradient
     * and Hessian and taking L2 regularization into account.
     *
     * @param gradient                  The gradient that corresponds to the label
     * @param hessian                   The Hessian that corresponds to the label
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The predicted score that has been calculated
     */
    static inline constexpr float64 calculateLabelWiseScore(float64 gradient, float64 hessian,
                                                            float64 l2RegularizationWeight) {
        return divideOrZero(-gradient, hessian + l2RegularizationWeight);
    }

    /**
     * Calculates and returns an overall quality score that assesses the quality of the scores that are predicted for
     * individual labels.
     *
     * @tparam ScoreIterator            The type of the iterator that provides access to the predicted scores
     * @param scoreIterator             An iterator that provides random access to the predicted scores
     * @param statisticIterator         An iterator that provides random access to the gradients and Hessians
     * @param numElements               The number of predicted scores
     * @param l2RegularizationWeight    The weight of the l2 regularization
     * @return                          The overall quality score that has been calculated
     */
    template<typename ScoreIterator>
    static inline constexpr float64 calculateLabelWiseQualityScore(
            ScoreIterator scoreIterator, DenseLabelWiseStatisticVector::const_iterator statisticIterator,
            uint32 numElements, float64 l2RegularizationWeight) {
        float64 overallQualityScore = 0;

        for (uint32 i = 0; i < numElements; i++) {
            const Tuple<float64>& tuple = statisticIterator[i];
            float64 gradient = tuple.first;
            float64 hessian = tuple.second;
            float64 score = scoreIterator[i];
            float64 scorePow = score * score;
            overallQualityScore += (gradient * score) + (0.5 * scorePow * hessian)
                                    + (0.5 * l2RegularizationWeight * scorePow);
        }

        return overallQualityScore;
    }


}
