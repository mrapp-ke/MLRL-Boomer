/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/data/statistic_vector_label_wise_dense.hpp"
#include "rule_evaluation_label_wise_common.hpp"


namespace boosting {

    /**
     * Calculates the score to be predicted for individual bins and returns an overall quality score that assesses the
     * quality of the predictions.
     *
     * @tparam ScoreIterator            The type of the iterator that provides access to the gradients and Hessians
     * @param statisticIterator         An iterator that provides random access to the gradients and Hessians
     * @param scoreIterator             An iterator, the calculated scores should be written to
     * @param weights                   An iterator that provides access to the weights of individual bins
     * @param numElements               The number of bins
     * @param l1RegularizationWeight    The L1 regularization weight
     * @param l2RegularizationWeight    The L2 regularization weight
     * @return                          The overall quality score that has been calculated
     */
    template<typename ScoreIterator>
    static inline float64 calculateBinnedScores(DenseLabelWiseStatisticVector::const_iterator statisticIterator,
                                                ScoreIterator scoreIterator, const uint32* weights, uint32 numElements,
                                                float64 l1RegularizationWeight, float64 l2RegularizationWeight) {
        float64 overallQualityScore = 0;

        for (uint32 i = 0; i < numElements; i++) {
            uint32 weight = weights[i];
            const Tuple<float64>& tuple = statisticIterator[i];
            float64 predictedScore = calculateLabelWiseScore(tuple.first, tuple.second, weight * l1RegularizationWeight,
                                                             weight * l2RegularizationWeight);
            scoreIterator[i] = predictedScore;
            overallQualityScore += calculateLabelWiseQualityScore(predictedScore, tuple.first, tuple.second,
                                                                  weight * l1RegularizationWeight,
                                                                  weight * l2RegularizationWeight);
        }

        return overallQualityScore;
    }

}
