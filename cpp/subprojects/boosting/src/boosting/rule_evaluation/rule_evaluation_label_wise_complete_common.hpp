/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "rule_evaluation_label_wise_common.hpp"
#include "boosting/data/statistic_vector_dense_label_wise.hpp"


namespace boosting {

    /**
     * Calculates the optimal scores to be predicted for several labels, based on the corresponding gradients and
     * Hessians and taking L2 regularization into account and writes them to an iterator.
     *
     * @tparam ScoreIterator            The type of the iterator, the calculated scores should be written to
     * @param statisticIterator         A `DenseLabelWiseStatisticVector::const_iterator` that provides access to the
     *                                  gradients and Hessians
     * @param scoreIterator             An iterator that allows to write the calculated scores
     * @param numElements               The number of scores to be calculated
     * @param l2RegularizationWeight    The weight of the L2 regularization
     */
    template<typename ScoreIterator>
    static inline void calculateLabelWiseScores(DenseLabelWiseStatisticVector::const_iterator statisticIterator,
                                                ScoreIterator scoreIterator, uint32 numElements,
                                                float64 l2RegularizationWeight) {
        for (uint32 i = 0; i < numElements; i++) {
            const Tuple<float64>& tuple = statisticIterator[i];
            scoreIterator[i] = calculateLabelWiseScore(tuple.first, tuple.second, l2RegularizationWeight);
        }
    }

    /**
     * Calculates and returns an overall quality score that assesses the quality of the scores that are predicted for
     * several labels.
     *
     * @tparam ScoreIterator            The type of the iterator that provides access to the predicted scores
     * @param statisticIterator         An iterator that provides random access to the gradients and Hessians
     * @param scoreIterator             An iterator that provides random access to the predicted scores
     * @param numElements               The number of predicted scores
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The overall quality score that has been calculated
     */
    template<typename ScoreIterator>
    static inline constexpr float64 calculateOverallQualityScore(
            DenseLabelWiseStatisticVector::const_iterator statisticIterator, ScoreIterator scoreIterator,
            uint32 numElements, float64 l2RegularizationWeight) {
        float64 overallQualityScore = 0;

        for (uint32 i = 0; i < numElements; i++) {
            const Tuple<float64>& tuple = statisticIterator[i];
            overallQualityScore += calculateLabelWiseQualityScore(scoreIterator[i], tuple.first, tuple.second,
                                                                  l2RegularizationWeight);
        }

        return overallQualityScore;
    }

}
