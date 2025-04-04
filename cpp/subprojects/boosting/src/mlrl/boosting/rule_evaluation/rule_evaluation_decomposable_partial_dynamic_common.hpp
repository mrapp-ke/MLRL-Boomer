/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/data/vector_statistic_decomposable_dense.hpp"
#include "mlrl/common/util/iterators.hpp"
#include "rule_evaluation_decomposable_common.hpp"

#include <utility>

namespace boosting {

    /**
     * Determines and returns the minimum and maximum absolute score to be predicted for an output.
     *
     * @tparam StatisticIterator        The type of the iterator that provides access to the gradients and Hessians
     * @param statisticIterator         An iterator that provides access to the gradients and Hessians for each output
     * @param numOutputs                The total number of available outputs
     * @param l1RegularizationWeight    The l2 regularization weight
     * @param l2RegularizationWeight    The L1 regularization weight
     * @return                          A `std::pair` that stores the minimum and maximum absolute score
     */
    template<typename StatisticIterator>
    static inline std::pair<typename util::iterator_value<StatisticIterator>::statistic_type,
                            typename util::iterator_value<StatisticIterator>::statistic_type>
      getMinAndMaxScore(StatisticIterator& statisticIterator, uint32 numOutputs, float32 l1RegularizationWeight,
                        float32 l2RegularizationWeight) {
        typedef typename util::iterator_value<StatisticIterator>::statistic_type statistic_type;
        const Statistic<statistic_type>& firstStatistic = statisticIterator[0];
        statistic_type maxAbsScore = std::abs(calculateOutputWiseScore(firstStatistic.gradient, firstStatistic.hessian,
                                                                       l1RegularizationWeight, l2RegularizationWeight));
        statistic_type minAbsScore = maxAbsScore;

        for (uint32 i = 1; i < numOutputs; i++) {
            const Statistic<statistic_type>& statistic = statisticIterator[i];
            statistic_type absScore = std::abs(calculateOutputWiseScore(
              statistic.gradient, statistic.hessian, l1RegularizationWeight, l2RegularizationWeight));

            if (absScore > maxAbsScore) {
                maxAbsScore = absScore;
            } else if (absScore < minAbsScore) {
                minAbsScore = absScore;
            }
        }

        return std::make_pair(minAbsScore, maxAbsScore);
    }

    /**
     * Calculates and returns the threshold that should be used to decide whether a rule should predict for an output or
     * not.
     *
     * @tparam T The type of the scores
     * @param minAbsScore   The minimum absolute score to be predicted for an output
     * @param maxAbsScore   The maximum absolute score to be predicted for an output
     * @param threshold     A threshold that affects for how many outputs the rule heads should predict
     * @param exponent      An exponent that is used to weigh the estimated predictive quality for individual outputs
     * @return              The threshold that has been calculated
     */
    template<typename T>
    static inline T calculateThreshold(T minAbsScore, T maxAbsScore, float32 threshold, float32 exponent) {
        return std::pow(maxAbsScore - minAbsScore, exponent) * threshold;
    }

    /**
     * Weighs and returns the score that is predicted for a particular output, depending on the minimum absolute score
     * that has been determined via the function `getMinMaxScore` and a given exponent.
     *
     * @tparam T            The type of the score
     * @param score         The score to be predicted
     * @param minAbsScore   The minimum absolute score to be predicted for an output
     * @param exponent      An exponent that is used to weigh the estimated predictive quality for individual outputs
     * @return              The weighted score that has been calculated
     */
    template<typename T>
    static inline T calculateWeightedScore(T score, T minAbsScore, float32 exponent) {
        return std::pow(std::abs(score) - minAbsScore, exponent);
    }

}
