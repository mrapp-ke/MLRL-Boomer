/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/math.hpp"

namespace boosting {

    /**
     * Returns the L1 regularization weight to be added to a specific gradient.
     *
     * @tparam StatisticType            The type of the gradient
     * @param gradient                  The gradient, the L1 regularization weight should be added to
     * @param l1RegularizationWeight    The L1 regularization weight
     * @return                          The L1 regularization weight to be added to the gradient
     */
    template<typename StatisticType>
    static inline constexpr float32 getL1RegularizationWeight(StatisticType gradient, float32 l1RegularizationWeight) {
        if (gradient > l1RegularizationWeight) {
            return -l1RegularizationWeight;
        } else if (gradient < -l1RegularizationWeight) {
            return l1RegularizationWeight;
        } else {
            return 0.0f;
        }
    }

    /**
     * Calculates and returns the optimal score to be predicted for a single output, based on the corresponding gradient
     * and Hessian and taking L1 and L2 regularization into account.
     *
     * @tparam StatisticType            The type of the gradient and Hessian
     * @param gradient                  The gradient that corresponds to the output
     * @param hessian                   The Hessian that corresponds to the output
     * @param l1RegularizationWeight    The weight of the L1 regularization
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The predicted score that has been calculated
     */
    template<typename StatisticType>
    static inline constexpr StatisticType calculateOutputWiseScore(StatisticType gradient, StatisticType hessian,
                                                                   float32 l1RegularizationWeight,
                                                                   float32 l2RegularizationWeight) {
        return util::divideOrZero(-gradient + getL1RegularizationWeight(gradient, l1RegularizationWeight),
                                  hessian + l2RegularizationWeight);
    }

    /**
     * Calculates and returns the quality of the prediction for a single output, taking L1 and L2 regularization into
     * account.
     *
     * @tparam StatisticType            The type of the predicted score, gradient and Hessian
     * @param score                     The predicted score
     * @param gradient                  The gradient
     * @param hessian                   The Hessian
     * @param l1RegularizationWeight    The weight of the L1 regularization
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The quality that has been calculated
     */
    template<typename StatisticType>
    static inline StatisticType calculateOutputWiseQuality(StatisticType score, StatisticType gradient,
                                                           StatisticType hessian, float32 l1RegularizationWeight,
                                                           float32 l2RegularizationWeight) {
        StatisticType scorePow = score * score;
        StatisticType quality = (gradient * score) + (0.5 * hessian * scorePow);
        StatisticType l1RegularizationTerm = l1RegularizationWeight * std::abs(score);
        StatisticType l2RegularizationTerm = 0.5 * l2RegularizationWeight * scorePow;
        return quality + l1RegularizationTerm + l2RegularizationTerm;
    }

}
