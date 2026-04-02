/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/math/scalar_math.hpp"

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
}
