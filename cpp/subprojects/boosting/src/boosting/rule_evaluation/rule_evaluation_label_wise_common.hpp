/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/math/math.hpp"


namespace boosting {

    /**
     * Calculates and returns the optimal score to be predicted for a single label, based on the corresponding gradient
     * and Hessian and taking L1 and L2 regularization into account.
     *
     * @param gradient                  The gradient that corresponds to the label
     * @param hessian                   The Hessian that corresponds to the label
     * @param l1RegularizationWeight    The weight of the L1 regularization
     * @param l2RegularizationWeight    The weight of the L2 regularization
     * @return                          The predicted score that has been calculated
     */
    static inline constexpr float64 calculateLabelWiseScore(float64 gradient, float64 hessian,
                                                            float64 l1RegularizationWeight,
                                                            float64 l2RegularizationWeight) {
        // TODO Take L1 regularization weight into account
        return divideOrZero(-gradient, hessian + l2RegularizationWeight);
    }

}
