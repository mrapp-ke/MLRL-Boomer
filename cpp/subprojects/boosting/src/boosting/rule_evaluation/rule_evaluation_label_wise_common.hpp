/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/math/math.hpp"


namespace boosting {

    static inline constexpr float64 calculateLabelWiseScore(float64 gradient, float64 hessian,
                                                            float64 l2RegularizationWeight) {
        return divideOrZero(-gradient, hessian + l2RegularizationWeight);
    }

}
