/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/math/scalar_math.hpp"

namespace seco {

    static inline constexpr float32 precision(float32 tp, float32 fp) {
        return math::divideOrZero(tp, tp + fp);
    }

    static inline constexpr float32 recall(float32 tp, float32 fn) {
        return math::divideOrZero(tp, tp + fn);
    }

    static inline constexpr float32 wra(float32 tp, float32 fp, float32 fn, float32 tn) {
        float32 numCovered = tp + fp;
        float32 numTotal = numCovered + fn + tn;

        if (numCovered > 0 && numTotal > 0) {
            return (numCovered / numTotal) * ((tp / numCovered) - ((fn + tp) / numTotal));
        }

        return 0;
    }

}
