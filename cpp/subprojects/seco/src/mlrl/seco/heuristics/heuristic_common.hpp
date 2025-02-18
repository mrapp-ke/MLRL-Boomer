/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/math.hpp"

namespace seco {

    static inline constexpr float32 precision(float32 cin, float32 cip, float32 crn, float32 crp) {
        float32 numCoveredCorrect = cin + crp;
        float32 numCovered = numCoveredCorrect + cip + crn;
        return util::divideOrZero(numCoveredCorrect, numCovered);
    }

    static inline constexpr float32 recall(float32 cin, float32 crp, float32 uin, float32 urp) {
        float32 numCoveredEqual = cin + crp;
        float32 numEqual = numCoveredEqual + uin + urp;
        return util::divideOrZero(numCoveredEqual, numEqual);
    }

    static inline constexpr float32 wra(float32 cin, float32 cip, float32 crn, float32 crp, float32 uin, float32 uip,
                                        float32 urn, float32 urp) {
        float32 numCoveredEqual = cin + crp;
        float32 numUncoveredEqual = uin + urp;
        float32 numEqual = numUncoveredEqual + numCoveredEqual;
        float32 numCovered = numCoveredEqual + cip + crn;
        float32 numUncovered = numUncoveredEqual + uip + urn;
        float32 numTotal = numCovered + numUncovered;

        if (numCovered > 0 && numTotal > 0) {
            return (numCovered / numTotal) * ((numCoveredEqual / numCovered) - (numEqual / numTotal));
        }

        return 0;
    }

}
