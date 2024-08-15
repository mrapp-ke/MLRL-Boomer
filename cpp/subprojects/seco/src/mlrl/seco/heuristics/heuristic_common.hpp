/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/util/math.hpp"

namespace seco {

    static inline constexpr float64 precision(float64 cin, float64 cip, float64 crn, float64 crp) {
        float64 numCoveredCorrect = cin + crp;
        float64 numCovered = numCoveredCorrect + cip + crn;
        return util::divideOrZero(numCoveredCorrect, numCovered);
    }

    static inline constexpr float64 recall(float64 cin, float64 crp, float64 uin, float64 urp) {
        float64 numCoveredEqual = cin + crp;
        float64 numEqual = numCoveredEqual + uin + urp;
        return util::divideOrZero(numCoveredEqual, numEqual);
    }

    static inline constexpr float64 wra(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                        float64 urn, float64 urp) {
        float64 numCoveredEqual = cin + crp;
        float64 numUncoveredEqual = uin + urp;
        float64 numEqual = numUncoveredEqual + numCoveredEqual;
        float64 numCovered = numCoveredEqual + cip + crn;
        float64 numUncovered = numUncoveredEqual + uip + urn;
        float64 numTotal = numCovered + numUncovered;

        if (numCovered > 0 && numTotal > 0) {
            return (numCovered / numTotal) * ((numCoveredEqual / numCovered) - (numEqual / numTotal));
        }

        return 0;
    }

}
