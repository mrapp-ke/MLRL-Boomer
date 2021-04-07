#include "seco/heuristics/heuristic_ripper.hpp"
#include "heuristic_common.hpp"


namespace seco {

    float64 RIPPER::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                          float64 uip, float64 urn, float64 urp) const {

        float64 numCoveredPositive = crn + crp;         // p
        float64 numCoveredNegative = cin + cip;         // n

        // inverse of (p - n) / (p + n) = (p + n) / (p - n)
        return (numCoveredPositive + numCoveredNegative) / (numCoveredPositive - numCoveredNegative);
    }

}