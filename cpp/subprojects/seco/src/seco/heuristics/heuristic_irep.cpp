#include "seco/heuristics/heuristic_irep.hpp"
#include "heuristic_common.hpp"


namespace seco {

    float64 IREP::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                               float64 uip, float64 urn, float64 urp) const {

        float64 numPositive = crn + crp + urn + urp;    // P
        float64 numNegative = cin + cip + uin + uip;    // N
        float64 numCoveredPositive = crn + crp;         // p
        float64 numCoveredNegative = cin + cip;         // n

        // inverse of (p + (N - n)) / (P + N) = (P + N) / (p + (N - n))
        return (numPositive + numNegative) / (numCoveredPositive + (numNegative - numCoveredNegative));
    }

}
