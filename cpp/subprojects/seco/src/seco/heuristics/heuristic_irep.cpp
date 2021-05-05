#include "seco/heuristics/heuristic_irep.hpp"
#include "heuristic_common.hpp"


namespace seco {

    float64 IREP::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                               float64 uip, float64 urn, float64 urp) const {

        float64 numUncoveredCorrect = urp + uin;
        float64 numCoveredIncorrect = cip + crn;
        float64 numTotal = numUncoveredCorrect + numCoveredIncorrect + cin + crp + uip + urn;

        // (p + (N - n)) / (P + N)
        // (numUncoveredCorrect + numCoveredIncorrect) / (numTotal)
        return (numUncoveredCorrect + numCoveredIncorrect) / numTotal;
    }

    std::string IREP::getName() const {
        return "IREP";
    }

}
