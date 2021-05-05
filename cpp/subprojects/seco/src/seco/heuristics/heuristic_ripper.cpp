#include "seco/heuristics/heuristic_ripper.hpp"
#include "heuristic_common.hpp"


namespace seco {

    float64 Ripper::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) const {

        float64 numCoveredCorrect = crp + cin;
        float64 numCoveredIncorrect = cip + crn;

        if (numCoveredIncorrect > numCoveredCorrect) {
            return 1;
        }
        float64 numCovered = numCoveredCorrect + numCoveredIncorrect;

        // (p - n) / (p + n)
        // 1 - ((numCoveredCorrect - numCoveredIncorrect) / (numCovered))
        return 1 - ((numCoveredCorrect - numCoveredIncorrect) / (numCovered));
    }

    std::string Ripper::getName() const {
        return "Ripper";
    }

}