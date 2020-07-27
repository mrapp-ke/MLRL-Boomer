#include "heuristics.h"

using namespace heuristics;


HeuristicFunction::~HeuristicFunction() {

}

float64 HeuristicFunction::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                                   float64 uip, float64 urn, float64 urp) {
    return 0;
}

PrecisionFunction::~PrecisionFunction() {

}

float64 PrecisionFunction::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                                   float64 uip, float64 urn, float64 urp) {
    return precision(cin, cip, crn, crp);
}

static float64 precision(float64 cin, float64 cip, float64 crn float64 crp) {
    float64 numCoveredIncorrect = cip + crn;
    float64 numCovered = numCoveredIncorrect + cin + crp;

    if (numCovered == 0) {
        return 1;
    }

    return numCoveredIncorrect / numCovered;
}
