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

RecallFunction::~RecallFunction() {

}

float64 RecallFunction::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                                float64 uip, float64 urn, float64 urp) {
    return recall(cin, crp, uin, urp);
}

WRAFunction::~WRAFunction() {

}

float64 WRAFunction::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                             float64 uip, float64 urn, float64 urp) {
    return wra(cin, cip, crn, crp, uin, uip, urn, urp);
}

static float64 precision(float64 cin, float64 cip, float64 crn float64 crp) {
    float64 numCoveredIncorrect = cip + crn;
    float64 numCovered = numCoveredIncorrect + cin + crp;

    if (numCovered == 0) {
        return 1;
    }

    return numCoveredIncorrect / numCovered;
}

static float64 recall(float64 cin, float64 crp, float64 uin, float64 urp) {
    float64 numUncoveredEqual = uin + urp;
    float64 numEqual = numUncoveredEqual + cin + crp;

    if (numEqual == 0) {
        return 1;
    }

    return numUncoveredEqual / numEqual;
}

static float64 wra(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip, float64 urn,
                   float64 urp) {
    float64 numCoveredEqual = cin + crp;
    float64 numUncoveredEqual = uin + urp;
    float64 numEqual = numUncoveredEqual + numCoveredEqual;
    float64 numCovered = numCoveredEqual + cip + crn;
    float64 numUncovered = numUncoveredEqual + uip + urn;
    float64 numTotal = numCovered + numUncovered;

    if (numCovered == 0 || numTotal == 0) {
        return 1;
    }

    return 1 - ((numCovered / numTotal) * ((numCoveredEqual / numCovered) - (numEqual / numTotal)));
}
