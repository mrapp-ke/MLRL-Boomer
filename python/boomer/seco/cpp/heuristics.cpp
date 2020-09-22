#include "heuristics.h"
#include <math.h>

using namespace seco;


static inline float64 precision(float64 cin, float64 cip, float64 crn, float64 crp) {
    float64 numCoveredIncorrect = cip + crn;
    float64 numCovered = numCoveredIncorrect + cin + crp;

    if (numCovered == 0) {
        return 1;
    }

    return numCoveredIncorrect / numCovered;
}

static inline float64 recall(float64 cin, float64 crp, float64 uin, float64 urp) {
    float64 numUncoveredEqual = uin + urp;
    float64 numEqual = numUncoveredEqual + cin + crp;

    if (numEqual == 0) {
        return 1;
    }

    return numUncoveredEqual / numEqual;
}

static inline float64 wra(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip, float64 urn,
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

float64 PrecisionImpl::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                               float64 uip, float64 urn, float64 urp) {
    return precision(cin, cip, crn, crp);
}

float64 RecallImpl::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                            float64 uip, float64 urn, float64 urp) {
    return recall(cin, crp, uin, urp);
}

float64 WRAImpl::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin, float64 uip,
                                         float64 urn, float64 urp) {
    return wra(cin, cip, crn, crp, uin, uip, urn, urp);
}

float64 HammingLossImpl::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                                 float64 uip, float64 urn, float64 urp) {
    float64 numCoveredIncorrect = cip + crn;
    float64 numCoveredCorrect = cin + crp;
    float64 numCovered = numCoveredIncorrect + numCoveredCorrect;

    if (numCovered == 0) {
        return 1;
    }

    float64 numIncorrect = numCoveredIncorrect + urn + urp;
    float64 numTotal = numIncorrect + numCoveredCorrect + uin + uip;
    return numIncorrect / numTotal;
}

FMeasureImpl::FMeasureImpl(float64 beta) {
    beta_ = beta;
}

float64 FMeasureImpl::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                              float64 uip, float64 urn, float64 urp) {
    if (isinf(beta_)) {
        // Equivalent to recall
        return recall(cin, crp, uin, urp);
    } else if (beta_ > 0) {
        // Weighted harmonic mean between precision and recall
        float64 numCoveredEqual = cin + crp;
        float64 betaPow = pow(beta_, 2);
        float64 numerator = (1 + betaPow) * numCoveredEqual;
        float64 denominator = numerator + (betaPow * (uin + urp)) + (cip + crn);

        if (denominator == 0) {
            return 1;
        }

        return 1 - (numerator / denominator);
    } else {
        // Equivalent to precision
        return precision(cin, cip, crn, crp);
    }
}

MEstimateImpl::MEstimateImpl(float64 m) {
    m_ = m;
}

float64 MEstimateImpl::evaluateConfusionMatrix(float64 cin, float64 cip, float64 crn, float64 crp, float64 uin,
                                               float64 uip, float64 urn, float64 urp) {
    if (isinf(m_)) {
        // Equivalent to weighted relative accuracy
        return wra(cin, cip, crn, crp, uin, uip, urn, urp);
    } else if (m_ > 0) {
        // Trade-off between precision and weighted relative accuracy
        float64 numCoveredEqual = cin + crp;
        float64 numCovered = numCoveredEqual + cip + crn;

        if (numCovered == 0) {
            return 1;
        }

        float64 numUncoveredEqual = uin + urp;
        float64 numTotal = numCovered + numUncoveredEqual + uip + urn;
        float64 numEqual = numCoveredEqual + numUncoveredEqual;
        return 1 - ((numCoveredEqual + (m_ * (numEqual / numTotal))) / (numCovered + m_));
    } else {
        // Equivalent to precision
        return precision(cin, cip, crn, crp);
    }
}
