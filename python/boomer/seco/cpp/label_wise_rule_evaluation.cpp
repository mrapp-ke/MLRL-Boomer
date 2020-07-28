#include "label_wise_rule_evaluation.h"
#include <cstddef>

using namespace rule_evaluation;


CppLabelWiseRuleEvaluation::CppLabelWiseRuleEvaluation(heuristics::HeuristicFunction* heuristicFunction) {
    heuristicFunction_ = heuristicFunction;
}

void CppLabelWiseRuleEvaluation::calculateLabelWisePrediction(const intp* labelIndices, const uint8* minorityLabels,
                                                              const float64* confusionMatricesTotal,
                                                              const float64* confusionMatricesSubset,
                                                              const float64* confusionMatricesCovered, bool uncovered,
                                                              LabelWisePrediction* prediction) {
    // Class members
    heuristics::HeuristicFunction* heuristicFunction = heuristicFunction_;
    // The number of labels to predict for
    intp numPredictions = prediction->numPredictions_;
    // The array that should be used to store the predicted scores
    float64* predictedScores = prediction->predictedScores_;
    // The array that should be used to store the quality scores
    float64* qualityScores = prediction->qualityScores_;
    // The overall quality score, i.e., the average of the quality scores for each label
    float64 overallQualityScore = 0;

    for (intp c = 0; c < numPredictions; c++) {
        intp l = labelIndices != NULL ? labelIndices[c] : c;

        // Set the score to be predicted for the current label...
        float64 score = (float64) minorityLabels[l];
        predictedScores[c] = score;

        // Calculate the quality score for the current label...
        intp offsetC = c * 4;
        intp offsetL = l * 4;
        intp uin, uip, urn, urp;

        intp cin = confusionMatricesCovered[offsetC + 0];  // <intp>Element.IN
        intp cip = confusionMatricesCovered[offsetC + 1];  // <intp>Element.IP
        intp crn = confusionMatricesCovered[offsetC + 2];  // <intp>Element.RN
        intp crp = confusionMatricesCovered[offsetC + 3];  // <intp>Element.RP

        if (uncovered) {
            uin = cin + confusionMatricesTotal[offsetL + 0] - confusionMatricesSubset[offsetL + 0];
            uip = cip + confusionMatricesTotal[offsetL + 1] - confusionMatricesSubset[offsetL + 1];
            urn = crn + confusionMatricesTotal[offsetL + 2] - confusionMatricesSubset[offsetL + 2];
            urp = crp + confusionMatricesTotal[offsetL + 3] - confusionMatricesSubset[offsetL + 3];
            cin = confusionMatricesSubset[offsetC + 0] - cin;
            cip = confusionMatricesSubset[offsetC + 1] - cip;
            crn = confusionMatricesSubset[offsetC + 2] - crn;
            crp = confusionMatricesSubset[offsetC + 3] - crp;
        } else {
            uin = confusionMatricesTotal[offsetL + 0] - cin;
            uip = confusionMatricesTotal[offsetL + 1] - cip;
            urn = confusionMatricesTotal[offsetL + 2] - crn;
            urp = confusionMatricesTotal[offsetL + 3] - crp;
        }

        score = heuristicFunction->evaluateConfusionMatrix(cin, cip, crn, crp, uin, uip, urn, urp);
        qualityScores[c] = score;
        overallQualityScore += score;
    }

    overallQualityScore /= numPredictions;
    prediction->overallQualityScore_ = overallQualityScore;
}
