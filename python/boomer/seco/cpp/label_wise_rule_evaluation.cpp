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
        intp uin, uip, urn, urp, cin, cip, crn, crp;

        cin = confusionMatricesCovered[c + (0 * numPredictions)];  // <intp>Element.IN
        cip = confusionMatricesCovered[c + (1 * numPredictions)];  // <intp>Element.IP
        crn = confusionMatricesCovered[c + (2 * numPredictions)];  // <intp>Element.RN
        crp = confusionMatricesCovered[c + (3 * numPredictions)];  // <intp>Element.RP

        if (uncovered) {
            uin = cin + confusionMatricesTotal[l + (0 * numPredictions)] - confusionMatricesSubset[l + (0 * numPredictions)];
            uip = cip + confusionMatricesTotal[l + (1 * numPredictions)] - confusionMatricesSubset[l + (1 * numPredictions)];
            urn = crn + confusionMatricesTotal[l + (2 * numPredictions)] - confusionMatricesSubset[l + (2 * numPredictions)];
            urp = crp + confusionMatricesTotal[l + (3 * numPredictions)] - confusionMatricesSubset[l + (3 * numPredictions)];
            cin = confusionMatricesSubset[c + (0 * numPredictions)] - cin;
            cip = confusionMatricesSubset[c + (1 * numPredictions)] - cip;
            crn = confusionMatricesSubset[c + (2 * numPredictions)] - crn;
            crp = confusionMatricesSubset[c + (3 * numPredictions)] - crp;
        } else {
            uin = confusionMatricesTotal[l + (0 * numPredictions)] - cin;
            uip = confusionMatricesTotal[l + (1 * numPredictions)] - cip;
            urn = confusionMatricesTotal[l + (2 * numPredictions)] - crn;
            urp = confusionMatricesTotal[l + (3 * numPredictions)] - crp;
        }

        score = heuristicFunction->evaluateConfusionMatrix(cin, cip, crn, crp, uin, uip, urn, urp);
        qualityScores[c] = score;
        overallQualityScore += score;
    }

    overallQualityScore /= numPredictions;
    prediction->overallQualityScore_ = overallQualityScore;
}
