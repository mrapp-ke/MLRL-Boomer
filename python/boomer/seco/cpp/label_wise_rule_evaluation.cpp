#include "label_wise_rule_evaluation.h"
#include <cstddef>

using namespace rule_evaluation;


LabelWiseRuleEvaluationImpl::LabelWiseRuleEvaluationImpl(heuristics::AbstractHeuristic* heuristic) {
    heuristic_ = heuristic;
}

void LabelWiseRuleEvaluationImpl::calculateLabelWisePrediction(const intp* labelIndices, const uint8* minorityLabels,
                                                               const float64* confusionMatricesTotal,
                                                               const float64* confusionMatricesSubset,
                                                               const float64* confusionMatricesCovered, bool uncovered,
                                                               LabelWisePrediction* prediction) {
    // Class members
    heuristics::AbstractHeuristic* heuristic = heuristic_;
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

        intp cin = confusionMatricesCovered[offsetC + heuristics::IN];
        intp cip = confusionMatricesCovered[offsetC + heuristics::IP];
        intp crn = confusionMatricesCovered[offsetC + heuristics::RN];
        intp crp = confusionMatricesCovered[offsetC + heuristics::RP];

        if (uncovered) {
            uin = cin + confusionMatricesTotal[offsetL + heuristics::IN] - confusionMatricesSubset[offsetL + heuristics::IN];
            uip = cip + confusionMatricesTotal[offsetL + heuristics::IP] - confusionMatricesSubset[offsetL + heuristics::IP];
            urn = crn + confusionMatricesTotal[offsetL + heuristics::RN] - confusionMatricesSubset[offsetL + heuristics::RN];
            urp = crp + confusionMatricesTotal[offsetL + heuristics::RP] - confusionMatricesSubset[offsetL + heuristics::RP];
            cin = confusionMatricesSubset[offsetC + heuristics::IN] - cin;
            cip = confusionMatricesSubset[offsetC + heuristics::IP] - cip;
            crn = confusionMatricesSubset[offsetC + heuristics::RN] - crn;
            crp = confusionMatricesSubset[offsetC + heuristics::RP] - crp;
        } else {
            uin = confusionMatricesTotal[offsetL + heuristics::IN] - cin;
            uip = confusionMatricesTotal[offsetL + heuristics::IP] - cip;
            urn = confusionMatricesTotal[offsetL + heuristics::RN] - crn;
            urp = confusionMatricesTotal[offsetL + heuristics::RP] - crp;
        }

        score = heuristic->evaluateConfusionMatrix(cin, cip, crn, crp, uin, uip, urn, urp);
        qualityScores[c] = score;
        overallQualityScore += score;
    }

    overallQualityScore /= numPredictions;
    prediction->overallQualityScore_ = overallQualityScore;
}
