#include "label_wise_rule_evaluation.h"
#include <cstddef>
#include <stdlib.h>

using namespace seco;


AbstractLabelWiseRuleEvaluation::~AbstractLabelWiseRuleEvaluation() {

}

void AbstractLabelWiseRuleEvaluation::calculateLabelWisePrediction(const uint32* labelIndices,
                                                                   const uint8* minorityLabels,
                                                                   const float64* confusionMatricesTotal,
                                                                   const float64* confusionMatricesSubset,
                                                                   const float64* confusionMatricesCovered,
                                                                   bool uncovered,
                                                                   LabelWisePredictionCandidate* prediction) {

}

HeuristicLabelWiseRuleEvaluationImpl::HeuristicLabelWiseRuleEvaluationImpl(
        std::shared_ptr<AbstractHeuristic> heuristicPtr, bool predictMajority) {
    heuristicPtr_ = heuristicPtr;
    predictMajority_ = predictMajority;
}

void HeuristicLabelWiseRuleEvaluationImpl::calculateLabelWisePrediction(const uint32* labelIndices,
                                                                        const uint8* minorityLabels,
                                                                        const float64* confusionMatricesTotal,
                                                                        const float64* confusionMatricesSubset,
                                                                        const float64* confusionMatricesCovered,
                                                                        bool uncovered,
                                                                        LabelWisePredictionCandidate* prediction) {
    // Class members
    AbstractHeuristic* heuristic = heuristicPtr_.get();
    bool predictMajority = predictMajority_;
    // The number of labels to predict for
    uint32 numPredictions = prediction->numPredictions_;
    // The array that should be used to store the predicted scores
    float64* predictedScores = prediction->predictedScores_;
    // The array that should be used to store the quality scores
    float64* qualityScores = prediction->qualityScores_;
    // The overall quality score, i.e., the average of the quality scores for each label
    float64 overallQualityScore = 0;

    for (uint32 c = 0; c < numPredictions; c++) {
        uint32 l = labelIndices != NULL ? labelIndices[c] : c;

        // Set the score to be predicted for the current label...
        uint8 minorityLabel = minorityLabels[l];
        float64 score = (float64) (predictMajority ? (minorityLabel > 0 ? 0 : 1) : minorityLabel);
        predictedScores[c] = score;

        // Calculate the quality score for the current label...
        uint32 offsetC = c * NUM_CONFUSION_MATRIX_ELEMENTS;
        uint32 offsetL = l * NUM_CONFUSION_MATRIX_ELEMENTS;
        uint32 uin, uip, urn, urp;

        uint32 cin = confusionMatricesCovered[offsetC + IN];
        uint32 cip = confusionMatricesCovered[offsetC + IP];
        uint32 crn = confusionMatricesCovered[offsetC + RN];
        uint32 crp = confusionMatricesCovered[offsetC + RP];

        if (uncovered) {
            uin = cin + confusionMatricesTotal[offsetL + IN] - confusionMatricesSubset[offsetL + IN];
            uip = cip + confusionMatricesTotal[offsetL + IP] - confusionMatricesSubset[offsetL + IP];
            urn = crn + confusionMatricesTotal[offsetL + RN] - confusionMatricesSubset[offsetL + RN];
            urp = crp + confusionMatricesTotal[offsetL + RP] - confusionMatricesSubset[offsetL + RP];
            cin = confusionMatricesSubset[offsetC + IN] - cin;
            cip = confusionMatricesSubset[offsetC + IP] - cip;
            crn = confusionMatricesSubset[offsetC + RN] - crn;
            crp = confusionMatricesSubset[offsetC + RP] - crp;
        } else {
            uin = confusionMatricesTotal[offsetL + IN] - cin;
            uip = confusionMatricesTotal[offsetL + IP] - cip;
            urn = confusionMatricesTotal[offsetL + RN] - crn;
            urp = confusionMatricesTotal[offsetL + RP] - crp;
        }

        score = heuristic->evaluateConfusionMatrix(cin, cip, crn, crp, uin, uip, urn, urp);
        qualityScores[c] = score;
        overallQualityScore += score;
    }

    overallQualityScore /= numPredictions;
    prediction->overallQualityScore_ = overallQualityScore;
}
