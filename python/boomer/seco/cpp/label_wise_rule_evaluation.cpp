#include "label_wise_rule_evaluation.h"

using namespace seco;


HeuristicLabelWiseRuleEvaluationImpl::HeuristicLabelWiseRuleEvaluationImpl(std::shared_ptr<IHeuristic> heuristicPtr,
                                                                           bool predictMajority)
    : heuristicPtr_(heuristicPtr), predictMajority_(predictMajority) {

}

void HeuristicLabelWiseRuleEvaluationImpl::calculateLabelWisePrediction(
        const uint32* labelIndices, const uint8* minorityLabels, const float64* confusionMatricesTotal,
        const float64* confusionMatricesSubset, const float64* confusionMatricesCovered, bool uncovered,
        LabelWiseEvaluatedPrediction& prediction) const {
    uint32 numPredictions = prediction.getNumElements();
    LabelWiseEvaluatedPrediction::iterator valueIterator = prediction.begin();
    LabelWiseEvaluatedPrediction::quality_score_iterator qualityScoreIterator = prediction.quality_scores_begin();
    float64 overallQualityScore = 0;

    for (uint32 c = 0; c < numPredictions; c++) {
        uint32 l = labelIndices != nullptr ? labelIndices[c] : c;

        // Set the score to be predicted for the current label...
        uint8 minorityLabel = minorityLabels[l];
        float64 score = (float64) (predictMajority_ ? (minorityLabel > 0 ? 0 : 1) : minorityLabel);
        valueIterator[c] = score;

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

        score = heuristicPtr_->evaluateConfusionMatrix(cin, cip, crn, crp, uin, uip, urn, urp);
        qualityScoreIterator[c] = score;
        overallQualityScore += score;
    }

    overallQualityScore /= numPredictions;
    prediction.overallQualityScore = overallQualityScore;
}
