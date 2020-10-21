#include "label_wise_rule_evaluation.h"
#include "linalg.h"
#include <cstddef>
#include <math.h>

using namespace boosting;


RegularizedLabelWiseRuleEvaluationImpl::RegularizedLabelWiseRuleEvaluationImpl(float64 l2RegularizationWeight)
    : l2RegularizationWeight_(l2RegularizationWeight) {

}

void RegularizedLabelWiseRuleEvaluationImpl::calculateLabelWisePrediction(
        const uint32* labelIndices, const float64* totalSumsOfGradients, float64* sumsOfGradients,
        const float64* totalSumsOfHessians, float64* sumsOfHessians, bool uncovered,
        LabelWiseEvaluatedPrediction& prediction) const {
    uint32 numPredictions = prediction.getNumElements();
    LabelWiseEvaluatedPrediction::iterator valueIterator = prediction.begin();
    LabelWiseEvaluatedPrediction::quality_score_iterator qualityScoreIterator = prediction.quality_scores_begin();
    float64 overallQualityScore = 0;

    // For each label, calculate a score to be predicted, as well as a corresponding quality score...
    for (uint32 c = 0; c < numPredictions; c++) {
        float64 sumOfGradients = sumsOfGradients[c];
        float64 sumOfHessians =  sumsOfHessians[c];

        if (uncovered) {
            uint32 l = labelIndices != NULL ? labelIndices[c] : c;
            sumOfGradients = totalSumsOfGradients[l] - sumOfGradients;
            sumOfHessians = totalSumsOfHessians[l] - sumOfHessians;
        }

        // Calculate the score to be predicted for the current label...
        float64 score = sumOfHessians + l2RegularizationWeight_;
        score = score != 0 ? -sumOfGradients / score : 0;
        valueIterator[c] = score;

        // Calculate the quality score for the current label...
        float64 scorePow = pow(score, 2);
        score = (sumOfGradients * score) + (0.5 * scorePow * sumOfHessians);
        qualityScoreIterator[c] = score + (0.5 * l2RegularizationWeight_ * scorePow);
        overallQualityScore += score;
    }

    // Add the L2 regularization term to the overall quality score...
    overallQualityScore += 0.5 * l2RegularizationWeight_ * linalg::l2NormPow(valueIterator, numPredictions);
    prediction.overallQualityScore = overallQualityScore;
}
