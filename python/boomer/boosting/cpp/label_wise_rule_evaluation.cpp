#include "label_wise_rule_evaluation.h"
#include "linalg.h"

using namespace boosting;


RegularizedLabelWiseRuleEvaluationImpl::RegularizedLabelWiseRuleEvaluationImpl(uint32 numPredictions,
                                                                               const uint32* labelIndices,
                                                                               float64 l2RegularizationWeight)
    : l2RegularizationWeight_(l2RegularizationWeight), labelIndices_(labelIndices),
      prediction_(LabelWiseEvaluatedPrediction(numPredictions)) {

}

const LabelWiseEvaluatedPrediction& RegularizedLabelWiseRuleEvaluationImpl::calculateLabelWisePrediction(
        const float64* totalSumsOfGradients, float64* sumsOfGradients, const float64* totalSumsOfHessians,
        float64* sumsOfHessians, bool uncovered) {
    uint32 numPredictions = prediction_.getNumElements();
    LabelWiseEvaluatedPrediction::iterator valueIterator = prediction_.begin();
    LabelWiseEvaluatedPrediction::quality_score_iterator qualityScoreIterator = prediction_.quality_scores_begin();
    float64 overallQualityScore = 0;

    // For each label, calculate a score to be predicted, as well as a corresponding quality score...
    for (uint32 c = 0; c < numPredictions; c++) {
        float64 sumOfGradients = sumsOfGradients[c];
        float64 sumOfHessians =  sumsOfHessians[c];

        if (uncovered) {
            uint32 l = labelIndices_ != nullptr ? labelIndices_[c] : c;
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
    prediction_.overallQualityScore = overallQualityScore;
    return prediction_;
}

RegularizedLabelWiseRuleEvaluationFactoryImpl::RegularizedLabelWiseRuleEvaluationFactoryImpl(
        float64 l2RegularizationWeight)
    : l2RegularizationWeight_(l2RegularizationWeight) {

}

std::unique_ptr<ILabelWiseRuleEvaluation> RegularizedLabelWiseRuleEvaluationFactoryImpl::create(
        const RangeIndexVector& indexVector, uint32 numLabelIndices, const uint32* labelIndices) const {
    return std::make_unique<RegularizedLabelWiseRuleEvaluationImpl>(numLabelIndices, labelIndices,
                                                                    l2RegularizationWeight_);
}

std::unique_ptr<ILabelWiseRuleEvaluation> RegularizedLabelWiseRuleEvaluationFactoryImpl::create(
        const DenseIndexVector& indexVector, uint32 numLabelIndices, const uint32* labelIndices) const {
    return std::make_unique<RegularizedLabelWiseRuleEvaluationImpl>(numLabelIndices, labelIndices,
                                                                    l2RegularizationWeight_);
}
