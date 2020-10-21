#include "example_wise_rule_evaluation.h"
#include "linalg.h"

using namespace boosting;


RegularizedExampleWiseRuleEvaluationImpl::RegularizedExampleWiseRuleEvaluationImpl(float64 l2RegularizationWeight,
                                                                                   std::shared_ptr<Blas> blasPtr,
                                                                                   std::shared_ptr<Lapack> lapackPtr)
    : l2RegularizationWeight_(l2RegularizationWeight), blasPtr_(std::move(blasPtr)), lapackPtr_(std::move(lapackPtr)) {

}

void RegularizedExampleWiseRuleEvaluationImpl::calculateLabelWisePrediction(
        const uint32* labelIndices, const float64* totalSumsOfGradients, float64* sumsOfGradients,
        const float64* totalSumsOfHessians, float64* sumsOfHessians, bool uncovered,
        LabelWiseEvaluatedPrediction& prediction) const {
    uint32 numPredictions = prediction.getNumElements();
    LabelWiseEvaluatedPrediction::iterator valueIterator = prediction.begin();
    LabelWiseEvaluatedPrediction::quality_score_iterator qualityScoreIterator = prediction.quality_scores_begin();
    float64 overallQualityScore = 0;

    // For each label, calculate the score to be predicted, as well as a quality score...
    for (uint32 c = 0; c < numPredictions; c++) {
        float64 sumOfGradients = sumsOfGradients[c];
        uint32 c2 = linalg::triangularNumber(c + 1) - 1;
        float64 sumOfHessians = sumsOfHessians[c2];

        if (uncovered) {
            uint32 l = labelIndices != nullptr ? labelIndices[c] : c;
            sumOfGradients = totalSumsOfGradients[l] - sumOfGradients;
            uint32 l2 = linalg::triangularNumber(l + 1) - 1;
            sumOfHessians = totalSumsOfHessians[l2] - sumOfHessians;
        }

        // Calculate the score to be predicted for the current label...
        float64 score = sumOfHessians + l2RegularizationWeight_;
        score = score != 0 ? -sumOfGradients / score : 0;
        valueIterator[c] = score;

        // Calculate the quality score for the current label...
        float64 scorePow = score * score;
        score = (sumOfGradients * score) + (0.5 * scorePow * sumOfHessians);
        qualityScoreIterator[c] = score + (0.5 * l2RegularizationWeight_ * scorePow);
        overallQualityScore += score;
    }

    // Add the L2 regularization term to the overall quality score...
    overallQualityScore += 0.5 * l2RegularizationWeight_ * linalg::l2NormPow(valueIterator, numPredictions);
    prediction.overallQualityScore = overallQualityScore;
}

void RegularizedExampleWiseRuleEvaluationImpl::calculateExampleWisePrediction(
        const uint32* labelIndices, const float64* totalSumsOfGradients, float64* sumsOfGradients,
        const float64* totalSumsOfHessians, float64* sumsOfHessians, float64* tmpGradients, float64* tmpHessians,
        int dsysvLwork, float64* dsysvTmpArray1, int* dsysvTmpArray2, double* dsysvTmpArray3, float64* dspmvTmpArray,
        bool uncovered, EvaluatedPrediction& prediction) const {
    uint32 numPredictions = prediction.getNumElements();
    EvaluatedPrediction::iterator valueIterator = prediction.begin();

    float64* gradients;
    float64* hessians;

    if (uncovered) {
        gradients = tmpGradients;
        hessians = tmpHessians;
        uint32 i = 0;

        for (uint32 c = 0; c < numPredictions; c++) {
            uint32 l = labelIndices != nullptr ? labelIndices[c] : c;
            gradients[c] = totalSumsOfGradients[l] - sumsOfGradients[c];
            uint32 offset = linalg::triangularNumber(l);

            for (uint32 c2 = 0; c2 < c + 1; c2++) {
                uint32 l2 = offset + (labelIndices != nullptr ? labelIndices[c2] : c2);
                hessians[i] = totalSumsOfHessians[l2] - sumsOfHessians[i];
                i++;
            }
        }
    } else {
        gradients = sumsOfGradients;
        hessians = sumsOfHessians;
    }

    // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
    lapackPtr_->dsysv(hessians, gradients, dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3, valueIterator,
                      numPredictions, dsysvLwork, l2RegularizationWeight_);

    // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
    float64 overallQualityScore = blasPtr_->ddot(valueIterator, gradients, numPredictions);
    blasPtr_->dspmv(hessians, valueIterator, dspmvTmpArray, numPredictions);
    overallQualityScore += 0.5 * blasPtr_->ddot(valueIterator, dspmvTmpArray, numPredictions);

    // Add the L2 regularization term to the overall quality score...
    overallQualityScore += 0.5 * l2RegularizationWeight_ * linalg::l2NormPow(valueIterator, numPredictions);
    prediction.overallQualityScore = overallQualityScore;
}
