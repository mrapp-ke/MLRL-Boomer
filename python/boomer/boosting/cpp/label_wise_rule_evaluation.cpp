#include "label_wise_rule_evaluation.h"
#include "linalg.h"
#include <cstddef>
#include <utility>

using namespace boosting;


LabelWiseDefaultRuleEvaluationImpl::LabelWiseDefaultRuleEvaluationImpl(
        std::shared_ptr<AbstractLabelWiseLoss> lossFunctionPtr, float64 l2RegularizationWeight) {
    lossFunctionPtr_ = lossFunctionPtr;
    l2RegularizationWeight_ = l2RegularizationWeight;
}

LabelWiseDefaultRuleEvaluationImpl::~LabelWiseDefaultRuleEvaluationImpl() {

}

Prediction* LabelWiseDefaultRuleEvaluationImpl::calculateDefaultPrediction(
        AbstractRandomAccessLabelMatrix* labelMatrix) {
    // Class members
    AbstractLabelWiseLoss* lossFunction = lossFunctionPtr_.get();
    float64 l2RegularizationWeight = l2RegularizationWeight_;
    // The number of examples
    intp numExamples = labelMatrix->numExamples_;
    // The number of labels
    intp numLabels = labelMatrix->numLabels_;
    // An array that stores the scores that are predicted by the default rule
    float64* predictedScores = (float64*) malloc(numLabels * sizeof(float64));

    for (intp c = 0; c < numLabels; c++) {
        float64 sumOfGradients = 0;
        float64 sumOfHessians = 0;

        for (intp r = 0; r < numExamples; r++) {
            // Calculate the gradient and Hessian for the current example and label...
            std::pair<float64, float64> pair = lossFunction->calculateGradientAndHessian(labelMatrix, r, c, 0);
            sumOfGradients += pair.first;
            sumOfHessians += pair.second;
        }

        // Calculate the score to be predicted by the default rule for the current label...
        predictedScores[c] = -sumOfGradients / (sumOfHessians + l2RegularizationWeight);
    }

    return new Prediction(numLabels, NULL, predictedScores);
}

LabelWiseRuleEvaluationImpl::LabelWiseRuleEvaluationImpl(float64 l2RegularizationWeight) {
    l2RegularizationWeight_ = l2RegularizationWeight;
}

void LabelWiseRuleEvaluationImpl::calculateLabelWisePrediction(const intp* labelIndices,
                                                               const float64* totalSumsOfGradients,
                                                               float64* sumsOfGradients,
                                                               const float64* totalSumsOfHessians,
                                                               float64* sumsOfHessians, bool uncovered,
                                                               LabelWisePredictionCandidate* prediction) {
    // Class members
    float64 l2RegularizationWeight = l2RegularizationWeight_;
    // The number of labels to predict for
    intp numPredictions = prediction->numPredictions_;
    // The array that should be used to store the predicted scores
    float64* predictedScores = prediction->predictedScores_;
    // The array that should be used to store the quality scores
    float64* qualityScores = prediction->qualityScores_;
    // The overall quality score, i.e., the sum of the quality scores for each label plus the L2 regularization term
    float64 overallQualityScore = 0;

    // For each label, calculate a score to be predicted, as well as a corresponding quality score...
    for (intp c = 0; c < numPredictions; c++) {
        float64 sumOfGradients = sumsOfGradients[c];
        float64 sumOfHessians =  sumsOfHessians[c];

        if (uncovered) {
            intp l = labelIndices != NULL ? labelIndices[c] : c;
            sumOfGradients = totalSumsOfGradients[l] - sumOfGradients;
            sumOfHessians = totalSumsOfHessians[l] - sumOfHessians;
        }

        // Calculate the score to be predicted for the current label...
        float64 score = sumOfHessians + l2RegularizationWeight;
        score = score != 0 ? -sumOfGradients / score : 0;
        predictedScores[c] = score;

        // Calculate the quality score for the current label...
        float64 scorePow = pow(score, 2);
        score = (sumOfGradients * score) + (0.5 * scorePow * sumOfHessians);
        qualityScores[c] = score + (0.5 * l2RegularizationWeight * scorePow);
        overallQualityScore += score;
    }

    // Add the L2 regularization term to the overall quality score...
    overallQualityScore += 0.5 * l2RegularizationWeight * linalg::l2NormPow(predictedScores, numPredictions);
    prediction->overallQualityScore_ = overallQualityScore;
}
