#include "example_wise_rule_evaluation.h"
#include "linalg.h"
#include "blas.h"
#include "lapack.h"
#include <cstddef>
#include <math.h>

using namespace boosting;


ExampleWiseRuleEvaluationImpl::ExampleWiseRuleEvaluationImpl(float64 l2RegularizationWeight) {
    l2RegularizationWeight_ = l2RegularizationWeight;
}

void ExampleWiseRuleEvaluationImpl::calculateLabelWisePrediction(const intp* labelIndices,
                                                                 const float64* totalSumsOfGradients,
                                                                 float64* sumsOfGradients,
                                                                 const float64* totalSumsOfHessians,
                                                                 float64* sumsOfHessians, bool uncovered,
                                                                 LabelWisePrediction* prediction) {
    // Class members
    float64 l2RegularizationWeight = l2RegularizationWeight_;
    // The number of elements in the arrays `predicted_scores` and `quality_scores`
    intp numPredictions = prediction->numPredictions_;
    // The array that should be used to store the predicted scores
    float64* predictedScores = prediction->predictedScores_;
    // The array that should be used to store the quality scores
    float64* qualityScores = prediction->qualityScores_;
    // The overall quality score, i.e. the sum of the quality scores for each label plus the L2 regularization term
    float64 overallQualityScore = 0;

    // To avoid array recreation each time this function is called, the array for storing the quality scores is only
    // initialized if it has not been initialized yet
    if (qualityScores == NULL) {
        qualityScores = arrays::mallocFloat64(numPredictions);
        prediction->qualityScores_ = qualityScores;
    }

    // For each label, calculate the score to be predicted, as well as a quality score...
    for (intp c = 0; c < numPredictions; c++) {
        float64 sumOfGradients = sumsOfGradients[c];
        intp c2 = linalg::triangularNumber(c + 1) - 1;
        float64 sumOfHessians = sumsOfHessians[c2];

        if (uncovered) {
            intp l = labelIndices != NULL ? labelIndices[c] : c;
            sumOfGradients = totalSumsOfGradients[l] - sumOfGradients;
            intp l2 = linalg::triangularNumber(l + 1) - 1;
            sumOfHessians = totalSumsOfHessians[l2] - sumOfHessians;
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

void ExampleWiseRuleEvaluationImpl::calculateExampleWisePrediction(const intp* labelIndices,
                                                                   const float64* totalSumsOfGradients,
                                                                   float64* sumsOfGradients,
                                                                   const float64* totalSumsOfHessians,
                                                                   float64* sumsOfHessians, bool uncovered,
                                                                   Prediction* prediction) {
    // Class members
    float64 l2RegularizationWeight = l2RegularizationWeight_;
    // The number of elements in the arrays `predicted_scores` and `quality_scores`
    intp numPredictions = prediction->numPredictions_;
    float64* gradients;
    float64* hessians;

    if (uncovered) {
        intp numHessians = linalg::triangularNumber(numPredictions);
        gradients = arrays::mallocFloat64(numPredictions);
        hessians = arrays::mallocFloat64(numHessians);
        intp i = 0;

        for (intp c = 0; c < numPredictions; c++) {
            intp l = labelIndices != NULL ? labelIndices[c] : c;
            gradients[c] = totalSumsOfGradients[l] - sumsOfGradients[c];
            intp offset = linalg::triangularNumber(l);

            for (intp c2 = 0; c2 < c + 1; c2++) {
                intp l2 = offset + (labelIndices != NULL ? labelIndices[c2] : c2);
                hessians[i] = totalSumsOfHessians[l2] - sumsOfHessians[i];
                i++;
            }
        }
    } else {
        gradients = sumsOfGradients;
        hessians = sumsOfHessians;
    }

    // Calculate the scores to be predicted for the individual labels by solving a system of linear equations...
    float64* predictedScores = dsysv(hessians, gradients, numPredictions, l2RegularizationWeight);
    prediction->predictedScores_ = predictedScores;

    // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
    float64 overallQualityScore = ddot(predictedScores, gradients, numPredictions);
    float64* tmp = dspmv(hessians, predictedScores, numPredictions);
    overallQualityScore += 0.5 * ddot(predictedScores, tmp, numPredictions);

    // Add the L2 regularization term to the overall quality score...
    overallQualityScore += 0.5 * l2RegularizationWeight * linalg::l2NormPow(predictedScores, numPredictions);
    prediction->overallQualityScore_ = overallQualityScore;
}
