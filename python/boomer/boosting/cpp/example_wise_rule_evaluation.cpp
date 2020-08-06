#include "example_wise_rule_evaluation.h"
#include "linalg.h"
#include "blas.h"
#include "lapack.h"
#include <cstddef>
#include <stdlib.h>
#include <math.h>

using namespace boosting;


ExampleWiseDefaultRuleEvaluationImpl::ExampleWiseDefaultRuleEvaluationImpl(AbstractExampleWiseLoss* lossFunction,
                                                                           float64 l2RegularizationWeight,
                                                                           Lapack* lapack) {
    lossFunction_ = lossFunction;
    l2RegularizationWeight_ = l2RegularizationWeight;
    lapack_ = lapack;
}

ExampleWiseDefaultRuleEvaluationImpl::~ExampleWiseDefaultRuleEvaluationImpl() {
    delete lapack_;
}

DefaultPrediction* ExampleWiseDefaultRuleEvaluationImpl::calculateDefaultPrediction(AbstractLabelMatrix* labelMatrix) {
    // Class members
    AbstractExampleWiseLoss* lossFunction = lossFunction_;
    float64 l2RegularizationWeight = l2RegularizationWeight_;
    // The number of examples
    intp numExamples = labelMatrix->numExamples_;
    // The number of labels
    intp numLabels = labelMatrix->numLabels_;
    // The number of hessians
    intp numHessians = linalg::triangularNumber(numLabels);
    // An array that stores the gradients for an example
    float64* gradients = (float64*) malloc(numLabels * sizeof(float64));
    // An array that stores the sums of gradients
    float64* sumsOfGradients = (float64*) malloc(numLabels * sizeof(float64));
    arrays::setToZeros(sumsOfGradients, numLabels);
    // An array that stores the Hessians for an example
    float64* hessians = (float64*) malloc(numHessians * sizeof(float64));
    // An array that stores the sums of Hessians
    float64* sumsOfHessians = (float64*) malloc(numHessians * sizeof(float64));
    arrays::setToZeros(sumsOfHessians, numHessians);
    // An array of zeros that stores the scores to be predicted by the default rule
    float64* predictedScores = (float64*) malloc(numLabels * sizeof(float64));
    arrays::setToZeros(predictedScores, numLabels);
    // Arrays that are used to temporarily store values that are computed by LAPACK's DSYSV routine
    float64* dsysvTmpArray1 = (float64*) malloc(numLabels * numLabels * sizeof(float64));
    int* dsysvTmpArray2 = (int*) malloc(numLabels * sizeof(int));


    for (intp r = 0; r < numExamples; r++) {
        // Calculate the gradients and Hessians for the current example...
        lossFunction->calculateGradientsAndHessians(labelMatrix, r, predictedScores, gradients, hessians);

        for (intp c = 0; c < numLabels; c++) {
            sumsOfGradients[c] += gradients[c];
        }

        for (intp c = 0; c < numHessians; c++) {
            sumsOfHessians[c] += hessians[c];
        }
    }

    // Query the optimal "lwork" parameter to be used by LAPACK'S DSYSV routine...
    int lwork = lapack_->queryDsysvLworkParameter(dsysvTmpArray1, predictedScores, numLabels);
    double* dsysvTmpArray3 = (double*) malloc(lwork * sizeof(double));

    // Calculate the scores to be predicted by the default rule by solving the system of linear equations...
    lapack_->dsysv(sumsOfHessians, sumsOfGradients, dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3, predictedScores,
                   numLabels, lwork, l2RegularizationWeight);

    // Free allocated memory...
    free(gradients);
    free(sumsOfGradients);
    free(hessians);
    free(sumsOfHessians);
    free(dsysvTmpArray1);
    free(dsysvTmpArray2);
    free(dsysvTmpArray3);

    return new DefaultPrediction(numLabels, predictedScores);
}

ExampleWiseRuleEvaluationImpl::ExampleWiseRuleEvaluationImpl(float64 l2RegularizationWeight, Blas* blas,
                                                             Lapack* lapack) {
    l2RegularizationWeight_ = l2RegularizationWeight;
    blas_ = blas;
    lapack_ = lapack;
    dsysvTmpArray1_ = NULL;
    dsysvTmpArray2_ = NULL;
    dsysvTmpArray3_ = NULL;
    dspmvTmpArray_ = NULL;
    tmpGradients_ = NULL;
    tmpHessians_ = NULL;
}

ExampleWiseRuleEvaluationImpl::~ExampleWiseRuleEvaluationImpl() {
    delete blas_;
    delete lapack_;
    free(dsysvTmpArray1_);
    free(dsysvTmpArray2_);
    free(dsysvTmpArray3_);
    free(dspmvTmpArray_);
    free(tmpGradients_);
    free(tmpHessians_);
}

void ExampleWiseRuleEvaluationImpl::calculateLabelWisePrediction(const intp* labelIndices,
                                                                 const float64* totalSumsOfGradients,
                                                                 float64* sumsOfGradients,
                                                                 const float64* totalSumsOfHessians,
                                                                 float64* sumsOfHessians, bool uncovered,
                                                                 LabelWisePrediction* prediction) {
    // Class members
    float64 l2RegularizationWeight = l2RegularizationWeight_;
    // The number of elements in the arrays `predictedScores` and `qualityScores`
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
        qualityScores = (float64*) malloc(numPredictions * sizeof(float64));
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
    // The number of elements in the arrays `predictedScores`
    intp numPredictions = prediction->numPredictions_;
    // The array that should be used to store the predicted scores
    float64* predictedScores = prediction->predictedScores_;
    // Arrays that are used to temporarily store values that are computed by the DSYSV or DSPMV routine
    float64* dsysvTmpArray1 = dsysvTmpArray1_;
    int* dsysvTmpArray2 = dsysvTmpArray2_;
    double* dsysvTmpArray3 = dsysvTmpArray3_;
    int dsysvLwork = dsysvLwork_;
    float64* dspmvTmpArray = dspmvTmpArray_;

    // To avoid array recreation each time this function is called, the arrays for temporarily storing values that are
    // computed by the DSYSV or DSPMV routine are only initialized if they have not been initialized yet
    if (dsysvTmpArray1 == NULL) {
        dsysvTmpArray1 = (float64*) malloc(numPredictions * numPredictions * sizeof(float64));
        dsysvTmpArray1_ = dsysvTmpArray1;
        dsysvTmpArray2 = (int*) malloc(numPredictions * sizeof(int));
        dsysvTmpArray2_ = dsysvTmpArray2;
        dspmvTmpArray = (float64*) malloc(numPredictions * sizeof(float64));
        dspmvTmpArray_ = dspmvTmpArray;

        // Query the optimal "lwork" parameter to be used by LAPACK'S DSYSV routine...
        dsysvLwork = lapack_->queryDsysvLworkParameter(dsysvTmpArray1, predictedScores, numPredictions);
        dsysvLwork_ = dsysvLwork;
        dsysvTmpArray3 = (double*) malloc(dsysvLwork * sizeof(double));
        dsysvTmpArray3_ = dsysvTmpArray3;
    }

    float64* gradients;
    float64* hessians;

    if (uncovered) {
        gradients = tmpGradients_;
        hessians = tmpHessians_;
        intp i = 0;

        // To avoid array recreation each time this function is called, the arrays for storing the gradients and
        // hessians are only initialized if they have not been initialized yet
        if (gradients == NULL) {
            gradients = (float64*) malloc(numPredictions * sizeof(float64));
            tmpGradients_ = gradients;
            intp numHessians = linalg::triangularNumber(numPredictions);
            hessians = (float64*) malloc(numHessians * sizeof(float64));
            tmpHessians_ = hessians;
        }

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
    lapack_->dsysv(hessians, gradients, dsysvTmpArray1, dsysvTmpArray2, dsysvTmpArray3, predictedScores, numPredictions,
                   dsysvLwork, l2RegularizationWeight);

    // Calculate overall quality score as (gradients * scores) + (0.5 * (scores * (hessians * scores)))...
    float64 overallQualityScore = blas_->ddot(predictedScores, gradients, numPredictions);
    blas_->dspmv(hessians, predictedScores, dspmvTmpArray, numPredictions);
    overallQualityScore += 0.5 * blas_->ddot(predictedScores, dspmvTmpArray, numPredictions);

    // Add the L2 regularization term to the overall quality score...
    overallQualityScore += 0.5 * l2RegularizationWeight * linalg::l2NormPow(predictedScores, numPredictions);
    prediction->overallQualityScore_ = overallQualityScore;
}
