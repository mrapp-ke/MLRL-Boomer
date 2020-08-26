#include "example_wise_statistics.h"
#include "linalg.h"
#include <stdlib.h>
#include <cstddef>

using namespace boosting;


ExampleWiseRefinementSearchImpl::ExampleWiseRefinementSearchImpl(
        std::shared_ptr<ExampleWiseRuleEvaluationImpl> ruleEvaluationPtr, std::shared_ptr<Lapack> lapackPtr,
        intp numPredictions, const intp* labelIndices, intp numLabels, const float64* gradients,
        const float64* totalSumsOfGradients, const float64* hessians, const float64* totalSumsOfHessians) {
    ruleEvaluationPtr_ = ruleEvaluationPtr;
    lapackPtr_ = lapackPtr;
    numPredictions_ = numPredictions;
    labelIndices_ = labelIndices;
    numLabels_ = numLabels;
    gradients_ = gradients;
    totalSumsOfGradients_ = totalSumsOfGradients;
    float64* sumsOfGradients = (float64*) malloc(numPredictions * sizeof(float64));
    arrays::setToZeros(sumsOfGradients, numPredictions);
    sumsOfGradients_ = sumsOfGradients;
    accumulatedSumsOfGradients_ = NULL;
    hessians_ = hessians;
    totalSumsOfHessians_ = totalSumsOfHessians;
    intp numHessians = linalg::triangularNumber(numPredictions);
    float64* sumsOfHessians = (float64*) malloc(numHessians * sizeof(float64));
    arrays::setToZeros(sumsOfHessians, numHessians);
    sumsOfHessians_ = sumsOfHessians;
    accumulatedSumsOfHessians_ = NULL;
    float64* predictedScores = (float64*) malloc(numPredictions * sizeof(float64));
    prediction_ = new LabelWisePredictionCandidate(numPredictions, NULL, predictedScores, NULL, 0);
    tmpGradients_ = NULL;
    tmpHessians_ = NULL;
    dsysvTmpArray1_ = NULL;
    dsysvTmpArray2_ = NULL;
    dsysvTmpArray3_ = NULL;
    dspmvTmpArray_ = NULL;
}

ExampleWiseRefinementSearchImpl::~ExampleWiseRefinementSearchImpl() {
    free(sumsOfGradients_);
    free(accumulatedSumsOfGradients_);
    free(sumsOfHessians_);
    free(accumulatedSumsOfHessians_);
    free(tmpGradients_);
    free(tmpHessians_);
    free(dsysvTmpArray1_);
    free(dsysvTmpArray2_);
    free(dsysvTmpArray3_);
    free(dspmvTmpArray_);
    delete prediction_;
}

void ExampleWiseRefinementSearchImpl::updateSearch(intp statisticIndex, uint32 weight) {
    // Add the gradients and Hessians of the example at the given index (weighted by the given weight) to the current
    // sum of gradients and Hessians...
    intp offsetGradients = statisticIndex * numLabels_;
    intp offsetHessians = statisticIndex * linalg::triangularNumber(numLabels_);
    intp i = 0;

    for (intp c = 0; c < numPredictions_; c++) {
        intp l = labelIndices_ != NULL ? labelIndices_[c] : c;
        sumsOfGradients_[c] += (weight * gradients_[offsetGradients + l]);
        intp triangularNumber = linalg::triangularNumber(l);

        for (intp c2 = 0; c2 < c + 1; c2++) {
            intp l2 = triangularNumber + (labelIndices_ != NULL ? labelIndices_[c2] : c2);
            sumsOfHessians_[i] += (weight * hessians_[offsetHessians + l2]);
            i++;
        }
    }
}

void ExampleWiseRefinementSearchImpl::resetSearch() {
    intp numHessians = linalg::triangularNumber(numPredictions_);

    // Allocate arrays for storing the accumulated sums of gradients and Hessians, if necessary...
    if (accumulatedSumsOfGradients_ == NULL) {
        accumulatedSumsOfGradients_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfGradients_, numPredictions_);
        accumulatedSumsOfHessians_ = (float64*) malloc(numHessians * sizeof(float64));
        arrays::setToZeros(accumulatedSumsOfHessians_, numHessians);
    }

    // Reset the sum of gradients and Hessians for each label to zero and add it to the accumulated sums of gradients
    // and Hessians...
    for (intp c = 0; c < numPredictions_; c++) {
        accumulatedSumsOfGradients_[c] += sumsOfGradients_[c];
        sumsOfGradients_[c] = 0;
    }

    for (intp c = 0; c < numHessians; c++) {
        accumulatedSumsOfHessians_[c] += sumsOfHessians_[c];
        sumsOfHessians_[c] = 0;
    }
}

LabelWisePredictionCandidate* ExampleWiseRefinementSearchImpl::calculateLabelWisePrediction(bool uncovered,
                                                                                            bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;
    ruleEvaluationPtr_.get()->calculateLabelWisePrediction(labelIndices_, totalSumsOfGradients_, sumsOfGradients,
                                                           totalSumsOfHessians_, sumsOfHessians, uncovered,
                                                           prediction_);
    return prediction_;
}

PredictionCandidate* ExampleWiseRefinementSearchImpl::calculateExampleWisePrediction(bool uncovered, bool accumulated) {
    float64* sumsOfGradients = accumulated ? accumulatedSumsOfGradients_ : sumsOfGradients_;
    float64* sumsOfHessians = accumulated ? accumulatedSumsOfHessians_ : sumsOfHessians_;

    // To avoid array recreation each time this function is called, the temporary arrays are only initialized if they
    // have not been initialized yet
    if (tmpGradients_ == NULL) {
        tmpGradients_ = (float64*) malloc(numPredictions_ * sizeof(float64));
        intp numHessians = linalg::triangularNumber(numPredictions_);
        tmpHessians_ = (float64*) malloc(numHessians * sizeof(float64));
        dsysvTmpArray1_ = (float64*) malloc(numPredictions_ * numPredictions_ * sizeof(float64));
        dsysvTmpArray2_ = (int*) malloc(numPredictions_ * sizeof(int));
        dspmvTmpArray_ = (float64*) malloc(numPredictions_ * sizeof(float64));

        // Query the optimal "lwork" parameter to be used by LAPACK'S DSYSV routine...
        dsysvLwork_ = lapackPtr_.get()->queryDsysvLworkParameter(dsysvTmpArray1_, prediction_->predictedScores_,
                                                                 numPredictions_);
        dsysvTmpArray3_ = (double*) malloc(dsysvLwork_ * sizeof(double));
    }

    ruleEvaluationPtr_.get()->calculateExampleWisePrediction(labelIndices_, totalSumsOfGradients_, sumsOfGradients,
                                                             totalSumsOfHessians_, sumsOfHessians, tmpGradients_,
                                                             tmpHessians_, dsysvLwork_, dsysvTmpArray1_,
                                                             dsysvTmpArray2_, dsysvTmpArray3_, dspmvTmpArray_,
                                                             uncovered, prediction_);
    return prediction_;
}

ExampleWiseStatisticsImpl::ExampleWiseStatisticsImpl(std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr,
                                                     std::shared_ptr<ExampleWiseRuleEvaluationImpl> ruleEvaluationPtr,
                                                     std::shared_ptr<Lapack> lapackPtr) {
    lossFunctionPtr_ = lossFunctionPtr;
    ruleEvaluationPtr_ = ruleEvaluationPtr;
    lapackPtr_ = lapackPtr;
    currentScores_ = NULL;
    gradients_ = NULL;
    totalSumsOfGradients_ = NULL;
    hessians_ = NULL;
    totalSumsOfHessians_ = NULL;
}

ExampleWiseStatisticsImpl::~ExampleWiseStatisticsImpl() {
    free(currentScores_);
    free(gradients_);
    free(totalSumsOfGradients_);
    free(hessians_);
    free(totalSumsOfHessians_);
}

void ExampleWiseStatisticsImpl::applyDefaultPrediction(std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr,
                                                       Prediction* defaultPrediction) {
    // Class members
    AbstractExampleWiseLoss* lossFunction = lossFunctionPtr_.get();
    // The number of examples
    intp numExamples = labelMatrixPtr.get()->numExamples_;
    // The number of labels
    intp numLabels = labelMatrixPtr.get()->numLabels_;
    // The number of hessians
    intp numHessians = linalg::triangularNumber(numLabels);
    // A matrix that stores the currently predicted scores for each example and label
    float64* currentScores = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // A matrix that stores the gradients for each example
    float64* gradients = (float64*) malloc(numExamples * numLabels * sizeof(float64));
    // An array that stores the column-wise sums of the matrix of gradients
    float64* totalSumsOfGradients = (float64*) malloc(numLabels * sizeof(float64));
    // A matrix that stores the Hessians for each example
    float64* hessians = (float64*) malloc(numExamples * numHessians * sizeof(float64));
    // An array that stores the column-wise sums of the matrix of Hessians
    float64* totalSumsOfHessians = (float64*) malloc(numHessians * sizeof(float64));

    for (intp r = 0; r < numExamples; r++) {
        intp offset = r * numLabels;

        for (intp c = 0; c < numLabels; c++) {
            // Store the score that is predicted by the default rule for the current example and label...
            currentScores[offset + c] = 0;
        }

        // Calculate the gradients and Hessians for the current example...
        lossFunction->calculateGradientsAndHessians(labelMatrixPtr.get(), r, &currentScores[offset], &gradients[offset],
                                                    &hessians[r * numHessians]);
    }

    // Store class members...
    labelMatrixPtr_ = labelMatrixPtr;
    currentScores_ = currentScores;
    gradients_ = gradients;
    totalSumsOfGradients_ = totalSumsOfGradients;
    hessians_ = hessians;
    totalSumsOfHessians_ = totalSumsOfHessians;
}

void ExampleWiseStatisticsImpl::resetCoveredStatistics() {
    intp numLabels = labelMatrixPtr_.get()->numLabels_;
    arrays::setToZeros(totalSumsOfGradients_, numLabels);
    intp numHessians = linalg::triangularNumber(numLabels);
    arrays::setToZeros(totalSumsOfHessians_, numHessians);
}

void ExampleWiseStatisticsImpl::updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove) {
    float64 signedWeight = remove ? -((float64) weight) : weight;
    intp numElements = labelMatrixPtr_.get()->numLabels_;
    intp offset = statisticIndex * numElements;

    // Add the gradients of the example at the given index (weighted by the given weight) to the total sums of
    // gradients...
    for (intp c = 0; c < numElements; c++) {
        totalSumsOfGradients_[c] += (signedWeight * gradients_[offset + c]);
    }

    numElements = linalg::triangularNumber(numElements);
    offset = statisticIndex * numElements;

    // Add the Hessians of the example at the given index (weighted by the given weight) to the total sums of
    // Hessians...
    for (intp c = 0; c < numElements; c++) {
        totalSumsOfHessians_[c] += (signedWeight * hessians_[offset + c]);
    }
}

AbstractRefinementSearch* ExampleWiseStatisticsImpl::beginSearch(intp numLabelIndices, const intp* labelIndices) {
    intp numLabels = labelMatrixPtr_.get()->numLabels_;
    intp numPredictions = labelIndices == NULL ? numLabels : numLabelIndices;
    return new ExampleWiseRefinementSearchImpl(ruleEvaluationPtr_, lapackPtr_, numPredictions, labelIndices, numLabels,
                                               gradients_, totalSumsOfGradients_, hessians_, totalSumsOfHessians_);
}

void ExampleWiseStatisticsImpl::applyPrediction(intp statisticIndex, Prediction* prediction) {
    AbstractExampleWiseLoss* lossFunction = lossFunctionPtr_.get();
    intp numPredictions = prediction->numPredictions_;
    const intp* labelIndices = prediction->labelIndices_;
    const float64* predictedScores = prediction->predictedScores_;
    intp numLabels = labelMatrixPtr_.get()->numLabels_;
    intp offset = statisticIndex * numLabels;
    intp numHessians = linalg::triangularNumber(numLabels);

    // Traverse the labels for which the new rule predicts to update the scores that are currently predicted for the
    // example at the given index...
    for (intp c = 0; c < numPredictions; c++) {
        intp l = labelIndices != NULL ? labelIndices[c] : c;
        currentScores_[offset + l] += predictedScores[c];
    }

    // Update the gradients and Hessians for the example at the given index...
    lossFunction->calculateGradientsAndHessians(labelMatrixPtr_.get(), statisticIndex, &currentScores_[offset],
                                                &gradients_[offset], &hessians_[statisticIndex * numHessians]);
}

ExampleWiseStatisticsFactoryImpl::ExampleWiseStatisticsFactoryImpl(
        std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr) {
    labelMatrixPtr_ = labelMatrixPtr;
}

ExampleWiseStatisticsFactoryImpl::~ExampleWiseStatisticsFactoryImpl() {

}

AbstractStatistics* ExampleWiseStatisticsFactoryImpl::create() {
    // TODO
    return NULL;
}
