#include "predictions.h"
#include <cstdlib>


Prediction::Prediction(uint32 numElements)
    : DenseVector<float64>(numElements) {
    predictedScores_ = nullptr;
    labelIndices_ = nullptr;
}

Prediction::~Prediction() {
    free(labelIndices_);
    free(predictedScores_);
}

PredictionCandidate::PredictionCandidate(uint32 numElements)
    : Prediction(numElements) {

}

FullPrediction::FullPrediction(uint32 numElements)
    : PredictionCandidate(numElements), RangeIndexVector(numElements) {

}

uint32 FullPrediction::getNumElements() const {
    return DenseVector<float64>::getNumElements();
}

void FullPrediction::setNumElements(uint32 numElements) {
    DenseVector<float64>::setNumElements(numElements);
    RangeIndexVector::setNumElements(numElements);
}

PartialPrediction::PartialPrediction(uint32 numElements)
    : PredictionCandidate(numElements), DenseIndexVector(numElements) {

}

uint32 PartialPrediction::getNumElements() const {
    return DenseVector<float64>::getNumElements();
}

void PartialPrediction::setNumElements(uint32 numElements) {
    DenseVector<float64>::setNumElements(numElements);
    DenseIndexVector::setNumElements(numElements);
}
