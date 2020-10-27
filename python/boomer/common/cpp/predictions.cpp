#include "predictions.h"
#include <cstdlib>


Prediction::Prediction(uint32 numPredictions)
    : numPredictions_(numPredictions) {
    predictedScores_ = nullptr;
    labelIndices_ = nullptr;
}

Prediction::~Prediction() {
    free(labelIndices_);
    free(predictedScores_);
}

PredictionCandidate::PredictionCandidate(uint32 numPredictions)
    : Prediction(numPredictions) {

}
