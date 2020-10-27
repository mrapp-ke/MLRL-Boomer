#include "predictions.h"
#include <cstdlib>


Prediction::Prediction(uint32 numPredictions, uint32* labelIndices, float64* predictedScores)
    : numPredictions_(numPredictions), labelIndices_(labelIndices), predictedScores_(predictedScores) {

}

Prediction::~Prediction() {
    free(labelIndices_);
    free(predictedScores_);
}

PredictionCandidate::PredictionCandidate(uint32 numPredictions, uint32* labelIndices, float64* predictedScores,
                                         float64 overallQualityScore)
    : Prediction(numPredictions, labelIndices, predictedScores), overallQualityScore_(overallQualityScore) {

}
