#include "head_refinement.h"
#include <stdlib.h>

using namespace head_refinement;


HeadCandidate::HeadCandidate(intp numPredictions, intp* labelIndices, float64* predictedScores, float64 qualityScore) {
    numPredictions_ = numPredictions;
    labelIndices_ = labelIndices;
    predictedScores_ = predictedScores;
    qualityScore_ = qualityScore;
}

HeadCandidate::~HeadCandidate() {
    free(labelIndices_);
    free(predictedScores_);
}
