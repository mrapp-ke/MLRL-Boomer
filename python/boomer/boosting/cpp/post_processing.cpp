#include "post_processing.h"


ConstantShrinkage::ConstantShrinkage(float64 shrinkage)
    : shrinkage_(shrinkage) {

}

void ConstantShrinkage::postProcess(Prediction& prediction) const {
    uint32 numPredictions = prediction.numPredictions_;
    float64* predictedScores = prediction.predictedScores_;

    for (uint32 i = 0; i < numPredictions; i++) {
        predictedScores[i] *= shrinkage_;
    }
}
