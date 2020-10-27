#include "post_processing.h"

using namespace boosting;


ConstantShrinkageImpl::ConstantShrinkageImpl(float64 shrinkage)
    : shrinkage_(shrinkage) {

}

void ConstantShrinkageImpl::postProcess(Prediction& prediction) const {
    uint32 numElements = prediction.getNumElements();
    Prediction::iterator iterator = prediction.begin();

    for (uint32 i = 0; i < numElements; i++) {
        iterator[i] *= shrinkage_;
    }

    // TODO Remove the following
    float64* predictedScores = prediction.predictedScores_;

    for (uint32 i = 0; i < numElements; i++) {
        predictedScores[i] *= shrinkage_;
    }
}
