#include "post_processing.h"

using namespace boosting;


ConstantShrinkageImpl::ConstantShrinkageImpl(float64 shrinkage)
    : shrinkage_(shrinkage) {

}

void ConstantShrinkageImpl::postProcess(AbstractPrediction& prediction) const {
    uint32 numElements = prediction.getNumElements();
    AbstractPrediction::iterator iterator = prediction.begin();

    for (uint32 i = 0; i < numElements; i++) {
        iterator[i] *= shrinkage_;
    }
}
