#include "prediction.h"


AbstractPrediction::AbstractPrediction(uint32 numElements)
    : predictedScoreVector_(DenseVector<float64>(numElements)) {

}

uint32 AbstractPrediction::getNumElements() const {
    return predictedScoreVector_.getNumElements();
}

void AbstractPrediction::setNumElements(uint32 numElements, bool freeMemory) {
    predictedScoreVector_.setNumElements(numElements, freeMemory);
}

AbstractPrediction::score_iterator AbstractPrediction::scores_begin() {
    return predictedScoreVector_.begin();
}

AbstractPrediction::score_iterator AbstractPrediction::scores_end() {
    return predictedScoreVector_.end();
}

AbstractPrediction::score_const_iterator AbstractPrediction::scores_cbegin() const {
    return predictedScoreVector_.cbegin();
}

AbstractPrediction::score_const_iterator AbstractPrediction::scores_cend() const {
    return predictedScoreVector_.cend();
}

void AbstractPrediction::set(AbstractPrediction::score_const_iterator begin, AbstractPrediction::score_const_iterator end) {
    uint32 numElements = predictedScoreVector_.getNumElements();
    DenseVector<float64>::iterator iterator = predictedScoreVector_.begin();

    for (uint32 i = 0; i < numElements; i++) {
        iterator[i] = begin[i];
    }
}
