#include "mlrl/common/rule_refinement/prediction.hpp"

#include "mlrl/common/util/arrays.hpp"

AbstractPrediction::AbstractPrediction(uint32 numElements) : predictedScoreVector_(numElements) {}

uint32 AbstractPrediction::getNumElements() const {
    return predictedScoreVector_.getNumElements();
}

AbstractPrediction::value_iterator AbstractPrediction::values_begin() {
    return predictedScoreVector_.begin();
}

AbstractPrediction::value_iterator AbstractPrediction::values_end() {
    return predictedScoreVector_.end();
}

AbstractPrediction::value_const_iterator AbstractPrediction::values_cbegin() const {
    return predictedScoreVector_.cbegin();
}

AbstractPrediction::value_const_iterator AbstractPrediction::values_cend() const {
    return predictedScoreVector_.cend();
}

void AbstractPrediction::set(DenseVector<float64>::const_iterator begin, DenseVector<float64>::const_iterator end) {
    copyArray(begin, predictedScoreVector_.begin(), predictedScoreVector_.getNumElements());
}

void AbstractPrediction::set(DenseBinnedVector<float64>::const_iterator begin,
                             DenseBinnedVector<float64>::const_iterator end) {
    copyArray(begin, predictedScoreVector_.begin(), predictedScoreVector_.getNumElements());
}
