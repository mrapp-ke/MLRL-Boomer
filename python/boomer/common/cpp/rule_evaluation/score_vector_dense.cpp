#include "score_vector_dense.h"
#include "score_processor.h"


DenseScoreVector::DenseScoreVector(uint32 numElements)
    : predictedScoreVector_(DenseVector<float64>(numElements)) {

}

DenseScoreVector::score_iterator DenseScoreVector::scores_begin() {
    return predictedScoreVector_.begin();
}

DenseScoreVector::score_iterator DenseScoreVector::scores_end() {
    return predictedScoreVector_.end();
}

DenseScoreVector::score_const_iterator DenseScoreVector::scores_cbegin() const {
    return predictedScoreVector_.cbegin();
}

DenseScoreVector::score_const_iterator DenseScoreVector::scores_cend() const {
    return predictedScoreVector_.cend();
}

uint32 DenseScoreVector::getNumElements() const {
    return predictedScoreVector_.getNumElements();
}

void DenseScoreVector::updatePrediction(AbstractPrediction& prediction) const {
    prediction.set(predictedScoreVector_.cbegin(), predictedScoreVector_.cend());
}

const AbstractEvaluatedPrediction* DenseScoreVector::processScores(const AbstractEvaluatedPrediction* bestHead,
                                                                   IScoreProcessor& scoreProcessor) const {
    return scoreProcessor.processScores(bestHead, *this);
}
