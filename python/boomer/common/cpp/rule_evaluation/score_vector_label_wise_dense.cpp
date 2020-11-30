#include "score_vector_label_wise_dense.h"
#include "score_processor_label_wise.h"


DenseLabelWiseScoreVector::DenseLabelWiseScoreVector(uint32 numElements)
    : DenseScoreVector(numElements), qualityScoreVector_(DenseVector<float64>(numElements)) {

}

DenseLabelWiseScoreVector::quality_score_iterator DenseLabelWiseScoreVector::quality_scores_begin() {
    return qualityScoreVector_.begin();
}

DenseLabelWiseScoreVector::quality_score_iterator DenseLabelWiseScoreVector::quality_scores_end() {
    return qualityScoreVector_.end();
}

DenseLabelWiseScoreVector::quality_score_const_iterator DenseLabelWiseScoreVector::quality_scores_cbegin() const {
    return qualityScoreVector_.cbegin();
}

DenseLabelWiseScoreVector::quality_score_const_iterator DenseLabelWiseScoreVector::quality_scores_cend() const {
    return qualityScoreVector_.cend();
}

const AbstractEvaluatedPrediction* DenseLabelWiseScoreVector::processScores(
        const AbstractEvaluatedPrediction* bestHead, ILabelWiseScoreProcessor& scoreProcessor) const {
    return scoreProcessor.processScores(bestHead, *this);
}
