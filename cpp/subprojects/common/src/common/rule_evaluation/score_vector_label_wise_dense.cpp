#include "common/rule_evaluation/score_vector_label_wise_dense.hpp"
#include "common/rule_evaluation/score_processor_label_wise.hpp"


template<typename T>
DenseLabelWiseScoreVector<T>::DenseLabelWiseScoreVector(const T& labelIndices)
    : DenseScoreVector<T>(labelIndices), qualityScoreVector_(DenseVector<float64>(labelIndices.getNumElements())) {

}

template<typename T>
typename DenseLabelWiseScoreVector<T>::quality_score_iterator DenseLabelWiseScoreVector<T>::quality_scores_begin() {
    return qualityScoreVector_.begin();
}

template<typename T>
typename DenseLabelWiseScoreVector<T>::quality_score_iterator DenseLabelWiseScoreVector<T>::quality_scores_end() {
    return qualityScoreVector_.end();
}

template<typename T>
typename DenseLabelWiseScoreVector<T>::quality_score_const_iterator DenseLabelWiseScoreVector<T>::quality_scores_cbegin() const {
    return qualityScoreVector_.cbegin();
}

template<typename T>
typename DenseLabelWiseScoreVector<T>::quality_score_const_iterator DenseLabelWiseScoreVector<T>::quality_scores_cend() const {
    return qualityScoreVector_.cend();
}

template<typename T>
const AbstractEvaluatedPrediction* DenseLabelWiseScoreVector<T>::processScores(
        const AbstractEvaluatedPrediction* bestHead, ILabelWiseScoreProcessor& scoreProcessor) const {
    return scoreProcessor.processScores(bestHead, *this);
}

template class DenseLabelWiseScoreVector<PartialIndexVector>;
template class DenseLabelWiseScoreVector<FullIndexVector>;
