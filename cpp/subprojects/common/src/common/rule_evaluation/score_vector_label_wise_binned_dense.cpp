#include "common/rule_evaluation/score_vector_label_wise_binned_dense.hpp"
#include "common/rule_evaluation/score_processor_label_wise.hpp"


template<typename T>
DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::QualityScoreIterator(
        const DenseBinnedLabelWiseScoreVector<T>& vector, uint32 index)
    : vector_(vector), index_(index) {

}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::reference DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator[](
        uint32 index) const {
    uint32 binIndex = vector_.indices_binned_cbegin()[index];
    return vector_.qualityScoreVector_.cbegin()[binIndex];
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::reference DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator*() const {
    uint32 binIndex = vector_.indices_binned_cbegin()[index_];
    return vector_.qualityScoreVector_.cbegin()[binIndex];
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator++() {
    ++index_;
    return *this;
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator++(
        int n) {
    index_++;
    return *this;
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator--() {
    --index_;
    return *this;
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator--(
        int n) {
    index_--;
    return *this;
}

template<typename T>
bool DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator!=(
        const DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& rhs) const {
    return index_ != rhs.index_;
}

template<typename T>
bool DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator==(
        const DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& rhs) const {
    return index_ == rhs.index_;
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::difference_type DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator-(
        const DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& rhs) const {
    return (difference_type) index_ - (difference_type) rhs.index_;
}

template<typename T>
DenseBinnedLabelWiseScoreVector<T>::DenseBinnedLabelWiseScoreVector(const T& labelIndices, uint32 numBins)
    : DenseBinnedScoreVector<T>(labelIndices, numBins), qualityScoreVector_(DenseVector<float64>(numBins)) {

}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::quality_score_const_iterator DenseBinnedLabelWiseScoreVector<T>::quality_scores_cbegin() const {
    return DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator(*this, 0);
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::quality_score_const_iterator DenseBinnedLabelWiseScoreVector<T>::quality_scores_cend() const {
    return DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator(*this, this->getNumElements());
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_iterator DenseBinnedLabelWiseScoreVector<T>::quality_scores_binned_begin() {
    return qualityScoreVector_.begin();
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_iterator DenseBinnedLabelWiseScoreVector<T>::quality_scores_binned_end() {
    return qualityScoreVector_.end();
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_const_iterator DenseBinnedLabelWiseScoreVector<T>::quality_scores_binned_cbegin() const {
    return qualityScoreVector_.cbegin();
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::quality_score_binned_const_iterator DenseBinnedLabelWiseScoreVector<T>::quality_scores_binned_cend() const {
    return qualityScoreVector_.cend();
}

template<typename T>
const AbstractEvaluatedPrediction* DenseBinnedLabelWiseScoreVector<T>::processScores(
        const AbstractEvaluatedPrediction* bestHead, ILabelWiseScoreProcessor& scoreProcessor) const {
    return scoreProcessor.processScores(bestHead, *this);
}

template class DenseBinnedLabelWiseScoreVector<PartialIndexVector>;
template class DenseBinnedLabelWiseScoreVector<FullIndexVector>;
