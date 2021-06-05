#include "common/rule_evaluation/score_vector_label_wise_binned_dense.hpp"
#include "common/rule_evaluation/score_processor_label_wise.hpp"


template<typename T>
DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::QualityScoreIterator(
        DenseVector<uint32>::const_iterator binIndexIterator, DenseVector<float64>::const_iterator qualityScoreIterator)
    : binIndexIterator_(binIndexIterator), qualityScoreIterator_(qualityScoreIterator) {

}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::reference DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator[](
        uint32 index) const {
    uint32 binIndex = binIndexIterator_[index];
    return qualityScoreIterator_[binIndex];
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::reference DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator*() const {
    uint32 binIndex = *binIndexIterator_;
    return qualityScoreIterator_[binIndex];
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator++() {
    ++binIndexIterator_;
    return *this;
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator++(
        int n) {
    binIndexIterator_++;
    return *this;
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator--() {
    --binIndexIterator_;
    return *this;
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator--(
        int n) {
    binIndexIterator_--;
    return *this;
}

template<typename T>
bool DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator!=(
        const DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& rhs) const {
    return binIndexIterator_ != rhs.binIndexIterator_;
}

template<typename T>
bool DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator==(
        const DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& rhs) const {
    return binIndexIterator_ == rhs.binIndexIterator_;
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::difference_type DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator::operator-(
        const DenseBinnedLabelWiseScoreVector<T>::QualityScoreIterator& rhs) const {
    return (difference_type) (binIndexIterator_ - rhs.binIndexIterator_);
}

template<typename T>
DenseBinnedLabelWiseScoreVector<T>::DenseBinnedLabelWiseScoreVector(const T& labelIndices, uint32 numBins)
    : DenseBinnedScoreVector<T>(labelIndices, numBins), qualityScoreVector_(DenseVector<float64>(numBins)) {

}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::quality_score_const_iterator DenseBinnedLabelWiseScoreVector<T>::quality_scores_cbegin() const {
    return QualityScoreIterator(this->indices_binned_cbegin(), qualityScoreVector_.cbegin());
}

template<typename T>
typename DenseBinnedLabelWiseScoreVector<T>::quality_score_const_iterator DenseBinnedLabelWiseScoreVector<T>::quality_scores_cend() const {
    return QualityScoreIterator(this->indices_binned_cend(), qualityScoreVector_.cbegin());
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
