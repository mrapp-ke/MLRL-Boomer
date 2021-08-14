#include "common/rule_evaluation/score_vector_binned_dense.hpp"
#include "common/rule_refinement/prediction.hpp"
#include "common/rule_refinement/score_processor.hpp"
#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"


template<typename T>
DenseBinnedScoreVector<T>::DenseBinnedScoreVector(const T& labelIndices, uint32 numBins)
    : labelIndices_(labelIndices), binnedVector_(DenseBinnedVector<float64>(labelIndices.getNumElements(), numBins)) {

}

template<typename T>
typename DenseBinnedScoreVector<T>::index_const_iterator DenseBinnedScoreVector<T>::indices_cbegin() const {
    return labelIndices_.cbegin();
}

template<typename T>
typename DenseBinnedScoreVector<T>::index_const_iterator DenseBinnedScoreVector<T>::indices_cend() const {
    return labelIndices_.cend();
}

template<typename T>
typename DenseBinnedScoreVector<T>::score_const_iterator DenseBinnedScoreVector<T>::scores_cbegin() const {
    return binnedVector_.cbegin();
}

template<typename T>
typename DenseBinnedScoreVector<T>::score_const_iterator DenseBinnedScoreVector<T>::scores_cend() const {
    return binnedVector_.cend();
}

template<typename T>
typename DenseBinnedScoreVector<T>::index_binned_iterator DenseBinnedScoreVector<T>::indices_binned_begin() {
    return binnedVector_.indices_binned_begin();
}

template<typename T>
typename DenseBinnedScoreVector<T>::index_binned_iterator DenseBinnedScoreVector<T>::indices_binned_end() {
    return binnedVector_.indices_binned_end();
}

template<typename T>
typename DenseBinnedScoreVector<T>::index_binned_const_iterator DenseBinnedScoreVector<T>::indices_binned_cbegin() const {
    return binnedVector_.indices_binned_cbegin();
}

template<typename T>
typename DenseBinnedScoreVector<T>::index_binned_const_iterator DenseBinnedScoreVector<T>::indices_binned_cend() const {
    return binnedVector_.indices_binned_cend();
}

template<typename T>
typename DenseBinnedScoreVector<T>::score_binned_iterator DenseBinnedScoreVector<T>::scores_binned_begin() {
    return binnedVector_.binned_begin();
}

template<typename T>
typename DenseBinnedScoreVector<T>::score_binned_iterator DenseBinnedScoreVector<T>::scores_binned_end() {
    return binnedVector_.binned_end();
}

template<typename T>
typename DenseBinnedScoreVector<T>::score_binned_const_iterator DenseBinnedScoreVector<T>::scores_binned_cbegin() const {
    return binnedVector_.binned_cbegin();
}

template<typename T>
typename DenseBinnedScoreVector<T>::score_binned_const_iterator DenseBinnedScoreVector<T>::scores_binned_cend() const {
    return binnedVector_.binned_cend();
}

template<typename T>
uint32 DenseBinnedScoreVector<T>::getNumElements() const {
    return binnedVector_.getNumElements();
}

template<typename T>
uint32 DenseBinnedScoreVector<T>::getNumBins() const {
    return binnedVector_.getNumBins();
}

template<typename T>
void DenseBinnedScoreVector<T>::setNumBins(uint32 numBins, bool freeMemory) {
    binnedVector_.setNumBins(numBins, freeMemory);
}

template<typename T>
bool DenseBinnedScoreVector<T>::isPartial() const {
    return labelIndices_.isPartial();
}

template<typename T>
void DenseBinnedScoreVector<T>::updatePrediction(AbstractPrediction& prediction) const {
    prediction.set(binnedVector_.cbegin(), binnedVector_.cend());
}

template<typename T>
const AbstractEvaluatedPrediction* DenseBinnedScoreVector<T>::processScores(const AbstractEvaluatedPrediction* bestHead,
                                                                            ScoreProcessor& scoreProcessor) const {
    return scoreProcessor.processScores(bestHead, *this);
}

template class DenseBinnedScoreVector<PartialIndexVector>;
template class DenseBinnedScoreVector<CompleteIndexVector>;
