#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/rule_refinement/prediction.hpp"
#include "mlrl/common/rule_refinement/score_processor.hpp"

template<typename IndexVector>
DenseBinnedScoreVector<IndexVector>::DenseBinnedScoreVector(const IndexVector& labelIndices, uint32 numBins,
                                                            bool sorted)
    : labelIndices_(labelIndices), binnedVector_(labelIndices.getNumElements(), numBins), sorted_(sorted) {}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::index_const_iterator DenseBinnedScoreVector<IndexVector>::indices_cbegin()
  const {
    return labelIndices_.cbegin();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::index_const_iterator DenseBinnedScoreVector<IndexVector>::indices_cend()
  const {
    return labelIndices_.cend();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_const_iterator DenseBinnedScoreVector<IndexVector>::values_cbegin()
  const {
    return binnedVector_.cbegin();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_const_iterator DenseBinnedScoreVector<IndexVector>::values_cend()
  const {
    return BinnedConstIterator<float64>(this->indices_binned_cend(), binnedVector_.values_cbegin());
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::index_binned_iterator
  DenseBinnedScoreVector<IndexVector>::indices_binned_begin() {
    return binnedVector_.indices_begin();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::index_binned_iterator
  DenseBinnedScoreVector<IndexVector>::indices_binned_end() {
    return &binnedVector_.indices_begin()[labelIndices_.getNumElements()];
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::index_binned_const_iterator
  DenseBinnedScoreVector<IndexVector>::indices_binned_cbegin() const {
    return binnedVector_.indices_cbegin();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::index_binned_const_iterator
  DenseBinnedScoreVector<IndexVector>::indices_binned_cend() const {
    return &binnedVector_.indices_cbegin()[labelIndices_.getNumElements()];
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_binned_iterator
  DenseBinnedScoreVector<IndexVector>::values_binned_begin() {
    return binnedVector_.values_begin();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_binned_iterator
  DenseBinnedScoreVector<IndexVector>::values_binned_end() {
    return binnedVector_.values_end();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_binned_const_iterator
  DenseBinnedScoreVector<IndexVector>::values_binned_cbegin() const {
    return binnedVector_.values_cbegin();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_binned_const_iterator
  DenseBinnedScoreVector<IndexVector>::values_binned_cend() const {
    return binnedVector_.values_cend();
}

template<typename IndexVector>
uint32 DenseBinnedScoreVector<IndexVector>::getNumElements() const {
    return labelIndices_.getNumElements();
}

template<typename IndexVector>
uint32 DenseBinnedScoreVector<IndexVector>::getNumBins() const {
    return binnedVector_.getNumBins();
}

template<typename IndexVector>
void DenseBinnedScoreVector<IndexVector>::setNumBins(uint32 numBins, bool freeMemory) {
    binnedVector_.setNumBins(numBins, freeMemory);
}

template<typename IndexVector>
bool DenseBinnedScoreVector<IndexVector>::isPartial() const {
    return labelIndices_.isPartial();
}

template<typename IndexVector>
bool DenseBinnedScoreVector<IndexVector>::isSorted() const {
    return sorted_;
}

template<typename IndexVector>
void DenseBinnedScoreVector<IndexVector>::updatePrediction(IPrediction& prediction) const {
    prediction.set(this->values_cbegin(), this->values_cend());
}

template<typename IndexVector>
void DenseBinnedScoreVector<IndexVector>::processScores(ScoreProcessor& scoreProcessor) const {
    scoreProcessor.processScores(*this);
}

template class DenseBinnedScoreVector<PartialIndexVector>;
template class DenseBinnedScoreVector<CompleteIndexVector>;
