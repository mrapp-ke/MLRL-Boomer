#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/rule_refinement/prediction.hpp"
#include "mlrl/common/rule_refinement/score_processor.hpp"

template<typename ScoreType, typename IndexVector>
DenseBinnedScoreVector<ScoreType, IndexVector>::DenseBinnedScoreVector(const IndexVector& outputIndices, uint32 numBins,
                                                                       bool sorted)
    : BinnedVectorDecorator<ViewDecorator<CompositeVector<AllocatedVector<uint32>, ResizableVector<ScoreType>>>>(
        CompositeVector<AllocatedVector<uint32>, ResizableVector<ScoreType>>(
          AllocatedVector<uint32>(outputIndices.getNumElements()), ResizableVector<ScoreType>(numBins))),
      outputIndices_(outputIndices), sorted_(sorted), maxCapacity_(numBins) {}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::indices_cbegin() const {
    return outputIndices_.cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::indices_cend() const {
    return outputIndices_.cend();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::values_cbegin() const {
    return value_const_iterator(View<const uint32>(this->bin_indices_cbegin()),
                                View<const ScoreType>(this->bin_values_cbegin()), 0);
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::values_cend() const {
    return value_const_iterator(View<const uint32>(this->bin_indices_cbegin()),
                                View<const ScoreType>(this->bin_values_cbegin()), this->getNumElements());
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_index_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_indices_begin() {
    return this->view.firstView.begin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_index_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_indices_end() {
    return &this->view.firstView.array[this->getNumElements()];
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_indices_cbegin() const {
    return this->view.firstView.cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_indices_cend() const {
    return &this->view.firstView.array[this->getNumElements()];
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_value_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_values_begin() {
    return this->view.secondView.begin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_value_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_values_end() {
    return this->view.secondView.end();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_values_cbegin() const {
    return this->view.secondView.cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_values_cend() const {
    return this->view.secondView.cend();
}

template<typename ScoreType, typename IndexVector>
uint32 DenseBinnedScoreVector<ScoreType, IndexVector>::getNumElements() const {
    return outputIndices_.getNumElements();
}

template<typename ScoreType, typename IndexVector>
void DenseBinnedScoreVector<ScoreType, IndexVector>::setNumBins(uint32 numBins, bool freeMemory) {
    this->view.secondView.resize(numBins, freeMemory);
}

template<typename ScoreType, typename IndexVector>
bool DenseBinnedScoreVector<ScoreType, IndexVector>::isPartial() const {
    return outputIndices_.isPartial();
}

template<typename ScoreType, typename IndexVector>
bool DenseBinnedScoreVector<ScoreType, IndexVector>::isSorted() const {
    return sorted_;
}

template<typename ScoreType, typename IndexVector>
void DenseBinnedScoreVector<ScoreType, IndexVector>::updatePrediction(IPrediction& prediction) const {
    prediction.set(this->values_cbegin(), this->values_cend());
}

template<typename ScoreType, typename IndexVector>
void DenseBinnedScoreVector<ScoreType, IndexVector>::processScores(ScoreProcessor& scoreProcessor) const {
    scoreProcessor.processScores(*this);
}

template class DenseBinnedScoreVector<float32, PartialIndexVector>;
template class DenseBinnedScoreVector<float64, PartialIndexVector>;
template class DenseBinnedScoreVector<float32, CompleteIndexVector>;
template class DenseBinnedScoreVector<float64, CompleteIndexVector>;
