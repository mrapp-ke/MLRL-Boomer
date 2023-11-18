#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"

#include "mlrl/common/indices/index_vector_complete.hpp"
#include "mlrl/common/indices/index_vector_partial.hpp"
#include "mlrl/common/rule_refinement/prediction.hpp"
#include "mlrl/common/rule_refinement/score_processor.hpp"

template<typename IndexVector>
DenseBinnedScoreVector<IndexVector>::DenseBinnedScoreVector(const IndexVector& labelIndices, uint32 numBins,
                                                            bool sorted)
    : BinnedVectorDecorator<CompositeVectorDecorator<AllocatedVector<uint32>, AllocatedVector<float64>>>(
      AllocatedVector<uint32>(labelIndices.getNumElements()), AllocatedVector<float64>(numBins)),
      labelIndices_(labelIndices), sorted_(sorted), maxCapacity_(numBins) {}

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
    return BinnedConstIterator<float64>(this->indices_binned_cbegin(), this->values_binned_cbegin());
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_const_iterator DenseBinnedScoreVector<IndexVector>::values_cend()
  const {
    return BinnedConstIterator<float64>(this->indices_binned_cend(), this->values_binned_cbegin());
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::index_binned_iterator
  DenseBinnedScoreVector<IndexVector>::indices_binned_begin() {
    return this->firstView_.begin();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::index_binned_iterator
  DenseBinnedScoreVector<IndexVector>::indices_binned_end() {
    return &this->firstView_.array[this->getNumElements()];
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::index_binned_const_iterator
  DenseBinnedScoreVector<IndexVector>::indices_binned_cbegin() const {
    return this->firstView_.cbegin();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::index_binned_const_iterator
  DenseBinnedScoreVector<IndexVector>::indices_binned_cend() const {
    return &this->firstView_.array[this->getNumElements()];
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_binned_iterator
  DenseBinnedScoreVector<IndexVector>::values_binned_begin() {
    return this->secondView_.begin();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_binned_iterator
  DenseBinnedScoreVector<IndexVector>::values_binned_end() {
    return this->secondView_.end();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_binned_const_iterator
  DenseBinnedScoreVector<IndexVector>::values_binned_cbegin() const {
    return this->secondView_.cbegin();
}

template<typename IndexVector>
typename DenseBinnedScoreVector<IndexVector>::value_binned_const_iterator
  DenseBinnedScoreVector<IndexVector>::values_binned_cend() const {
    return this->secondView_.cend();
}

template<typename IndexVector>
uint32 DenseBinnedScoreVector<IndexVector>::getNumElements() const {
    return labelIndices_.getNumElements();
}

template<typename IndexVector>
void DenseBinnedScoreVector<IndexVector>::setNumBins(uint32 numBins, bool freeMemory) {
    if (numBins < maxCapacity_) {
        if (freeMemory) {
            this->secondView_.array = reallocateMemory(this->secondView_.array, numBins);
            maxCapacity_ = numBins;
        }
    } else if (numBins > maxCapacity_) {
        this->secondView_.array = reallocateMemory(this->secondView_.array, numBins);
        maxCapacity_ = numBins;
    }

    this->secondView_.numElements = numBins;
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
