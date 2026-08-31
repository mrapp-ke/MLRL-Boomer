#include "mlrl/common/rule_evaluation/score_vector_bit.hpp"

static inline void visitInternally(const BitScoreVectorView<CompleteIndexVector>& scoreVector,
                                   IScoreVector::BitVisitor<CompleteIndexVector> completeBitVisitor,
                                   IScoreVector::BitVisitor<PartialIndexVector> partialBitVisitor) {
    completeBitVisitor(scoreVector);
}

static inline void visitInternally(const BitScoreVectorView<PartialIndexVector>& scoreVector,
                                   IScoreVector::BitVisitor<CompleteIndexVector> completeBitVisitor,
                                   IScoreVector::BitVisitor<PartialIndexVector> partialBitVisitor) {
    partialBitVisitor(scoreVector);
}

template<typename IndexVector>
BitScoreVector<IndexVector>::BitScoreVector(const IndexVector& outputIndices, bool sorted)
    : IndexableBitVectorDecorator<ViewDecorator<BitScoreVectorAllocator<BitScoreVectorView<IndexVector>>>>(
        BitScoreVectorAllocator<BitScoreVectorView<IndexVector>>(outputIndices, sorted)) {}

template<typename IndexVector>
typename BitScoreVector<IndexVector>::index_const_iterator BitScoreVector<IndexVector>::indices_cbegin() const {
    return this->view.indices_cbegin();
}

template<typename IndexVector>
typename BitScoreVector<IndexVector>::index_const_iterator BitScoreVector<IndexVector>::indices_cend() const {
    return this->view.indices_cend();
}

template<typename IndexVector>
typename BitScoreVector<IndexVector>::value_const_iterator BitScoreVector<IndexVector>::values_cbegin() const {
    return this->view.cbegin();
}

template<typename IndexVector>
typename BitScoreVector<IndexVector>::value_const_iterator BitScoreVector<IndexVector>::values_cend() const {
    return this->view.cend();
}

template<typename IndexVector>
uint32 BitScoreVector<IndexVector>::getNumElements() const {
    return this->view.getNumElements();
}

template<typename IndexVector>
bool BitScoreVector<IndexVector>::isPartial() const {
    return this->view.isPartial();
}

template<typename IndexVector>
void BitScoreVector<IndexVector>::setQuality(float64 quality) {
    this->view.quality = quality;
}

template<typename IndexVector>
float64 BitScoreVector<IndexVector>::getQuality() const {
    return this->view.quality;
}

template<typename IndexVector>
void BitScoreVector<IndexVector>::visit(
  BitVisitor<CompleteIndexVector> completeBitVisitor, BitVisitor<PartialIndexVector> partialBitVisitor,
  DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
  DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
  DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
  DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
  DenseBinnedVisitor<float32, CompleteIndexVector> completeDense32BitBinnedVisitor,
  DenseBinnedVisitor<float32, PartialIndexVector> partialDense32BitBinnedVisitor,
  DenseBinnedVisitor<float64, CompleteIndexVector> completeDense64BitBinnedVisitor,
  DenseBinnedVisitor<float64, PartialIndexVector> partialDense64BitBinnedVisitor) const {
    visitInternally(this->getView(), completeBitVisitor, partialBitVisitor);
}

template class BitScoreVector<CompleteIndexVector>;
template class BitScoreVector<PartialIndexVector>;
