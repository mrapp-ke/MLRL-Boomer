#include "mlrl/common/rule_evaluation/score_vector_bit.hpp"

#include "mlrl/common/simd/memory.hpp"

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

template<typename IndexVector, typename MemoryAllocator>
BitScoreVector<IndexVector, MemoryAllocator>::BitScoreVector(const IndexVector& outputIndices, bool sorted)
    : IndexableBitVectorDecorator<
        AbstractScoreVectorViewDecorator<BitScoreVectorAllocator<BitScoreVectorView<IndexVector>, MemoryAllocator>>>(
        BitScoreVectorAllocator<BitScoreVectorView<IndexVector>, MemoryAllocator>(outputIndices, sorted)) {}

template<typename IndexVector, typename MemoryAllocator>
typename BitScoreVector<IndexVector, MemoryAllocator>::index_const_iterator
  BitScoreVector<IndexVector, MemoryAllocator>::indices_cbegin() const {
    return this->view.indices_cbegin();
}

template<typename IndexVector, typename MemoryAllocator>
typename BitScoreVector<IndexVector, MemoryAllocator>::index_const_iterator
  BitScoreVector<IndexVector, MemoryAllocator>::indices_cend() const {
    return this->view.indices_cend();
}

template<typename IndexVector, typename MemoryAllocator>
typename BitScoreVector<IndexVector, MemoryAllocator>::value_const_iterator
  BitScoreVector<IndexVector, MemoryAllocator>::values_cbegin() const {
    return this->view.cbegin();
}

template<typename IndexVector, typename MemoryAllocator>
typename BitScoreVector<IndexVector, MemoryAllocator>::value_const_iterator
  BitScoreVector<IndexVector, MemoryAllocator>::values_cend() const {
    return this->view.cend();
}

template<typename IndexVector, typename MemoryAllocator>
void BitScoreVector<IndexVector, MemoryAllocator>::visit(
  IScoreVector::BitVisitor<CompleteIndexVector> completeBitVisitor,
  IScoreVector::BitVisitor<PartialIndexVector> partialBitVisitor,
  IScoreVector::DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
  IScoreVector::DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
  IScoreVector::DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
  IScoreVector::DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
  IScoreVector::DenseBinnedVisitor<float32, CompleteIndexVector> completeDense32BitBinnedVisitor,
  IScoreVector::DenseBinnedVisitor<float32, PartialIndexVector> partialDense32BitBinnedVisitor,
  IScoreVector::DenseBinnedVisitor<float64, CompleteIndexVector> completeDense64BitBinnedVisitor,
  IScoreVector::DenseBinnedVisitor<float64, PartialIndexVector> partialDense64BitBinnedVisitor) const {
    visitInternally(this->getView(), completeBitVisitor, partialBitVisitor);
}

template class BitScoreVector<CompleteIndexVector, DefaultMemoryAllocator>;
template class BitScoreVector<PartialIndexVector, DefaultMemoryAllocator>;

#if SIMD_SUPPORT_ENABLED
template class BitScoreVector<CompleteIndexVector, SimdMemoryAllocator>;
template class BitScoreVector<PartialIndexVector, SimdMemoryAllocator>;
#endif
