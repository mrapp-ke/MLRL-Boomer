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
    : IndexableBitVectorDecorator<
        AbstractScoreVectorViewDecorator<BitScoreVectorAllocator<BitScoreVectorView<IndexVector>>>>(
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
void BitScoreVector<IndexVector>::visit(
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

template class BitScoreVector<CompleteIndexVector>;
template class BitScoreVector<PartialIndexVector>;
