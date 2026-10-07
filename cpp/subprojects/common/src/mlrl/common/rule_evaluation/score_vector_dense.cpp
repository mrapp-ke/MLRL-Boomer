#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"

#include "mlrl/common/simd/memory.hpp"

static inline void visitInternally(const DenseScoreVectorView<float32, CompleteIndexVector>& scoreVector,
                                   IScoreVector::DenseVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    complete32BitVisitor(scoreVector);
}

static inline void visitInternally(const DenseScoreVectorView<float64, CompleteIndexVector>& scoreVector,
                                   IScoreVector::DenseVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    complete64BitVisitor(scoreVector);
}

static inline void visitInternally(const DenseScoreVectorView<float32, PartialIndexVector>& scoreVector,
                                   IScoreVector::DenseVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    partial32BitVisitor(scoreVector);
}

static inline void visitInternally(const DenseScoreVectorView<float64, PartialIndexVector>& scoreVector,
                                   IScoreVector::DenseVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    partial64BitVisitor(scoreVector);
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::DenseScoreVector(const IndexVector& outputIndices,
                                                                            bool sorted)
    : AbstractScoreVectorViewDecorator<
        DenseScoreVectorAllocator<DenseScoreVectorView<ScoreType, IndexVector>, MemoryAllocator>>(
        DenseScoreVectorAllocator<DenseScoreVectorView<ScoreType, IndexVector>, MemoryAllocator>(outputIndices,
                                                                                                 sorted)) {}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::index_const_iterator
  DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::indices_cbegin() const {
    return this->view.indices_cbegin();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::index_const_iterator
  DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::indices_cend() const {
    return this->view.indices_cend();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::value_iterator
  DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::values_begin() {
    return this->view.begin();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::value_iterator
  DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::values_end() {
    return this->view.end();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::value_const_iterator
  DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::values_cbegin() const {
    return this->view.cbegin();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::value_const_iterator
  DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::values_cend() const {
    return this->view.cend();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
void DenseScoreVector<ScoreType, IndexVector, MemoryAllocator>::visit(
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
    visitInternally(this->getView(), completeDense32BitVisitor, partialDense32BitVisitor, completeDense64BitVisitor,
                    partialDense64BitVisitor);
}

template class DenseScoreVector<float32, PartialIndexVector, DefaultMemoryAllocator>;
template class DenseScoreVector<float64, PartialIndexVector, DefaultMemoryAllocator>;
template class DenseScoreVector<float32, CompleteIndexVector, DefaultMemoryAllocator>;
template class DenseScoreVector<float64, CompleteIndexVector, DefaultMemoryAllocator>;

#if SIMD_SUPPORT_ENABLED
template class DenseScoreVector<float32, PartialIndexVector, SimdMemoryAllocator>;
template class DenseScoreVector<float64, PartialIndexVector, SimdMemoryAllocator>;
template class DenseScoreVector<float32, CompleteIndexVector, SimdMemoryAllocator>;
template class DenseScoreVector<float64, CompleteIndexVector, SimdMemoryAllocator>;
#endif
