#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"

#include "mlrl/common/simd/memory.hpp"

static inline void visitInternally(const DenseBinnedScoreVectorView<float32, CompleteIndexVector>& scoreVector,
                                   IScoreVector::DenseBinnedVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    complete32BitVisitor(scoreVector);
}

static inline void visitInternally(const DenseBinnedScoreVectorView<float64, CompleteIndexVector>& scoreVector,
                                   IScoreVector::DenseBinnedVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    complete64BitVisitor(scoreVector);
}

static inline void visitInternally(const DenseBinnedScoreVectorView<float32, PartialIndexVector>& scoreVector,
                                   IScoreVector::DenseBinnedVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    partial32BitVisitor(scoreVector);
}

static inline void visitInternally(const DenseBinnedScoreVectorView<float64, PartialIndexVector>& scoreVector,
                                   IScoreVector::DenseBinnedVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseBinnedVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    partial64BitVisitor(scoreVector);
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::DenseBinnedScoreVector(
  const IndexVector& outputIndices, uint32 numBins, bool sorted)
    : AbstractScoreVectorViewDecorator<
        DenseBinnedScoreVectorAllocator<DenseBinnedScoreVectorView<ScoreType, IndexVector>, MemoryAllocator>>(
        DenseBinnedScoreVectorAllocator<DenseBinnedScoreVectorView<ScoreType, IndexVector>, MemoryAllocator>(
          outputIndices, numBins, sorted)) {}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::indices_cbegin() const {
    return this->view.indices_cbegin();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::indices_cend() const {
    return this->view.indices_cend();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::cbegin() const {
    return this->view.cbegin();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::cend() const {
    return this->view.cend();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_index_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_indices_begin() {
    return this->view.bin_indices_begin();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_index_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_indices_end() {
    return this->view.bin_indices_end();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_indices_cbegin() const {
    return this->view.bin_indices_cbegin();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_indices_cend() const {
    return this->view.bin_indices_cend();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_value_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_values_begin() {
    return this->view.bin_values_begin();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_value_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_values_end() {
    return this->view.bin_values_end();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_values_cbegin() const {
    return this->view.bin_values_cbegin();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
typename DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::bin_values_cend() const {
    return this->view.bin_values_cend();
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
void DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::setNumBins(uint32 numBins, bool freeMemory) {
    this->view.resize(numBins, freeMemory);
}

template<typename ScoreType, typename IndexVector, typename MemoryAllocator>
void DenseBinnedScoreVector<ScoreType, IndexVector, MemoryAllocator>::visit(
  IScoreVector::BitVisitor<CompleteIndexVector> completeBitVisitor,
  IScoreVector::BitVisitor<PartialIndexVector> partialBitVisitor,
  IScoreVector::DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
  IScoreVector::DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
  IScoreVector::DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
  IScoreVector::DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
  IScoreVector::DenseBinnedVisitor<float32, CompleteIndexVector> completeDenseBinned32BitVisitor,
  IScoreVector::DenseBinnedVisitor<float32, PartialIndexVector> partialDenseBinned32BitVisitor,
  IScoreVector::DenseBinnedVisitor<float64, CompleteIndexVector> completeDenseBinned64BitVisitor,
  IScoreVector::DenseBinnedVisitor<float64, PartialIndexVector> partialDenseBinned64BitVisitor) const {
    visitInternally(this->getView(), completeDenseBinned32BitVisitor, partialDenseBinned32BitVisitor,
                    completeDenseBinned64BitVisitor, partialDenseBinned64BitVisitor);
}

template class DenseBinnedScoreVector<float32, PartialIndexVector, DefaultMemoryAllocator>;
template class DenseBinnedScoreVector<float64, PartialIndexVector, DefaultMemoryAllocator>;
template class DenseBinnedScoreVector<float32, CompleteIndexVector, DefaultMemoryAllocator>;
template class DenseBinnedScoreVector<float64, CompleteIndexVector, DefaultMemoryAllocator>;

#if SIMD_SUPPORT_ENABLED
template class DenseBinnedScoreVector<float32, PartialIndexVector, SimdMemoryAllocator>;
template class DenseBinnedScoreVector<float64, PartialIndexVector, SimdMemoryAllocator>;
template class DenseBinnedScoreVector<float32, CompleteIndexVector, SimdMemoryAllocator>;
template class DenseBinnedScoreVector<float64, CompleteIndexVector, SimdMemoryAllocator>;
#endif
