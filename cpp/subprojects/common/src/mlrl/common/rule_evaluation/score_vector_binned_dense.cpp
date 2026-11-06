#include "mlrl/common/rule_evaluation/score_vector_binned_dense.hpp"

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

template<typename ScoreType, typename IndexVector>
DenseBinnedScoreVector<ScoreType, IndexVector>::DenseBinnedScoreVector(const IndexVector& outputIndices, uint32 numBins,
                                                                       bool sorted)
    : AbstractScoreVectorViewDecorator<
        DenseBinnedScoreVectorAllocator<DenseBinnedScoreVectorView<ScoreType, IndexVector>>>(
        DenseBinnedScoreVectorAllocator<DenseBinnedScoreVectorView<ScoreType, IndexVector>>(outputIndices, numBins,
                                                                                            sorted)) {}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::indices_cbegin() const {
    return this->view.indices_cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::indices_cend() const {
    return this->view.indices_cend();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::cbegin() const {
    return this->view.cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::cend() const {
    return this->view.cend();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_index_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_indices_begin() {
    return this->view.bin_indices_begin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_index_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_indices_end() {
    return this->view.bin_indices_end();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_indices_cbegin() const {
    return this->view.bin_indices_cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_index_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_indices_cend() const {
    return this->view.bin_indices_cend();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_value_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_values_begin() {
    return this->view.bin_values_begin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_value_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_values_end() {
    return this->view.bin_values_end();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_values_cbegin() const {
    return this->view.bin_values_cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseBinnedScoreVector<ScoreType, IndexVector>::bin_value_const_iterator
  DenseBinnedScoreVector<ScoreType, IndexVector>::bin_values_cend() const {
    return this->view.bin_values_cend();
}

template<typename ScoreType, typename IndexVector>
void DenseBinnedScoreVector<ScoreType, IndexVector>::setNumBins(uint32 numBins, bool freeMemory) {
    this->view.resize(numBins, freeMemory);
}

template<typename ScoreType, typename IndexVector>
void DenseBinnedScoreVector<ScoreType, IndexVector>::visit(
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

template class DenseBinnedScoreVector<float32, PartialIndexVector>;
template class DenseBinnedScoreVector<float64, PartialIndexVector>;
template class DenseBinnedScoreVector<float32, CompleteIndexVector>;
template class DenseBinnedScoreVector<float64, CompleteIndexVector>;
