#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"

static inline void visitInternally(const DenseScoreVector<float32, CompleteIndexVector>& scoreVector,
                                   IScoreVector::DenseVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    complete32BitVisitor(scoreVector);
}

static inline void visitInternally(const DenseScoreVector<float64, CompleteIndexVector>& scoreVector,
                                   IScoreVector::DenseVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    complete64BitVisitor(scoreVector);
}

static inline void visitInternally(const DenseScoreVector<float32, PartialIndexVector>& scoreVector,
                                   IScoreVector::DenseVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    partial32BitVisitor(scoreVector);
}

static inline void visitInternally(const DenseScoreVector<float64, PartialIndexVector>& scoreVector,
                                   IScoreVector::DenseVisitor<float32, CompleteIndexVector> complete32BitVisitor,
                                   IScoreVector::DenseVisitor<float32, PartialIndexVector> partial32BitVisitor,
                                   IScoreVector::DenseVisitor<float64, CompleteIndexVector> complete64BitVisitor,
                                   IScoreVector::DenseVisitor<float64, PartialIndexVector> partial64BitVisitor) {
    partial64BitVisitor(scoreVector);
}

template<typename ScoreType, typename IndexVector>
DenseScoreVector<ScoreType, IndexVector>::DenseScoreVector(const IndexVector& outputIndices, bool sorted)
    : ViewDecorator<AllocatedView<ScoreType>>(AllocatedView<ScoreType>(outputIndices.getNumElements())),
      outputIndices_(outputIndices), sorted_(sorted) {}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::index_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::indices_cbegin() const {
    return outputIndices_.cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::index_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::indices_cend() const {
    return outputIndices_.cend();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_begin() {
    return this->view.begin();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_end() {
    return &this->view.array[this->getNumElements()];
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_cbegin() const {
    return this->view.cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_cend() const {
    return &this->view.array[this->getNumElements()];
}

template<typename ScoreType, typename IndexVector>
uint32 DenseScoreVector<ScoreType, IndexVector>::getNumElements() const {
    return outputIndices_.getNumElements();
}

template<typename ScoreType, typename IndexVector>
bool DenseScoreVector<ScoreType, IndexVector>::isPartial() const {
    return outputIndices_.isPartial();
}

template<typename ScoreType, typename IndexVector>
bool DenseScoreVector<ScoreType, IndexVector>::isSorted() const {
    return sorted_;
}

template<typename ScoreType, typename IndexVector>
void DenseScoreVector<ScoreType, IndexVector>::visit(
  DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
  DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
  DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
  DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
  DenseBinnedVisitor<float32, CompleteIndexVector> completeDense32BitBinnedVisitor,
  DenseBinnedVisitor<float32, PartialIndexVector> partialDense32BitBinnedVisitor,
  DenseBinnedVisitor<float64, CompleteIndexVector> completeDense64BitBinnedVisitor,
  DenseBinnedVisitor<float64, PartialIndexVector> partialDense64BitBinnedVisitor) const {
    visitInternally(*this, completeDense32BitVisitor, partialDense32BitVisitor, completeDense64BitVisitor,
                    partialDense64BitVisitor);
}

template class DenseScoreVector<float32, PartialIndexVector>;
template class DenseScoreVector<float64, PartialIndexVector>;
template class DenseScoreVector<float32, CompleteIndexVector>;
template class DenseScoreVector<float64, CompleteIndexVector>;
