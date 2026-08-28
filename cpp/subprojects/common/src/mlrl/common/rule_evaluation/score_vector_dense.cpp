#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"

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

template<typename ScoreType, typename IndexVector>
DenseScoreVector<ScoreType, IndexVector>::DenseScoreVector(const IndexVector& outputIndices, bool sorted)
    : ViewDecorator<DenseScoreVectorAllocator<DenseScoreVectorView<ScoreType, IndexVector>>>(
        DenseScoreVectorAllocator<DenseScoreVectorView<ScoreType, IndexVector>>(outputIndices, sorted)) {}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::index_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::indices_cbegin() const {
    return this->view.indices_cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::index_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::indices_cend() const {
    return this->view.indices_cend();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_begin() {
    return this->view.begin();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_end() {
    return this->view.end();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_cbegin() const {
    return this->view.cbegin();
}

template<typename ScoreType, typename IndexVector>
typename DenseScoreVector<ScoreType, IndexVector>::value_const_iterator
  DenseScoreVector<ScoreType, IndexVector>::values_cend() const {
    return this->view.cend();
}

template<typename ScoreType, typename IndexVector>
uint32 DenseScoreVector<ScoreType, IndexVector>::getNumElements() const {
    return this->view.getNumElements();
}

template<typename ScoreType, typename IndexVector>
bool DenseScoreVector<ScoreType, IndexVector>::isPartial() const {
    return this->view.isPartial();
}

template<typename ScoreType, typename IndexVector>
void DenseScoreVector<ScoreType, IndexVector>::setQuality(float64 quality) {
    this->view.quality = quality;
}

template<typename ScoreType, typename IndexVector>
float64 DenseScoreVector<ScoreType, IndexVector>::getQuality() const {
    return this->view.quality;
}

template<typename ScoreType, typename IndexVector>
void DenseScoreVector<ScoreType, IndexVector>::visit(
  BitVisitor<CompleteIndexVector> completeBitVisitor, BitVisitor<PartialIndexVector> partialBitVisitor,
  DenseVisitor<float32, CompleteIndexVector> completeDense32BitVisitor,
  DenseVisitor<float32, PartialIndexVector> partialDense32BitVisitor,
  DenseVisitor<float64, CompleteIndexVector> completeDense64BitVisitor,
  DenseVisitor<float64, PartialIndexVector> partialDense64BitVisitor,
  DenseBinnedVisitor<float32, CompleteIndexVector> completeDense32BitBinnedVisitor,
  DenseBinnedVisitor<float32, PartialIndexVector> partialDense32BitBinnedVisitor,
  DenseBinnedVisitor<float64, CompleteIndexVector> completeDense64BitBinnedVisitor,
  DenseBinnedVisitor<float64, PartialIndexVector> partialDense64BitBinnedVisitor) const {
    visitInternally(this->getView(), completeDense32BitVisitor, partialDense32BitVisitor, completeDense64BitVisitor,
                    partialDense64BitVisitor);
}

template class DenseScoreVector<float32, PartialIndexVector>;
template class DenseScoreVector<float64, PartialIndexVector>;
template class DenseScoreVector<float32, CompleteIndexVector>;
template class DenseScoreVector<float64, CompleteIndexVector>;
