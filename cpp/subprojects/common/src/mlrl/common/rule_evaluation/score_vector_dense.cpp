#include "mlrl/common/rule_evaluation/score_vector_dense.hpp"

static inline void visitInternally(const DenseScoreVector<CompleteIndexVector>& scoreVector,
                                   IScoreVector::DenseVisitor<CompleteIndexVector> completeVisitor,
                                   IScoreVector::DenseVisitor<PartialIndexVector> partialVisitor) {
    completeVisitor(scoreVector);
}

static inline void visitInternally(const DenseScoreVector<PartialIndexVector>& scoreVector,
                                   IScoreVector::DenseVisitor<CompleteIndexVector> completeVisitor,
                                   IScoreVector::DenseVisitor<PartialIndexVector> partialVisitor) {
    partialVisitor(scoreVector);
}

template<typename IndexVector>
DenseScoreVector<IndexVector>::DenseScoreVector(const IndexVector& outputIndices, bool sorted)
    : ViewDecorator<AllocatedView<float64>>(AllocatedView<float64>(outputIndices.getNumElements())),
      outputIndices_(outputIndices), sorted_(sorted) {}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::index_const_iterator DenseScoreVector<IndexVector>::indices_cbegin() const {
    return outputIndices_.cbegin();
}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::index_const_iterator DenseScoreVector<IndexVector>::indices_cend() const {
    return outputIndices_.cend();
}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::value_iterator DenseScoreVector<IndexVector>::values_begin() {
    return this->view.begin();
}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::value_iterator DenseScoreVector<IndexVector>::values_end() {
    return &this->view.array[this->getNumElements()];
}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::value_const_iterator DenseScoreVector<IndexVector>::values_cbegin() const {
    return this->view.cbegin();
}

template<typename IndexVector>
typename DenseScoreVector<IndexVector>::value_const_iterator DenseScoreVector<IndexVector>::values_cend() const {
    return &this->view.array[this->getNumElements()];
}

template<typename IndexVector>
uint32 DenseScoreVector<IndexVector>::getNumElements() const {
    return outputIndices_.getNumElements();
}

template<typename IndexVector>
bool DenseScoreVector<IndexVector>::isPartial() const {
    return outputIndices_.isPartial();
}

template<typename IndexVector>
bool DenseScoreVector<IndexVector>::isSorted() const {
    return sorted_;
}

template<typename IndexVector>
void DenseScoreVector<IndexVector>::visit(DenseVisitor<CompleteIndexVector> completeDenseVisitor,
                                          DenseVisitor<PartialIndexVector> partialDenseVisitor,
                                          DenseBinnedVisitor<CompleteIndexVector> completeDenseBinnedVisitor,
                                          DenseBinnedVisitor<PartialIndexVector> partialDenseBinnedVisitor) const {
    visitInternally(*this, completeDenseVisitor, partialDenseVisitor);
}

template class DenseScoreVector<PartialIndexVector>;
template class DenseScoreVector<CompleteIndexVector>;
