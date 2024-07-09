#include "mlrl/boosting/data/vector_statistic_non_decomposable_dense.hpp"

#include "mlrl/boosting/util/math.hpp"

namespace boosting {

    DenseNonDecomposableStatisticVector::DenseNonDecomposableStatisticVector(uint32 numGradients, bool init)
        : ClearableViewDecorator<ViewDecorator<CompositeVector<AllocatedVector<float64>, AllocatedVector<float64>>>>(
            CompositeVector<AllocatedVector<float64>, AllocatedVector<float64>>(
              AllocatedVector<float64>(numGradients, init),
              AllocatedVector<float64>(triangularNumber(numGradients), init))) {}

    DenseNonDecomposableStatisticVector::DenseNonDecomposableStatisticVector(
      const DenseNonDecomposableStatisticVector& other)
        : DenseNonDecomposableStatisticVector(other.getNumGradients()) {
        copyView(other.gradients_cbegin(), this->gradients_begin(), this->getNumGradients());
        copyView(other.hessians_cbegin(), this->hessians_begin(), this->getNumHessians());
    }

    DenseNonDecomposableStatisticVector::gradient_iterator DenseNonDecomposableStatisticVector::gradients_begin() {
        return this->view.firstView.begin();
    }

    DenseNonDecomposableStatisticVector::gradient_iterator DenseNonDecomposableStatisticVector::gradients_end() {
        return this->view.firstView.end();
    }

    DenseNonDecomposableStatisticVector::gradient_const_iterator DenseNonDecomposableStatisticVector::gradients_cbegin()
      const {
        return this->view.firstView.cbegin();
    }

    DenseNonDecomposableStatisticVector::gradient_const_iterator DenseNonDecomposableStatisticVector::gradients_cend()
      const {
        return this->view.firstView.cend();
    }

    DenseNonDecomposableStatisticVector::hessian_iterator DenseNonDecomposableStatisticVector::hessians_begin() {
        return this->view.secondView.begin();
    }

    DenseNonDecomposableStatisticVector::hessian_iterator DenseNonDecomposableStatisticVector::hessians_end() {
        return this->view.secondView.end();
    }

    DenseNonDecomposableStatisticVector::hessian_const_iterator DenseNonDecomposableStatisticVector::hessians_cbegin()
      const {
        return this->view.secondView.cbegin();
    }

    DenseNonDecomposableStatisticVector::hessian_const_iterator DenseNonDecomposableStatisticVector::hessians_cend()
      const {
        return this->view.secondView.cend();
    }

    DenseNonDecomposableStatisticVector::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticVector::hessians_diagonal_cbegin() const {
        return DiagonalConstIterator<float64>(this->hessians_cbegin(), 0);
    }

    DenseNonDecomposableStatisticVector::hessian_diagonal_const_iterator
      DenseNonDecomposableStatisticVector::hessians_diagonal_cend() const {
        return DiagonalConstIterator<float64>(this->hessians_cbegin(), this->getNumHessians());
    }

    uint32 DenseNonDecomposableStatisticVector::getNumGradients() const {
        return this->view.firstView.numElements;
    }

    uint32 DenseNonDecomposableStatisticVector::getNumHessians() const {
        return this->view.secondView.numElements;
    }

    void DenseNonDecomposableStatisticVector::add(const DenseNonDecomposableStatisticVector& view) {
        addToView(this->gradients_begin(), view.gradients_cbegin(), this->getNumGradients());
        addToView(this->hessians_begin(), view.hessians_cbegin(), this->getNumHessians());
    }

    void DenseNonDecomposableStatisticVector::add(const DenseNonDecomposableStatisticView& view, uint32 row) {
        addToView(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        addToView(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    void DenseNonDecomposableStatisticVector::add(const DenseNonDecomposableStatisticView& view, uint32 row,
                                                  float64 weight) {
        addToView(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(), weight);
        addToView(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(), weight);
    }

    void DenseNonDecomposableStatisticVector::remove(const DenseNonDecomposableStatisticView& view, uint32 row) {
        removeFromView(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        removeFromView(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    void DenseNonDecomposableStatisticVector::remove(const DenseNonDecomposableStatisticView& view, uint32 row,
                                                     float64 weight) {
        removeFromView(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(), weight);
        removeFromView(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(), weight);
    }

    void DenseNonDecomposableStatisticVector::addToSubset(const DenseNonDecomposableStatisticView& view, uint32 row,
                                                          const CompleteIndexVector& indices) {
        addToView(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients());
        addToView(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians());
    }

    void DenseNonDecomposableStatisticVector::addToSubset(const DenseNonDecomposableStatisticView& view, uint32 row,
                                                          const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(this->gradients_begin(), view.gradients_cbegin(row), indexIterator, this->getNumGradients());
        DenseNonDecomposableStatisticView::hessian_const_iterator hessiansBegin = view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            addToView(&this->hessians_begin()[triangularNumber(i)], &hessiansBegin[triangularNumber(index)],
                      indexIterator, i + 1);
        }
    }

    void DenseNonDecomposableStatisticVector::addToSubset(const DenseNonDecomposableStatisticView& view, uint32 row,
                                                          const CompleteIndexVector& indices, float64 weight) {
        addToView(this->gradients_begin(), view.gradients_cbegin(row), this->getNumGradients(), weight);
        addToView(this->hessians_begin(), view.hessians_cbegin(row), this->getNumHessians(), weight);
    }

    void DenseNonDecomposableStatisticVector::addToSubset(const DenseNonDecomposableStatisticView& view, uint32 row,
                                                          const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(this->gradients_begin(), view.gradients_cbegin(row), indexIterator, this->getNumGradients(), weight);
        DenseNonDecomposableStatisticView::hessian_const_iterator hessiansBegin = view.hessians_cbegin(row);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            addToView(&this->hessians_begin()[triangularNumber(i)], &hessiansBegin[triangularNumber(index)],
                      indexIterator, i + 1, weight);
        }
    }

    void DenseNonDecomposableStatisticVector::difference(const DenseNonDecomposableStatisticVector& first,
                                                         const CompleteIndexVector& firstIndices,
                                                         const DenseNonDecomposableStatisticVector& second) {
        setViewToDifference(this->gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(),
                            this->getNumGradients());
        setViewToDifference(this->hessians_begin(), first.hessians_cbegin(), second.hessians_cbegin(),
                            this->getNumHessians());
    }

    void DenseNonDecomposableStatisticVector::difference(const DenseNonDecomposableStatisticVector& first,
                                                         const PartialIndexVector& firstIndices,
                                                         const DenseNonDecomposableStatisticVector& second) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setViewToDifference(this->gradients_begin(), first.gradients_cbegin(), second.gradients_cbegin(), indexIterator,
                            this->getNumGradients());
        DenseNonDecomposableStatisticVector::hessian_const_iterator firstHessiansBegin = first.hessians_cbegin();
        DenseNonDecomposableStatisticVector::hessian_const_iterator secondHessiansBegin = second.hessians_cbegin();

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 offset = triangularNumber(i);
            uint32 index = indexIterator[i];
            setViewToDifference(&this->hessians_begin()[offset], &firstHessiansBegin[triangularNumber(index)],
                                &secondHessiansBegin[offset], indexIterator, i + 1);
        }
    }

}
