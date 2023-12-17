#include "mlrl/boosting/data/vector_statistic_example_wise_dense.hpp"

#include "mlrl/boosting/util/math.hpp"

namespace boosting {

    DenseExampleWiseStatisticVector::DenseExampleWiseStatisticVector(uint32 numGradients, bool init)
        : ClearableViewDecorator<ViewDecorator<CompositeVector<AllocatedVector<float64>, AllocatedVector<float64>>>>(
          CompositeVector<AllocatedVector<float64>, AllocatedVector<float64>>(
            AllocatedVector<float64>(numGradients, init),
            AllocatedVector<float64>(triangularNumber(numGradients), init))) {}

    DenseExampleWiseStatisticVector::DenseExampleWiseStatisticVector(const DenseExampleWiseStatisticVector& other)
        : DenseExampleWiseStatisticVector(other.getNumGradients()) {
        copyView(other.gradients_cbegin(), this->gradients_begin(), this->getNumGradients());
        copyView(other.hessians_cbegin(), this->hessians_begin(), this->getNumHessians());
    }

    DenseExampleWiseStatisticVector::gradient_iterator DenseExampleWiseStatisticVector::gradients_begin() {
        return this->view.firstView.begin();
    }

    DenseExampleWiseStatisticVector::gradient_iterator DenseExampleWiseStatisticVector::gradients_end() {
        return this->view.firstView.end();
    }

    DenseExampleWiseStatisticVector::gradient_const_iterator DenseExampleWiseStatisticVector::gradients_cbegin() const {
        return this->view.firstView.cbegin();
    }

    DenseExampleWiseStatisticVector::gradient_const_iterator DenseExampleWiseStatisticVector::gradients_cend() const {
        return this->view.firstView.cend();
    }

    DenseExampleWiseStatisticVector::hessian_iterator DenseExampleWiseStatisticVector::hessians_begin() {
        return this->view.secondView.begin();
    }

    DenseExampleWiseStatisticVector::hessian_iterator DenseExampleWiseStatisticVector::hessians_end() {
        return this->view.secondView.end();
    }

    DenseExampleWiseStatisticVector::hessian_const_iterator DenseExampleWiseStatisticVector::hessians_cbegin() const {
        return this->view.secondView.cbegin();
    }

    DenseExampleWiseStatisticVector::hessian_const_iterator DenseExampleWiseStatisticVector::hessians_cend() const {
        return this->view.secondView.cend();
    }

    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticVector::hessians_diagonal_cbegin() const {
        return DiagonalConstIterator<float64>(this->hessians_cbegin(), 0);
    }

    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticVector::hessians_diagonal_cend() const {
        return DiagonalConstIterator<float64>(this->hessians_cbegin(), this->getNumHessians());
    }

    uint32 DenseExampleWiseStatisticVector::getNumGradients() const {
        return this->view.firstView.numElements;
    }

    uint32 DenseExampleWiseStatisticVector::getNumHessians() const {
        return this->view.secondView.numElements;
    }

    void DenseExampleWiseStatisticVector::add(View<float64>::const_iterator gradientsBegin,
                                              View<float64>::const_iterator gradientsEnd,
                                              View<float64>::const_iterator hessiansBegin,
                                              View<float64>::const_iterator hessiansEnd) {
        addToView(this->gradients_begin(), gradientsBegin, this->getNumGradients());
        addToView(this->hessians_begin(), hessiansBegin, this->getNumHessians());
    }

    void DenseExampleWiseStatisticVector::add(View<float64>::const_iterator gradientsBegin,
                                              View<float64>::const_iterator gradientsEnd,
                                              View<float64>::const_iterator hessiansBegin,
                                              View<float64>::const_iterator hessiansEnd, float64 weight) {
        addToView(this->gradients_begin(), gradientsBegin, this->getNumGradients(), weight);
        addToView(this->hessians_begin(), hessiansBegin, this->getNumHessians(), weight);
    }

    void DenseExampleWiseStatisticVector::remove(View<float64>::const_iterator gradientsBegin,
                                                 View<float64>::const_iterator gradientsEnd,
                                                 View<float64>::const_iterator hessiansBegin,
                                                 View<float64>::const_iterator hessiansEnd) {
        removeFromView(this->gradients_begin(), gradientsBegin, this->getNumGradients());
        removeFromView(this->hessians_begin(), hessiansBegin, this->getNumHessians());
    }

    void DenseExampleWiseStatisticVector::remove(View<float64>::const_iterator gradientsBegin,
                                                 View<float64>::const_iterator gradientsEnd,
                                                 View<float64>::const_iterator hessiansBegin,
                                                 View<float64>::const_iterator hessiansEnd, float64 weight) {
        removeFromView(this->gradients_begin(), gradientsBegin, this->getNumGradients(), weight);
        removeFromView(this->hessians_begin(), hessiansBegin, this->getNumHessians(), weight);
    }

    void DenseExampleWiseStatisticVector::addToSubset(View<float64>::const_iterator gradientsBegin,
                                                      View<float64>::const_iterator gradientsEnd,
                                                      View<float64>::const_iterator hessiansBegin,
                                                      View<float64>::const_iterator hessiansEnd,
                                                      const CompleteIndexVector& indices) {
        addToView(this->gradients_begin(), gradientsBegin, this->getNumGradients());
        addToView(this->hessians_begin(), hessiansBegin, this->getNumHessians());
    }

    void DenseExampleWiseStatisticVector::addToSubset(View<float64>::const_iterator gradientsBegin,
                                                      View<float64>::const_iterator gradientsEnd,
                                                      View<float64>::const_iterator hessiansBegin,
                                                      View<float64>::const_iterator hessiansEnd,
                                                      const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(this->gradients_begin(), gradientsBegin, indexIterator, this->getNumGradients());

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            addToView(&this->hessians_begin()[triangularNumber(i)], &hessiansBegin[triangularNumber(index)],
                      indexIterator, i + 1);
        }
    }

    void DenseExampleWiseStatisticVector::addToSubset(View<float64>::const_iterator gradientsBegin,
                                                      View<float64>::const_iterator gradientsEnd,
                                                      View<float64>::const_iterator hessiansBegin,
                                                      View<float64>::const_iterator hessiansEnd,
                                                      const CompleteIndexVector& indices, float64 weight) {
        addToView(this->gradients_begin(), gradientsBegin, this->getNumGradients(), weight);
        addToView(this->hessians_begin(), hessiansBegin, this->getNumHessians(), weight);
    }

    void DenseExampleWiseStatisticVector::addToSubset(View<float64>::const_iterator gradientsBegin,
                                                      View<float64>::const_iterator gradientsEnd,
                                                      View<float64>::const_iterator hessiansBegin,
                                                      View<float64>::const_iterator hessiansEnd,
                                                      const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(this->gradients_begin(), gradientsBegin, indexIterator, this->getNumGradients(), weight);

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 index = indexIterator[i];
            addToView(&this->hessians_begin()[triangularNumber(i)], &hessiansBegin[triangularNumber(index)],
                      indexIterator, i + 1, weight);
        }
    }

    void DenseExampleWiseStatisticVector::difference(
      View<float64>::const_iterator firstGradientsBegin, View<float64>::const_iterator firstGradientsEnd,
      View<float64>::const_iterator firstHessiansBegin, View<float64>::const_iterator firstHessiansEnd,
      const CompleteIndexVector& firstIndices, View<float64>::const_iterator secondGradientsBegin,
      View<float64>::const_iterator secondGradientsEnd, View<float64>::const_iterator secondHessiansBegin,
      View<float64>::const_iterator secondHessiansEnd) {
        setViewToDifference(this->gradients_begin(), firstGradientsBegin, secondGradientsBegin,
                            this->getNumGradients());
        setViewToDifference(this->hessians_begin(), firstHessiansBegin, secondHessiansBegin, this->getNumHessians());
    }

    void DenseExampleWiseStatisticVector::difference(
      View<float64>::const_iterator firstGradientsBegin, View<float64>::const_iterator firstGradientsEnd,
      View<float64>::const_iterator firstHessiansBegin, View<float64>::const_iterator firstHessiansEnd,
      const PartialIndexVector& firstIndices, View<float64>::const_iterator secondGradientsBegin,
      View<float64>::const_iterator secondGradientsEnd, View<float64>::const_iterator secondHessiansBegin,
      View<float64>::const_iterator secondHessiansEnd) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setViewToDifference(this->gradients_begin(), firstGradientsBegin, secondGradientsBegin, indexIterator,
                            this->getNumGradients());

        for (uint32 i = 0; i < this->getNumGradients(); i++) {
            uint32 offset = triangularNumber(i);
            uint32 index = indexIterator[i];
            setViewToDifference(&this->hessians_begin()[offset], &firstHessiansBegin[triangularNumber(index)],
                                &secondHessiansBegin[offset], indexIterator, i + 1);
        }
    }

}
