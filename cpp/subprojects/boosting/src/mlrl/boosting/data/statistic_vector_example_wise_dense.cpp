#include "mlrl/boosting/data/statistic_vector_example_wise_dense.hpp"

#include "mlrl/boosting/util/math.hpp"
#include "mlrl/common/util/view_functions.hpp"

namespace boosting {

    DenseExampleWiseStatisticVector::DenseExampleWiseStatisticVector(uint32 numGradients, bool init)
        : gradients_(numGradients, init), hessians_(triangularNumber(numGradients), init) {}

    DenseExampleWiseStatisticVector::DenseExampleWiseStatisticVector(const DenseExampleWiseStatisticVector& other)
        : DenseExampleWiseStatisticVector(other.getNumElements()) {
        copyView(other.gradients_cbegin(), gradients_.begin(), gradients_.getNumElements());
        copyView(other.hessians_cbegin(), hessians_.begin(), hessians_.getNumElements());
    }

    DenseExampleWiseStatisticVector::gradient_iterator DenseExampleWiseStatisticVector::gradients_begin() {
        return gradients_.begin();
    }

    DenseExampleWiseStatisticVector::gradient_iterator DenseExampleWiseStatisticVector::gradients_end() {
        return gradients_.end();
    }

    DenseExampleWiseStatisticVector::gradient_const_iterator DenseExampleWiseStatisticVector::gradients_cbegin() const {
        return gradients_.cbegin();
    }

    DenseExampleWiseStatisticVector::gradient_const_iterator DenseExampleWiseStatisticVector::gradients_cend() const {
        return gradients_.cend();
    }

    DenseExampleWiseStatisticVector::hessian_iterator DenseExampleWiseStatisticVector::hessians_begin() {
        return hessians_.begin();
    }

    DenseExampleWiseStatisticVector::hessian_iterator DenseExampleWiseStatisticVector::hessians_end() {
        return hessians_.end();
    }

    DenseExampleWiseStatisticVector::hessian_const_iterator DenseExampleWiseStatisticVector::hessians_cbegin() const {
        return hessians_.cbegin();
    }

    DenseExampleWiseStatisticVector::hessian_const_iterator DenseExampleWiseStatisticVector::hessians_cend() const {
        return hessians_.cend();
    }

    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticVector::hessians_diagonal_cbegin() const {
        return DiagonalConstIterator<float64>(hessians_.cbegin(), 0);
    }

    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticVector::hessians_diagonal_cend() const {
        return DiagonalConstIterator<float64>(hessians_.cbegin(), gradients_.getNumElements());
    }

    uint32 DenseExampleWiseStatisticVector::getNumElements() const {
        return gradients_.getNumElements();
    }

    void DenseExampleWiseStatisticVector::clear() {
        setViewToZeros(gradients_.begin(), gradients_.getNumElements());
        setViewToZeros(hessians_.begin(), hessians_.getNumElements());
    }

    void DenseExampleWiseStatisticVector::add(gradient_const_iterator gradientsBegin,
                                              gradient_const_iterator gradientsEnd,
                                              hessian_const_iterator hessiansBegin,
                                              hessian_const_iterator hessiansEnd) {
        addToView(gradients_.begin(), gradientsBegin, gradients_.getNumElements());
        addToView(hessians_.begin(), hessiansBegin, hessians_.getNumElements());
    }

    void DenseExampleWiseStatisticVector::add(gradient_const_iterator gradientsBegin,
                                              gradient_const_iterator gradientsEnd,
                                              hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd,
                                              float64 weight) {
        addToView(gradients_.begin(), gradientsBegin, gradients_.getNumElements(), weight);
        addToView(hessians_.begin(), hessiansBegin, hessians_.getNumElements(), weight);
    }

    void DenseExampleWiseStatisticVector::remove(gradient_const_iterator gradientsBegin,
                                                 gradient_const_iterator gradientsEnd,
                                                 hessian_const_iterator hessiansBegin,
                                                 hessian_const_iterator hessiansEnd) {
        removeFromView(gradients_.begin(), gradientsBegin, gradients_.getNumElements());
        removeFromView(hessians_.begin(), hessiansBegin, hessians_.getNumElements());
    }

    void DenseExampleWiseStatisticVector::remove(gradient_const_iterator gradientsBegin,
                                                 gradient_const_iterator gradientsEnd,
                                                 hessian_const_iterator hessiansBegin,
                                                 hessian_const_iterator hessiansEnd, float64 weight) {
        removeFromView(gradients_.begin(), gradientsBegin, gradients_.getNumElements(), weight);
        removeFromView(hessians_.begin(), hessiansBegin, hessians_.getNumElements(), weight);
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const CompleteIndexVector& indices) {
        addToView(gradients_.begin(), gradientsBegin, gradients_.getNumElements());
        addToView(hessians_.begin(), hessiansBegin, hessians_.getNumElements());
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(gradients_.begin(), gradientsBegin, indexIterator, gradients_.getNumElements());

        for (uint32 i = 0; i < gradients_.getNumElements(); i++) {
            uint32 index = indexIterator[i];
            addToView(&hessians_.begin()[triangularNumber(i)], &hessiansBegin[triangularNumber(index)], indexIterator,
                      i + 1);
        }
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const CompleteIndexVector& indices, float64 weight) {
        addToView(gradients_.begin(), gradientsBegin, hessians_.getNumElements(), weight);
        addToView(hessians_.begin(), hessiansBegin, hessians_.getNumElements(), weight);
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(gradients_.begin(), gradientsBegin, indexIterator, gradients_.getNumElements(), weight);

        for (uint32 i = 0; i < gradients_.getNumElements(); i++) {
            uint32 index = indexIterator[i];
            addToView(&hessians_.begin()[triangularNumber(i)], &hessiansBegin[triangularNumber(index)], indexIterator,
                      i + 1, weight);
        }
    }

    void DenseExampleWiseStatisticVector::difference(
      gradient_const_iterator firstGradientsBegin, gradient_const_iterator firstGradientsEnd,
      hessian_const_iterator firstHessiansBegin, hessian_const_iterator firstHessiansEnd,
      const CompleteIndexVector& firstIndices, gradient_const_iterator secondGradientsBegin,
      gradient_const_iterator secondGradientsEnd, hessian_const_iterator secondHessiansBegin,
      hessian_const_iterator secondHessiansEnd) {
        setViewToDifference(gradients_.begin(), firstGradientsBegin, secondGradientsBegin, gradients_.getNumElements());
        setViewToDifference(hessians_.begin(), firstHessiansBegin, secondHessiansBegin, hessians_.getNumElements());
    }

    void DenseExampleWiseStatisticVector::difference(
      gradient_const_iterator firstGradientsBegin, gradient_const_iterator firstGradientsEnd,
      hessian_const_iterator firstHessiansBegin, hessian_const_iterator firstHessiansEnd,
      const PartialIndexVector& firstIndices, gradient_const_iterator secondGradientsBegin,
      gradient_const_iterator secondGradientsEnd, hessian_const_iterator secondHessiansBegin,
      hessian_const_iterator secondHessiansEnd) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setViewToDifference(gradients_.begin(), firstGradientsBegin, secondGradientsBegin, indexIterator,
                            gradients_.getNumElements());

        for (uint32 i = 0; i < gradients_.getNumElements(); i++) {
            uint32 offset = triangularNumber(i);
            uint32 index = indexIterator[i];
            setViewToDifference(&hessians_.begin()[offset], &firstHessiansBegin[triangularNumber(index)],
                                &secondHessiansBegin[offset], indexIterator, i + 1);
        }
    }

}
