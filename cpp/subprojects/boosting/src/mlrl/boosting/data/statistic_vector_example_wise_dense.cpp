#include "mlrl/boosting/data/statistic_vector_example_wise_dense.hpp"

#include "mlrl/boosting/util/math.hpp"
#include "mlrl/boosting/util/view_functions.hpp"
#include "mlrl/common/util/memory.hpp"
#include "mlrl/common/util/view_functions.hpp"

namespace boosting {

    DenseExampleWiseStatisticVector::DenseExampleWiseStatisticVector(uint32 numGradients, bool init)
        : numGradients_(numGradients), numHessians_(triangularNumber(numGradients)),
          gradients_(allocateMemory<float64>(numGradients, init)),
          hessians_(allocateMemory<float64>(numHessians_, init)) {}

    DenseExampleWiseStatisticVector::DenseExampleWiseStatisticVector(const DenseExampleWiseStatisticVector& vector)
        : DenseExampleWiseStatisticVector(vector.numGradients_) {
        copyView(vector.gradients_, gradients_, numGradients_);
        copyView(vector.hessians_, hessians_, numHessians_);
    }

    DenseExampleWiseStatisticVector::~DenseExampleWiseStatisticVector() {
        freeMemory(gradients_);
        freeMemory(hessians_);
    }

    DenseExampleWiseStatisticVector::gradient_iterator DenseExampleWiseStatisticVector::gradients_begin() {
        return gradients_;
    }

    DenseExampleWiseStatisticVector::gradient_iterator DenseExampleWiseStatisticVector::gradients_end() {
        return &gradients_[numGradients_];
    }

    DenseExampleWiseStatisticVector::gradient_const_iterator DenseExampleWiseStatisticVector::gradients_cbegin() const {
        return gradients_;
    }

    DenseExampleWiseStatisticVector::gradient_const_iterator DenseExampleWiseStatisticVector::gradients_cend() const {
        return &gradients_[numGradients_];
    }

    DenseExampleWiseStatisticVector::hessian_iterator DenseExampleWiseStatisticVector::hessians_begin() {
        return hessians_;
    }

    DenseExampleWiseStatisticVector::hessian_iterator DenseExampleWiseStatisticVector::hessians_end() {
        return &hessians_[numHessians_];
    }

    DenseExampleWiseStatisticVector::hessian_const_iterator DenseExampleWiseStatisticVector::hessians_cbegin() const {
        return hessians_;
    }

    DenseExampleWiseStatisticVector::hessian_const_iterator DenseExampleWiseStatisticVector::hessians_cend() const {
        return &hessians_[numHessians_];
    }

    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticVector::hessians_diagonal_cbegin() const {
        return DiagonalConstIterator<float64>(hessians_, 0);
    }

    DenseExampleWiseStatisticVector::hessian_diagonal_const_iterator
      DenseExampleWiseStatisticVector::hessians_diagonal_cend() const {
        return DiagonalConstIterator<float64>(hessians_, numGradients_);
    }

    uint32 DenseExampleWiseStatisticVector::getNumElements() const {
        return numGradients_;
    }

    void DenseExampleWiseStatisticVector::clear() {
        setViewToZeros(gradients_, numGradients_);
        setViewToZeros(hessians_, numHessians_);
    }

    void DenseExampleWiseStatisticVector::add(gradient_const_iterator gradientsBegin,
                                              gradient_const_iterator gradientsEnd,
                                              hessian_const_iterator hessiansBegin,
                                              hessian_const_iterator hessiansEnd) {
        addToView(gradients_, gradientsBegin, numGradients_);
        addToView(hessians_, hessiansBegin, numHessians_);
    }

    void DenseExampleWiseStatisticVector::add(gradient_const_iterator gradientsBegin,
                                              gradient_const_iterator gradientsEnd,
                                              hessian_const_iterator hessiansBegin, hessian_const_iterator hessiansEnd,
                                              float64 weight) {
        addToView(gradients_, gradientsBegin, numGradients_, weight);
        addToView(hessians_, hessiansBegin, numHessians_, weight);
    }

    void DenseExampleWiseStatisticVector::remove(gradient_const_iterator gradientsBegin,
                                                 gradient_const_iterator gradientsEnd,
                                                 hessian_const_iterator hessiansBegin,
                                                 hessian_const_iterator hessiansEnd) {
        removeFromView(gradients_, gradientsBegin, numGradients_);
        removeFromView(hessians_, hessiansBegin, numHessians_);
    }

    void DenseExampleWiseStatisticVector::remove(gradient_const_iterator gradientsBegin,
                                                 gradient_const_iterator gradientsEnd,
                                                 hessian_const_iterator hessiansBegin,
                                                 hessian_const_iterator hessiansEnd, float64 weight) {
        removeFromView(gradients_, gradientsBegin, numGradients_, weight);
        removeFromView(hessians_, hessiansBegin, numHessians_, weight);
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const CompleteIndexVector& indices) {
        addToView(gradients_, gradientsBegin, numGradients_);
        addToView(hessians_, hessiansBegin, numHessians_);
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const PartialIndexVector& indices) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(gradients_, gradientsBegin, indexIterator, numGradients_);

        for (uint32 i = 0; i < numGradients_; i++) {
            uint32 index = indexIterator[i];
            addToView(&hessians_[triangularNumber(i)], &hessiansBegin[triangularNumber(index)], indexIterator, i + 1);
        }
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const CompleteIndexVector& indices, float64 weight) {
        addToView(gradients_, gradientsBegin, numGradients_, weight);
        addToView(hessians_, hessiansBegin, numHessians_, weight);
    }

    void DenseExampleWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                      gradient_const_iterator gradientsEnd,
                                                      hessian_const_iterator hessiansBegin,
                                                      hessian_const_iterator hessiansEnd,
                                                      const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToView(gradients_, gradientsBegin, indexIterator, numGradients_, weight);

        for (uint32 i = 0; i < numGradients_; i++) {
            uint32 index = indexIterator[i];
            addToView(&hessians_[triangularNumber(i)], &hessiansBegin[triangularNumber(index)], indexIterator, i + 1,
                      weight);
        }
    }

    void DenseExampleWiseStatisticVector::difference(
      gradient_const_iterator firstGradientsBegin, gradient_const_iterator firstGradientsEnd,
      hessian_const_iterator firstHessiansBegin, hessian_const_iterator firstHessiansEnd,
      const CompleteIndexVector& firstIndices, gradient_const_iterator secondGradientsBegin,
      gradient_const_iterator secondGradientsEnd, hessian_const_iterator secondHessiansBegin,
      hessian_const_iterator secondHessiansEnd) {
        setViewToDifference(gradients_, firstGradientsBegin, secondGradientsBegin, numGradients_);
        setViewToDifference(hessians_, firstHessiansBegin, secondHessiansBegin, numHessians_);
    }

    void DenseExampleWiseStatisticVector::difference(
      gradient_const_iterator firstGradientsBegin, gradient_const_iterator firstGradientsEnd,
      hessian_const_iterator firstHessiansBegin, hessian_const_iterator firstHessiansEnd,
      const PartialIndexVector& firstIndices, gradient_const_iterator secondGradientsBegin,
      gradient_const_iterator secondGradientsEnd, hessian_const_iterator secondHessiansBegin,
      hessian_const_iterator secondHessiansEnd) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setViewToDifference(gradients_, firstGradientsBegin, secondGradientsBegin, indexIterator, numGradients_);

        for (uint32 i = 0; i < numGradients_; i++) {
            uint32 offset = triangularNumber(i);
            uint32 index = indexIterator[i];
            setViewToDifference(&hessians_[offset], &firstHessiansBegin[triangularNumber(index)],
                                &secondHessiansBegin[offset], indexIterator, i + 1);
        }
    }

}
