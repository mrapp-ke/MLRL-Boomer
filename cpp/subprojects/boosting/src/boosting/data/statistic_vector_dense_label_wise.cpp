#include "boosting/data/statistic_vector_dense_label_wise.hpp"
#include "boosting/data/arrays.hpp"
#include "common/data/arrays.hpp"
#include <cstdlib>


namespace boosting {

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(uint32 numElements)
        : DenseLabelWiseStatisticVector(numElements, false) {

    }

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(uint32 numElements, bool init)
        : numElements_(numElements),
          statistics_((Tuple<float64>*) (init ? calloc(numElements, sizeof(Tuple<float64>))
                                              : malloc(numElements * sizeof(Tuple<float64>)))),
          gradients_((float64*) (init ? calloc(numElements, sizeof(float64)) : malloc(numElements * sizeof(float64)))),
          hessians_((float64*) (init ? calloc(numElements, sizeof(float64)) : malloc(numElements * sizeof(float64)))) {

    }

    DenseLabelWiseStatisticVector::DenseLabelWiseStatisticVector(const DenseLabelWiseStatisticVector& vector)
        : DenseLabelWiseStatisticVector(vector.numElements_) {
        copyArray(vector.statistics_, statistics_, numElements_);
        copyArray(vector.gradients_, gradients_, numElements_);
        copyArray(vector.hessians_, hessians_, numElements_);
    }

    DenseLabelWiseStatisticVector::~DenseLabelWiseStatisticVector() {
        free(statistics_);
        free(gradients_);
        free(hessians_);
    }

    DenseLabelWiseStatisticVector::const_iterator DenseLabelWiseStatisticVector::cbegin() const {
        return statistics_;
    }

    DenseLabelWiseStatisticVector::const_iterator DenseLabelWiseStatisticVector::cend() const {
        return &statistics_[numElements_];
    }

    DenseLabelWiseStatisticVector::gradient_iterator DenseLabelWiseStatisticVector::gradients_begin() {
        return gradients_;
    }

    DenseLabelWiseStatisticVector::gradient_iterator DenseLabelWiseStatisticVector::gradients_end() {
        return &gradients_[numElements_];
    }

    DenseLabelWiseStatisticVector::gradient_const_iterator DenseLabelWiseStatisticVector::gradients_cbegin() const {
        return gradients_;
    }

    DenseLabelWiseStatisticVector::gradient_const_iterator DenseLabelWiseStatisticVector::gradients_cend() const {
        return &gradients_[numElements_];
    }

    DenseLabelWiseStatisticVector::hessian_iterator DenseLabelWiseStatisticVector::hessians_begin() {
        return hessians_;
    }

    DenseLabelWiseStatisticVector::hessian_iterator DenseLabelWiseStatisticVector::hessians_end() {
        return &hessians_[numElements_];
    }

    DenseLabelWiseStatisticVector::hessian_const_iterator DenseLabelWiseStatisticVector::hessians_cbegin() const {
        return hessians_;
    }

    DenseLabelWiseStatisticVector::hessian_const_iterator DenseLabelWiseStatisticVector::hessians_cend() const {
        return &hessians_[numElements_];
    }

    uint32 DenseLabelWiseStatisticVector::getNumElements() const {
        return numElements_;
    }

    void DenseLabelWiseStatisticVector::clear() {
        setArrayToZeros(gradients_, numElements_);
        setArrayToZeros(hessians_, numElements_);
    }

    void DenseLabelWiseStatisticVector::add(gradient_const_iterator gradientsBegin,
                                            gradient_const_iterator gradientsEnd, hessian_const_iterator hessiansBegin,
                                            hessian_const_iterator hessiansEnd) {
        addToArray(gradients_, gradientsBegin, numElements_);
        addToArray(hessians_, hessiansBegin, numElements_);
    }

    void DenseLabelWiseStatisticVector::add(gradient_const_iterator gradientsBegin,
                                            gradient_const_iterator gradientsEnd, hessian_const_iterator hessiansBegin,
                                            hessian_const_iterator hessiansEnd, float64 weight) {
        addToArray(gradients_, gradientsBegin, numElements_, weight);
        addToArray(hessians_, hessiansBegin, numElements_, weight);
    }

    void DenseLabelWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                    gradient_const_iterator gradientsEnd,
                                                    hessian_const_iterator hessiansBegin,
                                                    hessian_const_iterator hessiansEnd, const FullIndexVector& indices,
                                                    float64 weight) {
        addToArray(gradients_, gradientsBegin, numElements_, weight);
        addToArray(hessians_, hessiansBegin, numElements_, weight);
    }

    void DenseLabelWiseStatisticVector::addToSubset(gradient_const_iterator gradientsBegin,
                                                    gradient_const_iterator gradientsEnd,
                                                    hessian_const_iterator hessiansBegin,
                                                    hessian_const_iterator hessiansEnd,
                                                    const PartialIndexVector& indices, float64 weight) {
        PartialIndexVector::const_iterator indexIterator = indices.cbegin();
        addToArray(gradients_, gradientsBegin, indexIterator, numElements_, weight);
        addToArray(hessians_, hessiansBegin, indexIterator, numElements_, weight);
    }

    void DenseLabelWiseStatisticVector::difference(gradient_const_iterator firstGradientsBegin,
                                                   gradient_const_iterator firstGradientsEnd,
                                                   hessian_const_iterator firstHessiansBegin,
                                                   hessian_const_iterator firstHessiansEnd,
                                                   const FullIndexVector& firstIndices,
                                                   gradient_const_iterator secondGradientsBegin,
                                                   gradient_const_iterator secondGradientsEnd,
                                                   hessian_const_iterator secondHessiansBegin,
                                                   hessian_const_iterator secondHessiansEnd) {
        setArrayToDifference(gradients_, firstGradientsBegin, secondGradientsBegin, numElements_);
        setArrayToDifference(hessians_, firstHessiansBegin, secondHessiansBegin, numElements_);
    }

    void DenseLabelWiseStatisticVector::difference(gradient_const_iterator firstGradientsBegin,
                                                   gradient_const_iterator firstGradientsEnd,
                                                   hessian_const_iterator firstHessiansBegin,
                                                   hessian_const_iterator firstHessiansEnd,
                                                   const PartialIndexVector& firstIndices,
                                                   gradient_const_iterator secondGradientsBegin,
                                                   gradient_const_iterator secondGradientsEnd,
                                                   hessian_const_iterator secondHessiansBegin,
                                                   hessian_const_iterator secondHessiansEnd) {
        PartialIndexVector::const_iterator indexIterator = firstIndices.cbegin();
        setArrayToDifference(gradients_, firstGradientsBegin, secondGradientsBegin, indexIterator, numElements_);
        setArrayToDifference(hessians_, firstHessiansBegin, secondHessiansBegin, indexIterator, numElements_);
    }

}
