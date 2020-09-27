#include "sub_sampling.h"


template<class T>
DenseWeightVector<T>::DenseWeightVector(DenseVector<T>* weights, uint32 sumOfWeights) {
    weights_ = weights;
    sumOfWeights = sumOfWeights;
}

template<class T>
DenseWeightVector<T>::~DenseWeightVector() {
    delete weights_;
}

template<class T>
uint32 DenseWeightVector<T>::getNumElements() {
    return weights_->getNumElements();
}

template<class T>
bool DenseWeightVector<T>::hasZeroElements() {
    return false;
}

template<class T>
uint32 DenseWeightVector<T>::getValue(uint32 pos) {
    return (uint32) weights_->getValue(pos);
}

template<class T>
uint32 DenseWeightVector<T>::getSumOfWeights() {
    return sumOfWeights_;
}

EqualWeightVector::EqualWeightVector(uint32 numElements) {
    numElements_ = numElements;
}

uint32 EqualWeightVector::getNumElements() {
    return numElements_;
}

bool EqualWeightVector::hasZeroElements() {
    return false;
}

uint32 EqualWeightVector::getValue(uint32 pos) {
    return 1;
}

uint32 EqualWeightVector::getSumOfWeights() {
    return numElements_;
}

IWeightVector* NoInstanceSubSamplingImpl::subSample(uint32 numExamples, RNG* rng) {
    return new EqualWeightVector(numExamples);
}

IIndexVector* NoFeatureSubSamplingImpl::subSample(uint32 numFeatures, RNG* rng) {
    return new RangeIndexVector(numFeatures);
}


IIndexVector* NoLabelSubSamplingImpl::subSample(uint32 numLabels, RNG* rng) {
    return new RangeIndexVector(numLabels);
}
