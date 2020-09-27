#include "sub_sampling.h"


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
