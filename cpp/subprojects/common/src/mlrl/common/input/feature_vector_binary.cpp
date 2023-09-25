#include "mlrl/common/input/feature_vector_binary.hpp"

BinaryFeatureVector::BinaryFeatureVector(uint32 numExamples, int32 minorityValue, int32 majorityValue)
    : NominalFeatureVector(1, numExamples, majorityValue) {
    this->values_begin()[0] = minorityValue;
    this->indptr_begin()[0] = 0;
}
