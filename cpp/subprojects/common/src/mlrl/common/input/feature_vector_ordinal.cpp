#include "mlrl/common/input/feature_vector_ordinal.hpp"

OrdinalFeatureVector::OrdinalFeatureVector(uint32 numValues, uint32 numExamples, int32 majorityValue)
    : NominalFeatureVector(numValues, numExamples, majorityValue) {}
