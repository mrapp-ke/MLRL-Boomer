#include "mlrl/common/input/feature_vector_ordinal.hpp"

OrdinalFeatureVector::OrdinalFeatureVector(uint32 numValues, uint32 numExamples, int32 majorityValue)
    : NominalFeatureVector(numValues, numExamples, majorityValue), order_(new uint32[numValues]) {}

OrdinalFeatureVector::~OrdinalFeatureVector() {
    delete[] order_;
}

OrdinalFeatureVector::index_iterator OrdinalFeatureVector::order_begin(uint32 index) {
    return order_;
}

OrdinalFeatureVector::index_iterator OrdinalFeatureVector::order_end(uint32 index) {
    return &order_[this->getNumElements()];
}

OrdinalFeatureVector::index_const_iterator OrdinalFeatureVector::order_cbegin(uint32 index) const {
    return order_;
}

OrdinalFeatureVector::index_const_iterator OrdinalFeatureVector::order_cend(uint32 index) const {
    return &order_[this->getNumElements()];
}
