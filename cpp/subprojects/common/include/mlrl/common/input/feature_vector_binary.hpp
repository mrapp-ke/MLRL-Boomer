/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_nominal.hpp"

/**
 * A feature vector that stores the indices of all examples that are associated with the minority value, i.e., the least
 * frequent value, of a binary feature.
 */
class BinaryFeatureVector final : public NominalFeatureVector {
    public:

        /**
         * @param numMinorityExamples   The number of elements in the vector, i.e., the number of examples associated
         *                              with the minority value
         * @param minorityValue         The minority value, i.e., the least frequent value, of the binary feature
         * @param majorityValue         The majority value, i.e., the most frequent value, of the binary feature
         */
        BinaryFeatureVector(uint32 numMinorityExamples, int32 minorityValue, int32 majorityValue);
};
