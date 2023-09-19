/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector_common.hpp"

/**
 * A feature vector that does not actually store any values. It is used in cases where all training examples have the
 * same value for a certain feature.
 */
class EqualFeatureVector final : public AbstractFeatureVector {
    public:

        uint32 getNumElements() const override;
};
