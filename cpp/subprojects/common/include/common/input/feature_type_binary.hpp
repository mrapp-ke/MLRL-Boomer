/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/input/feature_type.hpp"

/**
 * Represents a binary feature.
 */
class BinaryFeatureType final : public IFeatureType {
    public:

        bool isNumerical() const override;
};
