/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_type.hpp"

/**
 * Represents a numerical/ordinal feature.
 */
class NumericalFeatureType final : public IFeatureType {
    public:

        bool isOrdinal() const override;

        bool isNominal() const override;

        std::unique_ptr<IFeatureVector> createFeatureVector(
          uint32 featureIndex, const FortranContiguousConstView<const float32>& featureMatrix) const override;

        std::unique_ptr<IFeatureVector> createFeatureVector(
          uint32 featureIndex, const CscConstView<const float32>& featureMatrix) const override;
};
