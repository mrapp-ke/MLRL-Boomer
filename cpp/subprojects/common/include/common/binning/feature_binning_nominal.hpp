/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/binning/feature_binning.hpp"


/**
 * Assigns nominal feature values to bins, such that each bin contains one of the available values.
 */
class NominalFeatureBinning final : public IFeatureBinning {

    public:

        Result createBins(FeatureVector& featureVector, uint32 numExamples) const override;

};

/**
 * Allows to create instances of the type `IFeatureBinning` that assign nominal feature values to bins, such that each
 * bin contains one of the available values.
 */
class NominalFeatureBinningFactory final : public IFeatureBinningFactory {

    public:

        std::unique_ptr<IFeatureBinning> create() const override;

};
