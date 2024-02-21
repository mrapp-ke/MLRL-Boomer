/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_binning.hpp"
#include "mlrl/common/thresholds/feature_space.hpp"

/**
 * Allows to create objects of type `IFeatureSpace` that provide access to a tabular feature space.
 */
class TabularFeatureSpaceFactory final : public IFeatureSpaceFactory {
    private:

        const std::unique_ptr<IFeatureBinningFactory> featureBinningFactoryPtr_;

        const uint32 numThreads_;

    public:

        /**
         * @param featureBinningFactoryPtr  An unique pointer to an object of type `IFeatureBinningFactory` that allows
         *                                  to create implementations of the binning method to be used for assigning
         *                                  numerical feature values to bins
         * @param numThreads                The number of CPU threads to be used to update statistics in parallel. Must
         *                                  be at least 1
         */
        TabularFeatureSpaceFactory(std::unique_ptr<IFeatureBinningFactory> featureBinningFactoryPtr, uint32 numThreads);

        std::unique_ptr<IFeatureSpace> create(const IColumnWiseFeatureMatrix& featureMatrix,
                                              const IFeatureInfo& featureInfo,
                                              IStatisticsProvider& statisticsProvider) const override;
};
