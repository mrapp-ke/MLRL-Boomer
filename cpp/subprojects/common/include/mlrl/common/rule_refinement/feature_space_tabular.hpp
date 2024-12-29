/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_binning.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/rule_refinement/feature_space.hpp"

#include <memory>

/**
 * Allows to create objects of type `IFeatureSpace` that provide access to a tabular feature space.
 */
class TabularFeatureSpaceFactory final : public IFeatureSpaceFactory {
    private:

        const std::unique_ptr<IFeatureBinningFactory> featureBinningFactoryPtr_;

        const MultiThreadingSettings multiThreadingSettings_;

    public:

        /**
         * @param featureBinningFactoryPtr  An unique pointer to an object of type `IFeatureBinningFactory` that allows
         *                                  to create implementations of the binning method to be used for assigning
         *                                  numerical feature values to bins
         * @param multiThreadingSettings    An object of type `MultiThreadingSettings` that stores the settings to be
         *                                  used for updating statistics in parallel
         */
        TabularFeatureSpaceFactory(std::unique_ptr<IFeatureBinningFactory> featureBinningFactoryPtr,
                                   MultiThreadingSettings multiThreadingSettings);

        std::unique_ptr<IFeatureSpace> create(const IColumnWiseFeatureMatrix& featureMatrix,
                                              const IFeatureInfo& featureInfo,
                                              IStatisticsProvider& statisticsProvider) const override;
};
