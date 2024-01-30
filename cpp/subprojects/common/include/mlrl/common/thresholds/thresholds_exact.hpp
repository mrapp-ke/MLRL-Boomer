/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/binning/feature_binning.hpp"
#include "mlrl/common/thresholds/thresholds.hpp"

/**
 * A factory that allows to create instances of the type `ExactThresholds`.
 */
class ExactThresholdsFactory final : public IThresholdsFactory {
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
        ExactThresholdsFactory(std::unique_ptr<IFeatureBinningFactory> featureBinningFactoryPtr, uint32 numThreads);

        std::unique_ptr<IThresholds> create(const IColumnWiseFeatureMatrix& featureMatrix,
                                            const IFeatureInfo& featureInfo,
                                            IStatisticsProvider& statisticsProvider) const override;
};
