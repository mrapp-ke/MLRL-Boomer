/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/multi_threading/multi_threading.hpp"

/**
 * Defines an interface for all classes that allow to configure the multi-threading behavior of a parallelizable
 * algorithm by manually specifying the number of threads to be used.
 */
class MLRLCOMMON_API IManualMultiThreadingConfig {
    public:

        virtual ~IManualMultiThreadingConfig() {}

        /**
         * Returns the number of preferred threads.
         *
         * @return The number of preferred threads or 0, if all available CPU cores are utilized
         */
        virtual uint32 getNumPreferredThreads() const = 0;

        /**
         * Sets the number of preferred threads. If not enough CPU cores are available or if multi-threading support was
         * disabled at compile-time, as many threads as possible will be used.
         *
         * @param numPreferredThreads   The preferred number of threads. Must be at least 1 or 0, if all available CPU
         *                              cores should be utilized
         * @return                      A reference to an object of type `IManualMultiThreadingConfig` that allows
         *                              further configuration of the multi-threading behavior
         */
        virtual IManualMultiThreadingConfig& setNumPreferredThreads(uint32 numPreferredThreads) = 0;
};

/**
 * Allows to configure the multi-threading behavior of a parallelizable algorithm by manually specifying the number of
 * threads to be used.
 */
class ManualMultiThreadingConfig final : public IMultiThreadingConfig,
                                         public IManualMultiThreadingConfig {
    private:

        uint32 numPreferredThreads_;

    public:

        ManualMultiThreadingConfig();

        uint32 getNumPreferredThreads() const override;

        IManualMultiThreadingConfig& setNumPreferredThreads(uint32 numPreferredThreads) override;

        MultiThreadingSettings getSettings(const IFeatureMatrix& featureMatrix, uint32 numOutputs) const override;
};
