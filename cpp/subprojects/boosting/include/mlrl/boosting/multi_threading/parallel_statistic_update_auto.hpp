/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

#include <memory>

namespace boosting {

    /**
     * Allows to configure the multi-threading behavior that is used for the parallel update of statistics by
     * automatically deciding for the number of threads to be used.
     */
    class AutoParallelStatisticUpdateConfig final : public IMultiThreadingConfig {
        private:

            const ReadableProperty<ILossConfig> lossConfig_;

        public:

            /**
             * @param lossConfigGetter A `ReadableProperty` that allows to access the `ILossConfig` that stores the
             *                         configuration of the loss function
             */
            AutoParallelStatisticUpdateConfig(const ReadableProperty<ILossConfig> lossConfigGetter);

            /**
             * @see `IMultiThreadingConfig::getNumThreads`
             */
            uint32 getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numOutputs) const override;
    };

}
