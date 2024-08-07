/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/sampling/feature_sampling.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure the multi-threading behavior that is used for the parallel refinement of rules by
     * automatically deciding for the number of threads to be used.
     */
    class AutoParallelRuleRefinementConfig final : public IMultiThreadingConfig {
        private:

            const ReadableProperty<ILossConfig> lossConfig_;

            const ReadableProperty<IHeadConfig> headConfig_;

            const ReadableProperty<IFeatureSamplingConfig> featureSamplingConfig_;

        public:

            /**
             * @param lossConfig            A `ReadableProperty` that allows to access the `ILossConfig` that stores the
             *                              configuration of the loss function
             * @param headConfig            A `ReadableProperty` that allows to access the `IHeadConfig` that stores the
             *                              configuration of the rule heads
             * @param featureSamplingConfig A `ReadableProperty` that allows to access the `IFeatureSamplingConfig` that
             *                              stores the configuration of the method for sampling features
             */
            AutoParallelRuleRefinementConfig(ReadableProperty<ILossConfig> lossConfig,
                                             ReadableProperty<IHeadConfig> headConfig,
                                             ReadableProperty<IFeatureSamplingConfig> featureSamplingConfig);

            /**
             * @see `IMultiThreadingConfig::getNumThreads`
             */
            uint32 getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numOutputs) const override;
    };

}
