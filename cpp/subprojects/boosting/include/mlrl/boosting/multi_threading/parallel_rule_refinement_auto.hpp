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

            const ReadableProperty<IClassificationLossConfig> lossConfig_;

            const ReadableProperty<IHeadConfig> headConfig_;

            const ReadableProperty<IFeatureSamplingConfig> featureSamplingConfig_;

        public:

            /**
             * @param lossConfigGetter              A `ReadableProperty` that allows to access the
             *                                      `IClassificationLossConfig` that stores the configuration of the
             *                                      loss function
             * @param headConfigGetter              A `ReadableProperty` that allows to access the `IHeadConfig` that
             *                                      stores the configuration of the rule heads
             * @param featureSamplingConfigGetter   A `ReadableProperty` that allows to access the
             *                                      `IFeatureSamplingConfig` that stores the configuration of the method
             *                                      for sampling features
             */
            AutoParallelRuleRefinementConfig(ReadableProperty<IClassificationLossConfig> lossConfigGetter,
                                             ReadableProperty<IHeadConfig> headConfigGetter,
                                             ReadableProperty<IFeatureSamplingConfig> featureSamplingConfigGetter);

            /**
             * @see `IMultiThreadingConfig::getNumThreads`
             */
            uint32 getNumThreads(const IFeatureMatrix& featureMatrix, uint32 numOutputs) const override;
    };

}
