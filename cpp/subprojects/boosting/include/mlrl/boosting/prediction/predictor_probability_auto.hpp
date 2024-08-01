/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/prediction/predictor_probability.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a predictor that automatically decides for a method that is used to predict probabilities for
     * given query examples, which estimate the chance of individual labels to be relevant.
     */
    class AutomaticProbabilityPredictorConfig final : public IProbabilityPredictorConfig {
        private:

            const ReadableProperty<ILossConfig> lossConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

        public:

            /**
             * @param lossConfigGetter              A `ReadableProperty` that allows to access the `ILossConfig` that
             *                                      stores the configuration of the loss function
             * @param multiThreadingConfigGetter    A `ReadableProperty` that allows to access the
             *                                      `IMultiThreadingConfig` that stores the configuration of the
             *                                      multi-threading behavior that should be used to predict for several
             *                                      query examples in parallel
             */
            AutomaticProbabilityPredictorConfig(ReadableProperty<ILossConfig> lossConfigGetter,
                                                ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter);

            /**
             * @see `IProbabilityPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<IProbabilityPredictorFactory> createPredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numOutputs) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
