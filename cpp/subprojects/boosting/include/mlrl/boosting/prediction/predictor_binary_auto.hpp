/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a predictor that automatically decides for a method that is used to predict whether
     * individual labels of given query examples are relevant or not.
     */
    class AutomaticBinaryPredictorConfig : public IBinaryPredictorConfig {
        private:

            const ReadableProperty<IClassificationLossConfig> lossConfig_;

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

        public:

            /**
             * @param lossConfig            A `ReadableProperty` that allows to access the `IClassificationLossConfig`
             *                              that stores the configuration of the loss function
             * @param multiThreadingConfig  A `ReadableProperty` that allows to access the `IMultiThreadingConfig` that
             *                              stores the configuration of the multi-threading behavior that should be used
             *                              to predict for several query examples in parallel
             */
            AutomaticBinaryPredictorConfig(ReadableProperty<IClassificationLossConfig> lossConfig,
                                           ReadableProperty<IMultiThreadingConfig> multiThreadingConfig);

            /**
             * @see `IPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<IBinaryPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                            uint32 numOutputs) const override;

            /**
             * @see `IBinaryPredictorConfig::createSparsePredictorFactory`
             */
            std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
              const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
