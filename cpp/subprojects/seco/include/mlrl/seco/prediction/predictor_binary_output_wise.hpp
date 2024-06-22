/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/prediction/predictor_binary.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

#include <memory>

namespace seco {

    /**
     * Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
     * irrelevant by processing rules of an existing rule-based model in the order they have been learned. If a rule
     * covers an example, its prediction (1 if the label is relevant, 0 otherwise) is applied to each label
     * individually, if none of the previous rules has already predicted for a particular example and label.
     */
    class OutputWiseBinaryPredictorConfig final : public IBinaryPredictorConfig {
        private:

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

        public:

            /**
             * @param multiThreadingConfigGetter A `ReadableProperty` that allows to access the `IMultiThreadingConfig`
             *                                   that stores the configuration of the multi-threading behavior that
             *                                   should be used to predict for several query examples in parallel
             */
            OutputWiseBinaryPredictorConfig(ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter);

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
