/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/prediction/predictor_score.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a predictor that predicts output-wise scores for given query examples by summing up the
     * scores that are provided by individual rules for each output individually.
     */
    class OutputWiseScorePredictorConfig final : public IScorePredictorConfig {
        private:

            const ReadableProperty<IMultiThreadingConfig> multiThreadingConfig_;

        public:

            /**
             * @param multiThreadingConfigGetter A `ReadableProperty` that allows to access the `IMultiThreadingConfig`
             *                                   that stores the configuration of the multi-threading behavior that
             *                                   should be used to predict for several query examples in parallel
             */
            OutputWiseScorePredictorConfig(ReadableProperty<IMultiThreadingConfig> multiThreadingConfigGetter);

            /**
             * @see `IPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<IScorePredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                           uint32 numOutputs) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
