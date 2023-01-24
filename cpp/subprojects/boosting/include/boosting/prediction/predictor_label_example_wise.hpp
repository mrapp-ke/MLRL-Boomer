/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/predictor_label.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "boosting/losses/loss.hpp"


namespace boosting {

    /**
     * Allows to configure a predictor that predicts known label vectors for given query examples by summing up the
     * scores that are provided by an existing rule-based model and comparing the aggregated score vector to the known
     * label vectors according to a certain distance measure. The label vector that is closest to the aggregated score
     * vector is finally predicted.
     */
    class ExampleWiseLabelPredictorConfig final : public ILabelPredictorConfig {

        private:

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

        public:

            /**
             * @param lossConfigPtr             A reference to an unique pointer that stores the configuration of the
             *                                  loss function
             * @param multiThreadingConfigPtr   A reference to an unique pointer that stores the configuration of the
             *                                  multi-threading behavior that should be used to predict for several
             *                                  query examples in parallel
             */
            ExampleWiseLabelPredictorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            /**
             * @see `IPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<ILabelPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                           uint32 numLabels) const override;

            /**
             * @see `ILabelPredictorConfig::createSparsePredictorFactory`
             */
            std::unique_ptr<ISparseLabelPredictorFactory> createSparsePredictorFactory(
                const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;

    };

}
