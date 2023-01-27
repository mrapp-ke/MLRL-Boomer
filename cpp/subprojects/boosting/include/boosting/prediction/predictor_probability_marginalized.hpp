/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "common/prediction/predictor_probability.hpp"

namespace boosting {

    /**
     * Allows to configure a predictor that predicts marginalized probabilities for given query examples, which estimate
     * the chance of individual labels to be relevant, by summing up the scores that are provided by individual rules of
     * an existing rule-based model and comparing the aggregated score vector to the known label vectors according to a
     * certain distance measure. The probability for an individual label calculates as the sum of the distances that
     * have been obtained for all label vectors, where the respective label is specified to be relevant, divided by the
     * total sum of all distances.
     */
    class MarginalizedProbabilityPredictorConfig final : public IProbabilityPredictorConfig {
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
            MarginalizedProbabilityPredictorConfig(
                const std::unique_ptr<ILossConfig>& lossConfigPtr,
                const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            /**
             * @see `IProbabilityPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<IProbabilityPredictorFactory> createPredictorFactory(
                const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
