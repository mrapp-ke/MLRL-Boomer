/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/losses/loss.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "common/prediction/predictor_binary.hpp"

namespace boosting {

    /**
     * Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
     * irrelevant by summing up the scores that are provided by the individual rules of an existing rule-based model and
     * transforming them into binary values according to a certain threshold that is applied to each label individually
     * (1 if a score exceeds the threshold, i.e., the label is relevant, 0 otherwise).
     */
    class LabelWiseBinaryPredictorConfig final : public IBinaryPredictorConfig {
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
            LabelWiseBinaryPredictorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                           const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            /**
             * @see `IPredictorFactory::createPredictorFactory`
             */
            std::unique_ptr<IBinaryPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                            uint32 numLabels) const override;

            /**
             * @see `IBinaryPredictorFactory::createSparsePredictorFactory`
             */
            std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
                const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
