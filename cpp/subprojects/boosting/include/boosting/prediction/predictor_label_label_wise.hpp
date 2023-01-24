/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/prediction/predictor_label.hpp"
#include "common/multi_threading/multi_threading.hpp"
#include "boosting/losses/loss.hpp"


namespace boosting {

    /**
     * Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
     * irrelevant by summing up the scores that are provided by the individual rules of an existing rule-based model and
     * transforming them into binary values according to a certain threshold that is applied to each label individually
     * (1 if a score exceeds the threshold, i.e., the label is relevant, 0 otherwise).
     */
    class LabelWiseLabelPredictorConfig final : public ILabelPredictorConfig {

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
            LabelWiseLabelPredictorConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr,
                                          const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            /**
             * @see `IPredictorFactory::createPredictorFactory`
             */
            std::unique_ptr<ILabelPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                           uint32 numLabels) const override;

            /**
             * @see `ILabelPredictorFactory::createSparsePredictorFactory`
             */
            std::unique_ptr<ISparseLabelPredictorFactory> createSparsePredictorFactory(
                const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;

    };

}
