/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/multi_threading/multi_threading.hpp"
#include "common/prediction/predictor_binary.hpp"

namespace seco {

    /**
     * Allows to configure a predictor that predicts whether individual labels of given query examples are relevant or
     * irrelevant by processing rules of an existing rule-based model in the order they have been learned. If a rule
     * covers an example, its prediction (1 if the label is relevant, 0 otherwise) is applied to each label
     * individually, if none of the previous rules has already predicted for a particular example and label.
     */
    class LabelWiseBinaryPredictorConfig final : public IBinaryPredictorConfig {
        private:

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

        public:

            /**
             * @param multiThreadingConfigPtr A reference to an unique pointer that stores the configuration of the
             *                                multi-threading behavior that should be used to predict for several query
             *                                examples in parallel
             */
            LabelWiseBinaryPredictorConfig(const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            std::unique_ptr<IBinaryPredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                            uint32 numLabels) const override;

            std::unique_ptr<ISparseBinaryPredictorFactory> createSparsePredictorFactory(
                const IRowWiseFeatureMatrix& featureMatrix, uint32 numLabels) const override;

            bool isLabelVectorSetNeeded() const override;
    };

}
