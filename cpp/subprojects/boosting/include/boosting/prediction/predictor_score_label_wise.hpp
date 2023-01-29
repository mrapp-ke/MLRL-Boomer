/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/multi_threading/multi_threading.hpp"
#include "common/prediction/predictor_score.hpp"

namespace boosting {

    /**
     * Allows to configure predictors that predict label-wise regression scores for given query examples by summing up
     * the scores that are provided by the individual rules of an existing rule-based model for each label individually.
     */
    class LabelWiseScorePredictorConfig final : public IScorePredictorConfig {
        private:

            const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr_;

        public:

            /**
             * @param multiThreadingConfigPtr A reference to an unique pointer that stores the configuration of the
             *                                multi-threading behavior that should be used to predict for several query
             *                                examples in parallel
             */
            LabelWiseScorePredictorConfig(const std::unique_ptr<IMultiThreadingConfig>& multiThreadingConfigPtr);

            /**
             * @see `IPredictorConfig::createPredictorFactory`
             */
            std::unique_ptr<IScorePredictorFactory> createPredictorFactory(const IRowWiseFeatureMatrix& featureMatrix,
                                                                           uint32 numLabels) const override;

            /**
             * @see `IPredictorConfig::isLabelVectorSetNeeded`
             */
            bool isLabelVectorSetNeeded() const override;
    };

}
