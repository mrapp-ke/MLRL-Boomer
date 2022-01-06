/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/learner.hpp"


namespace boosting {

    /**
     * A rule learner that makes use of gradient boosting.
     */
    class BoostingRuleLearner final : public AbstractRuleLearner {

        public:

            /**
             * Allows to configure a rule learner that makes use of gradient boosting.
             */
            class Config : public AbstractRuleLearner::Config {

            };

        protected:

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory() const override;

            std::unique_ptr<IModelBuilder> createModelBuilder() const override;

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override;

    };

}
