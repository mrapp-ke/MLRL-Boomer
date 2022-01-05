/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/learner.hpp"


namespace boosting {

    class BoostingRuleLearner final : public AbstractRuleLearner {

        protected:

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory() const override;

            std::unique_ptr<IModelBuilder> createModelBuilder() const override;

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override;

    };

}
