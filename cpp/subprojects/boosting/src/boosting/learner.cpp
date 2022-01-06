#include "boosting/learner.hpp"


namespace boosting {

    BoostingRuleLearner::BoostingRuleLearner(Config config)
        : AbstractRuleLearner(config) {

    }

    std::unique_ptr<IStatisticsProviderFactory> BoostingRuleLearner::createStatisticsProviderFactory() const {
        // TODO Implement
        return nullptr;
    }

    std::unique_ptr<IModelBuilder> BoostingRuleLearner::createModelBuilder() const {
        // TODO Implement
        return nullptr;
    }

    std::unique_ptr<IClassificationPredictorFactory> BoostingRuleLearner::createClassificationPredictorFactory() const {
        // TODO Implement
        return nullptr;
    }

}
