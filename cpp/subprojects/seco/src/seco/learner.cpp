#include "seco/learner.hpp"


namespace seco {

    SeCoRuleLearner::SeCoRuleLearner(std::unique_ptr<Config> configPtr)
        : AbstractRuleLearner(std::move(configPtr)) {

    }

    std::unique_ptr<IStatisticsProviderFactory> SeCoRuleLearner::createStatisticsProviderFactory() const {
        // TODO Implement
        return nullptr;
    }

    std::unique_ptr<IModelBuilder> SeCoRuleLearner::createModelBuilder() const {
        // TODO Implement
        return nullptr;
    }

    std::unique_ptr<IClassificationPredictorFactory> SeCoRuleLearner::createClassificationPredictorFactory() const {
        // TODO Implement
        return nullptr;
    }

}
