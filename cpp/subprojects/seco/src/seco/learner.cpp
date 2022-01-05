#include "seco/learner.hpp"


namespace seco {

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
