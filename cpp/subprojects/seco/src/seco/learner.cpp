#include "seco/learner.hpp"


namespace seco {

    /**
     * An implementation of the type `ISeCoRuleLearner`.
     */
    class SeCoRuleLearner final : public AbstractRuleLearner, virtual public ISeCoRuleLearner {

        public:

            /**
             * Allows to configure a rule learner that makes use of the separate-and-conquer (SeCo) paradigm.
             */
            class Config : public AbstractRuleLearner::Config, virtual public ISeCoRuleLearner::IConfig {

            };

        protected:

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory() const override {
                // TODO Implement
                return nullptr;
            }

            std::unique_ptr<IModelBuilder> createModelBuilder() const override {
                // TODO Implement
                return nullptr;
            }

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override {
                // TODO Implement
                return nullptr;
            }

        public:

            /**
             * @param configPtr An unique pointer to an object of type `ISeCoRuleLearner::IConfig` that specifies the
             *                  configuration that should be used by the rule learner
             */
            SeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr)
                : AbstractRuleLearner(std::move(configPtr)) {

            }

    };

    std::unique_ptr<ISeCoRuleLearner::IConfig> createSeCoRuleLearnerConfig() {
        return std::make_unique<SeCoRuleLearner::Config>();
    }

    std::unique_ptr<ISeCoRuleLearner> createSeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr) {
        return std::make_unique<SeCoRuleLearner>(std::move(configPtr));
    }

}
