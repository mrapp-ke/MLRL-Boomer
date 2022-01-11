/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/learner.hpp"


namespace seco {

    /**
     * Defines an interface for all rule learners that make use of the separate-and-conquer (SeCo) paradigm.
     */
    class ISeCoRuleLearner : virtual public IRuleLearner {

        public:

            /**
             * Defines an interface for all classes that allow to configure a rule learner that makes use of the
             * separate-and-conquer (SeCo) paradigm.
             */
            class IConfig : virtual public IRuleLearner::IConfig {

                friend class SeCoRuleLearner;

                public:

                    virtual ~IConfig() override { };

            };

            virtual ~ISeCoRuleLearner() override { };

    };

    /**
     * An implementation of the type `ISeCoRuleLearner`.
     */
    class SeCoRuleLearner final : public AbstractRuleLearner, virtual public ISeCoRuleLearner {

        public:

            /**
             * Allows to configure a rule learner that makes use of the separate-and-conquer (SeCo) paradigm.
             */
            class Config : public AbstractRuleLearner::Config, virtual public ISeCoRuleLearner::IConfig {

                public:

                    Config();

            };

        protected:

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory() const override;

            std::unique_ptr<IModelBuilder> createModelBuilder() const override;

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override;

        public:

            /**
             * @param configPtr An unique pointer to an object of type `ISeCoRuleLearner::IConfig` that specifies the
             *                  configuration that should be used by the rule learner
             */
            SeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr);

    };

    /**
     * Creates and returns a new object of type `ISeCoRuleLearner::IConfig`.
     *
     * @return An unique pointer to an object of type `ISeCoRuleLearner::IConfig` that has been created
     */
    std::unique_ptr<ISeCoRuleLearner::IConfig> createSeCoRuleLearnerConfig();

    /**
     * Creates and returns a new object of type `ISeCoRuleLearner`.
     *
     * @param configPtr An unique pointer to an object of type `ISeCoRuleLearner::IConfig` that specifies the
     *                  configuration that should be used by the rule learner.
     * @return          An unique pointer to an object of type `ISeCoRuleLearner` that has been created
     */
    std::unique_ptr<ISeCoRuleLearner> createSeCoRuleLearner(std::unique_ptr<ISeCoRuleLearner::IConfig> configPtr);

}
