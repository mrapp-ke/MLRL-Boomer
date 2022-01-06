/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/learner.hpp"


namespace seco {

    /**
     * Defines an interface for all rule learners that make use of the separate-and-conquer (SeCo) paradigm.
     */
    class ISeCoRuleLearner : public IRuleLearner {

        public:

            /**
             * Defines an interface for all classes that allow to configure a rule learner that makes use of the
             * separate-and-conquer (SeCo) paradigm.
             */
            class IConfig : public IRuleLearner::IConfig {

                public:

                    virtual ~IConfig() override { };

            };

            virtual ~ISeCoRuleLearner() override { };

    };

    /**
     * A rule learner that makes use of the separate-and-conquer (SeCo) paradigm.
     */
    class SeCoRuleLearner final : public AbstractRuleLearner, public ISeCoRuleLearner {

        public:

            /**
             * Allows to configure a rule learner that makes use of the separate-and-conquer (SeCo) paradigm.
             */
            class Config : public AbstractRuleLearner::Config, public ISeCoRuleLearner::IConfig {

            };

        protected:

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory() const override;

            std::unique_ptr<IModelBuilder> createModelBuilder() const override;

            std::unique_ptr<IClassificationPredictorFactory> createClassificationPredictorFactory() const override;

        public:

            /**
             * @param configPtr An unique pointer to the configuration that should be used by the rule learner
             */
            SeCoRuleLearner(std::unique_ptr<Config> configPtr);

    };

}
