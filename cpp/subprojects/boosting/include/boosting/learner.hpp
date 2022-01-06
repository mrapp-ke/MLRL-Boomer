/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/learner.hpp"


namespace boosting {

    /**
     * Defines an interface for all rule learners that make use of gradient boosting.
     */
    class IBoostingRuleLearner : virtual public IRuleLearner {

        public:

            /**
             * Defines an interface for all classes that allow to configure a rule learner that makes use of gradient
             * boosting.
             */
            class IConfig : virtual public IRuleLearner::IConfig {

                public:

                    virtual ~IConfig() override { };

            };

            virtual ~IBoostingRuleLearner() override { };

    };

    /**
     * Creates and returns a new object of type `IBoostingRuleLearner::IConfig`.
     *
     * @return An unique pointer to an object of type `IBoostingRuleLearner::IConfig` that has been created
     */
    std::unique_ptr<IBoostingRuleLearner::IConfig> createBoostingRuleLearnerConfig();

    /**
     * Creates and returns a new object of type `IBoostingRuleLearner`.
     *
     * @param configPtr An unique pointer to an object of type `IBoostingRuleLearner::IConfig` that specifies the
     *                  configuration that should be used by the rule learner.
     * @return          An unique pointer to an object of type `IBoostingRuleLearner` that has been created
     */
    std::unique_ptr<IBoostingRuleLearner> createBoostingRuleLearner(
        std::unique_ptr<IBoostingRuleLearner::IConfig> configPtr);

}
