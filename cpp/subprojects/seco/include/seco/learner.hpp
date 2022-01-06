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

                public:

                    virtual ~IConfig() override { };

            };

            virtual ~ISeCoRuleLearner() override { };

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
