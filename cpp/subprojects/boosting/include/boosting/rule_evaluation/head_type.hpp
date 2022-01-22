/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/statistics/statistics_provider.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to configure the heads of the rules that should be induced by a
     * rule learner.
     */
    class IHeadConfig {

        public:

            virtual ~IHeadConfig() { };

            /**
             * Creates and returns a new object of type `IStatisticsProviderFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `IStatisticsProviderFactory` that has been created
             */
            virtual std::unique_ptr<IStatisticsProviderFactory> configure() const = 0;

    };

}
