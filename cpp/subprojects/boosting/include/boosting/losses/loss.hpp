/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/measures/measure_evaluation.hpp"
#include "common/measures/measure_similarity.hpp"
#include "common/statistics/statistics_provider.hpp"


namespace boosting {

    /**
     * Defines an interface for all loss functions.
     */
    class ILoss : public IEvaluationMeasure, public ISimilarityMeasure {

        public:

            virtual ~ILoss() override { };

    };

    /**
     * Defines an interface for all classes that allow to configure a loss function.
     */
    class ILossConfig {

        public:

            virtual ~ILossConfig() { };

            /**
             * Creates and returns a new object of type `IStatisticsProviderFactory` according to the specified
             * configuration.
             *
             * @return An unique pointer to an object of type `IStatisticsProviderFactory` that has been created
             */
            virtual std::unique_ptr<IStatisticsProviderFactory> configure() const = 0;

    };

};
