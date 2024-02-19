/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/feature_vector.hpp"
#include "mlrl/common/statistics/statistics_weighted_immutable.hpp"

/**
 * Defines an interface for callbacks that may be invoked by subclasses of the the class `IRuleRefinement` in order to
 * retrieve the information that is required to search for potential refinements. It consists of
 * `IImmutableWeightedStatistics`, as well as an `IFeatureVector` that allows to determine the thresholds that may be
 * used by potential conditions.
 */
class IRuleRefinementCallback {
    public:

        /**
         * The data that is provided via the callback's `get` function.
         */
        struct Result final {
            public:

                /**
                 * @param statistics    A reference to an object of type `IImmutableWeightedStatistics` that should be
                 *                      used to search for potential refinements
                 * @param featureVector A reference to an object of type `IFeatureVector` that should be used to search
                 *                      for potential refinements
                 */
                Result(const IImmutableWeightedStatistics& statistics, const IFeatureVector& featureVector)
                    : statistics(statistics), featureVector(featureVector) {}

                /**
                 * A reference to an object of type `IImmutableWeightedStatistics` that should be used to search for
                 * potential refinements.
                 */
                const IImmutableWeightedStatistics& statistics;

                /**
                 * A reference to an object of type `IFeatureVector` that should be used to search for potential
                 * refinements.
                 */
                const IFeatureVector& featureVector;
        };

        virtual ~IRuleRefinementCallback() {}

        /**
         * Invokes the callback and returns its result.
         *
         * @return An object of type `Result` that stores references to the statistics and the feature vector that may
         *         be used to search for potential refinements
         */
        virtual Result get() = 0;
};
