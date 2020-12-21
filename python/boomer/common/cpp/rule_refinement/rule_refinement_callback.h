/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../statistics/statistics_immutable.h"
#include "../sampling/weight_vector.h"
#include <memory>


/**
 * Defines an interface for callbacks that may be invoked by subclasses of the the class `IRuleRefinement` in order to
 * retrieve the data, consisting of statistics, weights and a vector, that is required to search for potential
 * refinements.
 *
 * @tparam T The type of the vector that is returned by the callback
 */
template<class T>
class IRuleRefinementCallback {

    public:

        /**
         * The data that is provided via the callback's `get` function.
         */
        class Result final {

            public:

                /**
                 * @param statistics        A reference to an object of type `IImmutableStatistics` that should be used
                 *                          to search for potential refinements
                 * @param weights           A reference to an object of type `IWeightVector` that provides access to the
                 *                          weights of individual training examples
                 * @param totalSumOfWeights The total sum of the weights of the examples that are currently covered
                 * @param vector            A reference to an object of template type `T` that should be used to search
                 *                          for potential refinements
                 */
                Result(const IImmutableStatistics& statistics, const IWeightVector& weights, uint32 totalSumOfWeights,
                       const T& vector)
                    : statistics_(statistics), weights_(weights), totalSumOfWeights_(totalSumOfWeights),
                      vector_(vector) {

                }

                const IImmutableStatistics& statistics_;

                const IWeightVector& weights_;

                uint32 totalSumOfWeights_;

                const T& vector_;

        };

        virtual ~IRuleRefinementCallback() { };

        /**
         * Invokes the callback and returns its result.
         *
         * @return An unique pointer to an object of type `Result` that stores references to the statistics and the
         *         vector that may be used to search for potential refinements
         */
        virtual std::unique_ptr<Result> get() = 0;

};
