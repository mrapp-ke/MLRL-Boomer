/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../statistics/histogram.h"
#include <memory>
#include <utility>


/**
 * Defines an interface for callbacks that may be invoked by subclasses of the the class `IRuleRefinement` in order to
 * retrieve the data, consisting of statistics and a vector, that is required to search for potential refinements.
 *
 * @tparam T The type of the vector that is returned by the callback
 */
template<class T>
class IRuleRefinementCallback {

    public:

        virtual ~IRuleRefinementCallback() { };

        typedef std::pair<const IHistogram&, const T&> Result;

        /**
         * Invokes the callback and returns its result.
         *
         * @return An unique pointer to an object of type `Result` that stores references to the statistics and the
         *         vector that may be used to search for potential refinements
         */
        virtual std::unique_ptr<Result> get() = 0;

};

