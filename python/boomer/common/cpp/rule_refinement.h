/**
 * Implements classes that allow to find the best refinement of rules.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "predictions.h"
#include "rules.h"
#include <memory>


/**
 * Stores information about a potential refinement of a rule.
 */
class Refinement : public Condition {

    public:

        /**
         * Returns whether this refinement is better than another one.
         *
         * @param   A reference to an object of type `Refinement` to be compared to
         * @return  True, if this refinement is better than the given one, false otherwise
         */
        bool isBetterThan(const Refinement& another) const;

        std::unique_ptr<AbstractEvaluatedPrediction> headPtr;

        intp previous;

};

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

/**
 * Defines an interface for all classes that allow to find the best refinement of existing rules.
 */
class IRuleRefinement {

    public:

        virtual ~IRuleRefinement() { };

        /**
         * Finds the best refinement of an existing rule.
         *
         * @param currentHead A pointer to an object of type `AbstractEvaluatedPrediction`, representing the head of the
         *                    existing rule or a null pointer, if no rule exists yet
         */
        virtual void findRefinement(const AbstractEvaluatedPrediction* currentHead) = 0;

        /**
         * Returns the best refinement that has been found by the function `findRefinement`.
         *
         * @return An unique pointer to an object of type `Refinement` that stores information about the best refinement
         *         that has been found
         */
        virtual std::unique_ptr<Refinement> pollRefinement() = 0;

};
