#pragma once

#include <memory>


/**
 * Defines an interface for all classes that allow to keep track of the examples that are covered by a rule.
 */
class ICoverageState {

    public:

        virtual ~ICoverageState() { };

        /**
         * Creates and returns a deep copy of the coverage state.
         *
         * @return An unique pointer to an object of type `ICoverageState` that has been created
         */
        virtual std::unique_ptr<ICoverageState> copy() const = 0;

};
