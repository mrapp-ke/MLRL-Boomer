/**
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "head_refinement.h"


/**
 * Defines an interface for all factories that allow to create instances of the type `IHeadRefinement`.
 */
class IHeadRefinementFactory {

    public:

        virtual ~IHeadRefinementFactory() { };

        /**
         * Creates and returns a new object of type `IHeadRefinement` that allows to find the best head.
         *
         * @return An unique pointer to an object of type `IHeadRefinement` that has been created
         */
        virtual std::unique_ptr<IHeadRefinement> create() const = 0;

};
