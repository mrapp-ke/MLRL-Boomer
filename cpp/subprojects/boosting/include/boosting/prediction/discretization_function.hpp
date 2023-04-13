/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/data/types.hpp"

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that allow to discretize regression scores.
     */
    class IDiscretizationFunction {
        public:

            virtual ~IDiscretizationFunction() {};

            /**
             * Discretizes a given regression score.
             *
             * @param score The regression score to be discretized
             * @return      A binary value the given regression score has been turned into
             */
            virtual bool discretizeScore(float64 score) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IDiscretizationFunction`.
     */
    class IDiscretizationFunctionFactory {
        public:

            virtual ~IDiscretizationFunctionFactory() {};

            /**
             * Creates and returns a new object of the type `IDiscretizationFunction`.
             *
             * @return An unique pointer to an object of type `IDiscretizationFunction` that has been created
             */
            virtual std::unique_ptr<IDiscretizationFunction> create() const = 0;
    };

}
