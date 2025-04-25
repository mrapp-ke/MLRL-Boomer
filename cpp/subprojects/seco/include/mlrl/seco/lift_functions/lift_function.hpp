/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/input/label_matrix_row_wise.hpp"

#include <memory>

namespace seco {

    /**
     * An abstract base class for all lift functions that affect the quality of rules, depending on the number of labels
     * for which they predict.
     */
    class ILiftFunction {
        public:

            virtual ~ILiftFunction() {}

            /**
             * Calculates and returns the lift for a specific number of labels.
             *
             * @param numLabels The number of labels for which the lift should be calculated
             * @return          The lift that has been calculated
             */
            virtual float32 calculateLift(uint32 numLabels) const = 0;

            /**
             * Returns the maximum lift that is possible by adding additional labels to a head of a given size.
             *
             * @param numLabels The number of labels for which the current head predicts
             * @return          The maximum lift that is possible by adding additional labels to a head of the given
             *                  size
             */
            virtual float32 getMaxLift(uint32 numLabels) const = 0;
    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `ILiftFunction`.
     */
    class ILiftFunctionFactory {
        public:

            virtual ~ILiftFunctionFactory() {}

            /**
             * Creates and returns a new object of type `ILiftFunction`.
             *
             * @return An unique pointer to an object of type `ILiftFunction` that has been created
             */
            virtual std::unique_ptr<ILiftFunction> create() const = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a lift function.
     */
    class ILiftFunctionConfig {
        public:

            virtual ~ILiftFunctionConfig() {}

            /**
             * Creates and returns a new object of type `ILiftFunctionFactory` according to the specified configuration.
             *
             * @param labelMatrix   A reference to an object of type `IRowWiseLabelMatrix` that provides row-wise access
             *                      to the labels of the training examples
             * @return              An unique pointer to an object of type `ILiftFunctionFactory` that has been created
             */
            virtual std::unique_ptr<ILiftFunctionFactory> createLiftFunctionFactory(
              const IRowWiseLabelMatrix& labelMatrix) const = 0;
    };

}
