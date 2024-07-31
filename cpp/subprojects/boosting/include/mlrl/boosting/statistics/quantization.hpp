/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include <memory>

namespace boosting {

    /**
     * Defines an interface for all classes that implement a method for quantizing statistics about the quality of
     * predictions for training examples.
     */
    class IQuantization {
        public:

            virtual ~IQuantization() {}
    };

    /**
     * Defines an interface for all factories that allows to create instances of the type `IQuantization`.
     */
    class IQuantizationFactory {
        public:

            virtual ~IQuantizationFactory() {}

            /**
             * Creates and returns a new object of type `IQuantization`.
             *
             * @return An unique pointer to an object of type `IQuantization` that has been created
             */
            virtual std::unique_ptr<IQuantization> create() const = 0;
    };

    /**
     * Defines an interface for all classes that allow to configure a method for quantizing statistics about the quality
     * of predictions for training examples.
     */
    class IQuantizationConfig {
        public:

            virtual ~IQuantizationConfig() {}

            /**
             * Creates and returns a new object of type `IQuantizationFactory` according to the specified configuration.
             *
             * @return An unique pointer to an object of type `IQuantizationFactory` that has been created
             */
            virtual std::unique_ptr<IQuantizationFactory> createQuantizationFactory() const = 0;
    };

}
