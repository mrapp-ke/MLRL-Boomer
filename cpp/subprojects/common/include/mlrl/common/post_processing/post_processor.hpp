/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/common/data/view.hpp"

#include <memory>

/**
 * Defines an interface for all classes that allow to post-process the predictions of rules once they have been learned.
 */
class IPostProcessor {
    public:

        virtual ~IPostProcessor() {}

        /**
         * Post-processes the prediction, represented by 32-bit floating point values, of a rule.
         *
         * @param begin An iterator to the beginning of the predictions
         * @param end   An iterator to the end of the predictions
         */
        virtual void postProcess(View<float32>::iterator begin, View<float32>::iterator end) const = 0;

        /**
         * Post-processes the prediction, represented by 64-bit floating point values, of a rule.
         *
         * @param begin An iterator to the beginning of the predictions
         * @param end   An iterator to the end of the predictions
         */
        virtual void postProcess(View<float64>::iterator begin, View<float64>::iterator end) const = 0;
};

/**
 * Defines an interface for all factories that allow to create instances of the type `IPostProcessor`.
 */
class IPostProcessorFactory {
    public:

        virtual ~IPostProcessorFactory() {}

        /**
         * Creates and returns a new object of type `IPostProcessor`.
         *
         * @return An unique pointer to an object of type `IPostProcessor` that has been created
         */
        virtual std::unique_ptr<IPostProcessor> create() const = 0;
};

/**
 * Defines an interface for all classes that allow to configure a method that post-processes the predictions of rules
 * once they have been learned.
 */
class IPostProcessorConfig {
    public:

        virtual ~IPostProcessorConfig() {}

        /**
         * Creates and returns a new object of type `IPostProcessorFactory` according to the specified configuration.
         *
         * @return An unique pointer to an object of type `IPostProcessorFactory` that has been created
         */
        virtual std::unique_ptr<IPostProcessorFactory> createPostProcessorFactory() const = 0;
};
