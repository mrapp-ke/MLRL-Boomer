/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_probability.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that allow to transform the scores that are predicted for individual labels
     * into probabilities.
     */
    class IProbabilityFunction {

        public:

            virtual ~IProbabilityFunction() { };

            /**
             * Transforms the score that is predicted for an individual label into a probability.
             *
             * @param predictedScore    The predicted score
             * @return                  The probability
             */
            virtual float64 transform(float64 predictedScore) const = 0;

    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `IProbabilityFunction`.
     */
    class IProbabilityFunctionFactory {

        public:

            virtual ~IProbabilityFunctionFactory() { };

            /**
             * Creates and returns a new object of the type `IProbabilityFunction`.
             *
             * @return An unique pointer to an object of type `IProbabilityFunction` that has been created
             */
            virtual std::unique_ptr<IProbabilityFunction> create() const = 0;

    };

    /**
     * Allows to create instances of the type `IProbabilityFunction` that transform the score that is predicted for an
     * individual label into a probability by applying the logistic sigmoid function.
     */
    class LogisticFunctionFactory final : public IProbabilityFunctionFactory {

        public:

            std::unique_ptr<IProbabilityFunction> create() const override;

    };

    /**
     * Allows to create instances of the type `IProbabilityPredictor` that allow to predict label-wise probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based models and transforming the aggregated scores
     * into probabilities in [0, 1] according to a certain transformation function that is applied to each label
     * individually.
     */
    class LabelWiseProbabilityPredictorFactory final : public IProbabilityPredictorFactory {

        private:

            std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param probabilityFunctionFactoryPtr An unique pointer to an object of type `IProbabilityFunctionFactory`
             *                                      that allows to create implementations of the transformation function
             *                                      to be used to transform predicted scores into probabilities
             * @param numThreads                    The number of CPU threads to be used to make predictions for
             *                                      different query examples in parallel. Must be at least 1
             */
            LabelWiseProbabilityPredictorFactory(
                    std::unique_ptr<IProbabilityFunctionFactory> probabilityFunctionFactoryPtr, uint32 numThreads);

            std::unique_ptr<IProbabilityPredictor> create(const RuleList& model) const override;

    };

}
