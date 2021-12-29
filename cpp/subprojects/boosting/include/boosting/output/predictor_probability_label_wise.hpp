/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "common/output/predictor_probability.hpp"


namespace boosting {

    /**
     * Defines an interface for all classes that implement a transformation function that is applied to the scores that
     * are predicted for individual labels.
     */
    class ILabelWiseTransformationFunction {

        public:

            virtual ~ILabelWiseTransformationFunction() { };

            /**
             * Transforms the score that is predicted for an individual label.
             *
             * @param predictedScore    The predicted score
             * @return                  The result of the transformation
             */
            virtual float64 transform(float64 predictedScore) const = 0;

    };

    /**
     * Allows to transform the score that is predicted for an individual label into a probability by applying the
     * logistic sigmoid function.
     */
    class LogisticFunction : public ILabelWiseTransformationFunction {

        public:

            float64 transform(float64 predictedScore) const override;

    };

    /**
     * An implementation of the type `ILabelWiseProbabilityPredictor` that allows to predict label-wise probabilities
     * for given query examples, which estimate the chance of individual labels to be relevant, by summing up the scores
     * that are provided by individual rules of an existing rule-based models and transforming the aggregated scores
     * into probabilities in [0, 1] according to a certain transformation function that is applied to each label
     * individually.
     */
    class LabelWiseProbabilityPredictor final : public IProbabilityPredictor {

        private:

            std::unique_ptr<ILabelWiseTransformationFunction> transformationFunctionPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param transformationFunctionPtr An unique pointer to an object of type
             *                                  `ILabelWiseTransformationFunction` that should be used to transform
             *                                  predicted scores into probabilities
             * @param numThreads                The number of CPU threads to be used to make predictions for different
             *                                  query examples in parallel. Must be at least 1
             */
            LabelWiseProbabilityPredictor(std::unique_ptr<ILabelWiseTransformationFunction> transformationFunctionPtr,
                                          uint32 numThreads);

            void predict(const CContiguousFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const RuleModel& model, const LabelVectorSet* labelVectors) const override;

            void predict(const CsrFeatureMatrix& featureMatrix, CContiguousView<float64>& predictionMatrix,
                         const RuleModel& model, const LabelVectorSet* labelVectors) const override;

    };

}
