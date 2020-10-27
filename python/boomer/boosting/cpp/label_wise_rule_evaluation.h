/**
 * Implements classes for calculating the predictions of rules, as well as corresponding quality scores, based on the
 * gradients and Hessians that have been calculated according to a loss function that is applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/rule_evaluation.h"
#include "../../common/cpp/indices.h"
#include <memory>


namespace boosting {

    /**
     * Defines an interface for all classes that allow to calculate the predictions of rule, as well as corresponding
     * quality scores, based on the gradients and Hessians that have been calculated according to a loss function that
     * is applied label-wise.
     */
    class ILabelWiseRuleEvaluation {

        public:

            virtual ~ILabelWiseRuleEvaluation() { };

            /**
             * Calculates the scores to be predicted by a rule, as well as corresponding quality scores, based on the
             * label-wise sums of gradients and Hessians that are covered by the rule. The predicted scores and quality
             * scores are stored in a given object of type `LabelWiseEvaluatedPrediction`.
             *
             * If the argument `uncovered` is True, the rule is considered to cover the difference between the sums of
             * gradients and Hessians that are stored in the arrays `totalSumsOfGradients` and `sumsOfGradients` and
             * `totalSumsOfHessians` and `sumsOfHessians`, respectively.
             *
             * @param totalSumsOfGradients  A pointer to an array of type `float64`, shape `(num_labels), representing
             *                              the total sums of gradients for individual labels
             * @param sumsOfGradients       A pointer to an array of type `float64`, shape
             *                              `(prediction.numPredictions_)`, representing the sums of gradients for
             *                              individual labels
             * @param totalSumsOfHessians   A pointer to an array of type `float64`, shape `(num_labels)`, representing
             *                              the total sums of Hessians for individual labels
             * @param sumsOfHessians        A pointer to an array of type `float64`, shape
             *                              `(prediction.numPredictions_)`, representing the sums of Hessians for
             *                              individual labels
             * @param uncovered             False, if the rule covers the sums of gradient and Hessians that are stored
             *                              in the array `sumsOfGradients` and `sumsOfHessians`, True, if the rule
             *                              covers the difference between the sums of gradients and Hessians that are
             *                              stored in the arrays `totalSumsOfGradients` and `sumsOfGradients` and
             *                              `totalSumsOfHessians` and `sumsOfHessians`, respectively
             * @return                      A reference to an object of type `LabelWiseEvaluatedPrediction` that stores
             *                              the predicted scores and quality scores
             */
            virtual const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(
                const float64* totalSumsOfGradients, float64* sumsOfGradients, const float64* totalSumsOfHessians,
                float64* sumsOfHessians, bool uncovered) = 0;

    };

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, based on the gradients and
     * Hessians that have been calculated according to a loss function that is applied label-wise using L2
     * regularization.
     *
     * @tparam T The type of the vector that provides access to the labels for which predictions should be calculated
     */
    template<class T>
    class RegularizedLabelWiseRuleEvaluationImpl : virtual public ILabelWiseRuleEvaluation {

        private:

            const T& labelIndices_;

            float64 l2RegularizationWeight_;

            LabelWiseEvaluatedPrediction prediction_;

        public:

            /**
             * @param labelIndices              A reference to an object of template type `T` that provides access to
             *                                  the indices of the labels for which the rules may predict
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            RegularizedLabelWiseRuleEvaluationImpl(const T& labelIndices, float64 l2RegularizationWeight);

            const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(
                const float64* totalSumsOfGradients, float64* sumsOfGradients, const float64* totalSumsOfHessians,
                float64* sumsOfHessians, bool uncovered) override;

    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `ILabelWiseRuleEvaluation`.
     */
    class ILabelWiseRuleEvaluationFactory {

        public:

            virtual ~ILabelWiseRuleEvaluationFactory() { };

            /**
             * Creates a new instance of the class `ILabelWiseRuleEvaluation` that allows to calculate the predictions
             * of rules that predict for all available labels.
             *
             * @param indexVector   A reference to an object of the type `RangeIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<ILabelWiseRuleEvaluation> create(const RangeIndexVector& indexVector) const = 0;

            /**
             * Creates a new instance of the class `ILabelWiseRuleEvaluation` that allows to calculate the predictions
             * of rules that predict for a subset of the available labels.
             *
             * @param indexVector   A reference to an object of the type `DenseIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<ILabelWiseRuleEvaluation> create(const DenseIndexVector& indexVector) const = 0;


    };

    /**
     * Allows to create instances of the class `RegularizedLabelWiseRuleEvaluation`.
     */
    class RegularizedLabelWiseRuleEvaluationFactoryImpl : virtual public ILabelWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param l2RegularizationWeight The weight of the L2 regularization that is applied for calculating the
             *                               scores to be predicted by rules
             */
            RegularizedLabelWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight);

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const RangeIndexVector& indexVector) const override;

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const DenseIndexVector& indexVector) const override;

    };

}
