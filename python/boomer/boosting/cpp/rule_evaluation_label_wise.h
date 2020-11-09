/**
 * Implements classes for calculating the predictions of rules, as well as corresponding quality scores, based on the
 * gradients and Hessians that have been calculated according to a loss function that is applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/rule_evaluation.h"
#include "../../common/cpp/data_numeric.h"
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
             * @param sumsOfGradients   A reference to an object of type `DenseFloat64Vector` that stores the sums of
             *                          gradients for individual labels
             * @param sumsOfHessians    A reference to an object of type `DenseFloat64Vector` that stores the sums of
             *                          Hessians for individual labels
             * @return                  A reference to an object of type `LabelWiseEvaluatedPrediction` that stores the
             *                          predicted scores and quality scores
             */
            virtual const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(
                const float64* sumsOfGradients, const float64* sumsOfHessians) = 0;

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
             * @param indexVector   A reference to an object of the type `FullIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<ILabelWiseRuleEvaluation> create(const FullIndexVector& indexVector) const = 0;

            /**
             * Creates a new instance of the class `ILabelWiseRuleEvaluation` that allows to calculate the predictions
             * of rules that predict for a subset of the available labels.
             *
             * @param indexVector   A reference to an object of the type `PartialIndexVector` that provides access to
             *                      the indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<ILabelWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const = 0;


    };

    /**
     * Allows to create instances of the class `RegularizedLabelWiseRuleEvaluation`.
     */
    class RegularizedLabelWiseRuleEvaluationFactoryImpl : public ILabelWiseRuleEvaluationFactory {

        private:

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param l2RegularizationWeight The weight of the L2 regularization that is applied for calculating the
             *                               scores to be predicted by rules
             */
            RegularizedLabelWiseRuleEvaluationFactoryImpl(float64 l2RegularizationWeight);

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const FullIndexVector& indexVector) const override;

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
