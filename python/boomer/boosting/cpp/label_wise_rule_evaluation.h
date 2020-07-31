/**
 * Implements classes for calculating the predictions of rules, as well as corresponding quality scores, such that they
 * minimize a loss function that is applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/rule_evaluation.h"
#include "label_wise_losses.h"


namespace boosting {

    /**
     * Allows to calculate the predictions of a default rule such that it minimizes a loss function that is applied
     * label-wise.
     */
    class LabelWiseDefaultRuleEvaluationImpl : public AbstractDefaultRuleEvaluation {

        private:

            losses::AbstractLabelWiseLoss* lossFunction_;

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param lossFunction              A pointer to an object of type `AbstractLabelWiseLoss`, representing the
                                                loss function to be minimized by the default rule
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by the default rule
             */
            LabelWiseDefaultRuleEvaluationImpl(losses::AbstractLabelWiseLoss* lossFunction,
                                               float64 l2RegularizationWeight);

            ~LabelWiseDefaultRuleEvaluationImpl();

            DefaultPrediction* calculateDefaultPrediction(input::AbstractLabelMatrix* labelMatrix) override;

    };

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, such that they minimize a
     * loss function that is applied label-wise.
     */
    class LabelWiseRuleEvaluationImpl {

        private:

            float64 l2RegularizationWeight_;

        public:

            /**
             * @param l2RegularizationWeight    The weight of the L2 regularization that is applied for calculating the
             *                                  scores to be predicted by rules
             */
            LabelWiseRuleEvaluationImpl(float64 l2RegularizationWeight);

            /**
             * Calculates the scores to be predicted by a rule, as well as corresponding quality scores, based on the
             * label-wise sums of gradients and Hessians that are covered by the rule. The predicted scores and quality
             * scores are stored in a given object of type `LabelWisePrediction`.
             *
             * If the argument `uncovered` is True, the rule is considered to cover the difference between the sums of
             * gradients and Hessians that are stored in the arrays `totalSumsOfGradients` and `sumsOfGradients` and
             * `totalSumsOfHessians` and `sumsOfHessians`, respectively.
             *
             * @param labelIndices          A pointer to an array of type `intp`, shape `(prediction.numPredictions_)`,
             *                              representing the indices of the labels for which the rule should predict or
             *                              NULL, if the rule should predict for all labels
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
             * @param prediction            A pointer to an object of type `LabelWisePrediction` that should be used to
             *                              store the predicted scores and quality scores
             */
            void calculateLabelWisePrediction(const intp* labelIndices, const float64* totalSumsOfGradients,
                                              const float64* sumsOfGradients, const float64* totalSumsOfHessians,
                                              const float64* sumsOfHessians, bool uncovered,
                                              LabelWisePrediction* prediction);

    };

}
