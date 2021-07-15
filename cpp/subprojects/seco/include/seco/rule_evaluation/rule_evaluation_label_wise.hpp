/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/indices/index_vector_complete.hpp"
#include "common/indices/index_vector_partial.hpp"
#include "common/rule_evaluation/score_vector_label_wise.hpp"
#include "seco/data/vector_dense_confusion_matrices.hpp"
#include <memory>


namespace seco {

    /**
     * Defines an interface for all classes that allow to calculate the predictions of rules, as well as corresponding
     * quality scores, based on confusion matrices that have been computed for each label individually.
     */
    // TODO Rename to `IRuleEvaluation`
    class ILabelWiseRuleEvaluation {

        public:

            virtual ~ILabelWiseRuleEvaluation() { };

            /**
             * Calculates the scores to be predicted by a rule, as well as corresponding quality scores, based on
             * label-wise confusion matrices.
             *
             * @param majorityLabelVector       A reference to an object of type `DenseVector` that stores the
             *                                  predictions of the default rule
             * @param confusionMatricesTotal    A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores confusion matrices that take into account all examples
             * @param confusionMatricesSubset   A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores confusion matrices that take into account all all examples, which
             *                                  are covered by the previous refinement of the rule
             * @param confusionMatricesCovered  A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores a confusion matrices that take into account all examples, which
             *                                  are covered by the rule
             * @param uncovered                 False, if the confusion matrices in `confusion_matrices_covered`
             *                                  correspond to the examples that are covered by rule, true, if they
             *                                  correspond to the examples that are not covered by the rule
             * @return                          A reference to an object of type `ILabelWiseScoreVector` that stores the
             *                                  predicted scores and quality scores
             */
            // TODO Remove
            virtual const ILabelWiseScoreVector& calculateLabelWisePrediction(
                const BinarySparseArrayVector& majorityLabelVector,
                const DenseConfusionMatrixVector& confusionMatricesTotal,
                const DenseConfusionMatrixVector& confusionMatricesSubset,
                const DenseConfusionMatrixVector& confusionMatricesCovered, bool uncovered) = 0;

            /**
             * Calculates the scores to be predicted by a rule, as well as an overall quality score, based on label-wise
             * confusion matrices.
             *
             * @param majorityLabelVector       A reference to an object of type `DenseVector` that stores the
             *                                  predictions of the default rule
             * @param confusionMatricesTotal    A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores confusion matrices that take into account all examples
             * @param confusionMatricesSubset   A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores confusion matrices that take into account all all examples, which
             *                                  are covered by the previous refinement of the rule
             * @param confusionMatricesCovered  A reference to an object of type `DenseConfusionMatrixVector` that
             *                                  stores a confusion matrices that take into account all examples, which
             *                                  are covered by the rule
             * @param uncovered                 False, if the confusion matrices in `confusion_matrices_covered`
             *                                  correspond to the examples that are covered by rule, true, if they
             *                                  correspond to the examples that are not covered by the rule
             * @return                          A reference to an object of type `IScoreVector` that stores the
             *                                  predicted scores, as well as an overall quality score
             */
            virtual const IScoreVector& calculatePrediction(const BinarySparseArrayVector& majorityLabelVector,
                                                            const DenseConfusionMatrixVector& confusionMatricesTotal,
                                                            const DenseConfusionMatrixVector& confusionMatricesSubset,
                                                            const DenseConfusionMatrixVector& confusionMatricesCovered,
                                                            bool uncovered) = 0;

    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `ILabelWiseRuleEvaluation`.
     */
    class ILabelWiseRuleEvaluationFactory {

        public:

            virtual ~ILabelWiseRuleEvaluationFactory() { };

            /**
             * Creates and returns a new object of type `ILabelWiseRuleEvaluation` that allows to calculate the
             * predictions of rules that predict for all available labels.
             *
             * @param indexVector   A reference to an object of type `CompleteIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<ILabelWiseRuleEvaluation> create(const CompleteIndexVector& indexVector) const = 0;

            /**
             * Creates and returns a new object of type `ILabelWiseRuleEvaluation` that allows to calculate the
             * predictions of rules that predict for a subset of the available labels.
             *
             * @param indexVector   A reference to an object of type `PartialIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<ILabelWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const = 0;

    };

}
