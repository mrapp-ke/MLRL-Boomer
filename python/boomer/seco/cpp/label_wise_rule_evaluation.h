/**
 * Implements classes for calculating the predictions of rules, as well as corresponding quality scores, based on
 * confusion matrices that have been computed for each label individually.
 *
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/rule_evaluation.h"
#include "../../common/cpp/indices.h"
#include "heuristics.h"
#include <memory>


namespace seco {

    /**
     * Defines an interface for all classes that allow to calculate the predictions of rules, as well as corresponding
     * quality scores, based on confusion matrices that have been computed for each label individually.
     */
    class ILabelWiseRuleEvaluation {

        public:

            virtual ~ILabelWiseRuleEvaluation() { };

            /**
             * Calculates the scores to be predicted by a rule, as well as corresponding quality scores, based on
             * confusion matrices. The predicted scores and quality scores are stored in a given object of type
             * `LabelWiseEvaluatedPrediction`.
             *
             * @param minorityLabels            A pointer to an array of type `uint8`, shape `(num_labels)`, indicating
             *                                  whether the rule should predict individual labels as positive (1) or
             *                                  negative (0)
             * @param confusionMatricesTotal    A pointer to a C-contiguous array of type `float64`, shape
             *                                  `(num_labels, NUM_CONFUSION_MATRIX_ELEMENTS)`, storing a confusion
             *                                  matrix that takes into account all examples for each label
             * @param confusionMatricesSubset   A pointer to a C-contiguous array of type `float64`, shape
             *                                  `(num_labels, NUM_CONFUSION_MATRIX_ELEMENTS)`, storing a confusion
             *                                  matrix that takes into account all all examples, which are covered by
             *                                  the previous refinement of the rule, for each label
             * @param confusionMatricesCovered  A pointer to a C-contiguous array of type `float64`, shape
             *                                  `(prediction.numPredictions_, NUM_CONFUSION_MATRIX_ELEMENTS)`, storing a
             *                                  confusion matrix that takes into account all examples, which are covered
             *                                  by the rule, for each label
             * @param uncovered                 False, if the confusion matrices in `confusion_matrices_covered`
             *                                  correspond to the examples that are covered by rule, True, if they
             *                                  correspond to the examples that are not covered by the rule
             * @param return                    A reference to an object of type `LabelWiseEvaluatedPrediction` that
             *                                  stores the predicted scores and quality scores
             */
            virtual const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(
                const uint8* minorityLabels, const float64* confusionMatricesTotal,
                const float64* confusionMatricesSubset, const float64* confusionMatricesCovered, bool uncovered) = 0;

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
             * @param indexVector   A reference to an object of type `FullIndexVector` that provides access to the
             *                      indices of the labels for which the rules may predict
             * @return              An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                      created
             */
            virtual std::unique_ptr<ILabelWiseRuleEvaluation> create(const FullIndexVector& indexVector) const = 0;

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

    /**
     * Allows to create instances of the class `RegularizedLabelWiseRuleEvaluation`.
     */
    class HeuristicLabelWiseRuleEvaluationFactoryImpl : virtual public ILabelWiseRuleEvaluationFactory {

        private:

            std::shared_ptr<IHeuristic> heuristicPtr_;

            bool predictMajority_;

        public:

            /**
             * @param heuristicPtr      A shared pointer to an object of type `IHeuristic`, representing the heuristic
             *                          to be optimized
             * @param predictMajority   True, if for each label the majority label should be predicted, false, if the
             *                          minority label should be predicted
             */
            HeuristicLabelWiseRuleEvaluationFactoryImpl(std::shared_ptr<IHeuristic> heuristicPtr, bool predictMajority);

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const FullIndexVector& indexVector) const override;

            std::unique_ptr<ILabelWiseRuleEvaluation> create(const PartialIndexVector& indexVector) const override;

    };

}
