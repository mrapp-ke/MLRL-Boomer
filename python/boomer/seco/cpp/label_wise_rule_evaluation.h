/**
 * Implements classes for calculating the predictions of rules, as well as corresponding quality scores, based on
 * confusion matrices that have been computed for each label individually.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/rule_evaluation.h"
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
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, such that they optimize a
     * heuristic that is applied using label-wise averaging.
     */
    class HeuristicLabelWiseRuleEvaluationImpl : virtual public ILabelWiseRuleEvaluation {

        private:

            std::shared_ptr<IHeuristic> heuristicPtr_;

            bool predictMajority_;

            const uint32* labelIndices_;

            LabelWiseEvaluatedPrediction prediction_;

        public:

            /**
             * @param numPredictions    The number of labels for which the rules should predict
             * @param labelIndices      A pointer to an array of type `uint32` that stores the indices of the labels
             *                          for which the rules should predict or a null pointer, if the rules should
             *                          predict for all labels
             * @param heuristicPtr      A shared pointer to an object of type `IHeuristic`, representing the heuristic
             *                          to be optimized
             * @param predictMajority   True, if for each label the majority label should be predicted, false, if the
             *                          minority label should be predicted
             */
            HeuristicLabelWiseRuleEvaluationImpl(uint32 numPredictions, const uint32* labelIndices,
                                                 std::shared_ptr<IHeuristic> heuristicPtr, bool predictMajority);

            const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(
                const uint8* minorityLabels, const float64* confusionMatricesTotal,
                const float64* confusionMatricesSubset, const float64* confusionMatricesCovered,
                bool uncovered) override;

    };

    /**
     * Defines an interface for all factories that allow to create instances of the type `ILabelWiseRuleEvaluation`.
     */
    class ILabelWiseRuleEvaluationFactory {

        public:

            virtual ~ILabelWiseRuleEvaluationFactory() { };

            /**
             * Creates and returns a new object of type `ILabelWiseRuleEvaluation` that allows to calculate the
             * predictions of rules for several labels.
             *
             * @param numLabelIndices   The number of labels for which the rules should predict
             * @param labelIndices      A pointer to an array of type `uint32` that stores the indices of the labels for
             *                          which the rules should predict or a null pointer, if the rules should predict
             *                          for all available labels
             * @return                  An unique pointer to an object of type `ILabelWiseRuleEvaluation` that has been
             *                          created
             */
            virtual std::unique_ptr<ILabelWiseRuleEvaluation> create(uint32 numLabelIndices,
                                                                     const uint32* labelIndices) const = 0;

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

            std::unique_ptr<ILabelWiseRuleEvaluation> create(uint32 numLabelIndices,
                                                             const uint32* labelIndices) const override;

    };

}
