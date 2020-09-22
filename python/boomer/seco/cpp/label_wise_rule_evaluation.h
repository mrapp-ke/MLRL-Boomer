/**
 * Implements classes for calculating the predictions of rules, as well as corresponding quality scores, based on
 * confusion matrices that have been computed for each label individually.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/predictions.h"
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
             * `LabelWisePredictionCandidate`.
             *
             * @param labelIndices              A pointer to an array of type `uint32`, shape
             *                                  `(prediction.numPredictions_)`, representing the indices of the labels
             *                                  for which the rule should predict or NULL, if the rule should predict
             *                                  for all labels
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
             * @param prediction                A pointer to an object of type `LabelWisePredictionCandidate` that
             *                                  should be used to store the predicted scores and quality scores
             */
            virtual void calculateLabelWisePrediction(const uint32* labelIndices, const uint8* minorityLabels,
                                                      const float64* confusionMatricesTotal,
                                                      const float64* confusionMatricesSubset,
                                                      const float64* confusionMatricesCovered, bool uncovered,
                                                      LabelWisePredictionCandidate* prediction) = 0;

    };

    /**
     * Allows to calculate the predictions of rules, as well as corresponding quality scores, such that they optimize a
     * heuristic that is applied using label-wise averaging.
     */
    class HeuristicLabelWiseRuleEvaluationImpl : virtual public ILabelWiseRuleEvaluation {

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
            HeuristicLabelWiseRuleEvaluationImpl(std::shared_ptr<IHeuristic> heuristicPtr, bool predictMajority);

            void calculateLabelWisePrediction(const uint32* labelIndices, const uint8* minorityLabels,
                                              const float64* confusionMatricesTotal,
                                              const float64* confusionMatricesSubset,
                                              const float64* confusionMatricesCovered, bool uncovered,
                                              LabelWisePredictionCandidate* prediction) override;

    };

}
