/**
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../../common/cpp/rule_evaluation/score_vector_label_wise.h"


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
             * confusion matrices.
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
             * @param return                    A reference to an object of type `ILabelWiseScoreVector` that stores the
             *                                  predicted scores and quality scores
             */
            virtual const ILabelWiseScoreVector& calculateLabelWisePrediction(const uint8* minorityLabels,
                                                                              const float64* confusionMatricesTotal,
                                                                              const float64* confusionMatricesSubset,
                                                                              const float64* confusionMatricesCovered,
                                                                              bool uncovered) = 0;

    };

}
