/**
 * Provides classes that allow to store the elements of confusion matrices that are computed independently for each
 * label.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/input_data.h"
#include "../../common/cpp/statistics.h"
#include "label_wise_rule_evaluation.h"


namespace statistics {

    /**
     * Allows to search for the best refinement of a rule based on the confusion matrices previously stored by an object
     * of type `LabelWiseStatisticsImpl`.
     */
    class LabelWiseRefinementSearchImpl : public AbstractDecomposableRefinementSearch {

        private:

            rule_evaluation::LabelWiseRuleEvaluationImpl* ruleEvaluation_;

            intp numPredictions_;

            const intp* labelIndices_;

            AbstractLabelMatrix* labelMatrix_;

            const float64* uncoveredLabels_;

            const uint8* minorityLabels_;

            const float64* confusionMatricesTotal_;

            const float64* confusionMatricesSubset_;

            float64* confusionMatricesCovered_;

            float64* accumulatedConfusionMatricesCovered_;

            rule_evaluation::LabelWisePrediction* prediction_;

        public:

            /**
             * @param ruleEvaluation            A pointer to an object of type `LabelWiseRuleEvaluationImpl` to be used
             *                                  for calculating the predictions, as well as corresponding quality
             *                                  scores, of rules
             * @param numPredictions            The number of labels to be considered by the search
             * @param labelIndices              An array of type `intp`, shape `(numPredictions)`, representing the
             *                                  indices of the labels that should be considered by the search or NULL,
             *                                  if all labels should be considered
             * @param labelMatrix               A pointer to an object of type `AbstractLabelMatrix` that provides
             *                                  random access to the labels of the training examples
             * @param uncoveredLabels           A pointer to an array of type `float64`, shape
             *                                  `(numExamples, numLabels)`, indicating which examples and labels remain
             *                                  to be covered
             * @param minorityLabels            A pointer to an array of type `uint8`, shape `(numLabels)`, indicating
             *                                  whether rules should predict individual labels as relevant (1) or
             *                                  irrelevant (0)
             * @param confusionMatricesTotal    A pointer to a C-contiguous array of type `float64`, shape
             *                                  `(num_labels, 4)`, storing a confusion matrix that takes into account
             *                                  all examples for each label
             * @param confusionMatricesSubset   A pointer to a C-contiguous array of type `float64`, shape
             *                                  `(num_labels, 4)`, storing a confusion matrix that takes into account
             *                                  all all examples, which are covered by the previous refinement of the
             *                                  rule, for each label
             */
            LabelWiseRefinementSearchImpl(rule_evaluation::LabelWiseRuleEvaluationImpl* ruleEvaluation,
                                          intp numPredictions, const intp* labelIndices,
                                          AbstractLabelMatrix* labelMatrix, const float64* uncoveredLabels,
                                          const uint8* minorityLabels, const float64* confusionMatricesTotal,
                                          const float64* confusionMatricesSubset);

            ~LabelWiseRefinementSearchImpl();

            void updateSearch(intp statisticIndex, uint32 weight) override;

            void resetSearch() override;

            rule_evaluation::LabelWisePrediction* calculateLabelWisePrediction(bool uncovered,
                                                                               bool accumulated) override;

    };

}
