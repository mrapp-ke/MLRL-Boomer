/**
 * Provides classes that allow to store the elements of confusion matrices that are computed independently for each
 * label.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"
#include "label_wise_rule_evaluation.h"


namespace statistics {

    /**
     * Allows to search for the best refinement of a rule based on the confusion matrices previously stored by an object
     * of type `LabelWiseStatisticsImpl`.
     */
    class LabelWiseRefinementSearchImpl {

        private:

            rule_evaluation::LabelWiseRuleEvaluationImpl* ruleEvaluation_;

            intp numLabels_;

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
             * @param numLabels                 The number of labels to be considered by the search
             * @param labelIndices              An array of type `intp`, shape `(numLabels)`, representing the indices
             *                                  of the labels that should be considered by the search or NULL, if all
             *                                  labels should be considered
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
                                          intp numLabels, const intp* labelIndices, AbstractLabelMatrix* labelMatrix,
                                          const float64* uncoveredLabels, const uint8* minorityLabels,
                                          const float64* confusionMatricesTotal,
                                          const float64* confusionMatricesSubset);

            ~LabelWiseRefinementSearchImpl();

            /**
             * Notifies the search that a specific statistic is covered by the condition that is currently considered
             * for refining a rule.
             *
             * This function must be called repeatedly for each statistic that is covered by the current condition,
             * immediately after the invocation of the function `Statistics#beginSearch`. Each of these statistics must
             * have been provided earlier via the function `Statistics#addSampledStatistic` or
             * `Statistics#updateCoveredStatistic`.
             *
             * This function is supposed to update any internal state of the search that relates to the examples that
             * are covered by the current condition, i.e., to compute and store local information that is required by
             * the other functions that will be called later. Any information computed by this function is expected to
             * be reset when invoking the function `resetSearch` for the next time.
             *
             * @param statistic_index   The index of the covered statistic
             * @param weight            The weight of the covered statistic
             */
            void updateSearch(intp statisticIndex, uint32 weight);

            /**
             * Resets the internal state of the search that has been updated via preceding calls to the function
             * `updateSearch` to the state when the search was started via the function `Statistics#beginSearch`. When
             * calling this function, the current state must not be purged entirely, but it must be cached and made
             * available for use by the functions `calculateExampleWisePrediction` and `calculateLabelWisePrediction`
             * (if the function argument `accumulated` is set accordingly).
             *
             * This function may be invoked multiple times with one or several calls to `updateSearch` in between, which
             * is supposed to update the previously cached state by accumulating the current one, i.e., the accumulated
             * cached state should be the same as if `resetSearch` would not have been called at all.
             */
            void resetSearch();

            /**
             * Calculates and returns the scores to be predicted by a rule that covers all statistics that have been
             * provided to the search so far via the function `updateSearch`.
             *
             * If the argument `uncovered` is 1, the rule is considered to cover all statistics that belong to the
             * difference between the statistics that have been provided via the function `addSampledStatistic` or
             * `updateCoveredStatistic` and the statistics that have been provided via the function `updateSearch`.
             *
             * If the argument `accumulated` is 1, all statistics that have been provided since the search has been
             * started via the function `Statistics#beginSearch` are taken into account even if the function
             * `resetSearch` has been called since then. If said function has not been invoked, this argument does not
             * have any effect.
             *
             * The calculated scores correspond to the subset of labels that have been provided when starting the search
             * via the function `Statistics#beginSearch`. The score to be predicted for an individual label is
             * calculated independently of the other labels, i.e., in the non-decomposable case it is assumed that the
             * rule will not predict for any other labels. In addition to each score, a quality score, which assesses
             * the quality of the prediction for the respective label, is returned.
             *
             * @param uncovered     0, if the rule covers all statistics that have been provided via the function
             *                      `updateSearch`, 1, if the rule covers all examples that belong to the difference
             *                      between the statistics that have been provided via the function
             *                      `Statistics#addSampledStatistic` or `Statistics#updateCoveredStatistic` and the
             *                      statistics that have been provided via the function `updateSearch`
             * @param accumulated   0, if the rule covers all statistics that have been provided via the function
             *                      `updateSearch` since the function `resetSearch` has been called for the last time,
             *                      1, if the rule covers all examples that have been provided since the search has been
             *                      started via the function `Statistics#beginSearch`
             * @return              A pointer to an object of type `LabelWisePrediction` that stores the scores to be
             *                      predicted by the rule for each considered label, as well as the corresponding
             *                      quality scores
             */
            rule_evaluation::LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated);

    };

}
