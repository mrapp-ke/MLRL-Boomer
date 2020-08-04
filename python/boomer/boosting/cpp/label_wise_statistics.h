/**
 * Provides classes that allow to store gradients and Hessians that are calculated according to a (decomposable) loss
 * function that is applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"
#include "label_wise_rule_evaluation.h"


namespace boosting {

    /**
     * Allows to search for the best refinement of a rule based on the gradients and Hessians previously stored by an
     * object of type `LabelWiseStatisticsImpl`.
     */
    class LabelWiseRefinementSearchImpl : public AbstractDecomposableRefinementSearch {

        private:

            LabelWiseRuleEvaluationImpl* ruleEvaluation_;

            intp numPredictions_;

            const intp* labelIndices_;

            const float64* gradients_;

            const float64* totalSumsOfGradients_;

            const float64* hessians_;

            const float64* totalSumsOfHessians_;

            LabelWisePrediction* prediction_;

        public:

            /**
             * @param ruleEvaluation        A pointer to an object of type `LabelWiseRuleEvaluationImpl` to be used for
             *                              calculating the predictions, as well as corresponding quality scores of
             *                              rules
             * @param numPredictions        The number of labels to be considered by the search
             * @param labelIndices          A pointer to an array of type `intp`, shape `(numPredictions)`, representing
             *                              the indices of the labels that should be considered by the search or NULL,
             *                              if all labels should be considered
             * @param gradients             a pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                              representing the gradient for each example and label
             * @param totalSumsOfGradients  A pointer to an array of type `float64`, shape `(num_labels)`, representing
             *                              the sum of the gradients of all examples, which should be considered by the
             *                              search, for each label
             * @param hessians              A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                              representing the Hessian for each example and label
             * @param totalSumsOfHessians   A pointer to an array of type `float64`, shape `(num_labels)`, representing
             *                              the sum of the Hessians of all examples, which should be considered by the
             *                              search, for each label
             */
            LabelWiseRefinementSearchImpl(LabelWiseRuleEvaluationImpl* ruleEvaluation, intp numPredictions,
                                          const intp* labelIndices, const float64* gradients,
                                          const float64* totalSumsOfGradients, const float64* hessians,
                                          const float64* totalSumsOfHessians);

            ~LabelWiseRefinementSearchImpl();

            void updateSearch(intp statisticIndex, uint32 weight) override;

            void resetSearch() override;

            LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated) override;

    };

}
