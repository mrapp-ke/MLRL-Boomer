/**
 * Provides classes that allow to store gradients and Hessians that are calculated according to a (non-decomposable)
 * loss function that is applied example-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"
#include "example_wise_rule_evaluation.h"


namespace boosting {

    /**
     * Allows to search for the best refinement of a rule based on the gradients and Hessians previously stored by an
    `* object of type `ExampleWiseStatisticsImpl`.
     */
    class ExampleWiseRefinementSearchImpl : public AbstractRefinementSearch {

        private:

            ExampleWiseRuleEvaluationImpl* ruleEvaluation_;

            intp numPredictions_;

            const intp* labelIndices_;

            intp numLabels_;

            const float64* gradients_;

            const float64* totalSumsOfGradients_;

            float64* sumsOfGradients_;

            float64* accumulatedSumsOfGradients_;

            const float64* hessians_;

            const float64* totalSumsOfHessians_;

            float64* sumsOfHessians_;

            float64* accumulatedSumsOfHessians_;

            LabelWisePrediction* prediction_;

        public:

            /**
             * @param ruleEvaluation        A pointer to an object of type `ExampleWiseRuleEvaluationImpl` to be used
             *                              for calculating the predictions, as well as corresponding quality scores of
             *                              rules
             * @param numPredictions        The number of labels to be considered by the search
             * @param labelIndices          A pointer to an array of type `intp`, shape `(numPredictions)`, representing
             *                              the indices of the labels that should be considered by the search or NULL,
             *                              if all labels should be considered
             * @param numLabels             The total number of labels
             * @param gradients             A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                              representing the gradients for each example
             * @param totalSumsOfGradients  A pointer to an array of type `float64`, shape `(num_labels)`, representing
             *                              the sum of the gradients of all examples, which should be considered by the
             *                              search
             * @param hessians              A pointer to an array of type `float64`, shape
             *                              `(num_examples, (num_labels * (num_labels + 1)) / 2)`, representing the
             *                              Hessians for each example
             * @param totalSumsOfHessians   A pointer to an array of type `float64`, shape
             *                              `((num_labels * (num_labels + 1)) / 2)`, representing the sum of the
             *                              Hessians of all examples, which should be considered by the
             *                              search
             */
            ExampleWiseRefinementSearchImpl(ExampleWiseRuleEvaluationImpl* ruleEvaluation, intp numPredictions,
                                            const intp* labelIndices, intp numLabels, const float64* gradients,
                                            const float64* totalSumsOfGradients, const float64* hessians,
                                            const float64* totalSumsOfHessians);

            ~ExampleWiseRefinementSearchImpl();

            void updateSearch(intp statisticIndex, uint32 weight) override;

            void resetSearch() override;

            LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated) override;

            Prediction* calculateExampleWisePrediction(bool uncovered, bool accumulated) override;

    };

}
