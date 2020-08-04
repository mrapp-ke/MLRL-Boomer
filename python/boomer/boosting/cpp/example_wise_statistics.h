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

            const float64* hessians_;

            const float64* totalSumsOfHessians_;

        public:

            /**
             * @param ruleEvaluation        TODO
             * @param numPredictions        TODO
             * @param labelIndices          TODO
             * @param numLabels             TODO
             * @param gradients             TODO
             * @param totalSumsOfGradients  TODO
             * @param hessians              TODO
             * @param totalSumsOfHessians   TODO
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
