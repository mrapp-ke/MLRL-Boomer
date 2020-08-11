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
#include "example_wise_losses.h"
#include "statistics.h"
#include <memory>


namespace boosting {

    /**
     * Allows to search for the best refinement of a rule based on the gradients and Hessians previously stored by an
    `* object of type `ExampleWiseStatisticsImpl`.
     */
    class ExampleWiseRefinementSearchImpl : public AbstractRefinementSearch {

        private:

            std::shared_ptr<ExampleWiseRuleEvaluationImpl> ruleEvaluationPtr_;

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
             * @param ruleEvaluationPtr     A shared pointer to an object of type `ExampleWiseRuleEvaluationImpl` to be
             *                              used for calculating the predictions, as well as corresponding quality
             *                              scores of rules
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
            ExampleWiseRefinementSearchImpl(std::shared_ptr<ExampleWiseRuleEvaluationImpl> ruleEvaluationPtr,
                                            intp numPredictions, const intp* labelIndices, intp numLabels,
                                            const float64* gradients, const float64* totalSumsOfGradients,
                                            const float64* hessians, const float64* totalSumsOfHessians);

            ~ExampleWiseRefinementSearchImpl();

            void updateSearch(intp statisticIndex, uint32 weight) override;

            void resetSearch() override;

            LabelWisePrediction* calculateLabelWisePrediction(bool uncovered, bool accumulated) override;

            Prediction* calculateExampleWisePrediction(bool uncovered, bool accumulated) override;

    };

    /**
     * Allows to store gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied example-wise.
     */
    class ExampleWiseStatisticsImpl : public AbstractGradientStatistics {

        private:

            std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<ExampleWiseRuleEvaluationImpl> ruleEvaluationPtr_;

            std::shared_ptr<AbstractLabelMatrix> labelMatrixPtr_;

            float64* currentScores_;

            float64* gradients_;

            float64* totalSumsOfGradients_;

            float64* hessians_;

            float64* totalSumsOfHessians_;

        public:

            /**
             * @param lossFunctionPtr   A shared pointer to an object of type `AbstractExampleWiseLoss`, representing
             *                          the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationPtr A shared pointer to an object of type `ExampleWiseRuleEvaluationImpl`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             */
            ExampleWiseStatisticsImpl(std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr,
                                      std::shared_ptr<ExampleWiseRuleEvaluationImpl> ruleEvaluationPtr);

            ~ExampleWiseStatisticsImpl();

            void applyDefaultPrediction(std::shared_ptr<AbstractLabelMatrix> labelMatrixPtr,
                                        DefaultPrediction* defaultPrediction) override;

            void resetCoveredStatistics() override;

            void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove) override;

            AbstractRefinementSearch* beginSearch(intp numLabelIndices, const intp* labelIndices) override;

            void applyPrediction(intp statisticIndex, const intp* labelIndices, HeadCandidate* head) override;

    };

}
