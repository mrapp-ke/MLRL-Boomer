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
#include "statistics.h"
#include <memory>


namespace seco {

    /**
     * Allows to search for the best refinement of a rule based on the confusion matrices previously stored by an object
     * of type `DenseLabelWiseStatisticsImpl`.
     */
    class DenseLabelWiseRefinementSearchImpl : public AbstractDecomposableRefinementSearch {

        private:

            std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr_;

            uint32 numPredictions_;

            const uint32* labelIndices_;

            std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr_;

            const float64* uncoveredLabels_;

            const uint8* minorityLabels_;

            const float64* confusionMatricesTotal_;

            const float64* confusionMatricesSubset_;

            float64* confusionMatricesCovered_;

            float64* accumulatedConfusionMatricesCovered_;

            LabelWisePredictionCandidate* prediction_;

        public:

            /**
             * @param ruleEvaluationPtr         A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation`
             *                                  to be used for calculating the predictions, as well as corresponding
             *                                  quality scores, of rules
             * @param numPredictions            The number of labels to be considered by the search
             * @param labelIndices              An array of type `uint32`, shape `(numPredictions)`, representing the
             *                                  indices of the labels that should be considered by the search or NULL,
             *                                  if all labels should be considered
             * @param labelMatrix               A shared pointer to an object of type `AbstractRandomAccessLabelMatrix`
             *                                  that provides random access to the labels of the training examples
             * @param uncoveredLabels           A pointer to an array of type `float64`, shape
             *                                  `(numExamples, numLabels)`, indicating which examples and labels remain
             *                                  to be covered
             * @param minorityLabels            A pointer to an array of type `uint8`, shape `(numLabels)`, indicating
             *                                  whether rules should predict individual labels as relevant (1) or
             *                                  irrelevant (0)
             * @param confusionMatricesTotal    A pointer to a C-contiguous array of type `float64`, shape
             *                                  `(num_labels, NUM_CONFUSION_MATRIX_ELEMENTS)`, storing a confusion
             *                                  matrix that takes into account all examples for each label
             * @param confusionMatricesSubset   A pointer to a C-contiguous array of type `float64`, shape
             *                                  `(num_labels, NUM_CONFUSION_MATRIX_ELEMENTS)`, storing a confusion
             *                                  matrix that takes into account all all examples, which are covered by
             *                                  the previous refinement of the rule, for each label
             */
            DenseLabelWiseRefinementSearchImpl(std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
                                               uint32 numPredictions, const uint32* labelIndices,
                                               std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr,
                                               const float64* uncoveredLabels, const uint8* minorityLabels,
                                               const float64* confusionMatricesTotal,
                                               const float64* confusionMatricesSubset);

            ~DenseLabelWiseRefinementSearchImpl();

            void updateSearch(intp statisticIndex, uint32 weight) override;

            void resetSearch() override;

            LabelWisePredictionCandidate* calculateLabelWisePrediction(bool uncovered, bool accumulated) override;

    };

    /**
     * An abstract base class for all classes that allow to store the elements of confusion matrices that are computed
     * independently for each label.
     */
    class AbstractLabelWiseStatistics : public AbstractCoverageStatistics {

        public:

            std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr_;

            /**
             * @param numStatistics     The number of statistics
             * @param numLabels         The number of labels
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             */
            AbstractLabelWiseStatistics(uint32 numStatistics, uint32 numLabels,
                                        std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr);

            /**
             * Sets the implementation to be used for calculating the predictions, as well as corresponding quality
             * scores, of rules.
             *
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation` to be
             *                          set
             */
            void setRuleEvaluation(std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr);

    };

    /**
     * Allows to store the elements of confusion matrices that are computed independently for each label using dense
     * data structures.
     */
    class DenseLabelWiseStatisticsImpl : public AbstractLabelWiseStatistics {

        private:

            std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr_;

            float64* uncoveredLabels_;

            uint8* minorityLabels_;

            float64* confusionMatricesTotal_;

            float64* confusionMatricesSubset_;

        public:

            /**
             * @param ruleEvaluationPtr     A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation` to
             *                              be used for calculating the predictions, as well as corresponding quality
             *                              scores, of rules
             * @param labelMatrixPtr        A shared pointer to an object of type `AbstractRandomAccessLabelMatrix` that
             *                              provides random access to the labels of the training examples
             * @param uncoveredLabels       A pointer to an array of type `float64`, shape `(numExamples, numLabels)`,
             *                              indicating which examples and labels remain to be covered
             * @param sumUncoveredLabels    The sum of weights of all labels that remain to be covered
             * @param minorityLabels        A pointer to an array of type `uint8`, shape `(numLabels)`, indicating
             *                              whether rules should predict individual labels as relevant (1) or irrelevant
             *                              (0)
             */
            DenseLabelWiseStatisticsImpl(std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
                                         std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr,
                                         float64* uncoveredLabels, float64 sumUncoveredLabels, uint8* minorityLabels);

            ~DenseLabelWiseStatisticsImpl();

            void resetSampledStatistics() override;

            void addSampledStatistic(intp statisticIndex, uint32 weight) override;

            void resetCoveredStatistics() override;

            void updateCoveredStatistic(intp statisticIndex, uint32 weight, bool remove) override;

            AbstractRefinementSearch* beginSearch(uint32 numLabelIndices, const uint32* labelIndices) override;

            void applyPrediction(intp statisticIndex, Prediction* prediction) override;

    };

    /**
     * An abstract base class for all classes that allow to create new instances of the class
     * `AbstractLabelWiseStatistics`.
     */
    class AbstractLabelWiseStatisticsFactory {

        public:

            virtual ~AbstractLabelWiseStatisticsFactory();

            /**
             * Creates a new instance of the class `AbstractLabelWiseStatistics`.
             *
             * @return A pointer to an object of type `AbstractLabelWiseStatistics` that has been created
             */
            virtual AbstractLabelWiseStatistics* create();

    };

    /**
     * A factory that allows to create new instances of the class `DenseLabelWiseStatisticsImpl`.
     */
    class DenseLabelWiseStatisticsFactoryImpl : public AbstractLabelWiseStatisticsFactory {

        private:

            std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr_;

            std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation` to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             * @param labelMatrixPtr    A shared pointer to an object of type `AbstractRandomAccessLabelMatrix` that
             *                          provides random access to the labels of the training examples
             */
            DenseLabelWiseStatisticsFactoryImpl(std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
                                                std::shared_ptr<AbstractRandomAccessLabelMatrix> labelMatrixPtr);

            ~DenseLabelWiseStatisticsFactoryImpl();

            AbstractLabelWiseStatistics* create() override;

    };

}
