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
     * An abstract base class for all classes that allow to store the elements of confusion matrices that are computed
     * independently for each label.
     */
    class AbstractLabelWiseStatistics : public AbstractCoverageStatistics {

        private:

            std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr_;

        public:

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
     * Provides access to the elements of confusion matrices that are computed independently for each label using dense
     * data structures.
     */
    class DenseLabelWiseStatisticsImpl : public AbstractLabelWiseStatistics {

        private:

            /**
             * Provides access to a subset of the confusion matrices that are stored by an instance of the class
             * `DenseLabelWiseStatisticsImpl`.
             */
            class StatisticsSubsetImpl : public AbstractDecomposableStatisticsSubset {

                private:

                    DenseLabelWiseStatisticsImpl* statistics_;

                    uint32 numPredictions_;

                    const uint32* labelIndices_;

                    float64* confusionMatricesCovered_;

                    float64* accumulatedConfusionMatricesCovered_;

                    LabelWisePredictionCandidate* prediction_;

                public:

                    /**
                     * @param statistics        A pointer to an object of type `DenseLabelWiseStatisticsImpl` that
                     *                          stores the confusion matrices
                     * @param numPredictions    The number of elements in the array `labelIndices`
                     * @param labelIndices      An array of type `uint32`, shape `(numPredictions)`, representing the
                     *                          indices of the labels that should be included in the subset or NULL,
                     *                          if all labels should be considered
                     */
                    StatisticsSubsetImpl(DenseLabelWiseStatisticsImpl* statistics, uint32 numPredictions,
                                         const uint32* labelIndices);

                    ~StatisticsSubsetImpl();

                    void addToSubset(uint32 statisticIndex, uint32 weight) override;

                    void resetSubset() override;

                    LabelWisePredictionCandidate* calculateLabelWisePrediction(bool uncovered,
                                                                               bool accumulated) override;

            };

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

            float64* uncoveredLabels_;

            uint8* minorityLabels_;

            float64* confusionMatricesTotal_;

            float64* confusionMatricesSubset_;

        public:

            /**
             * @param ruleEvaluationPtr     A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation` to
             *                              be used for calculating the predictions, as well as corresponding quality
             *                              scores, of rules
             * @param labelMatrixPtr        A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                              provides random access to the labels of the training examples
             * @param uncoveredLabels       A pointer to an array of type `float64`, shape `(numExamples, numLabels)`,
             *                              indicating which examples and labels remain to be covered
             * @param sumUncoveredLabels    The sum of weights of all labels that remain to be covered
             * @param minorityLabels        A pointer to an array of type `uint8`, shape `(numLabels)`, indicating
             *                              whether rules should predict individual labels as relevant (1) or irrelevant
             *                              (0)
             */
            DenseLabelWiseStatisticsImpl(std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
                                         std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                         float64* uncoveredLabels, float64 sumUncoveredLabels, uint8* minorityLabels);

            ~DenseLabelWiseStatisticsImpl();

            void resetSampledStatistics() override;

            void addSampledStatistic(uint32 statisticIndex, uint32 weight) override;

            void resetCoveredStatistics() override;

            void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override;

            IStatisticsSubset* createSubset(uint32 numLabelIndices, const uint32* labelIndices) override;

            void applyPrediction(uint32 statisticIndex, Prediction* prediction) override;

    };

    /**
     * Defines an interface for all classes that allow to create new instances of the class
     * `AbstractLabelWiseStatistics`.
     */
    class ILabelWiseStatisticsFactory {

        public:

            virtual ~ILabelWiseStatisticsFactory() = { };

            /**
             * Creates a new instance of the class `AbstractLabelWiseStatistics`.
             *
             * @return A pointer to an object of type `AbstractLabelWiseStatistics` that has been created
             */
            virtual AbstractLabelWiseStatistics* create() = 0;

    };

    /**
     * A factory that allows to create new instances of the class `DenseLabelWiseStatisticsImpl`.
     */
    class DenseLabelWiseStatisticsFactoryImpl : virtual public ILabelWiseStatisticsFactory {

        private:

            std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractLabelWiseRuleEvaluation` to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             * @param labelMatrixPtr    A shared pointer to an object of type `IRandomAccessLabelMatrix` that provides
             *                          random access to the labels of the training examples
             */
            DenseLabelWiseStatisticsFactoryImpl(std::shared_ptr<AbstractLabelWiseRuleEvaluation> ruleEvaluationPtr,
                                                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr);

            AbstractLabelWiseStatistics* create() override;

    };

}
