/**
 * Provides classes that allow to store the elements of confusion matrices that are computed independently for each
 * label.
 *
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

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

        protected:

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param numStatistics             The number of statistics
             * @param numLabels                 The number of labels
             * @param sumUncoveredLabels        The sum of weights of all labels that remain to be covered, initially
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             */
            AbstractLabelWiseStatistics(uint32 numStatistics, uint32 numLabels, float64 sumUncoveredLabels,
                                        std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr);

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             *
             * @param ruleEvaluationFactoryPtr A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                 to be set
             */
            void setRuleEvaluationFactory(std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr);

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
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<class T>
            class StatisticsSubsetImpl : public AbstractDecomposableStatisticsSubset {

                private:

                    const DenseLabelWiseStatisticsImpl& statistics_;

                    std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    float64* confusionMatricesCovered_;

                    float64* accumulatedConfusionMatricesCovered_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `DenseLabelWiseStatisticsImpl` that
                     *                          stores the confusion matrices
                     * @param ruleEvaluationPtr An unique pointer to an object of type `ILabelWiseRuleEvaluation` that
                     *                          should be used to calculate the predictions, as well as corresponding
                     *                          quality scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubsetImpl(const DenseLabelWiseStatisticsImpl& statistics,
                                         std::unique_ptr<ILabelWiseRuleEvaluation> ruleEvaluationPtr,
                                         const T& labelIndices);

                    ~StatisticsSubsetImpl();

                    void addToSubset(uint32 statisticIndex, uint32 weight) override;

                    void resetSubset() override;

                    const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(bool uncovered,
                                                                                     bool accumulated) override;

            };

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

            float64* uncoveredLabels_;

            uint8* minorityLabels_;

            float64* confusionMatricesTotal_;

            float64* confusionMatricesSubset_;

        public:

            /**
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             * @param uncoveredLabels           A pointer to an array of type `float64`, shape
             *                                  `(numExamples, numLabels)`, indicating which examples and labels remain
             *                                  to be covered
             * @param sumUncoveredLabels        The sum of weights of all labels that remain to be covered
             * @param minorityLabels            A pointer to an array of type `uint8`, shape `(numLabels)`, indicating
             *                                  whether rules should predict individual labels as relevant (1) or
             *                                  irrelevant (0)
             */
            DenseLabelWiseStatisticsImpl(std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                         std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr,
                                         float64* uncoveredLabels, float64 sumUncoveredLabels, uint8* minorityLabels);

            ~DenseLabelWiseStatisticsImpl();

            void resetSampledStatistics() override;

            void addSampledStatistic(uint32 statisticIndex, uint32 weight) override;

            void resetCoveredStatistics() override;

            void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override;

            std::unique_ptr<IStatisticsSubset> createSubset(const RangeIndexVector& labelIndices) const override;

            std::unique_ptr<IStatisticsSubset> createSubset(const DenseIndexVector& labelIndices) const override;

            void applyPrediction(uint32 statisticIndex, const Prediction& prediction) override;

            std::unique_ptr<IHistogramBuilder> buildHistogram(uint32 numBins) const override;

    };

    /**
     * Defines an interface for all classes that allow to create new instances of the class
     * `AbstractLabelWiseStatistics`.
     */
    class ILabelWiseStatisticsFactory {

        public:

            virtual ~ILabelWiseStatisticsFactory() { };

            /**
             * Creates a new instance of the class `AbstractLabelWiseStatistics`.
             *
             * @return An unique pointer to an object of type `AbstractLabelWiseStatistics` that has been created
             */
            virtual std::unique_ptr<AbstractLabelWiseStatistics> create() const = 0;

    };

    /**
     * A factory that allows to create new instances of the class `DenseLabelWiseStatisticsImpl`.
     */
    class DenseLabelWiseStatisticsFactoryImpl : virtual public ILabelWiseStatisticsFactory {

        private:

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             */
            DenseLabelWiseStatisticsFactoryImpl(
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr);

            std::unique_ptr<AbstractLabelWiseStatistics> create() const override;

    };

}
