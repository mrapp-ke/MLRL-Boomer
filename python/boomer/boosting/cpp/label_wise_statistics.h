/**
 * Provides classes that allow to store gradients and Hessians that are calculated according to a (decomposable) loss
 * function that is applied label-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/statistics.h"
#include "label_wise_rule_evaluation.h"
#include "label_wise_losses.h"
#include "statistics.h"


namespace boosting {

    /**
     * An abstract base class for all classes that store gradients and Hessians that are calculated according to a
     * differentiable loss function that is applied label-wise.
     */
    class AbstractLabelWiseStatistics : public AbstractGradientStatistics {

        protected:

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param numStatistics             The number of statistics
             * @param numLabels                 The number of labels
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`,
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             */
            AbstractLabelWiseStatistics(uint32 numStatistics, uint32 numLabels,
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
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied label-wise using dense data structures.
     */
    class DenseLabelWiseStatisticsImpl : public AbstractLabelWiseStatistics {

        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
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

                    float64* sumsOfGradients_;

                    float64* accumulatedSumsOfGradients_;

                    float64* sumsOfHessians_;

                    float64* accumulatedSumsOfHessians_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `DenseLabelWiseStatisticsImpl` that
                     *                          stores the gradients and Hessians
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

            /**
             * Allows to build a histogram based on the gradients and Hessians that are stored by an instance of the
             * class `DenseLabelWiseStatisticsImpl`.
             */
            class HistogramBuilderImpl : virtual public AbstractStatistics::IHistogramBuilder {

                private:

                    const DenseLabelWiseStatisticsImpl& statistics_;

                    uint32 numBins_;

                    float64* gradients_;

                    float64* hessians_;

                public:

                    /**
                     * @param statistics    A reference to an object of type `DenseLabelWiseStatisticsImpl` that stores
                     *                      the gradients and Hessians
                     * @param numBins       The number of bins, the histogram should consist of
                     */
                    HistogramBuilderImpl(const DenseLabelWiseStatisticsImpl& statistics, uint32 numBins);

                    void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) override;

                    std::unique_ptr<AbstractStatistics> build() const override;

            };

            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

            float64* gradients_;

            float64* hessians_;

            float64* currentScores_;

            float64* totalSumsOfGradients_;

            float64* totalSumsOfHessians_;

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `ILabelWiseLoss`, representing the
             *                                  loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`,
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             * @param gradients                 A pointer to an array of type `float64`, shape
             *                                  `(num_examples, num_labels)`, representing the gradients
             * @param hessians                  A pointer to an array of type `float64`, shape
             *                                  `(num_examples, num_labels)`, representing the Hessians
             * @param current_scores            A pointer to an array of type `float64`, shape
             *                                  `(num_examples, num_labels)`, representing the currently predicted
             *                                  scores
             */
            DenseLabelWiseStatisticsImpl(std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
                                         std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                         std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, float64* gradients,
                                         float64* hessians, float64* current_scores);

            ~DenseLabelWiseStatisticsImpl();

            void resetCoveredStatistics() override;

            void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override;

            std::unique_ptr<IStatisticsSubset> createSubset(const RangeIndexVector& indexVector) const override;

            std::unique_ptr<IStatisticsSubset> createSubset(const DenseIndexVector& indexVector) const override;

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

            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `ILabelWiseLoss`, representing the
             *                                  loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`
             *                                  that allows to create instances of the class that is used to calculate
             *                                  the predictions, as well as corresponding quality scores, of rules
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             */
            DenseLabelWiseStatisticsFactoryImpl(
                std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr);

            std::unique_ptr<AbstractLabelWiseStatistics> create() const override;

    };

}
