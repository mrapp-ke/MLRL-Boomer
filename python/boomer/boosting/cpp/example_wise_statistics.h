/**
 * Provides classes that allow to store gradients and Hessians that are calculated according to a (non-decomposable)
 * loss function that is applied example-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/statistics.h"
#include "example_wise_rule_evaluation.h"
#include "example_wise_losses.h"
#include "statistics.h"
#include "lapack.h"


namespace boosting {

    /**
     * An abstract base class for all classes that allow to store gradients and Hessians that are calculated according
     * to a differentiable loss function that is applied example-wise.
     */
    class AbstractExampleWiseStatistics : public AbstractGradientStatistics {

        protected:

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

        public:

            /**
             * @param numStatistics             The number of statistics
             * @param numLabels                 The number of labels
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory` that allows to create instances of
             *                                  the class that is used for calculating the predictions, as well as
             *                                  corresponding quality scores, of rules
             */
            AbstractExampleWiseStatistics(uint32 numStatistics, uint32 numLabels,
                                          std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr);

            /**
             * Sets the factory that allows to create instances of the class that is used for calculating the
             * predictions, as well as corresponding quality scores, of rules.
             *
             * @param ruleEvaluationFactoryPtr A shared pointer to an object of type `IExampleWiseRuleFactoryEvaluation`
             *                                 to be set
             */
            void setRuleEvaluationFactory(std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr);

    };

    /**
     * Provides access to gradients and Hessians that are calculated according to a differentiable loss function that is
     * applied example-wise using dense data structures.
     */
    class DenseExampleWiseStatisticsImpl : public AbstractExampleWiseStatistics {

        private:

            /**
             * Provides access to a subset of the gradients and Hessians that are stored by an instance of the class
             * `DenseExampleWiseStatisticsImpl`.
             *
             * @tparam T The type of the vector that provides access to the indices of the labels that are included in
             *           the subset
             */
            template<class T>
            class StatisticsSubsetImpl : virtual public IStatisticsSubset {

                private:

                    const DenseExampleWiseStatisticsImpl& statistics_;

                    std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr_;

                    const T& labelIndices_;

                    float64* sumsOfGradients_;

                    float64* accumulatedSumsOfGradients_;

                    float64* sumsOfHessians_;

                    float64* accumulatedSumsOfHessians_;

                    float64* tmpGradients_;

                    float64* tmpHessians_;

                    int dsysvLwork_;

                    float64* dsysvTmpArray1_;

                    int* dsysvTmpArray2_;

                    double* dsysvTmpArray3_;

                    float64* dspmvTmpArray_;

                public:

                    /**
                     * @param statistics        A reference to an object of type `DenseExampleWiseStatisticsImpl` that
                     *                          stores the gradients and Hessians
                     * @param ruleEvaluationPtr An unique pointer to an object of type `IExampleWiseRuleEvaluation` that
                     *                          should be used to calculate the predictions, as well as corresponding
                     *                          quality scores, of rules
                     * @param labelIndices      A reference to an object of template type `T` that provides access to
                     *                          the indices of the labels that are included in the subset
                     */
                    StatisticsSubsetImpl(const DenseExampleWiseStatisticsImpl& statistics,
                                         std::unique_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr,
                                         const T& labelIndices);

                    ~StatisticsSubsetImpl();

                    void addToSubset(uint32 statisticIndex, uint32 weight) override;

                    void resetSubset() override;

                    const LabelWiseEvaluatedPrediction& calculateLabelWisePrediction(bool uncovered,
                                                                                     bool accumulated) override;

                    const EvaluatedPrediction& calculateExampleWisePrediction(bool uncovered,
                                                                              bool accumulated) override;

            };

            /**
             * Allows to build a histogram based on the gradients and Hessians that are stored by an instance of the
             * class `DenseExampleWiseStatisticsImpl`.
             */
            class HistogramBuilderImpl : virtual public IHistogramBuilder {

                private:

                    const DenseExampleWiseStatisticsImpl& statistics_;

                    uint32 numBins_;

                    float64* gradients_;

                    float64* hessians_;

                public:

                    /**
                     * @param statistics    A reference to an object of type `DenseExampleWiseStatisticsImpl` that
                     *                      stores the gradients and Hessians
                     * @param numBins       The number of bins, the histogram should consist of
                     */
                    HistogramBuilderImpl(const DenseExampleWiseStatisticsImpl& statistics, uint32 numBins);

                    void onBinUpdate(uint32 binIndex, const FeatureVector::Entry& entry) override;

                    std::unique_ptr<AbstractStatistics> build() const override;

            };

            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

            float64* gradients_;

            float64* hessians_;

            float64* currentScores_;

            float64* totalSumsOfGradients_;

            float64* totalSumsOfHessians_;

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `IExampleWiseLoss`, representing
             *                                  the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory`, to be used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             * @param lapackPtr                 A shared pointer to an object of type `Lapack` that allows to execute
             *                                  different Lapack routines
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             * @param gradients                 A pointer to an array of type `float64`, shape
             *                                  `(num_examples, num_labels)`, representing the gradients
             * @param hessians                  A pointer to an array of type `float64`, shape
             *                                  `(num_examples, num_labels + (num_labels + 1) // 2)`, representing the
             *                                  Hessians
             * @param currentScores             A pointer to an array of type `float64`, shape
             *                                  `(num_examples, num_labels`), representing the currently predicted
             *                                  scores
             */
            DenseExampleWiseStatisticsImpl(std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                                          std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                          std::shared_ptr<Lapack> lapackPtr,
                                          std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, float64* gradients,
                                          float64* hessians, float64* currentScores);

            ~DenseExampleWiseStatisticsImpl();

            void resetCoveredStatistics() override;

            void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override;

            std::unique_ptr<IStatisticsSubset> createSubset(const RangeIndexVector& labelIndices) const override;

            std::unique_ptr<IStatisticsSubset> createSubset(const DenseIndexVector& labelIndices) const override;

            void applyPrediction(uint32 statisticIndex, const FullPrediction& prediction) override;

            void applyPrediction(uint32 statisticIndex, const PartialPrediction& prediction) override;

            std::unique_ptr<IHistogramBuilder> buildHistogram(uint32 numBins) const override;

    };

    /**
     * Defines an interface for all classes that allow to create new instances of the class
     * `AbstractExampleWiseStatistics`.
     */
    class IExampleWiseStatisticsFactory {

        public:

            virtual ~IExampleWiseStatisticsFactory() { };

            /**
             * Creates a new instance of the class `AbstractExampleWiseStatistics`.
             *
             * @return An unique pointer to an object of type `AbstractExampleWiseStatistics` that has been created
             */
            virtual std::unique_ptr<AbstractExampleWiseStatistics> create() const = 0;

    };

    /**
     * A factory that allows to create new instances of the class `DenseExampleWiseStatisticsImpl`.
     */
    class DenseExampleWiseStatisticsFactoryImpl : virtual public IExampleWiseStatisticsFactory {

        private:

            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `IExampleWiseLoss`, representing
             *                                  the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory`, to be used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             * @param lapackPtr                 An unique pointer to an object of type `Lapack` that allows to execute
             *                                  different Lapack routines
             * @param labelMatrixPtr            A shared pointer to an object of type `IRandomAccessLabelMatrix` that
             *                                  provides random access to the labels of the training examples
             */
            DenseExampleWiseStatisticsFactoryImpl(
                    std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                    std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                    std::unique_ptr<Lapack> lapackPtr, std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr);

            std::unique_ptr<AbstractExampleWiseStatistics> create() const override;

    };

}
