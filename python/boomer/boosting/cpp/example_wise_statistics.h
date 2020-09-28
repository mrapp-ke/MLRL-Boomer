/**
 * Provides classes that allow to store gradients and Hessians that are calculated according to a (non-decomposable)
 * loss function that is applied example-wise.
 *
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "../../common/cpp/arrays.h"
#include "../../common/cpp/statistics.h"
#include "../../common/cpp/binning.h"
#include "example_wise_rule_evaluation.h"
#include "example_wise_losses.h"
#include "statistics.h"
#include "lapack.h"
#include <memory>


namespace boosting {

    /**
     * An abstract base class for all classes that allow to store gradients and Hessians that are calculated according
     * to a differentiable loss function that is applied example-wise.
     */
    class AbstractExampleWiseStatistics : public AbstractGradientStatistics {

        protected:

            std::shared_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr_;

        public:

            /**
             * @param numStatistics     The number of statistics
             * @param numLabels         The number of labels
             * @param ruleEvaluationPtr A shared pointer to an object of type `IExampleWiseRuleEvaluation`, to be used
             *                          for calculating the predictions, as well as corresponding quality scores, of
             *                          rules
             */
            AbstractExampleWiseStatistics(uint32 numStatistics, uint32 numLabels,
                                          std::shared_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr);

            /**
             * Sets the implementation to be used for calculating the predictions, as well as corresponding quality
             * scores, of rules.
             *
             * @param ruleEvaluationPtr A shared pointer to an object of type `IExampleWiseRuleEvaluation` to be set
             */
            void setRuleEvaluation(std::shared_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr);

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
             */
            class StatisticsSubsetImpl : virtual public IStatisticsSubset {

                private:

                    DenseExampleWiseStatisticsImpl* statistics_;

                    uint32 numPredictions_;

                    const uint32* labelIndices_;

                    float64* sumsOfGradients_;

                    float64* accumulatedSumsOfGradients_;

                    float64* sumsOfHessians_;

                    float64* accumulatedSumsOfHessians_;

                    LabelWisePredictionCandidate* prediction_;

                    float64* tmpGradients_;

                    float64* tmpHessians_;

                    int dsysvLwork_;

                    float64* dsysvTmpArray1_;

                    int* dsysvTmpArray2_;

                    double* dsysvTmpArray3_;

                    float64* dspmvTmpArray_;

                public:

                    /**
                     * @param statistics        A pointer to an object of type `DenseExampleWiseStatisticsImpl` that
                     *                          stores the gradients and Hessians
                     * @param numPredictions    The number of elements in the array `labelIndices`
                     * @param labelIndices      A pointer to an array of type `uint32`, shape `(numPredictions)`,
                     *                          representing the indices of the labels that should be included in the
                     *                          subset or NULL, if all labels should be considered
                     */
                    StatisticsSubsetImpl(DenseExampleWiseStatisticsImpl* statistics, uint32 numPredictions,
                                         const uint32* labelIndices);

                    ~StatisticsSubsetImpl();

                    void addToSubset(uint32 statisticIndex, uint32 weight) override;

                    void resetSubset() override;

                    LabelWisePredictionCandidate* calculateLabelWisePrediction(bool uncovered,
                                                                               bool accumulated) override;

                    PredictionCandidate* calculateExampleWisePrediction(bool uncovered, bool accumulated) override;

            };

            class DenseExampleWiseStatisticsBinsImpl : virtual public IHistogramBuilder {

                private:

                    DenseExampleWiseStatisticsImpl* statistics_;

                    uint32 numBins_;

                    float64* gradients_;

                    float64* hessians_;

                    Bin* bins;

                public:

                    DenseExampleWiseStatisticsBinsImpl(DenseExampleWiseStatisticsImpl* statistics, uint32 numBins);

                    void onBinUpdate(uint32 binIndex, IndexedFloat32* indexedValue) override;

                    AbstractStatistics* build();

            };

            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

            float64* currentScores_;

            float64* gradients_;

            float64* totalSumsOfGradients_;

            float64* hessians_;

            float64* totalSumsOfHessians_;

        public:

            /**
             * @param lossFunctionPtr   A shared pointer to an object of type `IExampleWiseLoss`, representing the loss
             *                          function to be used for calculating gradients and Hessians
             * @param ruleEvaluationPtr A shared pointer to an object of type `IExampleWiseRuleEvaluation`, to be used
             *                          for calculating the predictions, as well as corresponding quality scores, of
             *                          rules
             * @param lapackPtr         A shared pointer to an object of type `Lapack` that allows to execute different
             *                          Lapack routines
             * @param labelMatrixPtr    A shared pointer to an object of type `IRandomAccessLabelMatrix` that provides
             *                          random access to the labels of the training examples
             * @param gradients         A pointer to an array of type `float64`, shape `(num_examples, num_labels)`,
             *                          representing the gradients
             * @param hessians          A pointer to an array of type `float64`, shape
             *                          `(num_examples, num_labels + (num_labels + 1) // 2)`, representing the Hessians
             * @param currentScores     A pointer to an array of type `float64`, shape `(num_examples, num_labels`),
             *                          representing the currently predicted scores
             */
            DenseExampleWiseStatisticsImpl(std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                                          std::shared_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr,
                                          std::shared_ptr<Lapack> lapackPtr,
                                          std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, float64* gradients,
                                          float64* hessians, float64* currentScores);

            ~DenseExampleWiseStatisticsImpl();

            void resetCoveredStatistics() override;

            void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override;

            IStatisticsSubset* createSubset(uint32 numLabelIndices, const uint32* labelIndices) override;

            void applyPrediction(uint32 statisticIndex, Prediction* prediction) override;

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
             * @return A pointer to an object of type `AbstractExampleWiseStatistics` that has been created
             */
            virtual AbstractExampleWiseStatistics* create() = 0;

    };

    /**
     * A factory that allows to create new instances of the class `DenseExampleWiseStatisticsImpl`.
     */
    class DenseExampleWiseStatisticsFactoryImpl : virtual public IExampleWiseStatisticsFactory {

        private:

            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param lossFunctionPtr   A shared pointer to an object of type `IExampleWiseLoss`, representing the loss
             *                          function to be used for calculating gradients and Hessians
             * @param ruleEvaluationPtr A shared pointer to an object of type `IExampleWiseRuleEvaluation`, to be used
             *                          for calculating the predictions, as well as corresponding quality scores, of
             *                          rules
             * @param lapackPtr         A shared pointer to an object of type `Lapack` that allows to execute different
             *                          Lapack routines
             * @param labelMatrixPtr    A shared pointer to an object of type `IRandomAccessLabelMatrix` that provides
             *                          random access to the labels of the training examples
             */
            DenseExampleWiseStatisticsFactoryImpl(std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                                                  std::shared_ptr<IExampleWiseRuleEvaluation> ruleEvaluationPtr,
                                                  std::shared_ptr<Lapack> lapackPtr,
                                                  std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr);

            AbstractExampleWiseStatistics* create() override;

    };

}
