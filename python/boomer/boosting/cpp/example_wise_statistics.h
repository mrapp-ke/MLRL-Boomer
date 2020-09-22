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
#include "lapack.h"
#include <memory>


namespace boosting {

    /**
     * An abstract base class for all classes that allow to store gradients and Hessians that are calculated according
     * to a differentiable loss function that is applied example-wise.
     */
    class AbstractExampleWiseStatistics : public AbstractGradientStatistics {

        public:

            std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr_;

            /**
             * @param numStatistics     The number of statistics
             * @param numLabels         The number of labels
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractExampleWiseRuleEvaluation`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             */
            AbstractExampleWiseStatistics(uint32 numStatistics, uint32 numLabels,
                                          std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr);

            /**
             * Sets the implementation to be used for calculating the predictions, as well as corresponding quality
             * scores, of rules.
             *
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractExampleWiseRuleEvaluation` to be
             *                          set
             */
            void setRuleEvaluation(std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr);

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
            class StatisticsSubsetImpl : virtual public AbstractStatisticsSubset {

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

            std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

            float64* currentScores_;

            float64* gradients_;

            float64* totalSumsOfGradients_;

            float64* hessians_;

            float64* totalSumsOfHessians_;

        public:

            /**
             * @param lossFunctionPtr   A shared pointer to an object of type `AbstractExampleWiseLoss`, representing
             *                          the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractExampleWiseRuleEvaluation`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
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
            DenseExampleWiseStatisticsImpl(std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr,
                                          std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr,
                                          std::shared_ptr<Lapack> lapackPtr,
                                          std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, float64* gradients,
                                          float64* hessians, float64* currentScores);

            ~DenseExampleWiseStatisticsImpl();

            void resetCoveredStatistics() override;

            void updateCoveredStatistic(uint32 statisticIndex, uint32 weight, bool remove) override;

            AbstractStatisticsSubset* createSubset(uint32 numLabelIndices, const uint32* labelIndices) override;

            void applyPrediction(uint32 statisticIndex, Prediction* prediction) override;

    };

    /**
     * An abstract base class for all classes that allow to create new instances of the class
     * `AbstractExampleWiseStatistics`.
     */
    class AbstractExampleWiseStatisticsFactory {

        public:

            virtual ~AbstractExampleWiseStatisticsFactory();

            /**
             * Creates a new instance of the class `AbstractExampleWiseStatistics`.
             *
             * @return A pointer to an object of type `AbstractExampleWiseStatistics` that has been created
             */
            virtual AbstractExampleWiseStatistics* create();

    };

    /**
     * A factory that allows to create new instances of the class `DenseExampleWiseStatisticsImpl`.
     */
    class DenseExampleWiseStatisticsFactoryImpl : public AbstractExampleWiseStatisticsFactory {

        private:

            std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr_;

            std::shared_ptr<Lapack> lapackPtr_;

            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr_;

        public:

            /**
             * @param lossFunctionPtr   A shared pointer to an object of type `AbstractExampleWiseLoss`, representing
             *                          the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationPtr A shared pointer to an object of type `AbstractExampleWiseRuleEvaluation`, to be
             *                          used for calculating the predictions, as well as corresponding quality scores,
             *                          of rules
             * @param lapackPtr         A shared pointer to an object of type `Lapack` that allows to execute different
             *                          Lapack routines
             * @param labelMatrixPtr    A shared pointer to an object of type `IRandomAccessLabelMatrix` that provides
             *                          random access to the labels of the training examples
             */
            DenseExampleWiseStatisticsFactoryImpl(std::shared_ptr<AbstractExampleWiseLoss> lossFunctionPtr,
                                                  std::shared_ptr<AbstractExampleWiseRuleEvaluation> ruleEvaluationPtr,
                                                  std::shared_ptr<Lapack> lapackPtr,
                                                  std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr);

            ~DenseExampleWiseStatisticsFactoryImpl();

            AbstractExampleWiseStatistics* create() override;

    };

}
