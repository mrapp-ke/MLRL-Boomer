#include "boosting/statistics/statistics_example_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/statistic_vector_dense_example_wise.hpp"
#include "boosting/data/statistic_view_dense_example_wise.hpp"
#include "boosting/math/math.hpp"
#include "statistics_example_wise_common.hpp"
#include "statistics_example_wise_provider.hpp"
#include "omp.h"
#include <cstdlib>


namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a non-decomposable loss function
     * using C-contiguous arrays.
     */
    class DenseExampleWiseStatisticMatrix : public DenseExampleWiseStatisticView {

        public:

            /**
             * @param numRows       The number of rows in the matrix
             * @param numGradients  The number of gradients per row
             */
            DenseExampleWiseStatisticMatrix(uint32 numRows, uint32 numGradients)
                : DenseExampleWiseStatisticView(
                      numRows, numGradients, triangularNumber(numGradients),
                      (float64*) malloc(numRows * numGradients * sizeof(float64)),
                      (float64*) malloc(numRows * triangularNumber(numGradients) * sizeof(float64))) {

            }

    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss function
     * that is applied example-wise and are stored using dense data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<typename LabelMatrix>
    class DenseExampleWiseStatistics final : public AbstractExampleWiseStatistics<LabelMatrix,
                                                                                  DenseExampleWiseStatisticVector,
                                                                                  DenseExampleWiseStatisticView,
                                                                                  DenseExampleWiseStatisticMatrix,
                                                                                  NumericDenseMatrix<float64>,
                                                                                  IExampleWiseLoss> {

        public:

            /**
             * @param lossFunction          A reference to an object of type `IExampleWiseLoss`, representing the loss
             *                              function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `IExampleWiseRuleEvaluationFactory`, to be
             *                              used for calculating the predictions, as well as corresponding quality
             *                              scores, of rules
             * @param labelMatrix           A reference to an object of template type `LabelMatrix` that provides access
             *                              to the labels of the training examples
             * @param statisticViewPtr      An unique pointer to an object of type `DenseExampleWiseStatisticView` that
             *                              provides access to the gradients and Hessians
             * @param scoreMatrixPtr        An unique pointer to an object of type `NumericDenseMatrix` that stores the
             *                              currently predicted scores
             */
            DenseExampleWiseStatistics(const IExampleWiseLoss& lossFunction,
                                       const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                       const LabelMatrix& labelMatrix,
                                       std::unique_ptr<DenseExampleWiseStatisticView> statisticViewPtr,
                                       std::unique_ptr<NumericDenseMatrix<float64>> scoreMatrixPtr)
                : AbstractExampleWiseStatistics<LabelMatrix, DenseExampleWiseStatisticVector,
                                                DenseExampleWiseStatisticView, DenseExampleWiseStatisticMatrix,
                                                NumericDenseMatrix<float64>, IExampleWiseLoss>(
                      lossFunction, ruleEvaluationFactory, labelMatrix, std::move(statisticViewPtr),
                      std::move(scoreMatrixPtr)) {

            }

            void visit(IBoostingStatistics::DenseLabelWiseStatisticViewVisitor denseLabelWiseStatisticViewVisitor,
                       IBoostingStatistics::DenseExampleWiseStatisticViewVisitor denseExampleWiseStatisticViewVisitor) override {
                denseExampleWiseStatisticViewVisitor(this->statisticViewPtr_);
            }

    };

    template<typename LabelMatrix>
    static inline std::unique_ptr<IExampleWiseStatistics> createInternally(
            const IExampleWiseLoss& lossFunction, const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
            uint32 numThreads, const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseExampleWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseExampleWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericDenseMatrix<float64>> scoreMatrixPtr =
            std::make_unique<NumericDenseMatrix<float64>>(numExamples, numLabels, true);
        const IExampleWiseLoss* lossFunctionRawPtr = &lossFunction;
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const CContiguousConstView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseExampleWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossFunctionRawPtr) \
        firstprivate(labelMatrixPtr) firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) \
        schedule(dynamic) num_threads(numThreads)
        for (uint32 i = 0; i < numExamples; i++) {
            lossFunctionRawPtr->updateExampleWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr,
                                                            *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseExampleWiseStatistics<LabelMatrix>>(
            lossFunction, ruleEvaluationFactory, labelMatrix, std::move(statisticMatrixPtr),
            std::move(scoreMatrixPtr));
    }

    DenseExampleWiseStatisticsFactory::DenseExampleWiseStatisticsFactory(
            const IExampleWiseLoss& lossFunction, const  IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
            uint32 numThreads)
        : lossFunction_(lossFunction), ruleEvaluationFactory_(ruleEvaluationFactory), numThreads_(numThreads) {

    }

    std::unique_ptr<IExampleWiseStatistics> DenseExampleWiseStatisticsFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        return createInternally<CContiguousLabelMatrix>(lossFunction_, ruleEvaluationFactory_, numThreads_,
                                                        labelMatrix);
    }

    std::unique_ptr<IExampleWiseStatistics> DenseExampleWiseStatisticsFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        return createInternally<CsrLabelMatrix>(lossFunction_, ruleEvaluationFactory_, numThreads_, labelMatrix);
    }

    DenseExampleWiseStatisticsProviderFactory::DenseExampleWiseStatisticsProviderFactory(
            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), defaultRuleEvaluationFactoryPtr_(defaultRuleEvaluationFactoryPtr),
          ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), numThreads_(numThreads) {

    }

    std::unique_ptr<IStatisticsProvider> DenseExampleWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        DenseExampleWiseStatisticsFactory statisticsFactory(*lossFunctionPtr_, *defaultRuleEvaluationFactoryPtr_,
                                                            numThreads_);
        return std::make_unique<ExampleWiseStatisticsProvider>(*ruleEvaluationFactoryPtr_,
                                                               statisticsFactory.create(labelMatrix));
    }

    std::unique_ptr<IStatisticsProvider> DenseExampleWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        DenseExampleWiseStatisticsFactory statisticsFactory(*lossFunctionPtr_, *defaultRuleEvaluationFactoryPtr_,
                                                            numThreads_);
        return std::make_unique<ExampleWiseStatisticsProvider>(*ruleEvaluationFactoryPtr_,
                                                               statisticsFactory.create(labelMatrix));
    }

}
