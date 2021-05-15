#include "boosting/statistics/statistics_example_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/statistic_vector_dense_example_wise.hpp"
#include "boosting/data/statistic_view_dense_example_wise.hpp"
#include "boosting/data/statistic_view_dense_label_wise.hpp"
#include "statistics_example_wise_common.hpp"
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
                : DenseExampleWiseStatisticMatrix(numRows, numGradients, false) {

            }

            /**
             * @param numRows       The number of rows in the matrix
             * @param numGradients  The number of gradients per row
             * @param init          True, if the gradients and Hessiansshould be value-initialized, false otherwise
             */
            DenseExampleWiseStatisticMatrix(uint32 numRows, uint32 numGradients, bool init)
                : DenseExampleWiseStatisticView(numRows, numGradients,
                                                (float64*) (init ? calloc(numRows * numGradients, sizeof(float64))
                                                                 : malloc(numRows * numGradients * sizeof(float64))),
                                                (float64*) (init ? calloc(numRows * numHessians_, sizeof(float64))
                                                                 : malloc(numRows * numHessians_ * sizeof(float64)))) {

            }

            ~DenseExampleWiseStatisticMatrix() {
                free(gradients_);
                free(hessians_);
            }

    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss function
     * that is applied example-wise and are stored using dense data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<class LabelMatrix>
    class DenseExampleWiseStatistics final : public AbstractExampleWiseStatistics<LabelMatrix,
                                                                                  DenseExampleWiseStatisticVector,
                                                                                  DenseExampleWiseStatisticView,
                                                                                  DenseExampleWiseStatisticMatrix,
                                                                                  DenseNumericMatrix<float64>> {

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `IExampleWiseLoss`, representing
             *                                  the loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type
             *                                  `IExampleWiseRuleEvaluationFactory`, to be used for calculating the
             *                                  predictions, as well as corresponding quality scores, of rules
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param statisticViewPtr          An unique pointer to an object of template type `StatisticView` that
             *                                  provides access to the gradients and Hessians
             * @param scoreMatrixPtr            An unique pointer to an object of template type `ScoreMatrix` that
             *                                  stores the currently predicted scores
             */
            DenseExampleWiseStatistics(std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                                       std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                       const LabelMatrix& labelMatrix,
                                       std::unique_ptr<DenseExampleWiseStatisticView> statisticViewPtr,
                                       std::unique_ptr<DenseNumericMatrix<float64>> scoreMatrixPtr)
                : AbstractExampleWiseStatistics<LabelMatrix, DenseExampleWiseStatisticVector,
                                                DenseExampleWiseStatisticView, DenseExampleWiseStatisticMatrix,
                                                DenseNumericMatrix<float64>>(
                      lossFunctionPtr, ruleEvaluationFactoryPtr, labelMatrix, std::move(statisticViewPtr),
                      std::move(scoreMatrixPtr)) {

            }

            void visit(IBoostingStatistics::DenseLabelWiseStatisticViewVisitor denseLabelWiseStatisticViewVisitor,
                       IBoostingStatistics::DenseExampleWiseStatisticViewVisitor denseExampleWiseStatisticViewVisitor) override {
                denseExampleWiseStatisticViewVisitor(this->statisticViewPtr_);
            }

    };

    template<class LabelMatrix>
    static inline std::unique_ptr<IExampleWiseStatistics> createInternally(
            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads,
            const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseExampleWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseExampleWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<DenseNumericMatrix<float64>> scoreMatrixPtr =
            std::make_unique<DenseNumericMatrix<float64>>(numExamples, numLabels, true);
        const IExampleWiseLoss* lossFunctionRawPtr = lossFunctionPtr.get();
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
            lossFunctionPtr, ruleEvaluationFactoryPtr, labelMatrix, std::move(statisticMatrixPtr),
            std::move(scoreMatrixPtr));
    }

    DenseExampleWiseStatisticsFactory::DenseExampleWiseStatisticsFactory(
            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
          numThreads_(numThreads) {

    }

    std::unique_ptr<IExampleWiseStatistics> DenseExampleWiseStatisticsFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        return createInternally<CContiguousLabelMatrix>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, numThreads_,
                                                        labelMatrix);
    }

    std::unique_ptr<IExampleWiseStatistics> DenseExampleWiseStatisticsFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        return createInternally<CsrLabelMatrix>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, numThreads_, labelMatrix);
    }

}
