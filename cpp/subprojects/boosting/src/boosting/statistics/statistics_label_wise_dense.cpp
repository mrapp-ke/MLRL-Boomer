#include "boosting/statistics/statistics_label_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/statistic_vector_dense_label_wise.hpp"
#include "boosting/data/statistic_view_dense_label_wise.hpp"
#include "statistics_label_wise_common.hpp"
#include "statistics_label_wise_provider.hpp"
#include "omp.h"
#include <cstdlib>


namespace boosting {

    /**
     * A matrix that stores gradients and Hessians that have been calculated using a label-wise decomposable loss
     * function using C-contiguous arrays.
     */
    class DenseLabelWiseStatisticMatrix final : public DenseLabelWiseStatisticView {

        public:

            /**
             * @param numRows   The number of rows in the matrix
             * @param numCols   The number of columns in the matrix
             */
            DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols)
                : DenseLabelWiseStatisticView(numRows, numCols,
                                              (Tuple<float64>*) malloc(numRows * numCols * sizeof(Tuple<float64>))) {

            }

            ~DenseLabelWiseStatisticMatrix() {
                free(statistics_);
            }

    };

    /**
     * Provides access to gradients and Hessians that have been calculated according to a differentiable loss function
     * that is applied label-wise and are stored using dense data structures.
     *
     * @tparam LabelMatrix The type of the matrix that provides access to the labels of the training examples
     */
    template<typename LabelMatrix>
    class DenseLabelWiseStatistics final : public AbstractLabelWiseStatistics<LabelMatrix,
                                                                              DenseLabelWiseStatisticVector,
                                                                              DenseLabelWiseStatisticView,
                                                                              DenseLabelWiseStatisticMatrix,
                                                                              NumericDenseMatrix<float64>> {

        public:

            /**
             * @param lossFunctionPtr           A shared pointer to an object of type `ILabelWiseLoss`, representing the
             *                                  loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactoryPtr  A shared pointer to an object of type `ILabelWiseRuleEvaluationFactory`,
             *                                  that allows to create instances of the class that is used for
             *                                  calculating the predictions, as well as corresponding quality scores, of
             *                                  rules
             * @param labelMatrix               A reference to an object of template type `LabelMatrix` that provides
             *                                  access to the labels of the training examples
             * @param statisticViewPtr          An unique pointer to an object of type `DenseLabelWiseStatisticView`
             *                                  that provides access to the gradients and Hessians
             * @param scoreMatrixPtr            An unique pointer to an object of type `NumericDenseMatrix` that stores
             *                                  the currently predicted scores
             */
            DenseLabelWiseStatistics(std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
                                     std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
                                     const LabelMatrix& labelMatrix,
                                     std::unique_ptr<DenseLabelWiseStatisticView> statisticViewPtr,
                                     std::unique_ptr<NumericDenseMatrix<float64>> scoreMatrixPtr)
                : AbstractLabelWiseStatistics<LabelMatrix, DenseLabelWiseStatisticVector, DenseLabelWiseStatisticView,
                                              DenseLabelWiseStatisticMatrix, NumericDenseMatrix<float64>>(
                      lossFunctionPtr, ruleEvaluationFactoryPtr, labelMatrix, std::move(statisticViewPtr),
                      std::move(scoreMatrixPtr)) {

            }

            void visit(IBoostingStatistics::DenseLabelWiseStatisticViewVisitor denseLabelWiseStatisticViewVisitor,
                       IBoostingStatistics::DenseExampleWiseStatisticViewVisitor denseExampleWiseStatisticViewVisitor) override {
                denseLabelWiseStatisticViewVisitor(this->statisticViewPtr_);
            }

    };

    template<typename LabelMatrix>
    static inline std::unique_ptr<ILabelWiseStatistics> createInternally(
            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads,
            const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseLabelWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<NumericDenseMatrix<float64>> scoreMatrixPtr =
            std::make_unique<NumericDenseMatrix<float64>>(numExamples, numLabels, true);
        const ILabelWiseLoss* lossFunctionRawPtr = lossFunctionPtr.get();
        const LabelMatrix* labelMatrixPtr = &labelMatrix;
        const CContiguousConstView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseLabelWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossFunctionRawPtr) \
        firstprivate(labelMatrixPtr) firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) \
        schedule(dynamic) num_threads(numThreads)
        for (uint32 i = 0; i < numExamples; i++) {
            lossFunctionRawPtr->updateLabelWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                          IndexIterator(labelMatrixPtr->getNumCols()),
                                                          *statisticMatrixRawPtr);
        }

        return std::make_unique<DenseLabelWiseStatistics<LabelMatrix>>(lossFunctionPtr, ruleEvaluationFactoryPtr,
                                                                       labelMatrix, std::move(statisticMatrixPtr),
                                                                       std::move(scoreMatrixPtr));
    }

    DenseLabelWiseStatisticsFactory::DenseLabelWiseStatisticsFactory(
            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
          numThreads_(numThreads) {

    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        return createInternally<CContiguousLabelMatrix>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, numThreads_,
                                                        labelMatrix);
    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        return createInternally<CsrLabelMatrix>(lossFunctionPtr_, ruleEvaluationFactoryPtr_, numThreads_, labelMatrix);
    }

    DenseLabelWiseStatisticsProviderFactory::DenseLabelWiseStatisticsProviderFactory(
            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), defaultRuleEvaluationFactoryPtr_(defaultRuleEvaluationFactoryPtr),
          ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), numThreads_(numThreads) {

    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        DenseLabelWiseStatisticsFactory statisticsFactory(lossFunctionPtr_, defaultRuleEvaluationFactoryPtr_,
                                                          numThreads_);
        return std::make_unique<LabelWiseStatisticsProvider>(ruleEvaluationFactoryPtr_,
                                                             statisticsFactory.create(labelMatrix));
    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        DenseLabelWiseStatisticsFactory statisticsFactory(lossFunctionPtr_, defaultRuleEvaluationFactoryPtr_,
                                                          numThreads_);
        return std::make_unique<LabelWiseStatisticsProvider>(ruleEvaluationFactoryPtr_,
                                                             statisticsFactory.create(labelMatrix));
    }


}
