#include "boosting/statistics/statistics_label_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/statistic_vector_dense_label_wise.hpp"
#include "boosting/data/statistic_view_dense_label_wise.hpp"
#include "statistics_label_wise_common.hpp"
#include "omp.h"
#include <cstdlib>


namespace boosting {

    class DenseLabelWiseStatisticMatrix final : public DenseLabelWiseStatisticView {

        public:

            DenseLabelWiseStatisticMatrix(uint32 numRows, uint32 numCols)
                : DenseLabelWiseStatisticView(numRows, numCols, (float64*) malloc(numRows * numCols * sizeof(float64)),
                                              (float64*) malloc(numRows * numCols * sizeof(float64))) {

            }

            ~DenseLabelWiseStatisticMatrix() {
                free(gradients_);
                free(hessians_);
            }

    };

    template<class LabelMatrix>
    static inline std::unique_ptr<ILabelWiseStatistics> createInternally(
            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, uint32 numThreads,
            const LabelMatrix& labelMatrix) {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseLabelWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<DenseNumericMatrix<float64>> scoreMatrixPtr =
            std::make_unique<DenseNumericMatrix<float64>>(numExamples, numLabels, true);
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

        return std::make_unique<LabelWiseStatistics<LabelMatrix, DenseLabelWiseStatisticVector,
                                                    DenseLabelWiseStatisticView, DenseLabelWiseStatisticMatrix,
                                                    DenseNumericMatrix<float64>>>(
            lossFunctionPtr, ruleEvaluationFactoryPtr, labelMatrix, std::move(statisticMatrixPtr),
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

}
