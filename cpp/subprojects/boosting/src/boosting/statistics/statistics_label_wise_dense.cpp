#include "boosting/statistics/statistics_label_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/matrix_dense_label_wise.hpp"
#include "boosting/data/vector_dense_label_wise.hpp"
#include "statistics_label_wise_common.hpp"
#include "omp.h"


namespace boosting {

    template<class LabelMatrix>
    DenseLabelWiseStatisticsFactory<LabelMatrix>::DenseLabelWiseStatisticsFactory(
            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, const LabelMatrix& labelMatrix,
            uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
          labelMatrix_(labelMatrix), numThreads_(numThreads) {

    }

    template<class LabelMatrix>
    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory<LabelMatrix>::create() const {
        uint32 numExamples = labelMatrix_.getNumRows();
        uint32 numLabels = labelMatrix_.getNumCols();
        std::unique_ptr<DenseLabelWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<DenseNumericMatrix<float64>> scoreMatrixPtr =
            std::make_unique<DenseNumericMatrix<float64>>(numExamples, numLabels, true);
        const ILabelWiseLoss* lossFunctionPtr = lossFunctionPtr_.get();
        const LabelMatrix* labelMatrixPtr = &labelMatrix_;
        const CContiguousView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseLabelWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossFunctionPtr) firstprivate(labelMatrixPtr) \
        firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) \
        schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            lossFunctionPtr->updateLabelWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr, IndexIterator(),
                                                       IndexIterator(labelMatrixPtr->getNumCols()),
                                                       *statisticMatrixRawPtr);
        }

        return std::make_unique<LabelWiseStatistics<LabelMatrix, DenseLabelWiseStatisticVector, DenseLabelWiseStatisticMatrix, DenseNumericMatrix<float64>>>(
            lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrix_, std::move(statisticMatrixPtr),
            std::move(scoreMatrixPtr));
    }

    template class DenseLabelWiseStatisticsFactory<CContiguousLabelMatrix>;
    template class DenseLabelWiseStatisticsFactory<CsrLabelMatrix>;

}
