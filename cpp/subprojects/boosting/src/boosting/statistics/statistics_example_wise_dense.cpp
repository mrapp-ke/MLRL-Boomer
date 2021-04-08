#include "boosting/statistics/statistics_example_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/matrix_dense_example_wise.hpp"
#include "boosting/data/vector_dense_example_wise.hpp"
#include "statistics_example_wise_common.hpp"
#include "omp.h"


namespace boosting {

    template<class LabelMatrix>
    DenseExampleWiseStatisticsFactory<LabelMatrix>::DenseExampleWiseStatisticsFactory(
            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr, const LabelMatrix& labelMatrix,
            uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
          labelMatrix_(labelMatrix), numThreads_(numThreads) {

    }

    template<class LabelMatrix>
    std::unique_ptr<IExampleWiseStatistics> DenseExampleWiseStatisticsFactory<LabelMatrix>::create() const {
        uint32 numExamples = labelMatrix_.getNumRows();
        uint32 numLabels = labelMatrix_.getNumCols();
        std::unique_ptr<DenseExampleWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseExampleWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<DenseNumericMatrix<float64>> scoreMatrixPtr =
            std::make_unique<DenseNumericMatrix<float64>>(numExamples, numLabels, true);
        const IExampleWiseLoss* lossFunctionPtr = lossFunctionPtr_.get();
        const LabelMatrix* labelMatrixPtr = &labelMatrix_;
        const CContiguousView<float64>* scoreMatrixRawPtr = scoreMatrixPtr.get();
        DenseExampleWiseStatisticMatrix* statisticMatrixRawPtr = statisticMatrixPtr.get();

        #pragma omp parallel for firstprivate(numExamples) firstprivate(lossFunctionPtr) firstprivate(labelMatrixPtr) \
        firstprivate(scoreMatrixRawPtr) firstprivate(statisticMatrixRawPtr) schedule(dynamic) num_threads(numThreads_)
        for (uint32 i = 0; i < numExamples; i++) {
            lossFunctionPtr->updateExampleWiseStatistics(i, *labelMatrixPtr, *scoreMatrixRawPtr,
                                                         *statisticMatrixRawPtr);
        }

        return std::make_unique<ExampleWiseStatistics<LabelMatrix, DenseExampleWiseStatisticVector, DenseExampleWiseStatisticMatrix, DenseNumericMatrix<float64>>>(
            lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrix_, std::move(statisticMatrixPtr),
            std::move(scoreMatrixPtr));
    }

    template class DenseExampleWiseStatisticsFactory<CContiguousLabelMatrix>;
    template class DenseExampleWiseStatisticsFactory<CsrLabelMatrix>;

}
