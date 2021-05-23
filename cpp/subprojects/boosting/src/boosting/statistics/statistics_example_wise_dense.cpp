#include "boosting/statistics/statistics_example_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/statistic_vector_dense_example_wise.hpp"
#include "boosting/data/statistic_view_dense_example_wise.hpp"
#include "boosting/math/math.hpp"
#include "statistics_example_wise_common.hpp"
#include "omp.h"
#include <cstdlib>


namespace boosting {

    class DenseExampleWiseStatisticMatrix : public DenseExampleWiseStatisticView {

        public:

            DenseExampleWiseStatisticMatrix(uint32 numRows, uint32 numGradients)
                : DenseExampleWiseStatisticView(
                      numRows, numGradients, triangularNumber(numGradients),
                      (float64*) malloc(numRows * numGradients * sizeof(float64)),
                      (float64*) malloc(numRows * triangularNumber(numGradients) * sizeof(float64))) {

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

        return std::make_unique<ExampleWiseStatistics<LabelMatrix, DenseExampleWiseStatisticVector,
                                                      DenseExampleWiseStatisticView, DenseExampleWiseStatisticMatrix,
                                                      DenseNumericMatrix<float64>>>(
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
