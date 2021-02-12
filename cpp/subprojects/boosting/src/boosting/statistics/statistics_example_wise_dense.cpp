#include "boosting/statistics/statistics_example_wise_dense.hpp"
#include "boosting/data/matrix_dense_numeric.hpp"
#include "boosting/data/matrix_dense_example_wise.hpp"
#include "boosting/data/vector_dense_example_wise.hpp"
#include "statistics_example_wise_common.hpp"


namespace boosting {

    DenseExampleWiseStatisticsFactory::DenseExampleWiseStatisticsFactory(
            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr, uint32 numThreads)
        : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
          labelMatrixPtr_(labelMatrixPtr), numThreads_(numThreads) {

    }

    std::unique_ptr<IExampleWiseStatistics> DenseExampleWiseStatisticsFactory::create() const {
        uint32 numExamples = labelMatrixPtr_->getNumRows();
        uint32 numLabels = labelMatrixPtr_->getNumCols();
        std::unique_ptr<DenseExampleWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseExampleWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<DenseNumericMatrix<float64>> scoreMatrixPtr =
            std::make_unique<DenseNumericMatrix<float64>>(numExamples, numLabels, true);

        for (uint32 r = 0; r < numExamples; r++) {
            lossFunctionPtr_->updateExampleWiseStatistics(r, *labelMatrixPtr_, *scoreMatrixPtr, *statisticMatrixPtr);
        }

        return std::make_unique<ExampleWiseStatistics<DenseExampleWiseStatisticVector, DenseExampleWiseStatisticMatrix, DenseNumericMatrix<float64>>>(
            lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrixPtr_, std::move(statisticMatrixPtr),
            std::move(scoreMatrixPtr));
    }

}
