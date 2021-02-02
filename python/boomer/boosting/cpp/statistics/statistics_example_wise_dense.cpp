#include "statistics_example_wise_dense.h"
#include "statistics_example_wise_common.h"
#include "../data/matrix_dense_numeric.h"
#include "../data/matrix_dense_example_wise.h"
#include "../data/vector_dense_example_wise.h"


namespace boosting {

    DenseExampleWiseStatisticsFactory::DenseExampleWiseStatisticsFactory(
            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
            std::shared_ptr<IExampleWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr)
        : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
          labelMatrixPtr_(labelMatrixPtr) {

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
