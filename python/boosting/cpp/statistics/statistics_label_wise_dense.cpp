#include "statistics_label_wise_dense.hpp"
#include "statistics_label_wise_common.hpp"
#include "../data/matrix_dense_numeric.hpp"
#include "../data/matrix_dense_label_wise.hpp"
#include "../data/vector_dense_label_wise.hpp"


namespace boosting {

    DenseLabelWiseStatisticsFactory::DenseLabelWiseStatisticsFactory(
            std::shared_ptr<ILabelWiseLoss> lossFunctionPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr)
        : lossFunctionPtr_(lossFunctionPtr), ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr),
          labelMatrixPtr_(labelMatrixPtr) {

    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create() const {
        uint32 numExamples = labelMatrixPtr_->getNumRows();
        uint32 numLabels = labelMatrixPtr_->getNumCols();
        std::unique_ptr<DenseLabelWiseStatisticMatrix> statisticMatrixPtr =
            std::make_unique<DenseLabelWiseStatisticMatrix>(numExamples, numLabels);
        std::unique_ptr<DenseNumericMatrix<float64>> scoreMatrixPtr =
            std::make_unique<DenseNumericMatrix<float64>>(numExamples, numLabels, true);
        FullIndexVector labelIndices(numLabels);
        FullIndexVector::const_iterator labelIndicesBegin = labelIndices.cbegin();
        FullIndexVector::const_iterator labelIndicesEnd = labelIndices.cend();

        for (uint32 r = 0; r < numExamples; r++) {
            lossFunctionPtr_->updateLabelWiseStatistics(r, *labelMatrixPtr_, *scoreMatrixPtr, labelIndicesBegin,
                                                        labelIndicesEnd, *statisticMatrixPtr);
        }

        return std::make_unique<LabelWiseStatistics<DenseLabelWiseStatisticVector, DenseLabelWiseStatisticMatrix, DenseNumericMatrix<float64>>>(
            lossFunctionPtr_, ruleEvaluationFactoryPtr_, labelMatrixPtr_, std::move(statisticMatrixPtr),
            std::move(scoreMatrixPtr));
    }

}
