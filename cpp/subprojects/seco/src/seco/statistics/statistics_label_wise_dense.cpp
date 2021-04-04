#include "seco/statistics/statistics_label_wise_dense.hpp"
#include "seco/data/matrix_dense_weights.hpp"
#include "seco/data/vector_dense_confusion_matrices.hpp"
#include "statistics_label_wise_common.hpp"


namespace seco {

    DenseLabelWiseStatisticsFactory::DenseLabelWiseStatisticsFactory(
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
            std::shared_ptr<IRandomAccessLabelMatrix> labelMatrixPtr)
        : ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), labelMatrixPtr_(labelMatrixPtr) {

    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create() const {
        uint32 numExamples = labelMatrixPtr_->getNumRows();
        uint32 numLabels = labelMatrixPtr_->getNumCols();
        std::unique_ptr<DenseWeightMatrix> weightMatrixPtr = std::make_unique<DenseWeightMatrix>(numExamples,
                                                                                                 numLabels);
        std::unique_ptr<DenseVector<uint8>> majorityLabelVectorPtr = std::make_unique<DenseVector<uint8>>(numLabels);
        DenseVector<uint8>::iterator majorityIterator = majorityLabelVectorPtr->begin();
        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = 0;

            for (uint32 j = 0; j < numExamples; j++) {
                uint8 trueLabel = labelMatrixPtr_->getValue(j, i);
                numRelevant += trueLabel;
            }

            if (numRelevant > threshold) {
                majorityIterator[i] = 1;
                sumOfUncoveredWeights += (numExamples - numRelevant);
            } else {
                majorityIterator[i] = 0;
                sumOfUncoveredWeights += numRelevant;
            }
        }

        weightMatrixPtr->setSumOfUncoveredWeights(sumOfUncoveredWeights);
        return std::make_unique<LabelWiseStatistics<DenseWeightMatrix, DenseConfusionMatrixVector>>(
            ruleEvaluationFactoryPtr_, labelMatrixPtr_, std::move(weightMatrixPtr), std::move(majorityLabelVectorPtr));
    }

}
