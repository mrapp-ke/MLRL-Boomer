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
        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr = std::make_unique<BinarySparseArrayVector>(
            numLabels);
        BinarySparseArrayVector::index_iterator majorityIterator = majorityLabelVectorPtr->indices_begin();
        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = 0;

            for (uint32 j = 0; j < numExamples; j++) {
                uint8 trueLabel = labelMatrixPtr_->getValue(j, i);
                numRelevant += trueLabel;
            }

            if (numRelevant > threshold) {
                sumOfUncoveredWeights += (numExamples - numRelevant);
                majorityIterator[n] = i;
                n++;
            } else {
                sumOfUncoveredWeights += numRelevant;
            }
        }

        majorityLabelVectorPtr->setNumElements(n, true);
        weightMatrixPtr->setSumOfUncoveredWeights(sumOfUncoveredWeights);
        return std::make_unique<LabelWiseStatistics<IRandomAccessLabelMatrix, DenseWeightMatrix, DenseConfusionMatrixVector>>(
            ruleEvaluationFactoryPtr_, labelMatrixPtr_, std::move(weightMatrixPtr), std::move(majorityLabelVectorPtr));
    }

}
