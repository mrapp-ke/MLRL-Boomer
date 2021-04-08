#include "seco/statistics/statistics_label_wise_dense.hpp"
#include "seco/data/matrix_dense_weights.hpp"
#include "seco/data/vector_dense_confusion_matrices.hpp"
#include "statistics_label_wise_common.hpp"


namespace seco {

    DenseLabelWiseStatisticsFactory::DenseLabelWiseStatisticsFactory(
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> ruleEvaluationFactoryPtr,
            const CContiguousLabelMatrix& labelMatrix)
        : ruleEvaluationFactoryPtr_(ruleEvaluationFactoryPtr), labelMatrix_(labelMatrix) {

    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create() const {
        uint32 numExamples = labelMatrix_.getNumRows();
        uint32 numLabels = labelMatrix_.getNumCols();
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
                uint8 trueLabel = labelMatrix_.row_values_cbegin(j)[i];
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
        return std::make_unique<LabelWiseStatistics<CContiguousLabelMatrix, DenseWeightMatrix, DenseConfusionMatrixVector>>(
            ruleEvaluationFactoryPtr_, labelMatrix_, std::move(weightMatrixPtr), std::move(majorityLabelVectorPtr));
    }

}
