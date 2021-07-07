#include "seco/statistics/statistics_label_wise_dense.hpp"
#include "seco/data/matrix_dense_weights.hpp"
#include "seco/data/vector_dense_confusion_matrices.hpp"
#include "statistics_label_wise_common.hpp"
#include "statistics_label_wise_provider.hpp"


namespace seco {

    DenseLabelWiseStatisticsFactory::DenseLabelWiseStatisticsFactory(
            const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory)
        : ruleEvaluationFactory_(ruleEvaluationFactory) {

    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseWeightMatrix> weightMatrixPtr = std::make_unique<DenseWeightMatrix>(
            numExamples, numLabels);
        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr = std::make_unique<BinarySparseArrayVector>(
            numLabels);
        BinarySparseArrayVector::index_iterator majorityIterator = majorityLabelVectorPtr->indices_begin();
        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = 0;

            for (uint32 j = 0; j < numExamples; j++) {
                uint8 trueLabel = labelMatrix.row_values_cbegin(j)[i];
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
            ruleEvaluationFactory_, labelMatrix, std::move(weightMatrixPtr), std::move(majorityLabelVectorPtr));
    }

    std::unique_ptr<ILabelWiseStatistics> DenseLabelWiseStatisticsFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        uint32 numExamples = labelMatrix.getNumRows();
        uint32 numLabels = labelMatrix.getNumCols();
        std::unique_ptr<DenseWeightMatrix> weightMatrixPtr = std::make_unique<DenseWeightMatrix>(
            numExamples, numLabels);
        std::unique_ptr<BinarySparseArrayVector> majorityLabelVectorPtr = std::make_unique<BinarySparseArrayVector>(
            numLabels, true);
        BinarySparseArrayVector::index_iterator majorityIterator = majorityLabelVectorPtr->indices_begin();

        for (uint32 i = 0; i < numExamples; i++) {
            CsrLabelMatrix::index_const_iterator indexIterator = labelMatrix.row_indices_cbegin(i);
            uint32 numElements = labelMatrix.row_indices_cend(i) - indexIterator;

            for (uint32 j = 0; j < numElements; j++) {
                uint32 index = indexIterator[j];
                majorityIterator[index] += 1;
            }
        }

        float64 threshold = numExamples / 2.0;
        float64 sumOfUncoveredWeights = 0;
        uint32 n = 0;

        for (uint32 i = 0; i < numLabels; i++) {
            uint32 numRelevant = majorityIterator[i];

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
        return std::make_unique<LabelWiseStatistics<CsrLabelMatrix, DenseWeightMatrix, DenseConfusionMatrixVector>>(
            ruleEvaluationFactory_, labelMatrix, std::move(weightMatrixPtr), std::move(majorityLabelVectorPtr));
    }


    DenseLabelWiseStatisticsProviderFactory::DenseLabelWiseStatisticsProviderFactory(
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
            std::shared_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr)
        : defaultRuleEvaluationFactoryPtr_(defaultRuleEvaluationFactoryPtr),
          regularRuleEvaluationFactoryPtr_(regularRuleEvaluationFactoryPtr),
          pruningRuleEvaluationFactoryPtr_(pruningRuleEvaluationFactoryPtr) {

    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        DenseLabelWiseStatisticsFactory statisticsFactory(*defaultRuleEvaluationFactoryPtr_);
        return std::make_unique<LabelWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                             *pruningRuleEvaluationFactoryPtr_,
                                                             statisticsFactory.create(labelMatrix));
    }

    std::unique_ptr<IStatisticsProvider> DenseLabelWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        DenseLabelWiseStatisticsFactory statisticsFactory(*defaultRuleEvaluationFactoryPtr_);
        return std::make_unique<LabelWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                             *pruningRuleEvaluationFactoryPtr_,
                                                             statisticsFactory.create(labelMatrix));
    }

}
