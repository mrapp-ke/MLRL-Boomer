#include "boosting/statistics/statistics_provider_factory_label_wise_sparse.hpp"
#include "common/validation.hpp"


namespace boosting {

    SparseLabelWiseStatisticsProviderFactory::SparseLabelWiseStatisticsProviderFactory(
            std::unique_ptr<ISparseLabelWiseLoss> lossFunctionPtr,
            std::unique_ptr<ISparseEvaluationMeasure> evaluationMeasurePtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(std::move(lossFunctionPtr)), evaluationMeasurePtr_(std::move(evaluationMeasurePtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {
        assertNotNull("lossFunctionPtr", lossFunctionPtr_.get());
        assertNotNull("evaluationMeasurePtr", evaluationMeasurePtr_.get());
        assertNotNull("defaultRuleEvaluationFactoryPtr", defaultRuleEvaluationFactoryPtr_.get());
        assertNotNull("regularRuleEvaluationFactoryPtr", regularRuleEvaluationFactoryPtr_.get());
        assertNotNull("pruningRuleEvaluationFactoryPtr", pruningRuleEvaluationFactoryPtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        // TODO Implement
        return nullptr;
    }

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        // TODO Implement
        return nullptr;
    }

}
