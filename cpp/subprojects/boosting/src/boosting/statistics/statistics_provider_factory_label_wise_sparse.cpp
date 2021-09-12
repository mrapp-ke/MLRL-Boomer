#include "boosting/statistics/statistics_provider_factory_label_wise_sparse.hpp"
#include "common/validation.hpp"
#include "statistics_provider_label_wise.hpp"


namespace boosting {

    SparseLabelWiseStatisticsProviderFactory::SparseLabelWiseStatisticsProviderFactory(
            std::unique_ptr<ISparseLabelWiseLoss> lossFunctionPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
            std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads)
        : lossFunctionPtr_(std::move(lossFunctionPtr)),
          defaultRuleEvaluationFactoryPtr_(std::move(defaultRuleEvaluationFactoryPtr)),
          regularRuleEvaluationFactoryPtr_(std::move(regularRuleEvaluationFactoryPtr)),
          pruningRuleEvaluationFactoryPtr_(std::move(pruningRuleEvaluationFactoryPtr)), numThreads_(numThreads) {
        assertNotNull("lossFunctionPtr", lossFunctionPtr_.get());
        assertNotNull("defaultRuleEvaluationFactoryPtr", defaultRuleEvaluationFactoryPtr_.get());
        assertNotNull("regularRuleEvaluationFactoryPtr", regularRuleEvaluationFactoryPtr_.get());
        assertNotNull("pruningRuleEvaluationFactoryPtr", pruningRuleEvaluationFactoryPtr_.get());
        assertGreaterOrEqual<uint32>("numThreads", numThreads, 1);
    }

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
            const CContiguousLabelMatrix& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics> statisticsPtr; // TODO Initialize
        return std::make_unique<LabelWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                             *pruningRuleEvaluationFactoryPtr_,
                                                             std::move(statisticsPtr));
    }

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        std::unique_ptr<ILabelWiseStatistics> statisticsPtr; // TODO Initialize
        return std::make_unique<LabelWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                             *pruningRuleEvaluationFactoryPtr_,
                                                             std::move(statisticsPtr));
    }

}
