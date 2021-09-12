#include "boosting/statistics/statistics_provider_factory_label_wise_sparse.hpp"
#include "common/validation.hpp"
#include "statistics_provider_label_wise.hpp"


namespace boosting {

    /**
     * A factory that allows to create new instances of the class `ILabelWiseStatistics` that use sparse data structures
     * to store the statistics.
     */
    class SparseLabelWiseStatisticsFactory final : public ILabelWiseStatisticsFactory {

        private:

            const ISparseLabelWiseLoss& lossFunction_;

            const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFunction          A reference to an object of type `ISparseLabelWiseLoss`, representing the
             *                              loss function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `ILabelWiseRuleEvaluationFactory` that
             *                              allows to create instances of the class that is used to calculate the
             *                              predictions, as well as corresponding quality scores, of rules
             * @param numThreads            The number of CPU threads to be used to calculate the initial statistics in
             *                              parallel. Must be at least 1
             */
            SparseLabelWiseStatisticsFactory(const ISparseLabelWiseLoss& lossFunction,
                                             const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                             uint32 numThreads)
                : lossFunction_(lossFunction), ruleEvaluationFactory_(ruleEvaluationFactory), numThreads_(numThreads) {

            }

            std::unique_ptr<ILabelWiseStatistics> create(const CContiguousLabelMatrix& labelMatrix) const override {
                // TODO Implement
                return nullptr;
            }

            std::unique_ptr<ILabelWiseStatistics> create(const CsrLabelMatrix& labelMatrix) const override {
                // TODO Implement
                return nullptr;
            }

    };

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
        SparseLabelWiseStatisticsFactory statisticsFactory(*lossFunctionPtr_, *defaultRuleEvaluationFactoryPtr_,
                                                           numThreads_);
        return std::make_unique<LabelWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                             *pruningRuleEvaluationFactoryPtr_,
                                                             statisticsFactory.create(labelMatrix));
    }

    std::unique_ptr<IStatisticsProvider> SparseLabelWiseStatisticsProviderFactory::create(
            const CsrLabelMatrix& labelMatrix) const {
        SparseLabelWiseStatisticsFactory statisticsFactory(*lossFunctionPtr_, *defaultRuleEvaluationFactoryPtr_,
                                                           numThreads_);
        return std::make_unique<LabelWiseStatisticsProvider>(*regularRuleEvaluationFactoryPtr_,
                                                             *pruningRuleEvaluationFactoryPtr_,
                                                             statisticsFactory.create(labelMatrix));
    }

}
