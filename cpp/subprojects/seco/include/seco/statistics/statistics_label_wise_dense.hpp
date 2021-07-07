/*
 * @author Jakob Steeg (jakob.steeg@gmail.com)
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_provider_factory.hpp"
#include "seco/statistics/statistics_label_wise.hpp"


namespace seco {

    /**
     * A factory that allows to create new instances of the class `ILabelWiseStatistics` that use dense data structures
     * to store the statistics.
     */
    class DenseLabelWiseStatisticsFactory final : public ILabelWiseStatisticsFactory {

        private:

            const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory_;

        public:

            /**
             * @param ruleEvaluationFactory A reference to an object of type `ILabelWiseRuleEvaluationFactory` that
             *                              allows to create instances of the class that is used for calculating the
             *                              predictions, as well as corresponding quality scores, of rules
             */
            DenseLabelWiseStatisticsFactory(const ILabelWiseRuleEvaluationFactory& ruleEvaluationFactory);

            std::unique_ptr<ILabelWiseStatistics> create(const CContiguousLabelMatrix& labelMatrix) const override;

            std::unique_ptr<ILabelWiseStatistics> create(const CsrLabelMatrix& labelMatrix) const override;

    };

    /**
     * Allows to create instances of the class `IStatisticsProvider` that provide access to an object of type
     * `ILabelWiseStatistics`, which uses dense data structures to store the statistics.
     */
    class DenseLabelWiseStatisticsProviderFactory : public IStatisticsProviderFactory {

        private:

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            std::shared_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

        public:

            /**
             * @param defaultRuleEvaluationFactoryPtr   A shared pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param regularRuleEvaluationFactoryPtr   A shared pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   A shared pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, when pruning rules
             */
            DenseLabelWiseStatisticsProviderFactory(
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
                std::shared_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr);

            std::unique_ptr<IStatisticsProvider> create(const CContiguousLabelMatrix& labelMatrix) const override;

            std::unique_ptr<IStatisticsProvider> create(const CsrLabelMatrix& labelMatrix) const override;

    };

}
