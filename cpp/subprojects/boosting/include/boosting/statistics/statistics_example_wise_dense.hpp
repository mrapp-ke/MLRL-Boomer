/*
 * @author Michael Rapp (mrapp@ke.tu-darmstadt.de)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "boosting/statistics/statistics_example_wise.hpp"
#include "common/statistics/statistics_provider_factory.hpp"
#include "boosting/losses/loss_example_wise.hpp"


namespace boosting {

    /**
     * A factory that allows to create new instances of the class `IExampleWiseStatistics` that used dense data
     * structures to store the statistics.
     */
    class DenseExampleWiseStatisticsFactory final : public IExampleWiseStatisticsFactory {

        private:

            const IExampleWiseLoss& lossFunction_;

            const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFunction          A reference to an object of type `IExampleWiseLoss`, representing the loss
             *                              function to be used for calculating gradients and Hessians
             * @param ruleEvaluationFactory A reference to an object of type `IExampleWiseRuleEvaluationFactory`, to be
             *                              used for calculating the predictions, as well as corresponding quality
             *                              scores, of rules
             * @param numThreads            The number of CPU threads to be used to calculate the initial statistics in
             *                              parallel. Must be at least 1
             */
            DenseExampleWiseStatisticsFactory(const IExampleWiseLoss& lossFunction,
                                              const IExampleWiseRuleEvaluationFactory& ruleEvaluationFactory,
                                              uint32 numThreads);

            std::unique_ptr<IExampleWiseStatistics> create(const CContiguousLabelMatrix& labelMatrix) const override;

            std::unique_ptr<IExampleWiseStatistics> create(const CsrLabelMatrix& labelMatrix) const override;

    };

    /**
     * Allows to create instances of the class `IStatisticsProvider` that provide access to an object of type
     * `IExampleWiseStatistics`, which uses dense data structures to store the statistics.
     */
    class DenseExampleWiseStatisticsProviderFactory: public IStatisticsProviderFactory {

        private:

            std::shared_ptr<IExampleWiseLoss> lossFunctionPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            std::shared_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFunctionPtr                   A shared pointer to an object of type `IExampleWiseLoss` that
             *                                          should be used for calculating gradients and Hessians
             * @param defaultRuleEvaluationFactoryPtr   A shared pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param regularRuleEvaluationFactoryPtr   A shared pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   A shared pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, when pruning rules
             * @param numThreads                        The number of CPU threads to be used to calculate the initial
             *                                          statistics in parallel. Must be at least 1
             */
            DenseExampleWiseStatisticsProviderFactory(
                std::shared_ptr<IExampleWiseLoss> lossFunctionPtr,
                std::shared_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
                std::shared_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
                std::shared_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads);

            std::unique_ptr<IStatisticsProvider> create(const CContiguousLabelMatrix& labelMatrix) const override;

            std::unique_ptr<IStatisticsProvider> create(const CsrLabelMatrix& labelMatrix) const override;

    };

}
