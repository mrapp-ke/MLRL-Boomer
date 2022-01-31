/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "common/statistics/statistics_provider.hpp"
#include "boosting/statistics/statistics_example_wise.hpp"
#include "boosting/losses/loss_example_wise.hpp"


namespace boosting {

    /**
     * Allows to create instances of the class `IStatisticsProvider` that provide access to an object of type
     * `IExampleWiseStatistics`, which uses dense data structures to store the statistics.
     */
    class DenseExampleWiseStatisticsProviderFactory final : public IStatisticsProviderFactory {

        private:

            std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr_;

            std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr_;

            std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            std::unique_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            std::unique_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFactoryPtr                    An unique pointer to an object of type `IExampleWiseLossFactory`
             *                                          that allows to create implementations of the loss function that
             *                                          should be used for calculating gradients and Hessians
             * @param evaluationMeasureFactoryPtr       An unique pointer to an object of type
             *                                          `IEvaluationMeasureFactory` that allows to create
             *                                          implementations of the evaluation measure that should be used
             *                                          for assessing the quality of predictions
             * @param defaultRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param regularRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, when pruning rules
             * @param numThreads                        The number of CPU threads to be used to calculate the initial
             *                                          statistics in parallel. Must be at least 1
             */
            DenseExampleWiseStatisticsProviderFactory(
                std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr,
                std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
                std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
                std::unique_ptr<IExampleWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
                std::unique_ptr<IExampleWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads);

            /**
             * @see `IStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(
                const CContiguousConstView<const uint8>& labelMatrix) const override;

            /**
             * @see `IStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(const BinaryCsrConstView& labelMatrix) const override;

    };

    /**
     * Allows to create instances of the class `IStatisticsProvider` that provide access to an object of type
     * `IExampleWiseStatistics`, which uses dense data structures to store the statistics and can be converted into an
     * object of type `ILabelWiseStatistics`.
     */
    class DenseConvertibleExampleWiseStatisticsProviderFactory final : public IStatisticsProviderFactory {

        private:

            std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr_;

            std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr_;

            std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            uint32 numThreads_;

        public:

            /**
             * @param lossFactoryPtr                    An unique pointer to an object of type `IExampleWiseLossFactory`
             *                                          that allows to create implementations of the loss function that
             *                                          should be used for calculating gradients and Hessians
             * @param evaluationMeasureFactoryPtr       An unique pointer to an object of type
             *                                          `IEvaluationMeasureFactory` that allows to create
             *                                          implementations of the evaluation measure that should be used
             *                                          for assessing the quality of predictions
             * @param defaultRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `IExampleWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param regularRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `ILabelWiseRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, when pruning rules
             * @param numThreads                        The number of CPU threads to be used to calculate the initial
             *                                          statistics in parallel. Must be at least 1
             */
            DenseConvertibleExampleWiseStatisticsProviderFactory(
                std::unique_ptr<IExampleWiseLossFactory> lossFactoryPtr,
                std::unique_ptr<IEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
                std::unique_ptr<IExampleWiseRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
                std::unique_ptr<ILabelWiseRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
                std::unique_ptr<ILabelWiseRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr, uint32 numThreads);

            /**
             * @see `IStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(
                const CContiguousConstView<const uint8>& labelMatrix) const override;

            /**
             * @see `IStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(const BinaryCsrConstView& labelMatrix) const override;

    };

}
