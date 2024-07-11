/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_decomposable_sparse.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_sparse.hpp"
#include "mlrl/boosting/statistics/statistics_decomposable.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to create instances of the class `IStatisticsProvider` that provide access to an object of type
     * `IDecomposableStatistics`, which uses sparse data structures to store the statistics.
     */
    class SparseDecomposableStatisticsProviderFactory final : public IClassificationStatisticsProviderFactory,
                                                              public IRegressionStatisticsProviderFactory {
        private:

            const std::unique_ptr<ISparseDecomposableLossFactory> lossFactoryPtr_;

            const std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr_;

            const std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            const std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            const uint32 numThreads_;

        public:

            /**
             * @param lossFactoryPtr                    An unique pointer to an object of type
             *                                          `ISparseDecomposableLossFactory` that allows to create
             *                                          implementations of the loss function that should be used for
             *                                          calculating gradients and Hessians
             * @param evaluationMeasureFactoryPtr       An unique pointer to an object of type
             *                                          `ISparseEvaluationMeasureFactory` that allows to create
             *                                          implementations of the evaluation measure that should be used
             *                                          for assessing the quality of predictions
             * @param regularRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `ISparseDecomposableRuleEvaluationFactory` that should be used
             *                                          for calculating the predictions, as well as corresponding
             *                                          quality scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `ISparseDecomposableRuleEvaluationFactory` that should be used
             *                                          for calculating the predictions, as well as corresponding
             *                                          quality scores, when pruning rules
             * @param numThreads                        The number of CPU threads to be used to calculate the initial
             *                                          statistics in parallel. Must be at least 1
             */
            SparseDecomposableStatisticsProviderFactory(
              std::unique_ptr<ISparseDecomposableLossFactory> lossFactoryPtr,
              std::unique_ptr<ISparseEvaluationMeasureFactory> evaluationMeasureFactoryPtr,
              std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
              std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
              uint32 numThreads);

            /**
             * @see `IClassificationStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(const CContiguousView<const uint8>& labelMatrix) const override;

            /**
             * @see `IClassificationStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(const BinaryCsrView& labelMatrix) const override;

            /**
             * @see `IClassificationStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(
              const CContiguousView<const float32>& regressionMatrix) const override;

            /**
             * @see `IClassificationStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(const CsrView<const float32>& regressionMatrix) const override;
    };

}
