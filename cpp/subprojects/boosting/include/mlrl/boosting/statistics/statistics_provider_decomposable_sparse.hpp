/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_decomposable_sparse.hpp"
#include "mlrl/boosting/rule_evaluation/rule_evaluation_decomposable_sparse.hpp"
#include "mlrl/boosting/statistics/quantization.hpp"
#include "mlrl/boosting/statistics/statistics_decomposable.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to create instances of the class `IStatisticsProvider` that can be used in classification problems and
     * provide access to an object of type `IDecomposableStatistics` using sparse data structures for storing the
     * statistics.
     *
     * @tparam StatisticType The type of the statistics
     */
    template<typename StatisticType>
    class SparseDecomposableClassificationStatisticsProviderFactory final
        : public IClassificationStatisticsProviderFactory {
        private:

            const std::unique_ptr<IQuantizationFactory> quantizationFactoryPtr_;

            const std::unique_ptr<ISparseDecomposableClassificationLossFactory<StatisticType>> lossFactoryPtr_;

            const std::unique_ptr<ISparseEvaluationMeasureFactory<StatisticType>> evaluationMeasureFactoryPtr_;

            const std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            const std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param quantizationFactoryPtr            An unique pointer to an object of type `IQuantizationFactory`
             *                                          that allows to create implementations of the method that should
             *                                          be used for quantizing gradients and Hessians
             * @param lossFactoryPtr                    An unique pointer to an object of type
             *                                          `ISparseDecomposableClassificationLossFactory` that allows to
             *                                          create implementations of the loss function that should be used
             *                                          for calculating gradients and Hessians
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
             * @param multiThreadingSettings            An object of type `MultiThreadingSettings` that stores the
             *                                          settings to be used for calculating the initial statistics in
             *                                          parallel
             */
            SparseDecomposableClassificationStatisticsProviderFactory(
              std::unique_ptr<IQuantizationFactory> quantizationFactoryPtr,
              std::unique_ptr<ISparseDecomposableClassificationLossFactory<StatisticType>> lossFactoryPtr,
              std::unique_ptr<ISparseEvaluationMeasureFactory<StatisticType>> evaluationMeasureFactoryPtr,
              std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
              std::unique_ptr<ISparseDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
              MultiThreadingSettings multiThreadingSettings);

            /**
             * @see `IClassificationStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(const CContiguousView<const uint8>& labelMatrix) const override;

            /**
             * @see `IClassificationStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(const BinaryCsrView& labelMatrix) const override;
    };

}
