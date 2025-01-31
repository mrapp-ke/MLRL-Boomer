/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 * @author Lukas Johannes Eberle (lukasjohannes.eberle@stud.tu-darmstadt.de)
 */
#pragma once

#include "mlrl/boosting/losses/loss_non_decomposable.hpp"
#include "mlrl/boosting/statistics/statistics_non_decomposable.hpp"
#include "mlrl/common/multi_threading/multi_threading.hpp"
#include "mlrl/common/statistics/statistics_provider.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to create instances of the class `IStatisticsProvider` that can be used in classification problems and
     * provide access to an object of type `INonDecomposableStatistics` using dense data structures for storing the
     * statistics.
     */
    class DenseNonDecomposableClassificationStatisticsProviderFactory final
        : public IClassificationStatisticsProviderFactory {
        private:

            const std::unique_ptr<INonDecomposableClassificationLossFactory<float64>> lossFactoryPtr_;

            const std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr_;

            const std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            const std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            const std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param lossFactoryPtr                    An unique pointer to an object of type
             *                                          `INonDecomposableClassificationLossFactory` that allows to
             *                                          create implementations of the loss function that should be used
             *                                          for calculating gradients and Hessians
             * @param evaluationMeasureFactoryPtr       An unique pointer to an object of type
             *                                          `IClassificationEvaluationMeasureFactory` that allows to create
             *                                          implementations of the evaluation measure that should be used
             *                                          for assessing the quality of predictions
             * @param defaultRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `INonDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param regularRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `INonDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `INonDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, when pruning rules
             * @param multiThreadingSettings            An object of type `MultiThreadingSettings` that stores the
             *                                          settings to be used for calculating the initial statistics in
             *                                          parallel
             */
            DenseNonDecomposableClassificationStatisticsProviderFactory(
              std::unique_ptr<INonDecomposableClassificationLossFactory<float64>> lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr,
              std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
              std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
              std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
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

    /**
     * Allows to create instances of the class `IStatisticsProvider` that can be used in regression problems and provide
     * access to an object of type `INonDecomposableStatistics` using dense data structures for storing the statistics.
     */
    class DenseNonDecomposableRegressionStatisticsProviderFactory final : public IRegressionStatisticsProviderFactory {
        private:

            const std::unique_ptr<INonDecomposableRegressionLossFactory<float64>> lossFactoryPtr_;

            const std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr_;

            const std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            const std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            const std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param lossFactoryPtr                    An unique pointer to an object of type
             *                                          `INonDecomposableRegressionLossFactory` that allows to create
             *                                          implementations of the loss function that should be used for
             *                                          calculating gradients and Hessians
             * @param evaluationMeasureFactoryPtr       An unique pointer to an object of type
             *                                          `IRegressionEvaluationMeasureFactory` that allows to create
             *                                          implementations of the evaluation measure that should be used
             *                                          for assessing the quality of predictions
             * @param defaultRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `INonDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param regularRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `INonDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `INonDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, when pruning rules
             * @param multiThreadingSettings            An object of type `MultiThreadingSettings` that stores the
             *                                          settings to be used for calculating the initial statistics in
             *                                          parallel
             */
            DenseNonDecomposableRegressionStatisticsProviderFactory(
              std::unique_ptr<INonDecomposableRegressionLossFactory<float64>> lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr,
              std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
              std::unique_ptr<INonDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
              std::unique_ptr<INonDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
              MultiThreadingSettings multiThreadingSettings);

            /**
             * @see `IRegressionStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(
              const CContiguousView<const float32>& regressionMatrix) const override;

            /**
             * @see `IRegressionStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(const CsrView<const float32>& regressionMatrix) const override;
    };

    /**
     * Allows to create instances of the class `IStatisticsProvider` that provide access to an object of type
     * `INonDecomposableStatistics`, which uses dense data structures to store the statistics and can be converted into
     * an object of type `IDecomposableStatistics`.
     */
    class DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory final
        : public IClassificationStatisticsProviderFactory {
        private:

            const std::unique_ptr<INonDecomposableClassificationLossFactory<float64>> lossFactoryPtr_;

            const std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr_;

            const std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            const std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            const std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param lossFactoryPtr                    An unique pointer to an object of type
             *                                          `INonDecomposableClassificationLossFactory` that allows to
             *                                          create implementations of the loss function that should be used
             *                                          for calculating gradients and Hessians
             * @param evaluationMeasureFactoryPtr       An unique pointer to an object of type
             *                                          `IClassificationEvaluationMeasureFactory` that allows to create
             *                                          implementations of the evaluation measure that should be used
             *                                          for assessing the quality of predictions
             * @param defaultRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `INonDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param regularRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `IDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `IDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, when pruning rules
             * @param multiThreadingSettings            An object of type `MultiThreadingSettings` that stores the
             *                                          settings to be used for calculating the initial statistics in
             *                                          parallel
             */
            DenseConvertibleNonDecomposableClassificationStatisticsProviderFactory(
              std::unique_ptr<INonDecomposableClassificationLossFactory<float64>> lossFactoryPtr,
              std::unique_ptr<IClassificationEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr,
              std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
              std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
              std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
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

    /**
     * Allows to create instances of the class `IStatisticsProvider` that provide access to an object of type
     * `INonDecomposableStatistics`, which uses dense data structures to store the statistics and can be converted into
     * an object of type `IDecomposableStatistics`.
     */
    class DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory final
        : public IRegressionStatisticsProviderFactory {
        private:

            const std::unique_ptr<INonDecomposableRegressionLossFactory<float64>> lossFactoryPtr_;

            const std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr_;

            const std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr_;

            const std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr_;

            const std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr_;

            const MultiThreadingSettings multiThreadingSettings_;

        public:

            /**
             * @param lossFactoryPtr                    An unique pointer to an object of type
             *                                          `INonDecomposableRegressionLossFactory` that allows to create
             *                                          implementations of the loss function that should be used for
             *                                          calculating gradients and Hessians
             * @param evaluationMeasureFactoryPtr       An unique pointer to an object of type
             *                                          `IRegressionEvaluationMeasureFactory` that allows to create
             *                                          implementations of the evaluation measure that should be used
             *                                          for assessing the quality of predictions
             * @param defaultRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `INonDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of the default rule
             * @param regularRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `IDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, of all remaining rules
             * @param pruningRuleEvaluationFactoryPtr   An unique pointer to an object of type
             *                                          `IDecomposableRuleEvaluationFactory` that should be used for
             *                                          calculating the predictions, as well as corresponding quality
             *                                          scores, when pruning rules
             * @param multiThreadingSettings            An object of type `MultiThreadingSettings` that stores the
             *                                          settings to be used for calculating the initial statistics in
             *                                          parallel
             */
            DenseConvertibleNonDecomposableRegressionStatisticsProviderFactory(
              std::unique_ptr<INonDecomposableRegressionLossFactory<float64>> lossFactoryPtr,
              std::unique_ptr<IRegressionEvaluationMeasureFactory<float64>> evaluationMeasureFactoryPtr,
              std::unique_ptr<INonDecomposableRuleEvaluationFactory> defaultRuleEvaluationFactoryPtr,
              std::unique_ptr<IDecomposableRuleEvaluationFactory> regularRuleEvaluationFactoryPtr,
              std::unique_ptr<IDecomposableRuleEvaluationFactory> pruningRuleEvaluationFactoryPtr,
              MultiThreadingSettings multiThreadingSettings);

            /**
             * @see `IRegressionStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(
              const CContiguousView<const float32>& regressionMatrix) const override;

            /**
             * @see `IRegressionStatisticsProviderFactory::create`
             */
            std::unique_ptr<IStatisticsProvider> create(const CsrView<const float32>& regressionMatrix) const override;
    };

}
