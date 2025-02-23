/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_decomposable_sparse.hpp"
#include "mlrl/boosting/statistics/statistic_type.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a loss function that implements a multivariate variant of the squared hinge loss that is
     * decomposable.
     */
    class DecomposableSquaredHingeLossConfig final : public ISparseDecomposableClassificationLossConfig {
        private:

            const ReadableProperty<IStatisticTypeConfig> statisticTypeConfig_;

        public:

            /**
             *  @param statisticTypeConfig  A `ReadableProperty` that allows to access the `IStatisticTypeConfig` that
             *                              stores the configuration of the data type that should be used for
             *                              representing statistics about the quality of predictions for training
             *                              examples
             */
            DecomposableSquaredHingeLossConfig(ReadableProperty<IStatisticTypeConfig> statisticTypeConfig);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix,
              const BlasFactory& blasFactory, const LapackFactory& lapackFactory,
              bool preferSparseStatistics) const override;

            std::unique_ptr<IMarginalProbabilityFunctionFactory> createMarginalProbabilityFunctionFactory()
              const override;

            std::unique_ptr<IJointProbabilityFunctionFactory> createJointProbabilityFunctionFactory() const override;

            float64 getDefaultPrediction() const override;

            std::unique_ptr<ISparseDecomposableClassificationLossFactory<float64>>
              createSparseDecomposableClassificationLossFactory() const override;
    };

}
