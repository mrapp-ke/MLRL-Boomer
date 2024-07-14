/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_decomposable_sparse.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a loss function that implements a multivariate variant of the squared hinge loss that is
     * decomposable.
     */
    class DecomposableSquaredHingeLossConfig final : public ISparseDecomposableClassificationLossConfig {
        private:

            const ReadableProperty<IHeadConfig> headConfig_;

        public:

            /**
             * @param headConfigGetter A `ReadableProperty` that allows to access the `IHeadConfig` that stores the
             *                         configuration of rule heads
             */
            DecomposableSquaredHingeLossConfig(ReadableProperty<IHeadConfig> headConfigGetter);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack, bool preferSparseStatistics) const override;

            std::unique_ptr<IMarginalProbabilityFunctionFactory> createMarginalProbabilityFunctionFactory()
              const override;

            std::unique_ptr<IJointProbabilityFunctionFactory> createJointProbabilityFunctionFactory() const override;

            float64 getDefaultPrediction() const override;

            std::unique_ptr<ISparseDecomposableClassificationLossFactory>
              createSparseDecomposableClassificationLossFactory() const override;
    };

}
