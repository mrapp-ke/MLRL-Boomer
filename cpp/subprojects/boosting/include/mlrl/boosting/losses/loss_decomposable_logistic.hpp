/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss_decomposable.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a loss function that implements a multivariate variant of the logistic loss that is
     * decomposable.
     */
    class DecomposableLogisticLossConfig final : public IDecomposableClassificationLossConfig {
        private:

            const ReadableProperty<IHeadConfig> headConfig_;

        public:

            /**
             * @param headConfig A `ReadableProperty` that allows to access the `IHeadConfig` that stores the
             *                   configuration of rule heads
             */
            DecomposableLogisticLossConfig(ReadableProperty<IHeadConfig> headConfig);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack, bool preferSparseStatistics) const override;

            std::unique_ptr<IMarginalProbabilityFunctionFactory> createMarginalProbabilityFunctionFactory()
              const override;

            std::unique_ptr<IJointProbabilityFunctionFactory> createJointProbabilityFunctionFactory() const override;

            float64 getDefaultPrediction() const override;

            std::unique_ptr<IDecomposableClassificationLossFactory> createDecomposableClassificationLossFactory()
              const override;
    };

}
