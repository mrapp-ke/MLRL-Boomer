/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/boosting/statistics/statistic_format.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

namespace boosting {

    /**
     * Allows to configure a dense format for storing statistics about the quality of predictions for training examples
     * in classification problems.
     */
    class DenseClassificationStatisticsConfig final : public IClassificationStatisticsConfig {
        private:

            const GetterFunction<IClassificationLossConfig> lossConfigGetter_;

        public:

            /**
             * @param lossConfigGetter  A `GetterFunction` that allows to access the `IClassificationLossConfig` that
             *                          stores the configuration of the loss function that should be used in
             *                          classification problems
             * @param lossConfigGetter  A `GetterFunction` that allows to access the `IRegressionLossConfig` that stores
             *                          the configuration of the loss function that should be used in regression
             *                          problems
             */
            DenseClassificationStatisticsConfig(GetterFunction<IClassificationLossConfig> lossConfigGetter);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            bool isDense() const override;

            bool isSparse() const override;
    };

    /**
     * Allows to configure a dense format for storing statistics about the quality of predictions for training examples
     * in regression problems.
     */
    class DenseRegressionStatisticsConfig final : public IRegressionStatisticsConfig {
        private:

            const GetterFunction<IRegressionLossConfig> lossConfigGetter_;

        public:

            /**
             * @param lossConfigGetter  A `GetterFunction` that allows to access the `IRegressionLossConfig` that stores
             *                          the configuration of the loss function that should be used in regression
             *                          problems
             */
            DenseRegressionStatisticsConfig(GetterFunction<IRegressionLossConfig> lossConfigGetter);

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            bool isDense() const override;

            bool isSparse() const override;
    };

};
