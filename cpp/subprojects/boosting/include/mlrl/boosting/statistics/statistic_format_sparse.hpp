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
     * Allows to configure a sparse format for storing statistics about the quality of predictions for training
     * examples.
     */
    class SparseStatisticsConfig final : public IStatisticsConfig {
        private:

            const ReadableProperty<IClassificationLossConfig> lossConfig_;

        public:

            /**
             * @param lossConfigGetter A `ReadableProperty` that allows to access the `IClassificationLossConfig` that
             *                         stores the configuration of the loss function
             */
            SparseStatisticsConfig(ReadableProperty<IClassificationLossConfig> lossConfigGetter);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            bool isDense() const override;

            bool isSparse() const override;
    };

};
