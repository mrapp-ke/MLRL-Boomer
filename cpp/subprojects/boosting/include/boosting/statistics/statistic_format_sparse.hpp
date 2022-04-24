/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "boosting/statistics/statistic_format.hpp"
#include "boosting/losses/loss.hpp"


namespace boosting {

    class SparseStatisticsConfig : public IStatisticsConfig {

        private:

            const std::unique_ptr<ILossConfig>& lossConfigPtr_;

        public:

            /**
             * @param lossConfigPtr A reference to an unique pointer that stores the configuration of the loss function
             */
            SparseStatisticsConfig(const std::unique_ptr<ILossConfig>& lossConfigPtr);

            std::unique_ptr<IStatisticsProviderFactory> createStatisticsProviderFactory(
                const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
                const Lapack& lapack) const override;

    };

};
