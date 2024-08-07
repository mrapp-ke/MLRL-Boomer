/*
 * @author Michael Rapp (michael.rapp.ml@gmail.com)
 */
#pragma once

#include "mlrl/boosting/losses/loss.hpp"
#include "mlrl/boosting/rule_evaluation/head_type.hpp"
#include "mlrl/boosting/statistics/statistic_format.hpp"
#include "mlrl/common/rule_model_assemblage/default_rule.hpp"
#include "mlrl/common/util/properties.hpp"

#include <memory>

#include <memory>

namespace boosting {

    /**
     * Allows to configure a method that automatically decides for a format for storing statistics about the quality of
     * predictions for training examples.
     */
    class AutomaticStatisticsConfig final : public IClassificationStatisticsConfig,
                                            public IRegressionStatisticsConfig {
        private:

            const ReadableProperty<IClassificationLossConfig> classificationLossConfig_;

            const ReadableProperty<IRegressionLossConfig> regressionLossConfig_;

            const ReadableProperty<IHeadConfig> headConfig_;

            const ReadableProperty<IDefaultRuleConfig> defaultRuleConfig_;

        public:

            /**
             * @param classificationLossConfig  A `ReadableProperty` that allows to access the
             *                                  `IClassificationLossConfig` that stores the configuration of the loss
             *                                  function
             * @param regressionLossConfig      A `ReadableProperty` that allows to access the `IRegressionLossConfig`
             *                                  that stores the configuration of the loss function
             * @param headConfig                A `ReadableProperty` that allows to access the `IHeadConfig` that stores
             *                                  the configuration of the rule heads
             * @param defaultRuleConfig         A `ReadableProperty` that allows to access the `IDefaultRuleConfig` that
             *                                  stores the configuration of the default rule
             */
            AutomaticStatisticsConfig(ReadableProperty<IClassificationLossConfig> classificationLossConfig,
                                      ReadableProperty<IRegressionLossConfig> regressionLossConfig,
                                      ReadableProperty<IHeadConfig> headConfig,
                                      ReadableProperty<IDefaultRuleConfig> defaultRuleConfig);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            bool isDense() const override;

            bool isSparse() const override;
    };

}
