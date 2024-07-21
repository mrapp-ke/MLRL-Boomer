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

namespace boosting {

    /**
     * Allows to configure a method that automatically decides for a format for storing statistics about the quality of
     * predictions for training examples in classification problems.
     */
    class AutomaticClassificationStatisticsConfig final : public IClassificationStatisticsConfig {
        private:

            const ReadableProperty<IClassificationLossConfig> lossConfigGetter_;

            const ReadableProperty<IHeadConfig> headConfig_;

            const ReadableProperty<IDefaultRuleConfig> defaultRuleConfig_;

        public:

            /**
             * @param lossConfigGetter        A `GetterFunction` that allows to access the `IClassificationLossConfig`
             *                                that stores the configuration of the loss function
             * @param headConfigGetter        A `GetterFunction` that allows to access the `IHeadConfig` that stores the
             *                                configuration of the rule heads
             * @param defaultRuleConfigGetter A `GetterFunction` that allows to access the `IDefaultRuleConfig` that
             *                                stores the configuration of the default rule
             */
            AutomaticClassificationStatisticsConfig(ReadableProperty<IClassificationLossConfig> lossConfigGetter,
                                                    ReadableProperty<IHeadConfig> headConfigGetter,
                                                    ReadableProperty<IDefaultRuleConfig> defaultRuleConfigGetter);

            std::unique_ptr<IClassificationStatisticsProviderFactory> createClassificationStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseLabelMatrix& labelMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            bool isDense() const override;

            bool isSparse() const override;
    };

    /**
     * Allows to configure a method that automatically decides for a format for storing statistics about the quality of
     * predictions for training examples in regression problems.
     */
    class AutomaticRegressionStatisticsConfig final : public IRegressionStatisticsConfig {
        private:

            const ReadableProperty<IRegressionLossConfig> lossConfigGetter_;

            const ReadableProperty<IHeadConfig> headConfigGetter_;

            const ReadableProperty<IDefaultRuleConfig> defaultRuleConfigGetter_;

        public:

            /**
             * @param lossConfigGetter        A `GetterFunction` that allows to access the `IRegressionLossConfig` that
             *                                stores the configuration of the loss function that should be used in
             *                                regression problems
             * @param headConfigGetter        A `GetterFunction` that allows to access the `IHeadConfig` that stores the
             *                                configuration of the rule heads
             * @param defaultRuleConfigGetter A `GetterFunction` that allows to access the `IDefaultRuleConfig` that
             *                                stores the configuration of the default rule
             */
            AutomaticRegressionStatisticsConfig(ReadableProperty<IRegressionLossConfig> lossConfigGetter,
                                                ReadableProperty<IHeadConfig> headConfigGetter,
                                                ReadableProperty<IDefaultRuleConfig> defaultRuleConfigGetter);

            std::unique_ptr<IRegressionStatisticsProviderFactory> createRegressionStatisticsProviderFactory(
              const IFeatureMatrix& featureMatrix, const IRowWiseRegressionMatrix& regressionMatrix, const Blas& blas,
              const Lapack& lapack) const override;

            bool isDense() const override;

            bool isSparse() const override;
    };

};
